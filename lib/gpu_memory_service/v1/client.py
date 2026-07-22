# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot-resident exact allocation mapping manager."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol
from uuid import uuid4

from gpu_memory_service.common.locks import GrantedLockType

from .errors import FatalGMSError, GMSError
from .interfaces import AllocationService, VMM
from .protocol import AccessClass, Allocation, Generation, Mapping


class Manager:
    """Canonical VMM records restored in place by CRIU."""

    def __init__(
        self,
        service: AllocationService,
        vmm: VMM,
        device: int,
        *,
        artifact_id: str,
        generation_id: str | None = None,
    ):
        if not artifact_id:
            raise ValueError("artifact id must not be empty")
        self.service, self.vmm, self.device = service, vmm, device
        self.artifact_id = artifact_id
        vmm.ensure_initialized()
        self._granularity = int(vmm.get_allocation_granularity(device))
        self.generation = Generation(
            generation_id or f"generation-{uuid4()}", self._gpu_identity()
        )
        try:
            service.begin(self.generation)
        except (GMSError, ValueError):
            raise
        except Exception as cause:
            failures: list[Exception] = []
            self._cleanup(failures, service.retire, self.generation)
            if failures:
                raise FatalGMSError(
                    f"generation creation failed ({cause}) and cancellation "
                    f"failed: {failures[0]}"
                ) from cause
            raise
        self._mappings: dict[int, Mapping] = {}
        self._imports: dict[int, int] = {}
        self._unmapped_imports: set[int] = set()
        self.reader_id: str | None = None
        self._sealed = False
        self._fatal: FatalGMSError | None = None

    @property
    def mappings(self) -> tuple[Mapping, ...]:
        return tuple(self._mappings[base] for base in sorted(self._mappings))

    def allocate_parameter(self, size: int) -> int:
        return self._allocate(size, AccessClass.PARAMETER_RO)

    def allocate_private(self, size: int) -> int:
        return self._allocate(size, AccessClass.PRIVATE_RW)

    def _allocate(self, size: int, access: AccessClass) -> int:
        self._check_building()
        if size <= 0:
            raise ValueError("allocation size must be positive")
        allocation = Allocation(
            self.generation,
            f"allocation-{uuid4()}",
            size,
            self._align(size),
            access,
        )
        owner = self.generation.generation_id
        server_owned = reserved = False
        base = handle = 0
        try:
            try:
                self.service.allocate(owner, allocation)
                server_owned = True
            except (GMSError, MemoryError, ValueError):
                # A decoded server error means the mutation did not commit.
                raise
            except Exception:
                # A transport failure can happen after commit. The stable ID
                # makes an unconditional cleanup free safe.
                server_owned = True
                raise
            self._select_device()
            base = int(
                self.vmm.address_reserve(allocation.aligned_size, self._granularity)
            )
            reserved = True
            mapping = Mapping(allocation, base, allocation.aligned_size)
            fd = self.service.export(owner, allocation)
            handle = self._install(mapping, fd, GrantedLockType.RW)
        except Exception as cause:
            failures: list[Exception] = []
            if base in self._imports:
                self._drop_import(mapping, failures)
            if reserved:
                self._cleanup(
                    failures,
                    self.vmm.address_free,
                    base,
                    allocation.aligned_size,
                )
            if server_owned:
                self._cleanup(failures, self.service.free, owner, allocation)
            if failures or isinstance(cause, FatalGMSError):
                cleanup = failures[0] if failures else cause
                raise self._latch(
                    f"allocation failed ({cause}) and cleanup lost ownership",
                    cleanup,
                ) from cause
            raise
        self._mappings[base] = mapping
        self._imports[base] = handle
        return base

    def free(self, base: int) -> None:
        """Torch pool callback cleanup; any split ownership is terminal."""
        self._check_building()
        mapping = self._mappings.get(base)
        handle = self._imports.get(base)
        if mapping is None or handle is None:
            raise self._latch("allocator freed an unknown mapping")
        self._select_device()
        failures: list[Exception] = []
        dropped = self._drop_import(mapping, failures)
        if not dropped and base not in self._unmapped_imports:
            raise self._latch("allocator free cleanup failed", failures[0])
        self._cleanup(
            failures,
            self.service.free,
            self.generation.generation_id,
            mapping.allocation,
        )
        self._cleanup(failures, self.vmm.address_free, base, mapping.reservation_size)
        if failures:
            raise self._latch("allocator free cleanup failed", failures[0])
        del self._mappings[base]

    def seal(self) -> None:
        """Make the entire local parameter set RO before publishing the pin."""
        self._check_building()
        parameters = tuple(
            mapping
            for mapping in self.mappings
            if mapping.allocation.access is AccessClass.PARAMETER_RO
        )
        if not parameters:
            raise GMSError("generation has no parameter allocations")
        for mapping in self.mappings:
            if (
                self._imports.get(mapping.base) is None
                or mapping.allocation.generation != self.generation
                or mapping.reservation_size != mapping.allocation.aligned_size
            ):
                raise self._latch("local mapping preflight failed")

        self._select_device()
        changed: list[Mapping] = []
        try:
            for mapping in parameters:
                self.vmm.set_access(
                    mapping.base,
                    mapping.allocation.aligned_size,
                    self.device,
                    GrantedLockType.RO,
                )
                changed.append(mapping)
        except Exception as cause:
            # A partial protection transition is fail-stop even when the simple
            # best-effort RW restoration succeeds.
            for mapping in reversed(changed):
                try:
                    self.vmm.set_access(
                        mapping.base,
                        mapping.allocation.aligned_size,
                        self.device,
                        GrantedLockType.RW,
                    )
                except Exception:
                    pass
            self._abandon_capture(pinned=False)
            raise self._latch("parameter read-only transition failed", cause) from cause
        try:
            self.service.seal(self.generation, self.artifact_id)
        except Exception as cause:
            self._abandon_capture(pinned=True)
            raise self._latch("server seal response is unknown", cause) from cause
        self._sealed = True

    def sleep(self) -> None:
        """Unmap/release every unique canonical allocation exactly once."""
        self._check()
        if not self._sealed:
            raise GMSError("only a sealed generation may sleep")
        self._select_device()
        try:
            self.vmm.synchronize()
        except Exception as cause:
            self._abandon_capture()
            raise self._latch("capture synchronization failed", cause) from cause

        failures: list[Exception] = []
        for mapping in reversed(self.mappings):
            if mapping.base not in self._imports:
                failures.append(
                    GMSError(f"mapping 0x{mapping.base:x} has no imported handle")
                )
                continue
            if not self._drop_import(mapping, failures):
                continue
            if mapping.allocation.access is AccessClass.PRIVATE_RW:
                self._cleanup(
                    failures,
                    self.service.free,
                    self.generation.generation_id,
                    mapping.allocation,
                )
        if failures:
            self._abandon_capture(failures)
            raise self._latch("sleep cleanup failed", failures[0])

    def wake(self, reader_id: str) -> None:
        """Exact-ID attach, preserved parameter backing, fresh private backing."""
        self._check()
        if not self._sealed or self._imports or self.reader_id is not None:
            raise GMSError("snapshot manager is not fully asleep")
        if not reader_id:
            raise ValueError("reader id must not be empty")
        if self._gpu_identity() != self.generation.gpu:
            raise self._latch("restored process is on another physical GPU")
        parameters = tuple(
            mapping.allocation
            for mapping in self.mappings
            if mapping.allocation.access is AccessClass.PARAMETER_RO
        )
        try:
            self.service.attach(
                self.generation,
                self.artifact_id,
                reader_id,
                parameters,
            )
        except Exception as cause:
            failures: list[Exception] = []
            self._cleanup(failures, self.service.detach, self.generation, reader_id)
            self._abandon_capture(failures)
            raise self._latch("restore preflight failed", cause) from cause

        installed: list[tuple[Mapping, int]] = []
        fresh: list[Allocation] = []
        updated = dict(self._mappings)
        try:
            self._select_device()
            for original in self.mappings:
                mapping = original
                if original.allocation.access is AccessClass.PRIVATE_RW:
                    allocation = Allocation(
                        self.generation,
                        f"allocation-{uuid4()}",
                        original.allocation.requested_size,
                        original.allocation.aligned_size,
                        AccessClass.PRIVATE_RW,
                    )
                    fresh.append(allocation)
                    self.service.allocate(reader_id, allocation)
                    mapping = original.with_allocation(allocation)
                fd = self.service.export(reader_id, mapping.allocation)
                handle = self._install(
                    mapping,
                    fd,
                    GrantedLockType.RO
                    if mapping.allocation.access is AccessClass.PARAMETER_RO
                    else GrantedLockType.RW,
                )
                installed.append((mapping, handle))
                self._imports[mapping.base] = handle
                updated[mapping.base] = mapping
        except Exception as cause:
            rollback_failures: list[Exception] = []
            for mapping, _handle in reversed(installed):
                self._drop_import(mapping, rollback_failures)
            for allocation in reversed(fresh):
                self._cleanup(
                    rollback_failures, self.service.free, reader_id, allocation
                )
            self._cleanup(
                rollback_failures,
                self.service.detach,
                self.generation,
                reader_id,
            )
            self._abandon_capture(rollback_failures)
            detail = rollback_failures[0] if rollback_failures else cause
            raise self._latch(f"wake failed: {cause}", detail) from cause
        self._mappings = updated
        self.reader_id = reader_id

    def retire(self) -> None:
        """Release local restore imports, reader, artifact, then generation."""
        self._check()
        self._select_device()
        failures: list[Exception] = []
        try:
            self.vmm.synchronize()
        except Exception as exc:
            failures.append(exc)
        for mapping in reversed(self.mappings):
            if mapping.base not in self._imports:
                continue
            self._drop_import(mapping, failures)
        if self.reader_id is not None:
            self._cleanup(
                failures,
                self.service.detach,
                self.generation,
                self.reader_id,
            )
            self.reader_id = None
        self._cleanup(
            failures,
            self.service.release_artifact,
            self.generation,
            self.artifact_id,
        )
        self._cleanup(failures, self.service.retire, self.generation)
        for mapping in reversed(self.mappings):
            self._cleanup(
                failures,
                self.vmm.address_free,
                mapping.base,
                mapping.reservation_size,
            )
        if failures:
            raise self._latch("generation retirement failed", failures[0])
        self._mappings.clear()

    def _abandon_capture(
        self,
        failures: list[Exception] | None = None,
        *,
        pinned: bool = True,
    ) -> None:
        failures = failures if failures is not None else []
        for mapping in reversed(self.mappings):
            if mapping.base not in self._imports:
                continue
            self._drop_import(mapping, failures)
        if pinned:
            self._cleanup(
                failures,
                self.service.release_artifact,
                self.generation,
                self.artifact_id,
            )
        self._cleanup(failures, self.service.retire, self.generation)
        for mapping in reversed(self.mappings):
            self._cleanup(
                failures,
                self.vmm.address_free,
                mapping.base,
                mapping.reservation_size,
            )

    def _gpu_identity(self) -> str:
        import torch

        return str(torch.cuda.get_device_properties(self.device).uuid)

    def _install(self, mapping: Mapping, fd: int, protection: GrantedLockType) -> int:
        """Consume one export FD and install its exact mapping."""
        handle = 0
        mapped = False
        try:
            # Concrete VMMDevice imports close fd on success and failure.
            handle = int(self.vmm.import_shareable_handle_close_fd(fd))
            self.vmm.map(mapping.base, mapping.allocation.aligned_size, handle)
            mapped = True
            self.vmm.set_access(
                mapping.base,
                mapping.allocation.aligned_size,
                self.device,
                protection,
            )
            return handle
        except Exception:
            failures: list[Exception] = []
            if mapped:
                try:
                    self.vmm.unmap(mapping.base, mapping.allocation.aligned_size)
                except Exception as exc:
                    failures.append(exc)
                else:
                    mapped = False
            if handle and not mapped:
                try:
                    self.vmm.release(handle)
                except Exception as exc:
                    failures.append(exc)
            if failures and handle:
                self._imports[mapping.base] = handle
                if not mapped:
                    self._unmapped_imports.add(mapping.base)
            if failures:
                # The caller owns the outer transition and gets one more
                # deterministic cleanup pass using the retained exact state.
                raise
            raise

    def _select_device(self) -> None:
        self.vmm.runtime_set_device(self.device)

    def _drop_import(self, mapping: Mapping, failures: list[Exception]) -> bool:
        """Best-effort local import cleanup without splitting map ownership."""
        base = mapping.base
        handle = self._imports[base]
        if base not in self._unmapped_imports:
            try:
                self.vmm.unmap(base, mapping.allocation.aligned_size)
            except Exception as exc:
                failures.append(exc)
                return False
            self._unmapped_imports.add(base)
        try:
            self.vmm.release(handle)
        except Exception as exc:
            failures.append(exc)
            return False
        del self._imports[base]
        self._unmapped_imports.discard(base)
        return True

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _check_building(self) -> None:
        self._check()
        if self._sealed:
            raise GMSError("sealed pools cannot allocate or free segments")

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _latch(self, message: str, cause: Exception | None = None) -> FatalGMSError:
        if self._fatal is None:
            suffix = f": {cause}" if cause is not None else ""
            self._fatal = FatalGMSError(message + suffix)
        return self._fatal

    @staticmethod
    def _cleanup(failures: list[Exception], operation, *args: object) -> None:
        try:
            operation(*args)
        except Exception as exc:
            failures.append(exc)


_active: ContextVar[AccessClass | None] = ContextVar("gms_v1_pool", default=None)


class _PoolManager(Protocol):
    device: int
    _mappings: dict[int, Mapping]

    def allocate_parameter(self, size: int) -> int: ...

    def allocate_private(self, size: int) -> int: ...

    def free(self, base: int) -> None: ...


class AllocationPools:
    """Torch callback routing for the two exact allocation domains."""

    def __init__(self, manager: _PoolManager):
        self.manager = manager
        self.failure: Exception | None = None

    @contextmanager
    def parameter_pool(self):
        token = _active.set(AccessClass.PARAMETER_RO)
        try:
            yield
        finally:
            _active.reset(token)

    @contextmanager
    def private_pool(self):
        token = _active.set(AccessClass.PRIVATE_RW)
        try:
            yield
        finally:
            _active.reset(token)

    def malloc(self, size: int, device: int, _stream: int) -> int:
        try:
            if device != self.manager.device:
                raise GMSError(
                    f"allocator callback device {device} != {self.manager.device}"
                )
            access = _active.get()
            if access is None:
                raise GMSError("allocation occurred outside a V1 pool scope")
            if access is AccessClass.PARAMETER_RO:
                return self.manager.allocate_parameter(size)
            return self.manager.allocate_private(size)
        except Exception as exc:
            self.failure = self.failure or exc
            raise

    def free(self, base: int, size: int, device: int, _stream: int) -> None:
        try:
            if device != self.manager.device:
                raise GMSError(
                    f"allocator callback device {device} != {self.manager.device}"
                )
            mapping = self.manager._mappings.get(base)
            if mapping is None or size != mapping.allocation.requested_size:
                raise GMSError("allocator free does not match exact mapping")
            self.manager.free(base)
        except Exception as exc:
            # CUDAPluggableAllocator's free ABI returns void. Capture checks
            # this terminal latch immediately after pool-scoped eviction.
            self.failure = self.failure or exc
