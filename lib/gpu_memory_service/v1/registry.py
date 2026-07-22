# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent exact-ID allocation ownership for snapshot-only GMS V1."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from .errors import FatalGMSError, GMSError
from .interfaces import VMM
from .protocol import AccessClass, Allocation, Generation


@dataclass(frozen=True)
class _Owned:
    allocation: Allocation
    handle: int
    fd: int
    owner: str


@dataclass
class _Generation:
    identity: Generation
    sealed_artifact: str | None = None
    pinned: bool = False
    allocations: dict[str, _Owned] = field(default_factory=dict)
    freed: dict[str, tuple[Allocation, str]] = field(default_factory=dict)
    readers: dict[str, tuple[Allocation, ...]] = field(default_factory=dict)
    detached_readers: set[str] = field(default_factory=set)


class Registry:
    """One same-server, same-GPU allocation registry."""

    def __init__(self, gpu: str, vmm: VMM, device: int):
        if not gpu:
            raise ValueError("gpu identity must not be empty")
        self.gpu, self.vmm, self.device = gpu, vmm, device
        self.process = (os.getpid(), time.monotonic_ns())
        vmm.ensure_initialized()
        self._granularity = int(vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._generations: dict[str, _Generation] = {}
        self._retired: set[str] = set()
        self._fatal: FatalGMSError | None = None
        self._lock = threading.Lock()

    def process_evidence(self) -> tuple[int, int]:
        return self.process

    def begin(self, generation: Generation) -> None:
        with self._lock:
            self._check()
            if generation.gpu != self.gpu:
                raise GMSError("generation names another physical GPU")
            if generation.generation_id in self._retired:
                raise GMSError("retired generation ID cannot be reused")
            existing = self._generations.get(generation.generation_id)
            if existing is not None:
                if existing.identity != generation:
                    raise GMSError("generation identity mismatch")
                return
            self._generations[generation.generation_id] = _Generation(generation)

    def allocate(self, owner: str, expected: Allocation) -> None:
        with self._lock:
            self._check()
            generation = self._owner(expected.generation, owner)
            if expected.aligned_size != self._align(expected.requested_size):
                raise GMSError("allocation alignment does not match this server")
            if owner == generation.identity.generation_id:
                if generation.sealed_artifact is not None:
                    raise GMSError("sealed generations cannot allocate")
            elif expected.access is AccessClass.PARAMETER_RO:
                raise GMSError("readers cannot allocate parameters")
            existing = generation.allocations.get(expected.allocation_id)
            if existing is not None:
                if existing.allocation != expected or existing.owner != owner:
                    raise GMSError("allocation ID was reused with different metadata")
                return
            if expected.allocation_id in generation.freed:
                raise GMSError("freed allocation ID cannot be reused")

            allocated, handle = self.vmm.create_tolerate_oom(
                expected.aligned_size, self.device
            )
            if not allocated:
                raise MemoryError(f"cannot allocate {expected.aligned_size} GPU bytes")
            try:
                fd = int(self.vmm.export_to_shareable_handle(int(handle)))
            except Exception as cause:
                try:
                    self.vmm.release(int(handle))
                except Exception as cleanup:
                    raise self._latch(
                        "allocation export cleanup failed", cleanup
                    ) from cause
                raise
            generation.allocations[expected.allocation_id] = _Owned(
                expected, int(handle), fd, owner
            )

    def free(self, owner: str, expected: Allocation) -> None:
        """Free an exact allocation; repeated response-loss retries are no-ops."""
        with self._lock:
            self._check()
            generation = self._owner(expected.generation, owner)
            owned = generation.allocations.get(expected.allocation_id)
            if owned is None:
                prior = generation.freed.get(expected.allocation_id)
                if prior is not None and prior != (expected, owner):
                    raise GMSError("freed allocation metadata does not match")
                return
            if owned.allocation != expected:
                raise GMSError("exact allocation ID/class/sizes do not match")
            if owned.owner != owner:
                raise GMSError("private allocation belongs to another owner")
            if (
                expected.access is AccessClass.PARAMETER_RO
                and generation.sealed_artifact is not None
            ):
                raise GMSError("sealed parameter allocations cannot be freed")
            del generation.allocations[expected.allocation_id]
            generation.freed[expected.allocation_id] = (expected, owner)
            self._destroy(owned)

    def export(self, owner: str, expected: Allocation) -> int:
        with self._lock:
            self._check()
            generation = self._owner(expected.generation, owner)
            owned = self._exact(generation, expected)
            if owner == generation.identity.generation_id:
                if generation.sealed_artifact is not None:
                    raise GMSError("writer export is BUILDING-only")
            elif expected.access is AccessClass.PARAMETER_RO:
                if generation.sealed_artifact is None:
                    raise GMSError("parameter generation is not sealed")
            elif owned.owner != owner:
                raise GMSError("private allocation belongs to another reader")
            return os.dup(owned.fd)

    def seal(self, generation_id: Generation, artifact_id: str) -> None:
        if not artifact_id:
            raise ValueError("artifact id must not be empty")
        with self._lock:
            self._check()
            generation = self._generation(generation_id)
            if generation.sealed_artifact is not None:
                if generation.sealed_artifact != artifact_id:
                    raise GMSError("generation is sealed for another artifact")
                generation.pinned = True
                return
            if not any(
                item.allocation.access is AccessClass.PARAMETER_RO
                for item in generation.allocations.values()
            ):
                raise GMSError("generation has no parameter allocations")
            generation.sealed_artifact = artifact_id
            generation.pinned = True

    def attach(
        self,
        generation_id: Generation,
        artifact_id: str,
        reader_id: str,
        parameters: tuple[Allocation, ...],
    ) -> None:
        """Preflight the complete exact parameter set before any FD export."""
        if not reader_id:
            raise ValueError("reader id must not be empty")
        with self._lock:
            self._check()
            generation = self._generation(generation_id)
            if generation.sealed_artifact != artifact_id or not generation.pinned:
                raise GMSError("artifact does not pin this generation")
            existing = generation.readers.get(reader_id)
            if existing is not None:
                if existing != parameters:
                    raise GMSError("reader ID was reused with another parameter set")
                return
            if reader_id in generation.detached_readers:
                raise GMSError("detached reader ID cannot be reused")
            seen: set[str] = set()
            for expected in parameters:
                if expected.allocation_id in seen:
                    raise GMSError("duplicate allocation id in preflight")
                seen.add(expected.allocation_id)
                owned = self._exact(generation, expected)
                if owned.allocation.access is not AccessClass.PARAMETER_RO:
                    raise GMSError("restore set contains a non-parameter allocation")
            authoritative = {
                allocation_id
                for allocation_id, item in generation.allocations.items()
                if item.allocation.access is AccessClass.PARAMETER_RO
            }
            if seen != authoritative:
                raise GMSError("preflight set is not the complete parameter set")
            generation.readers[reader_id] = parameters

    def detach(self, generation_id: Generation, reader_id: str) -> None:
        with self._lock:
            self._check()
            generation = self._generation(generation_id)
            if reader_id in generation.detached_readers:
                return
            if reader_id not in generation.readers:
                generation.detached_readers.add(reader_id)
                return
            private = [
                allocation_id
                for allocation_id, item in generation.allocations.items()
                if item.owner == reader_id
            ]
            first: Exception | None = None
            for allocation_id in private:
                owned = generation.allocations.pop(allocation_id)
                generation.freed[allocation_id] = (owned.allocation, reader_id)
                try:
                    self._destroy(owned)
                except Exception as exc:
                    first = first or exc
            del generation.readers[reader_id]
            generation.detached_readers.add(reader_id)
            if first is not None:
                raise first

    def release_artifact(self, generation_id: Generation, artifact_id: str) -> None:
        with self._lock:
            self._check()
            generation = self._generation(generation_id)
            if generation.sealed_artifact != artifact_id:
                raise GMSError("artifact does not name this generation")
            generation.pinned = False

    def retire(self, generation_id: Generation) -> None:
        with self._lock:
            self._check()
            if generation_id.generation_id in self._retired:
                return
            generation = self._generation(generation_id)
            if generation.pinned or generation.readers:
                raise GMSError("generation still has an artifact or readers")
            del self._generations[generation_id.generation_id]
            self._retired.add(generation_id.generation_id)
            first: Exception | None = None
            for owned in generation.allocations.values():
                try:
                    self._destroy(owned)
                except Exception as exc:
                    first = first or exc
            if first is not None:
                raise first

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _latch(self, message: str, cause: Exception) -> FatalGMSError:
        if self._fatal is None:
            self._fatal = FatalGMSError(f"{message}: {cause}")
        return self._fatal

    def _generation(self, expected: Generation) -> _Generation:
        if expected.gpu != self.gpu:
            raise GMSError("server GPU identity mismatch")
        generation = self._generations.get(expected.generation_id)
        if generation is None or generation.identity != expected:
            raise GMSError("unknown generation")
        return generation

    def _owner(self, generation_id: Generation, owner: str) -> _Generation:
        generation = self._generation(generation_id)
        if (
            owner != generation.identity.generation_id
            and owner not in generation.readers
        ):
            raise GMSError("unknown allocation owner")
        return generation

    @staticmethod
    def _exact(generation: _Generation, expected: Allocation) -> _Owned:
        if expected.generation != generation.identity:
            raise GMSError("allocation names another generation")
        owned = generation.allocations.get(expected.allocation_id)
        if owned is None or owned.allocation != expected:
            raise GMSError("exact allocation ID/class/sizes do not match")
        return owned

    def _destroy(self, owned: _Owned) -> None:
        first: Exception | None = None
        try:
            os.close(owned.fd)
        except Exception as exc:
            first = exc
        try:
            self.vmm.release(owned.handle)
        except Exception as exc:
            first = first or exc
        if first is not None:
            raise self._latch("server allocation cleanup failed", first)
