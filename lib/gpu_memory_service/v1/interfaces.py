# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural interfaces consumed by snapshot-only GMS V1."""

from __future__ import annotations

from typing import Protocol

from gpu_memory_service.common.locks import GrantedLockType

from .protocol import Allocation, Generation


class VMM(Protocol):
    def ensure_initialized(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def get_allocation_granularity(self, device: int) -> int:
        pass

    def create_tolerate_oom(self, size: int, device: int) -> tuple[bool, int]:
        pass

    def release(self, handle: int) -> None:
        pass

    def export_to_shareable_handle(self, handle: int) -> int:
        pass

    def import_shareable_handle_close_fd(self, fd: int) -> int:
        pass

    def address_reserve(self, size: int, granularity: int) -> int:
        pass

    def address_free(self, va: int, size: int) -> None:
        pass

    def map(self, va: int, size: int, handle: int) -> None:
        pass

    def unmap(self, va: int, size: int) -> None:
        pass

    def set_access(
        self, va: int, size: int, device: int, access: GrantedLockType
    ) -> None:
        pass

    def runtime_set_device(self, device: int) -> None:
        pass


class AllocationService(Protocol):
    def begin(self, generation: Generation) -> None:
        pass

    def allocate(self, owner: str, allocation: Allocation) -> None:
        pass

    def free(self, owner: str, allocation: Allocation) -> None:
        pass

    def export(self, owner: str, allocation: Allocation) -> int:
        pass

    def seal(self, generation: Generation, artifact_id: str) -> None:
        pass

    def attach(
        self,
        generation: Generation,
        artifact_id: str,
        reader_id: str,
        parameters: tuple[Allocation, ...],
    ) -> None:
        pass

    def detach(self, generation: Generation, reader_id: str) -> None:
        pass

    def release_artifact(self, generation: Generation, artifact_id: str) -> None:
        pass

    def retire(self, generation: Generation) -> None:
        pass
