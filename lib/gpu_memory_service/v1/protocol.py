# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact allocation records checkpointed by snapshot-only GMS V1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class AccessClass(str, Enum):
    PARAMETER_RO = "parameter_ro"
    PRIVATE_RW = "private_rw"


@dataclass(frozen=True)
class Generation:
    generation_id: str
    gpu: str

    def __post_init__(self) -> None:
        if not self.generation_id or not self.gpu:
            raise ValueError("generation identity fields must not be empty")


@dataclass(frozen=True)
class Allocation:
    generation: Generation
    allocation_id: str
    requested_size: int
    aligned_size: int
    access: AccessClass

    def __post_init__(self) -> None:
        if not self.allocation_id:
            raise ValueError("allocation_id must not be empty")
        if self.requested_size <= 0 or self.aligned_size < self.requested_size:
            raise ValueError("invalid allocation size")


@dataclass(frozen=True)
class Mapping:
    """CRIU-preserved VA reservation for one allocator segment."""

    allocation: Allocation
    base: int
    reservation_size: int

    def __post_init__(self) -> None:
        if self.base <= 0:
            raise ValueError("mapping base must be positive")
        if self.reservation_size < self.allocation.aligned_size:
            raise ValueError("reservation does not cover allocation")

    @property
    def end(self) -> int:
        return self.base + self.allocation.aligned_size

    def with_allocation(self, allocation: Allocation) -> "Mapping":
        old = self.allocation
        if (
            allocation.generation != old.generation
            or allocation.requested_size != old.requested_size
            or allocation.aligned_size != old.aligned_size
            or allocation.access is not old.access
        ):
            raise ValueError("fresh allocation does not match preserved mapping")
        return replace(self, allocation=allocation)
