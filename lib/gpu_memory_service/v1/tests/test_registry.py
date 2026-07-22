# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_memory_service.v1.errors import FatalGMSError, GMSError
from gpu_memory_service.v1.protocol import AccessClass, Allocation, Generation
from gpu_memory_service.v1.registry import Registry
from gpu_memory_service.v1.tests.fakes import VMM

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

GENERATION = Generation("generation", "GPU-0")


def allocation(name="parameter", access=AccessClass.PARAMETER_RO):
    return Allocation(GENERATION, name, 65, 128, access)


def test_mutations_are_idempotent_with_caller_ids() -> None:
    vmm = VMM()
    registry = Registry("GPU-0", vmm, 0)
    parameter = allocation()

    registry.begin(GENERATION)
    registry.begin(GENERATION)
    registry.allocate("generation", parameter)
    registry.allocate("generation", parameter)
    assert len(vmm.server_handles) == 1

    registry.seal(GENERATION, "artifact")
    registry.seal(GENERATION, "artifact")
    registry.attach(GENERATION, "artifact", "reader", (parameter,))
    registry.attach(GENERATION, "artifact", "reader", (parameter,))

    private = allocation("private", AccessClass.PRIVATE_RW)
    registry.allocate("reader", private)
    registry.allocate("reader", private)
    registry.free("reader", private)
    registry.free("reader", private)
    registry.detach(GENERATION, "reader")
    registry.detach(GENERATION, "reader")
    registry.release_artifact(GENERATION, "artifact")
    registry.release_artifact(GENERATION, "artifact")
    registry.retire(GENERATION)
    registry.retire(GENERATION)
    assert not vmm.server_handles


def test_attach_requires_exact_complete_parameter_set() -> None:
    registry = Registry("GPU-0", VMM(), 0)
    first = allocation("first")
    second = allocation("second")
    registry.begin(GENERATION)
    registry.allocate("generation", first)
    registry.allocate("generation", second)
    registry.seal(GENERATION, "artifact")

    with pytest.raises(GMSError, match="complete parameter set"):
        registry.attach(GENERATION, "artifact", "reader", (first,))
    with pytest.raises(GMSError, match="exact allocation"):
        registry.attach(
            GENERATION,
            "artifact",
            "other",
            (first, replace(second, allocation_id="wrong")),
        )


def test_wrong_or_restarted_server_exact_ids_fail() -> None:
    registry = Registry("GPU-0", VMM(), 0)
    registry.begin(GENERATION)

    with pytest.raises(GMSError, match="GPU"):
        registry.allocate(
            "generation",
            replace(allocation(), generation=Generation("generation", "GPU-X")),
        )
    restarted = Registry("GPU-0", VMM(), 0)
    with pytest.raises(GMSError, match="unknown generation"):
        restarted.allocate("generation", allocation())


def test_server_cleanup_attempts_release_and_latches_fatal() -> None:
    vmm = VMM()
    registry = Registry("GPU-0", vmm, 0)
    item = allocation("private", AccessClass.PRIVATE_RW)
    registry.begin(GENERATION)
    registry.allocate("generation", item)
    handle = next(iter(vmm.server_handles))
    vmm.fail_release.add(handle)

    with pytest.raises(FatalGMSError, match="server allocation cleanup"):
        registry.free("generation", item)

    assert ("release", handle) in vmm.events
    with pytest.raises(FatalGMSError):
        registry.begin(Generation("other", "GPU-0"))


def test_failed_retire_replays_same_fatal_result() -> None:
    vmm = VMM()
    registry = Registry("GPU-0", vmm, 0)
    item = allocation()
    registry.begin(GENERATION)
    registry.allocate("generation", item)
    handle = next(iter(vmm.server_handles))
    vmm.fail_release.add(handle)

    with pytest.raises(FatalGMSError) as first:
        registry.retire(GENERATION)
    with pytest.raises(FatalGMSError) as replay:
        registry.retire(GENERATION)

    assert replay.value is first.value
    assert handle in vmm.server_handles
    assert GENERATION.generation_id in registry._retired
