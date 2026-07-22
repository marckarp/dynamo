# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.v1.client import AllocationPools, Manager
from gpu_memory_service.v1.errors import FatalGMSError
from gpu_memory_service.v1.registry import Registry
from gpu_memory_service.v1.tests.fakes import VMM

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def gpu_identity(monkeypatch):
    monkeypatch.setattr(Manager, "_gpu_identity", lambda self: "GPU-0")


def manager():
    vmm = VMM()
    registry = Registry("GPU-0", vmm, 3)
    result = Manager(
        registry, vmm, 3, artifact_id="artifact", generation_id="generation"
    )
    return vmm, registry, result


def test_sleep_wake_preserves_parameter_and_replaces_private() -> None:
    vmm, _, captured = manager()
    parameter = captured.allocate_parameter(65)
    private = captured.allocate_private(33)
    before = {item.base: item for item in captured.mappings}
    captured.seal()
    captured.sleep()

    assert set(vmm.reservations) == {parameter, private}
    assert not vmm.mapped
    assert len(vmm.server_handles) == 1

    captured.wake("reader")
    after = {item.base: item for item in captured.mappings}
    assert after[parameter].allocation == before[parameter].allocation
    assert (
        after[private].allocation.allocation_id
        != before[private].allocation.allocation_id
    )
    assert vmm.access[parameter] is GrantedLockType.RO
    assert vmm.access[private] is GrantedLockType.RW
    assert all(event[1] == 3 for event in vmm.events if event[0] == "device")


def test_reservation_failure_unwinds_server_allocation() -> None:
    vmm, registry, captured = manager()
    vmm.fail_reserve = True

    with pytest.raises(RuntimeError, match="reserve failed"):
        captured.allocate_parameter(64)

    assert not vmm.server_handles
    assert not registry._generations["generation"].allocations


def test_cleanup_failure_is_fatal_but_still_frees_server() -> None:
    vmm, registry, captured = manager()
    vmm.fail_reserve = True
    vmm.fail_address_free = True  # reserve never succeeds, so this is unused
    original_free = captured.service.free

    def free_then_fail(owner, allocation):
        original_free(owner, allocation)
        raise RuntimeError("response lost")

    captured.service.free = free_then_fail
    with pytest.raises(FatalGMSError, match="cleanup lost ownership"):
        captured.allocate_parameter(64)

    assert not vmm.server_handles
    assert not registry._generations["generation"].allocations


def test_allocator_free_failure_latches_and_aborts_capture() -> None:
    vmm, _, captured = manager()
    base = captured.allocate_parameter(64)
    handle = captured._imports[base]
    vmm.fail_release.add(handle)
    callbacks = AllocationPools(captured)

    callbacks.free(base, 64, 3, 0)

    assert isinstance(callbacks.failure, FatalGMSError)
    with pytest.raises(FatalGMSError):
        captured.seal()


def test_whole_set_ro_failure_never_seals_or_publishes() -> None:
    vmm, registry, captured = manager()
    captured.allocate_parameter(64)
    captured.allocate_parameter(64)
    # Two initial RW transitions, then fail the second seal-time RO transition.
    vmm.fail_access_call = vmm.access_calls + 2

    with pytest.raises(FatalGMSError, match="read-only transition"):
        captured.seal()

    assert "generation" not in registry._generations
    assert captured._sealed is False


def test_sleep_failure_is_terminal_and_processes_all_mappings() -> None:
    vmm, registry, captured = manager()
    first = captured.allocate_parameter(64)
    second = captured.allocate_parameter(64)
    captured.allocate_private(64)
    captured.seal()
    vmm.fail_unmap.add(second)

    with pytest.raises(FatalGMSError, match="sleep cleanup"):
        captured.sleep()

    unmaps = [event[1] for event in vmm.events if event[0] == "unmap"]
    assert first in unmaps and second in unmaps
    assert "generation" not in registry._generations
    with pytest.raises(FatalGMSError):
        captured.wake("reader")


def test_wake_failure_cleans_reader_backing_and_reservations() -> None:
    vmm, registry, captured = manager()
    captured.allocate_parameter(64)
    captured.allocate_private(64)
    captured.seal()
    captured.sleep()
    vmm.fail_map_call = vmm.map_calls + 2

    with pytest.raises(FatalGMSError, match="wake failed"):
        captured.wake("reader")

    assert "generation" not in registry._generations
    assert not vmm.server_handles
    assert not vmm.reservations


def test_allocator_callbacks_reject_wrong_device() -> None:
    _, _, captured = manager()
    callbacks = AllocationPools(captured)

    with callbacks.parameter_pool():
        with pytest.raises(Exception, match="callback device"):
            callbacks.malloc(64, 2, 0)

    assert callbacks.failure is not None
