# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import socketserver
import struct
import threading
from typing import cast

import pytest

from gpu_memory_service.v1.client import Manager
from gpu_memory_service.v1.errors import FatalGMSError, GMSError
from gpu_memory_service.v1.protocol import AccessClass, Allocation, Generation
from gpu_memory_service.v1.registry import Registry
from gpu_memory_service.v1.rpc import (
    RPCClient,
    RPCServer,
    _Handler,
    _receive,
    _send,
)
from gpu_memory_service.v1.tests.fakes import VMM

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]

GENERATION = Generation("generation", "GPU-0")


def parameter():
    return Allocation(GENERATION, "parameter", 65, 128, AccessClass.PARAMETER_RO)


def test_uds_passes_fd_and_runs_exact_id_lifecycle(tmp_path) -> None:
    path = str(tmp_path / "gms-v1.sock")
    with RPCServer(path, Registry("GPU-0", VMM(), 0)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RPCClient(path)
            item = parameter()
            client.begin(GENERATION)
            client.allocate("generation", item)
            fd = client.export("generation", item)
            os.fstat(fd)
            os.close(fd)
            client.seal(GENERATION, "artifact")
            client.attach(GENERATION, "artifact", "reader", (item,))
            fd = client.export("reader", item)
            os.fstat(fd)
            os.close(fd)
            client.detach(GENERATION, "reader")
            client.release_artifact(GENERATION, "artifact")
            client.retire(GENERATION)
            client.close()
        finally:
            server.shutdown()
            thread.join()


class _DropFirstResponse(_Handler):
    def handle(self):
        server = cast(_DropServer, self.server)
        request, received_fd = _receive(self.request)
        assert received_fd < 0
        result, export_fd = server.dispatch(request)
        assert export_fd < 0
        if server.drop:
            server.drop = False
            return
        _send(self.request, [True, result])
        super().handle()


class _DropServer(RPCServer):
    def __init__(self, path, registry):
        self.path, self.registry, self.drop = path, registry, True
        socketserver.ThreadingUnixStreamServer.__init__(self, path, _DropFirstResponse)
        os.chmod(path, 0o600)


class _DropRetireErrorResponse(_Handler):
    def handle(self):
        server = cast(_DropRetireErrorServer, self.server)
        while True:
            try:
                request, received_fd = _receive(self.request)
            except EOFError:
                return
            assert received_fd < 0
            export_fd = -1
            try:
                result, export_fd = server.dispatch(request)
                response = [True, result]
            except Exception as exc:
                response = [False, type(exc).__name__, str(exc)]
                server.errors.append(exc)
                if (
                    isinstance(request, list)
                    and request
                    and request[0] == "retire"
                    and server.drop
                ):
                    server.drop = False
                    return
            try:
                _send(self.request, response, export_fd)
            finally:
                if export_fd >= 0:
                    os.close(export_fd)


class _DropRetireErrorServer(RPCServer):
    def __init__(self, path, registry):
        self.path, self.registry, self.drop, self.errors = path, registry, True, []
        socketserver.ThreadingUnixStreamServer.__init__(
            self, path, _DropRetireErrorResponse
        )
        os.chmod(path, 0o600)


def test_response_loss_reconnects_and_replays_stable_mutation(tmp_path) -> None:
    path = str(tmp_path / "gms-v1.sock")
    registry = Registry("GPU-0", VMM(), 0)
    with _DropServer(path, registry) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RPCClient(path)
            client.begin(GENERATION)
            item = parameter()
            for operation in (
                lambda current: current.allocate("generation", item),
                lambda current: current.seal(GENERATION, "artifact"),
                lambda current: current.attach(
                    GENERATION, "artifact", "reader", (item,)
                ),
            ):
                client.close()
                server.drop = True
                client = RPCClient(path)
                operation(client)
            assert tuple(registry._generations) == ("generation",)
            owned = registry._generations["generation"]
            assert tuple(owned.allocations) == ("parameter",)
            assert tuple(owned.readers) == ("reader",)
            client.detach(GENERATION, "reader")
            client.release_artifact(GENERATION, "artifact")
            client.retire(GENERATION)
            client.close()
        finally:
            server.shutdown()
            thread.join()


def test_failed_retire_response_loss_replays_fatal_and_keeps_manager_ownership(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(Manager, "_gpu_identity", lambda self: "GPU-0")
    path = str(tmp_path / "gms-v1.sock")
    vmm = VMM()
    registry = Registry("GPU-0", vmm, 0)
    with _DropRetireErrorServer(path, registry) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = RPCClient(path)
        try:
            manager = Manager(
                client,
                vmm,
                0,
                artifact_id="artifact",
                generation_id="generation",
            )
            manager.allocate_parameter(64)
            manager.seal()
            before = manager.mappings
            server_handle = next(iter(vmm.server_handles))
            vmm.fail_release.add(server_handle)

            with pytest.raises(
                FatalGMSError, match="server allocation cleanup failed"
            ) as failure:
                manager.retire()

            assert manager._fatal is failure.value
            assert manager.mappings == before
            assert server_handle in vmm.server_handles
            assert registry._fatal is not None
            assert server.errors == [registry._fatal, registry._fatal]
            assert str(registry._fatal) in str(failure.value)
        finally:
            client.close()
            server.shutdown()
            thread.join()


def test_partial_header_eof_does_not_spin() -> None:
    sender, receiver = socket.socketpair()
    sender.sendall(b"\x00")
    sender.close()
    try:
        with pytest.raises(EOFError):
            _receive(receiver)
    finally:
        receiver.close()


def test_extra_rights_fds_are_all_closed(monkeypatch) -> None:
    sender, receiver = socket.socketpair()
    source = [os.open("/dev/null", os.O_RDONLY) for _ in range(2)]
    closed: list[int] = []
    real_close = os.close

    def record_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr("gpu_memory_service.v1.rpc.os.close", record_close)
    try:
        frame = struct.pack("!I", 2) + b"[]"
        sender.sendmsg(
            [frame],
            [
                (
                    socket.SOL_SOCKET,
                    socket.SCM_RIGHTS,
                    struct.pack("2i", *source),
                )
            ],
        )
        with pytest.raises(GMSError, match="multiple"):
            _receive(receiver)
        assert len(closed) == 2
    finally:
        sender.close()
        receiver.close()
        for fd in source:
            real_close(fd)


def test_truncated_rights_are_rejected_and_received_fds_closed(monkeypatch) -> None:
    sender, receiver = socket.socketpair()
    source = [os.open("/dev/null", os.O_RDONLY) for _ in range(32)]
    closed: list[int] = []
    real_close = os.close

    def record_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr("gpu_memory_service.v1.rpc.os.close", record_close)
    try:
        frame = struct.pack("!I", 2) + b"[]"
        sender.sendmsg(
            [frame],
            [
                (
                    socket.SOL_SOCKET,
                    socket.SCM_RIGHTS,
                    struct.pack(f"{len(source)}i", *source),
                )
            ],
        )
        with pytest.raises(GMSError, match="truncated"):
            _receive(receiver)
        assert closed
    finally:
        sender.close()
        receiver.close()
        for fd in source:
            real_close(fd)
