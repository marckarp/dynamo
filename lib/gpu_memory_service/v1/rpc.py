# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded JSON frames and SCM_RIGHTS for snapshot-only GMS V1."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import struct
import threading
from pathlib import Path

from .errors import GMSError
from .protocol import AccessClass, Allocation, Generation
from .registry import Registry

_MAX_FRAME = 1 << 20
_INT_SIZE = struct.calcsize("i")
_ANCILLARY_SIZE = socket.CMSG_SPACE(16 * _INT_SIZE)


def _generation(value: list) -> Generation:
    return Generation(str(value[0]), str(value[1]))


def _allocation(value: list) -> Allocation:
    return Allocation(
        _generation(value[:2]),
        str(value[2]),
        int(value[3]),
        int(value[4]),
        AccessClass(str(value[5])),
    )


def _gen_wire(value: Generation) -> list[object]:
    return [value.generation_id, value.gpu]


def _allocation_wire(value: Allocation) -> list[object]:
    return [
        value.generation.generation_id,
        value.generation.gpu,
        value.allocation_id,
        value.requested_size,
        value.aligned_size,
        value.access.value,
    ]


def _send(sock: socket.socket, value: object, fd: int = -1) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > _MAX_FRAME:
        raise GMSError("V1 RPC frame is too large")
    frame = struct.pack("!I", len(payload)) + payload
    if fd < 0:
        sock.sendall(frame)
        return
    sent = sock.sendmsg(
        [frame],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
    )
    if sent <= 0:
        raise ConnectionError("V1 RPC sendmsg made no progress")
    if sent < len(frame):
        sock.sendall(frame[sent:])


def _receive(sock: socket.socket) -> tuple[object, int]:
    received_fds: list[int] = []

    def read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk, ancillary, flags, _ = sock.recvmsg(size - len(data), _ANCILLARY_SIZE)
            for level, kind, raw in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    if len(raw) % _INT_SIZE:
                        raise GMSError("malformed V1 RPC file descriptor data")
                    count = len(raw) // _INT_SIZE
                    if count:
                        received_fds.extend(
                            struct.unpack(f"{count}i", raw[: count * _INT_SIZE])
                        )
            if flags & socket.MSG_CTRUNC:
                raise GMSError("V1 RPC ancillary data was truncated")
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    try:
        header = read_exact(4)
        (length,) = struct.unpack("!I", header)
        if length > _MAX_FRAME:
            raise GMSError("V1 RPC frame is too large")
        payload = read_exact(length)
        if len(received_fds) > 1:
            raise GMSError("V1 RPC received multiple file descriptors")
        value = json.loads(payload.decode())
        return value, received_fds.pop() if received_fds else -1
    except Exception:
        for fd in received_fds:
            os.close(fd)
        raise


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                request, received_fd = _receive(self.request)
            except EOFError:
                return
            except Exception:
                return
            if received_fd >= 0:
                os.close(received_fd)
                return
            export_fd = -1
            try:
                result, export_fd = self.server.dispatch(request)  # type: ignore[attr-defined]
                _send(self.request, [True, result], export_fd)
            except Exception as exc:
                try:
                    _send(self.request, [False, type(exc).__name__, str(exc)])
                except Exception:
                    return
            finally:
                if export_fd >= 0:
                    os.close(export_fd)


class RPCServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, registry: Registry):
        self.path, self.registry = path, registry
        super().__init__(path, _Handler)
        os.chmod(path, 0o600)

    def server_close(self) -> None:
        super().server_close()
        Path(self.path).unlink(missing_ok=True)

    def dispatch(self, request: object) -> tuple[object, int]:
        if not isinstance(request, list) or len(request) != 2:
            raise GMSError("invalid V1 RPC request")
        method, params = request
        if not isinstance(method, str) or not isinstance(params, list):
            raise GMSError("invalid V1 RPC request")
        if method == "begin":
            self.registry.begin(_generation(params[0]))
        elif method == "process_evidence":
            return list(self.registry.process_evidence()), -1
        elif method == "allocate":
            self.registry.allocate(str(params[0]), _allocation(params[1]))
        elif method == "free":
            self.registry.free(str(params[0]), _allocation(params[1]))
        elif method == "export":
            return None, self.registry.export(str(params[0]), _allocation(params[1]))
        elif method == "seal":
            self.registry.seal(_generation(params[0]), str(params[1]))
        elif method == "attach":
            self.registry.attach(
                _generation(params[0]),
                str(params[1]),
                str(params[2]),
                tuple(_allocation(item) for item in params[3]),
            )
        elif method == "detach":
            self.registry.detach(_generation(params[0]), str(params[1]))
        elif method == "release_artifact":
            self.registry.release_artifact(_generation(params[0]), str(params[1]))
        elif method == "retire":
            self.registry.retire(_generation(params[0]))
        else:
            raise GMSError(f"unknown V1 RPC method {method!r}")
        return None, -1


class RPCClient:
    """One serialized stream; failed transports reconnect and repeat exact IDs."""

    def __init__(self, path: str):
        self.path = path
        self._socket = self._connect()
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.path)
        except Exception:
            sock.close()
            raise
        return sock

    def close(self) -> None:
        self._socket.close()

    def _call(self, method: str, params: list[object], fd: bool = False) -> object:
        with self._lock:
            for attempt in range(2):
                try:
                    _send(self._socket, [method, params])
                    response, received_fd = _receive(self._socket)
                    break
                except (EOFError, OSError):
                    self._socket.close()
                    if attempt:
                        raise
                    self._socket = self._connect()
            else:  # pragma: no cover - loop always breaks or raises
                raise AssertionError
        if not isinstance(response, list) or not response:
            if received_fd >= 0:
                os.close(received_fd)
            raise GMSError("invalid V1 RPC response")
        if not response[0]:
            if received_fd >= 0:
                os.close(received_fd)
            raise GMSError(f"{response[1]}: {response[2]}")
        if fd and received_fd < 0:
            raise GMSError(f"{method} did not return an FD")
        if not fd and received_fd >= 0:
            os.close(received_fd)
            raise GMSError(f"{method} returned an unexpected FD")
        return received_fd if fd else response[1]

    def begin(self, generation: Generation) -> None:
        self._call("begin", [_gen_wire(generation)])

    def process_evidence(self) -> tuple[int, int]:
        result = self._call("process_evidence", [])
        assert isinstance(result, list) and len(result) == 2
        return int(result[0]), int(result[1])

    def allocate(self, owner: str, allocation: Allocation) -> None:
        self._call("allocate", [owner, _allocation_wire(allocation)])

    def free(self, owner: str, allocation: Allocation) -> None:
        self._call("free", [owner, _allocation_wire(allocation)])

    def export(self, owner: str, allocation: Allocation) -> int:
        result = self._call("export", [owner, _allocation_wire(allocation)], fd=True)
        assert isinstance(result, int)
        return result

    def seal(self, generation: Generation, artifact_id: str) -> None:
        self._call("seal", [_gen_wire(generation), artifact_id])

    def attach(
        self,
        generation: Generation,
        artifact_id: str,
        reader_id: str,
        parameters: tuple[Allocation, ...],
    ) -> None:
        self._call(
            "attach",
            [
                _gen_wire(generation),
                artifact_id,
                reader_id,
                [_allocation_wire(item) for item in parameters],
            ],
        )

    def detach(self, generation: Generation, reader_id: str) -> None:
        self._call("detach", [_gen_wire(generation), reader_id])

    def release_artifact(self, generation: Generation, artifact_id: str) -> None:
        self._call("release_artifact", [_gen_wire(generation), artifact_id])

    def retire(self, generation: Generation) -> None:
        self._call("retire", [_gen_wire(generation)])
