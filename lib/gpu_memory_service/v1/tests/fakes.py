# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os


class VMM:
    def __init__(self, granularity: int = 64):
        self.granularity = granularity
        self.next_server_handle = 10
        self.next_import = 100
        self.next_base = 0x100000
        self.server_handles: set[int] = set()
        self.imports: set[int] = set()
        self.reservations: dict[int, int] = {}
        self.mapped: dict[int, tuple[int, int]] = {}
        self.access: dict[int, object] = {}
        self.events: list[tuple[object, ...]] = []
        self.fail_reserve = False
        self.fail_address_free = False
        self.fail_import = False
        self.fail_map_call: int | None = None
        self.map_calls = 0
        self.fail_access_call: int | None = None
        self.access_calls = 0
        self.fail_unmap: set[int] = set()
        self.fail_release: set[int] = set()

    def ensure_initialized(self):
        return None

    def get_allocation_granularity(self, device):
        self.events.append(("granularity", device))
        return self.granularity

    def create_tolerate_oom(self, size, device):
        assert size % self.granularity == 0
        handle = self.next_server_handle
        self.next_server_handle += 1
        self.server_handles.add(handle)
        self.events.append(("create", size, device, handle))
        return True, handle

    def export_to_shareable_handle(self, handle):
        assert handle in self.server_handles
        return os.open("/dev/null", os.O_RDONLY)

    def import_shareable_handle_close_fd(self, fd):
        try:
            if self.fail_import:
                self.fail_import = False
                raise RuntimeError("import failed")
            handle = self.next_import
            self.next_import += 1
            self.imports.add(handle)
            self.events.append(("import", handle))
            return handle
        finally:
            os.close(fd)

    def release(self, handle):
        self.events.append(("release", handle))
        if handle in self.fail_release:
            self.fail_release.remove(handle)
            raise RuntimeError("release failed")
        if handle >= 100:
            self.imports.remove(handle)
        else:
            self.server_handles.remove(handle)

    def address_reserve(self, size, granularity):
        if self.fail_reserve:
            self.fail_reserve = False
            raise RuntimeError("reserve failed")
        assert granularity == self.granularity
        base = self.next_base
        self.next_base += size + 0x1000
        self.reservations[base] = size
        self.events.append(("reserve", base, size))
        return base

    def address_free(self, base, size):
        self.events.append(("address_free", base, size))
        if self.fail_address_free:
            self.fail_address_free = False
            raise RuntimeError("reservation free failed")
        if base in self.mapped:
            raise RuntimeError("reservation remains mapped")
        assert self.reservations.pop(base) == size

    def map(self, base, size, handle):
        self.map_calls += 1
        if self.map_calls == self.fail_map_call:
            raise RuntimeError("map failed")
        assert handle in self.imports
        self.mapped[base] = (size, handle)
        self.events.append(("map", base, size))

    def unmap(self, base, size):
        self.events.append(("unmap", base, size))
        if base in self.fail_unmap:
            self.fail_unmap.remove(base)
            raise RuntimeError("unmap failed")
        assert self.mapped.pop(base)[0] == size
        self.access.pop(base, None)

    def set_access(self, base, size, device, access):
        self.access_calls += 1
        self.events.append(("access", base, size, device, access))
        if self.access_calls == self.fail_access_call:
            raise RuntimeError("access failed")
        assert self.mapped[base][0] == size
        self.access[base] = access

    def synchronize(self):
        self.events.append(("synchronize",))

    def runtime_set_device(self, device):
        self.events.append(("device", device))
