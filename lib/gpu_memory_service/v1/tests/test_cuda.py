# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def test_real_cuda_pool_teardown_preserves_live_storages() -> None:
    """Use a subprocess because the allocator callback singleton is process-wide."""
    code = textwrap.dedent(
        """
        import os
        import tempfile
        import threading

        import torch

        from gpu_memory_service.common.vmm import get_vmm
        from gpu_memory_service.v1.client import Manager
        from gpu_memory_service.v1.registry import Registry
        from gpu_memory_service.v1.rpc import RPCClient, RPCServer
        from gpu_memory_service.v1.torch import TorchPools, discover_parameter_mappings

        torch.cuda.set_device(0)
        vmm = get_vmm()
        gpu = str(torch.cuda.get_device_properties(0).uuid)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "gms-v1.sock")
            with RPCServer(path, Registry(gpu, vmm, 0)) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    rpc = RPCClient(path)
                    manager = Manager(
                        rpc,
                        vmm,
                        0,
                        artifact_id="cuda-test",
                        generation_id="cuda-test",
                    )
                    pools = TorchPools(manager)
                    with pools.parameter_pool():
                        base = torch.arange(16, device="cuda", dtype=torch.float32)
                        first = torch.nn.Parameter(base.view(4, 4))
                        second = torch.nn.Parameter(base[4:12], requires_grad=False)
                    with pools.private_pool():
                        private = torch.ones(4, device="cuda")

                    class Model(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.first = first
                            self.second = second

                    model = Model()
                    before = (
                        id(first),
                        id(second),
                        int(first.untyped_storage()._cdata),
                        int(second.untyped_storage()._cdata),
                        first.data_ptr(),
                        second.data_ptr(),
                        private.data_ptr(),
                    )
                    pools.collect_and_destroy()
                    records = discover_parameter_mappings(model, manager.mappings)
                    assert torch.equal(
                        first,
                        torch.arange(
                            16, device="cuda", dtype=torch.float32
                        ).view(4, 4),
                    )
                    assert torch.equal(private, torch.ones(4, device="cuda"))
                    after = (
                        id(first),
                        id(second),
                        int(first.untyped_storage()._cdata),
                        int(second.untyped_storage()._cdata),
                        first.data_ptr(),
                        second.data_ptr(),
                        private.data_ptr(),
                    )
                    assert before == after
                    assert len(records) == 1
                    assert before[2] == before[3]
                finally:
                    server.shutdown()
                    thread.join()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
