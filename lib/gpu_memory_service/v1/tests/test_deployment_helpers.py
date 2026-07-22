# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gpu_memory_service.v1.tests.test_deployment import _checkpoint_cleanup_pod, _render

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_render_uses_image_pull_secret() -> None:
    manifests = _render(
        name="gms-v1-test",
        namespace="test",
        node="gpu-node",
        image="registry.example/gms:test",
        image_pull_secret="registry-secret",
        checkpoint_pvc="snapshot-pvc",
        checkpoint_path="/checkpoints",
        device_class="gpu.nvidia.com",
        runtime_class="nvidia",
    )

    assert manifests[1]["spec"]["imagePullSecrets"] == [{"name": "registry-secret"}]


def test_checkpoint_cleanup_pod_uses_image_pull_secret() -> None:
    pod = _checkpoint_cleanup_pod(
        name="gms-v1-test",
        namespace="test",
        node="gpu-node",
        image="registry.example/gms:test",
        image_pull_secret="registry-secret",
        checkpoint_pvc="snapshot-pvc",
        checkpoint_path="/checkpoints",
    )

    assert pod["spec"]["imagePullSecrets"] == [{"name": "registry-secret"}]
