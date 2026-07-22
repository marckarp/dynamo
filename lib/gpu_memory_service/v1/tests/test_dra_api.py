# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from gpu_memory_service.v1.tests.test_deployment import (
    _create_resource_claim_template,
    _delete_resource_claim_template,
    _get_resource_claim_template,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.asyncio
async def test_resource_claim_template_uses_generic_ga_api(monkeypatch) -> None:
    from kubernetes_asyncio import client

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def call_api(*args: Any, **kwargs: Any) -> dict[str, bool]:
        calls.append((args, kwargs))
        return {"ok": True}

    api_client = client.ApiClient()
    monkeypatch.setattr(api_client, "call_api", call_api)
    custom = client.CustomObjectsApi(api_client)
    body = {
        "apiVersion": "resource.k8s.io/v1",
        "kind": "ResourceClaimTemplate",
        "metadata": {"name": "claim"},
    }
    try:
        await _create_resource_claim_template(custom, "test", body)
        await _get_resource_claim_template(custom, "test", "claim")
        await _delete_resource_claim_template(custom, "test", "claim")
    finally:
        await api_client.close()

    expected_path_parameters = {
        "group": "resource.k8s.io",
        "version": "v1",
        "namespace": "test",
        "plural": "resourceclaimtemplates",
    }
    assert [(args[0], args[1]) for args, _ in calls] == [
        ("/apis/{group}/{version}/namespaces/{namespace}/{plural}", "POST"),
        (
            "/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}",
            "GET",
        ),
        (
            "/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}",
            "DELETE",
        ),
    ]
    assert calls[0][0][2] == expected_path_parameters
    assert calls[0][1]["body"] == body
    for args, _ in calls[1:]:
        assert args[2] == {**expected_path_parameters, "name": "claim"}
