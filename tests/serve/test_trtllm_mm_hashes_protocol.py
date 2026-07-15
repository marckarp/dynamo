# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin TRT-LLM's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed tensorrt_llm module and verify two
contracts the dynamo Rust frontend depends on for TRT-LLM MM-aware KV routing:

  1. The Qwen2-VL family's synthetic image marker is ``vocab_size + 1``
     (``Qwen2VLInputProcessorBase.tllm_multimodal_token_id``). Dynamo resolves the
     same marker in ``trtllm/workers/llm_worker.py::_resolve_image_token_id`` and
     the KV-event publisher normalizes those runs to ``pad_value``. A future
     TRT-LLM that shifts the convention would silently misroute, so pin it here
     so the drift fails-closed at PR time.

  2. ``multi_modal_uuids`` is a field on TRT-LLM's ``TokensPrompt`` / ``TextPrompt``.
     Dynamo forwards the frontend's ``mm_hash`` through it so TRT-LLM echoes the
     routing hash back into its KV events (``trtllm/multimodal_processor.py``). An
     upstream rename or removal would break MM-aware routing.
"""
from __future__ import annotations

import inspect

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_trtllm_qwen2vl_image_marker_is_vocab_size_plus_one() -> None:
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase

    src = inspect.getsource(Qwen2VLInputProcessorBase.__init__)
    assert "self.tllm_multimodal_token_id = self.get_vocab_size() + 1" in src, (
        "TRT-LLM's Qwen2-VL image marker is no longer `vocab_size + 1`; "
        "dynamo's _resolve_image_token_id and _QWEN2_VL_FAMILY_MODEL_TYPES gate "
        "(components/src/dynamo/trtllm/workers/llm_worker.py) must be updated in "
        "lockstep or MM-aware KV routing will misroute."
    )


def test_trtllm_multi_modal_uuids_field_present() -> None:
    from tensorrt_llm.inputs.data import TextPrompt, TokensPrompt

    for cls in (TextPrompt, TokensPrompt):
        assert "multi_modal_uuids" in cls.__annotations__, (
            f"tensorrt_llm.inputs.data.{cls.__name__}.multi_modal_uuids is gone; "
            "dynamo forwards the frontend's mm_hash through it "
            "(components/src/dynamo/trtllm/multimodal_processor.py) so TRT-LLM "
            "echoes the routing hash in its KV events. An upstream rename breaks "
            "MM-aware routing."
        )
