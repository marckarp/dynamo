#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${KUBE_CONTEXT:?set KUBE_CONTEXT}"
: "${NAMESPACE:?set NAMESPACE}"
: "${NODE:?set NODE}"
: "${IMAGE:?set IMAGE}"
: "${IMAGE_PULL_SECRET:?set IMAGE_PULL_SECRET}"
: "${CHECKPOINT_PVC:?set CHECKPOINT_PVC}"

exec python -m pytest \
  lib/gpu_memory_service/v1/tests/test_deployment.py \
  -m dynamocheckpoint \
  --gms-v1-kube-context "${KUBE_CONTEXT}" \
  --gms-v1-namespace "${NAMESPACE}" \
  --gms-v1-node "${NODE}" \
  --gms-v1-image "${IMAGE}" \
  --gms-v1-image-pull-secret "${IMAGE_PULL_SECRET}" \
  --gms-v1-checkpoint-pvc "${CHECKPOINT_PVC}" \
  --gms-v1-checkpoint-path "${CHECKPOINT_PATH:-/checkpoints}" \
  --gms-v1-device-class "${DEVICE_CLASS:-gpu.nvidia.com}" \
  --gms-v1-runtime-class "${RUNTIME_CLASS:-nvidia}"
