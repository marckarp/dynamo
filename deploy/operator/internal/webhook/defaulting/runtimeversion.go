/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func defaultAlphaRuntimeVersion(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	if spec == nil || strings.TrimSpace(spec.RuntimeVersion) != "" {
		return false
	}
	container := common.AlphaMainContainer(spec)
	if container == nil {
		return false
	}
	version, err := runtimeversion.ParseImageVersion(container.Image)
	if err != nil {
		return false
	}
	spec.RuntimeVersion = version.String()
	return true
}

func defaultBetaRuntimeVersion(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if spec == nil || strings.TrimSpace(spec.RuntimeVersion) != "" {
		return false
	}
	container := common.BetaMainContainer(spec)
	if container == nil {
		return false
	}
	version, err := runtimeversion.ParseImageVersion(container.Image)
	if err != nil {
		return false
	}
	spec.RuntimeVersion = version.String()
	return true
}
