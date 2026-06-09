/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func defaultAlphaRuntimeVersion(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	if spec == nil || strings.TrimSpace(spec.RuntimeVersion) != "" {
		return false
	}
	version, err := runtimeversion.ParseImageVersion(alphaMainContainerImage(spec))
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
	version, err := runtimeversion.ParseImageVersion(betaMainContainerImage(spec))
	if err != nil {
		return false
	}
	spec.RuntimeVersion = version.String()
	return true
}

func alphaMainContainerImage(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) string {
	if spec == nil || spec.ExtraPodSpec == nil || spec.ExtraPodSpec.MainContainer == nil {
		return ""
	}
	return spec.ExtraPodSpec.MainContainer.Image
}

func betaMainContainerImage(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) string {
	if spec == nil || spec.PodTemplate == nil {
		return ""
	}
	for i := range spec.PodTemplate.Spec.Containers {
		if spec.PodTemplate.Spec.Containers[i].Name == nvidiacomv1beta1.MainContainerName {
			return spec.PodTemplate.Spec.Containers[i].Image
		}
	}
	return ""
}
