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
	if spec == nil {
		return false
	}
	container := common.AlphaMainContainer(spec)
	if container == nil {
		return false
	}
	return setDefaultRuntimeVersionFromImage(&spec.RuntimeVersion, container.Image)
}

func defaultAlphaRuntimeVersionForImageUpdate(oldSpec, newSpec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	if oldSpec == nil || newSpec == nil {
		return false
	}
	oldImage := ""
	if oldContainer := common.AlphaMainContainer(oldSpec); oldContainer != nil {
		oldImage = oldContainer.Image
	}
	newImage := ""
	if newContainer := common.AlphaMainContainer(newSpec); newContainer != nil {
		newImage = newContainer.Image
	}
	return setDefaultRuntimeVersionForImageUpdate(
		oldSpec.RuntimeVersion,
		&newSpec.RuntimeVersion,
		oldImage,
		newImage,
	)
}

func defaultBetaRuntimeVersion(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if spec == nil {
		return false
	}
	container := common.BetaMainContainer(spec)
	if container == nil {
		return false
	}
	return setDefaultRuntimeVersionFromImage(&spec.RuntimeVersion, container.Image)
}

func defaultBetaRuntimeVersionForImageUpdate(oldSpec, newSpec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if oldSpec == nil || newSpec == nil {
		return false
	}
	oldImage := ""
	if oldContainer := common.BetaMainContainer(oldSpec); oldContainer != nil {
		oldImage = oldContainer.Image
	}
	newImage := ""
	if newContainer := common.BetaMainContainer(newSpec); newContainer != nil {
		newImage = newContainer.Image
	}
	return setDefaultRuntimeVersionForImageUpdate(
		oldSpec.RuntimeVersion,
		&newSpec.RuntimeVersion,
		oldImage,
		newImage,
	)
}

func setDefaultRuntimeVersionFromImage(runtimeVersion *string, image string) bool {
	if runtimeVersion == nil || strings.TrimSpace(*runtimeVersion) != "" {
		return false
	}
	version, err := runtimeversion.ParseImageVersion(image)
	if err != nil {
		return false
	}
	*runtimeVersion = version.String()
	return true
}

func setDefaultRuntimeVersionForImageUpdate(oldRuntimeVersion string, newRuntimeVersion *string, oldImage, newImage string) bool {
	if newRuntimeVersion == nil || oldRuntimeVersion != *newRuntimeVersion {
		return false
	}
	if strings.TrimSpace(oldImage) == strings.TrimSpace(newImage) {
		return false
	}
	version, err := runtimeversion.ParseImageVersion(newImage)
	if err != nil {
		return false
	}
	normalized := version.String()
	if *newRuntimeVersion == normalized {
		return false
	}
	*newRuntimeVersion = normalized
	return true
}
