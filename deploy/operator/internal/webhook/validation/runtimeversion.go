/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func validateAlphaRuntimeVersion(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, fieldPath string) error {
	if spec == nil {
		return nil
	}
	image := alphaMainContainerImage(spec)
	imageField := "extraPodSpec.mainContainer.image"

	if spec.RuntimeVersion == "" && image == "" {
		return fmt.Errorf("%s.runtimeVersion is required because %s is not set; set runtimeVersion explicitly for SHA/custom tags",
			fieldPath, imageField)
	}
	if spec.RuntimeVersion == "" {
		if _, err := runtimeversion.ParseImageVersion(image); err != nil {
			return fmt.Errorf("%s.runtimeVersion is required because %s %q does not contain a parseable semver tag; set runtimeVersion explicitly for SHA/custom tags",
				fieldPath, imageField, image)
		}
		return nil
	}

	explicitVersion, err := runtimeversion.Parse(spec.RuntimeVersion)
	if err != nil {
		return fmt.Errorf("%s.runtimeVersion has invalid value %q: must be a semantic version such as \"1.1.0\"",
			fieldPath, spec.RuntimeVersion)
	}

	imageVersion, err := runtimeversion.ParseImageVersion(image)
	if err != nil {
		return nil
	}
	if explicitVersion != imageVersion {
		return fmt.Errorf("%s.runtimeVersion has invalid value %q: runtime version %q does not match image tag runtime version %q derived from %s %q",
			fieldPath, spec.RuntimeVersion, explicitVersion.String(), imageVersion.String(), imageField, image)
	}
	return nil
}

func alphaMainContainerImage(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) string {
	if spec == nil || spec.ExtraPodSpec == nil || spec.ExtraPodSpec.MainContainer == nil {
		return ""
	}
	return spec.ExtraPodSpec.MainContainer.Image
}
