/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"errors"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func validateAlphaRuntimeVersion(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, fieldPath string) error {
	if spec == nil {
		return nil
	}
	image := alphaMainContainerImage(spec)
	if _, err := runtimeversion.Resolve(spec.RuntimeVersion, image); err != nil {
		return runtimeVersionAdmissionError(err, spec.RuntimeVersion, image, fieldPath, "extraPodSpec.mainContainer.image")
	}
	return nil
}

func runtimeVersionAdmissionError(err error, runtimeVersion, image, fieldPath, imageField string) error {
	var resolveErr *runtimeversion.ResolutionError
	if !errors.As(err, &resolveErr) {
		return fmt.Errorf("%s.runtimeVersion: %w", fieldPath, err)
	}

	switch resolveErr.Kind {
	case runtimeversion.ErrorInvalidExplicit:
		return fmt.Errorf("%s.runtimeVersion has invalid value %q: must be a semantic version such as \"1.1\" or \"1.1.0\"",
			fieldPath, runtimeVersion)
	case runtimeversion.ErrorMismatch:
		return fmt.Errorf("%s.runtimeVersion has invalid value %q: runtime version %q does not match image tag runtime version %q derived from %s %q",
			fieldPath, runtimeVersion, resolveErr.ExplicitVersion, resolveErr.ImageVersion, imageField, image)
	case runtimeversion.ErrorUnresolved:
		if image == "" {
			return fmt.Errorf("%s.runtimeVersion is required because %s is not set; set runtimeVersion explicitly for SHA/custom tags",
				fieldPath, imageField)
		}
		return fmt.Errorf("%s.runtimeVersion is required because %s %q does not contain a parseable semver tag; set runtimeVersion explicitly for SHA/custom tags",
			fieldPath, imageField, image)
	default:
		return fmt.Errorf("%s.runtimeVersion: %w", fieldPath, err)
	}
}

func alphaMainContainerImage(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) string {
	if spec == nil || spec.ExtraPodSpec == nil || spec.ExtraPodSpec.MainContainer == nil {
		return ""
	}
	return spec.ExtraPodSpec.MainContainer.Image
}
