/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import (
	"fmt"
	"regexp"
	"strings"

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

var runtimeVersionPattern = regexp.MustCompile(`^[vV]?[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?(?:\+[0-9A-Za-z][0-9A-Za-z.-]*)?$`)

// Resolution describes how a component's runtime version was determined.
type Resolution struct {
	RuntimeVersion string
	ImageVersion   string
	Derived        bool
}

// Normalize converts a runtime semver string to the minor compatibility form
// used in the DGD API, for example "1.1.0-rc1" -> "1.1".
func Normalize(value string) (string, error) {
	trimmed := strings.TrimSpace(value)
	if !runtimeVersionPattern.MatchString(trimmed) {
		return "", fmt.Errorf("must be a semantic version such as \"1.1\" or \"1.1.0\"")
	}
	version, err := semver.NewVersion(canonicalSemverInput(trimmed))
	if err != nil {
		return "", fmt.Errorf("must be a semantic version such as \"1.1\" or \"1.1.0\"")
	}
	return fmt.Sprintf("%d.%d", version.Major(), version.Minor()), nil
}

// ParseImage derives the runtime minor version from a container image tag.
func ParseImage(image string) (string, bool) {
	tag := imageTag(image)
	if tag == "" {
		return "", false
	}
	version, err := Normalize(tag)
	if err != nil {
		return "", false
	}
	return version, true
}

// Resolve applies runtimeVersion precedence and mismatch detection.
func Resolve(runtimeVersion, image string) (Resolution, error) {
	explicit := strings.TrimSpace(runtimeVersion)
	imageVersion, imageOK := ParseImage(image)
	if explicit == "" {
		if !imageOK {
			return Resolution{}, nil
		}
		return Resolution{RuntimeVersion: imageVersion, ImageVersion: imageVersion, Derived: true}, nil
	}

	normalizedExplicit, err := Normalize(explicit)
	if err != nil {
		return Resolution{}, err
	}
	if imageOK && normalizedExplicit != imageVersion {
		return Resolution{}, fmt.Errorf("runtime version %q does not match image tag runtime version %q", normalizedExplicit, imageVersion)
	}
	return Resolution{RuntimeVersion: normalizedExplicit, ImageVersion: imageVersion}, nil
}

// ValidateAlphaSharedSpec validates explicit and image-derived runtime version
// consistency for a v1alpha1 component shared spec.
func ValidateAlphaSharedSpec(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, fieldPath string) error {
	if spec == nil {
		return nil
	}
	return validate(spec.RuntimeVersion, AlphaMainContainerImage(spec), fieldPath, "extraPodSpec.mainContainer.image")
}

// ValidateBetaSharedSpec validates explicit and image-derived runtime version
// consistency for a v1beta1 component shared spec.
func ValidateBetaSharedSpec(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, fieldPath string) error {
	if spec == nil {
		return nil
	}
	return validate(spec.RuntimeVersion, BetaMainContainerImage(spec), fieldPath, "podTemplate.spec.containers[name=main].image")
}

// DefaultAlphaSharedSpec persists a derived runtimeVersion when a v1alpha1
// component has a parseable main container image tag.
func DefaultAlphaSharedSpec(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	if spec == nil || strings.TrimSpace(spec.RuntimeVersion) != "" {
		return false
	}
	version, ok := ParseImage(AlphaMainContainerImage(spec))
	if !ok {
		return false
	}
	spec.RuntimeVersion = version
	return true
}

// DefaultBetaSharedSpec persists a derived runtimeVersion when a v1beta1
// component has a parseable main container image tag.
func DefaultBetaSharedSpec(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if spec == nil || strings.TrimSpace(spec.RuntimeVersion) != "" {
		return false
	}
	version, ok := ParseImage(BetaMainContainerImage(spec))
	if !ok {
		return false
	}
	spec.RuntimeVersion = version
	return true
}

// AlphaMainContainerImage returns the v1alpha1 semantic main container image.
func AlphaMainContainerImage(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) string {
	if spec == nil || spec.ExtraPodSpec == nil || spec.ExtraPodSpec.MainContainer == nil {
		return ""
	}
	return spec.ExtraPodSpec.MainContainer.Image
}

// BetaMainContainerImage returns the v1beta1 well-known main container image.
func BetaMainContainerImage(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) string {
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

func validate(runtimeVersion, image, fieldPath, imageField string) error {
	resolution, err := Resolve(runtimeVersion, image)
	if err != nil {
		if strings.TrimSpace(runtimeVersion) != "" {
			if strings.Contains(err.Error(), "does not match") {
				return fmt.Errorf("%s.runtimeVersion has invalid value %q: %s derived from %s %q",
					fieldPath, runtimeVersion, err, imageField, image)
			}
			return fmt.Errorf("%s.runtimeVersion has invalid value %q: %s", fieldPath, runtimeVersion, err)
		}
		return err
	}
	if strings.TrimSpace(runtimeVersion) == "" && resolution.RuntimeVersion == "" {
		if image == "" {
			return fmt.Errorf("%s.runtimeVersion is required because %s is not set; set runtimeVersion explicitly for SHA/custom tags",
				fieldPath, imageField)
		}
		return fmt.Errorf("%s.runtimeVersion is required because %s %q does not contain a parseable semver tag; set runtimeVersion explicitly for SHA/custom tags",
			fieldPath, imageField, image)
	}
	return nil
}

func imageTag(image string) string {
	ref := strings.TrimSpace(image)
	if ref == "" {
		return ""
	}
	if digest := strings.Index(ref, "@"); digest >= 0 {
		ref = ref[:digest]
	}
	lastSlash := strings.LastIndex(ref, "/")
	lastColon := strings.LastIndex(ref, ":")
	if lastColon <= lastSlash {
		return ""
	}
	return ref[lastColon+1:]
}

func canonicalSemverInput(value string) string {
	out := strings.TrimPrefix(strings.TrimPrefix(value, "v"), "V")
	suffixStart := len(out)
	if idx := strings.IndexAny(out, "-+"); idx >= 0 {
		suffixStart = idx
	}
	core := out[:suffixStart]
	suffix := out[suffixStart:]
	if strings.Count(core, ".") == 1 {
		return core + ".0" + suffix
	}
	return out
}
