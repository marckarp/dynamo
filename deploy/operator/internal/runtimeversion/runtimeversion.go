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
)

var runtimeVersionPattern = regexp.MustCompile(`^[vV]?[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?(?:\+[0-9A-Za-z][0-9A-Za-z.-]*)?$`)

// ErrorKind identifies why a runtime version could not be resolved.
type ErrorKind string

const (
	ErrorInvalidExplicit ErrorKind = "InvalidExplicit"
	ErrorUnresolved      ErrorKind = "Unresolved"
	ErrorMismatch        ErrorKind = "Mismatch"
)

// ResolutionError carries structured details for callers that need to format
// API-specific admission errors.
type ResolutionError struct {
	Kind            ErrorKind
	Explicit        string
	Image           string
	ExplicitVersion string
	ImageVersion    string
}

func (e *ResolutionError) Error() string {
	switch e.Kind {
	case ErrorInvalidExplicit:
		return "runtimeVersion must be a semantic version such as \"1.1\" or \"1.1.0\""
	case ErrorUnresolved:
		return "runtimeVersion could not be resolved"
	case ErrorMismatch:
		return fmt.Sprintf("runtime version %q does not match image tag runtime version %q", e.ExplicitVersion, e.ImageVersion)
	default:
		return "runtimeVersion resolution failed"
	}
}

// Resolve returns the normalized runtime compatibility version (major.minor)
// from either an explicit runtimeVersion or a parseable image tag.
func Resolve(runtimeVersion, image string) (string, error) {
	explicit := strings.TrimSpace(runtimeVersion)
	imageVersion, imageOK := parseImage(image)
	if explicit == "" {
		if !imageOK {
			return "", &ResolutionError{
				Kind:  ErrorUnresolved,
				Image: image,
			}
		}
		return imageVersion, nil
	}

	explicitVersion, err := normalize(explicit)
	if err != nil {
		return "", &ResolutionError{
			Kind:     ErrorInvalidExplicit,
			Explicit: runtimeVersion,
			Image:    image,
		}
	}
	if imageOK && explicitVersion != imageVersion {
		return "", &ResolutionError{
			Kind:            ErrorMismatch,
			Explicit:        runtimeVersion,
			Image:           image,
			ExplicitVersion: explicitVersion,
			ImageVersion:    imageVersion,
		}
	}
	return explicitVersion, nil
}

func normalize(value string) (string, error) {
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

func parseImage(image string) (string, bool) {
	tag := imageTag(image)
	if tag == "" {
		return "", false
	}
	version, err := normalize(tag)
	if err != nil {
		return "", false
	}
	return version, true
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
