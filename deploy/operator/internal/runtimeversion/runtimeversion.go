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

// Normalize returns the runtime compatibility version (major.minor) for a
// semver-like runtimeVersion value.
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

// ParseImageVersion returns the normalized runtime compatibility version
// (major.minor) from a parseable image tag.
func ParseImageVersion(image string) (string, error) {
	tag := imageTag(image)
	if tag == "" {
		return "", fmt.Errorf("image %q does not contain a tag", image)
	}
	version, err := Normalize(tag)
	if err != nil {
		return "", fmt.Errorf("image tag %q: %w", tag, err)
	}
	return version, nil
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
