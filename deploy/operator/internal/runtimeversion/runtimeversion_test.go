/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import "testing"

func TestParse(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		want    Version
		wantErr bool
	}{
		{name: "empty", value: "", wantErr: true},
		{name: "core", value: "1.1.0", want: Version{Major: 1, Minor: 1, Patch: 0}},
		{name: "major minor", value: "1.1", wantErr: true},
		{name: "leading v", value: "v1.2.3", wantErr: true},
		{name: "prerelease", value: "1.3.0-rc1", wantErr: true},
		{name: "build metadata", value: "1.4.0+build.7", wantErr: true},
		{name: "custom tag", value: "latest", wantErr: true},
		{name: "sha", value: "sha256-deadbeef", wantErr: true},
		{name: "missing minor", value: "1", wantErr: true},
		{name: "leading zero", value: "01.2.3", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Parse(tt.value)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Parse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("Parse() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseImageVersion(t *testing.T) {
	tests := []struct {
		name    string
		image   string
		want    Version
		wantErr bool
	}{
		{name: "simple tag", image: "vllm-runtime:1.1.0", want: Version{Major: 1, Minor: 1, Patch: 0}},
		{name: "registry port", image: "localhost:5000/ns/vllm-runtime:1.2.3", want: Version{Major: 1, Minor: 2, Patch: 3}},
		{name: "tag plus digest", image: "nvcr.io/nvidia/vllm-runtime:1.3.0@sha256:abc", want: Version{Major: 1, Minor: 3, Patch: 0}},
		{name: "backend suffix", image: "rohanv672/dynamo:v0.5.1-trtllm", want: Version{Major: 0, Minor: 5, Patch: 1}},
		{name: "prerelease suffix", image: "vllm-runtime:1.3.0-nemotron-ultra-dev.1", want: Version{Major: 1, Minor: 3, Patch: 0}},
		{name: "build metadata", image: "vllm-runtime:1.4.0+build.7", want: Version{Major: 1, Minor: 4, Patch: 0}},
		{name: "latest", image: "vllm-runtime:latest", wantErr: true},
		{name: "digest only", image: "vllm-runtime@sha256:abc", wantErr: true},
		{name: "no tag", image: "localhost:5000/ns/vllm-runtime", wantErr: true},
		{name: "missing patch", image: "vllm-runtime:1.1", wantErr: true},
		{name: "leading zero", image: "vllm-runtime:01.2.3", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseImageVersion(tt.image)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseImageVersion() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("ParseImageVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}
