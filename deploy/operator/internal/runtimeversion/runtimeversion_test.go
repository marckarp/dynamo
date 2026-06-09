/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import "testing"

func TestNormalize(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		want    string
		wantErr bool
	}{
		{name: "major minor", value: "1.1", want: "1.1"},
		{name: "patch", value: "1.1.0", want: "1.1"},
		{name: "leading v", value: "v1.2.3", want: "1.2"},
		{name: "prerelease", value: "1.3.0-rc1", want: "1.3"},
		{name: "build metadata", value: "1.4.0+build.7", want: "1.4"},
		{name: "custom tag", value: "latest", wantErr: true},
		{name: "sha", value: "sha256-deadbeef", wantErr: true},
		{name: "missing minor", value: "1", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Normalize(tt.value)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Normalize() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("Normalize() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseImageVersion(t *testing.T) {
	tests := []struct {
		name    string
		image   string
		want    string
		wantErr bool
	}{
		{name: "simple tag", image: "vllm-runtime:1.1.0", want: "1.1"},
		{name: "registry port", image: "localhost:5000/ns/vllm-runtime:1.2.3", want: "1.2"},
		{name: "tag plus digest", image: "nvcr.io/nvidia/vllm-runtime:1.3.0@sha256:abc", want: "1.3"},
		{name: "backend suffix", image: "rohanv672/dynamo:v0.5.1-trtllm", want: "0.5"},
		{name: "latest", image: "vllm-runtime:latest", wantErr: true},
		{name: "digest only", image: "vllm-runtime@sha256:abc", wantErr: true},
		{name: "no tag", image: "localhost:5000/ns/vllm-runtime", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseImageVersion(tt.image)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseImageVersion() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("ParseImageVersion() = %q, want %q", got, tt.want)
			}
		})
	}
}
