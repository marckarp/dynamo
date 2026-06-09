/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import (
	"errors"
	"testing"
)

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
			got, err := normalize(tt.value)
			if (err != nil) != tt.wantErr {
				t.Fatalf("normalize() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("normalize() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseImage(t *testing.T) {
	tests := []struct {
		name  string
		image string
		want  string
		ok    bool
	}{
		{name: "simple tag", image: "vllm-runtime:1.1.0", want: "1.1", ok: true},
		{name: "registry port", image: "localhost:5000/ns/vllm-runtime:1.2.3", want: "1.2", ok: true},
		{name: "tag plus digest", image: "nvcr.io/nvidia/vllm-runtime:1.3.0@sha256:abc", want: "1.3", ok: true},
		{name: "backend suffix", image: "rohanv672/dynamo:v0.5.1-trtllm", want: "0.5", ok: true},
		{name: "latest", image: "vllm-runtime:latest", ok: false},
		{name: "digest only", image: "vllm-runtime@sha256:abc", ok: false},
		{name: "no tag", image: "localhost:5000/ns/vllm-runtime", ok: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := parseImage(tt.image)
			if ok != tt.ok {
				t.Fatalf("parseImage() ok = %v, want %v", ok, tt.ok)
			}
			if got != tt.want {
				t.Fatalf("parseImage() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResolve(t *testing.T) {
	tests := []struct {
		name           string
		runtimeVersion string
		image          string
		want           string
		wantErrKind    ErrorKind
	}{
		{
			name:  "derives from image",
			image: "vllm-runtime:1.1.0",
			want:  "1.1",
		},
		{
			name:           "explicit wins for custom image",
			runtimeVersion: "1.2",
			image:          "vllm-runtime:latest",
			want:           "1.2",
		},
		{
			name:           "explicit matches parseable image",
			runtimeVersion: "1.2",
			image:          "vllm-runtime:1.2.3",
			want:           "1.2",
		},
		{
			name:           "explicit mismatch",
			runtimeVersion: "1.2",
			image:          "vllm-runtime:1.3.0",
			wantErrKind:    ErrorMismatch,
		},
		{
			name:        "unresolved",
			image:       "vllm-runtime:latest",
			wantErrKind: ErrorUnresolved,
		},
		{
			name:           "invalid explicit",
			runtimeVersion: "latest",
			image:          "vllm-runtime:latest",
			wantErrKind:    ErrorInvalidExplicit,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Resolve(tt.runtimeVersion, tt.image)
			if tt.wantErrKind == "" {
				if err != nil {
					t.Fatalf("Resolve() error = %v, want nil", err)
				}
				if got != tt.want {
					t.Fatalf("Resolve() = %q, want %q", got, tt.want)
				}
				return
			}

			var resolveErr *ResolutionError
			if !errors.As(err, &resolveErr) {
				t.Fatalf("Resolve() error = %T, want *ResolutionError", err)
			}
			if resolveErr.Kind != tt.wantErrKind {
				t.Fatalf("Resolve() error kind = %q, want %q", resolveErr.Kind, tt.wantErrKind)
			}
		})
	}
}
