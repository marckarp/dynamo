/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package defaulting

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestDCDDefaulter_DefaultsComponentNameOnCreate(t *testing.T) {
	tests := []struct {
		name    string
		ctx     context.Context
		dcd     *nvidiacomv1beta1.DynamoComponentDeployment
		want    string
		wantErr bool
	}{
		{
			name: "CREATE defaults empty spec name from metadata name",
			ctx:  admissionCtx(admissionv1.Create, nvidiacomv1beta1.DynamoComponentDeploymentGVK),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
			},
			want: "worker",
		},
		{
			name: "CREATE preserves explicit spec name",
			ctx:  admissionCtx(admissionv1.Create, nvidiacomv1beta1.DynamoComponentDeploymentGVK),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentName: "custom",
					},
				},
			},
			want: "custom",
		},
		{
			name: "UPDATE does not default empty spec name",
			ctx:  admissionCtx(admissionv1.Update, nvidiacomv1beta1.DynamoComponentDeploymentGVK),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
			},
			want: "",
		},
		{
			name: "missing admission request fails closed",
			ctx:  context.Background(),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDCDDefaulter()

			err := defaulter.Default(tt.ctx, tt.dcd)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Default() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			if got := tt.dcd.Spec.ComponentName; got != tt.want {
				t.Fatalf("spec.name = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDCDDefaulter_DefaultRejectsWrongType(t *testing.T) {
	defaulter := NewDCDDefaulter()

	if err := defaulter.Default(admissionCtx(admissionv1.Create, nvidiacomv1beta1.DynamoComponentDeploymentGVK), &corev1.Pod{}); err == nil {
		t.Fatal("Default() error = nil, want type error")
	}
}

func TestDCDDefaulter_DefaultReturnsErrorForInvalidOldObject(t *testing.T) {
	dcd := betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0", "1.1.0")
	ctx := admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Update,
			OldObject: runtime.RawExtension{Raw: []byte("{")},
		},
	})

	err := NewDCDDefaulter().Default(ctx, dcd)
	if err == nil {
		t.Fatal("Default() error = nil, want old object decode error")
	}
	if !strings.Contains(err.Error(), "failed to decode old DCD object") {
		t.Fatalf("Default() error = %q, want old object decode error", err.Error())
	}
}

func TestDCDDefaulter_DefaultsRuntimeVersion(t *testing.T) {
	tests := []struct {
		name string
		ctx  context.Context
		dcd  *nvidiacomv1beta1.DynamoComponentDeployment
		want string
	}{
		{
			name: "CREATE derives runtimeVersion from semver image tag",
			ctx:  admissionCtx(admissionv1.Create),
			dcd:  betaDCDWithImage("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0"),
			want: "1.1.0",
		},
		{
			name: "UPDATE derives runtimeVersion",
			ctx:  admissionCtx(admissionv1.Update),
			dcd:  betaDCDWithImage("nvcr.io/nvidia/ai-dynamo/vllm-runtime:v1.2.3"),
			want: "1.2.3",
		},
		{
			name: "preserves explicit runtimeVersion",
			ctx:  admissionCtx(admissionv1.Create),
			dcd: func() *nvidiacomv1beta1.DynamoComponentDeployment {
				dcd := betaDCDWithImage("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0")
				dcd.Spec.RuntimeVersion = "1.1.0"
				return dcd
			}(),
			want: "1.1.0",
		},
		{
			name: "does not default unparseable image tag",
			ctx:  admissionCtx(admissionv1.Create),
			dcd:  betaDCDWithImage("nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest"),
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := NewDCDDefaulter().Default(tt.ctx, tt.dcd); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}
			if got := tt.dcd.Spec.RuntimeVersion; got != tt.want {
				t.Fatalf("runtimeVersion = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDCDDefaulter_DefaultsRuntimeVersionForImageUpdate(t *testing.T) {
	tests := []struct {
		name   string
		oldDCD *nvidiacomv1beta1.DynamoComponentDeployment
		newDCD *nvidiacomv1beta1.DynamoComponentDeployment
		want   string
	}{
		{
			name:   "UPDATE refreshes runtimeVersion when only image changes",
			oldDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0", "1.1.0"),
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0", "1.1.0"),
			want:   "1.2.0",
		},
		{
			name:   "UPDATE refreshes runtimeVersion when old image was unparseable",
			oldDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest", "1.1.0"),
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0", "1.1.0"),
			want:   "1.2.0",
		},
		{
			name: "UPDATE refreshes runtimeVersion when old image was unset",
			oldDCD: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentName:  "worker",
						RuntimeVersion: "1.1.0",
					},
				},
			},
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0", "1.1.0"),
			want:   "1.2.0",
		},
		{
			name:   "UPDATE preserves runtimeVersion changed by user",
			oldDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0", "1.1.0"),
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0", "1.3.0"),
			want:   "1.3.0",
		},
		{
			name:   "UPDATE preserves runtimeVersion when new image is unparseable",
			oldDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0", "1.1.0"),
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest", "1.1.0"),
			want:   "1.1.0",
		},
		{
			name:   "UPDATE preserves runtimeVersion when old object is unavailable",
			oldDCD: nil,
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0", "1.1.0"),
			want:   "1.1.0",
		},
		{
			name:   "UPDATE preserves runtimeVersion when image is unchanged",
			oldDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0", "1.1.0"),
			newDCD: betaDCDWithImageAndRuntimeVersion("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0", "1.1.0"),
			want:   "1.1.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := admissionCtx(admissionv1.Update)
			if tt.oldDCD != nil {
				ctx = admissionCtxWithOldDCD(t, admissionv1.Update, tt.oldDCD)
			}

			if err := NewDCDDefaulter().Default(ctx, tt.newDCD); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}
			if got := tt.newDCD.Spec.RuntimeVersion; got != tt.want {
				t.Fatalf("runtimeVersion = %q, want %q", got, tt.want)
			}
		})
	}
}

func admissionCtxWithOldDCD(t *testing.T, op admissionv1.Operation, oldObj *nvidiacomv1beta1.DynamoComponentDeployment) context.Context {
	t.Helper()

	raw, err := json.Marshal(oldObj)
	if err != nil {
		t.Fatalf("marshal old object: %v", err)
	}
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
			OldObject: runtime.RawExtension{Raw: raw},
		},
	})
}

func betaDCDWithImageAndRuntimeVersion(image, runtimeVersion string) *nvidiacomv1beta1.DynamoComponentDeployment {
	dcd := betaDCDWithImage(image)
	dcd.Spec.RuntimeVersion = runtimeVersion
	return dcd
}

func betaDCDWithImage(image string) *nvidiacomv1beta1.DynamoComponentDeployment {
	return &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "worker"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: nvidiacomv1beta1.MainContainerName, Image: image}},
					},
				},
			},
		},
	}
}
