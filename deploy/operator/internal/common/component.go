package common

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

func AlphaMainContainer(spec *v1alpha1.DynamoComponentDeploymentSharedSpec) *corev1.Container {
	if spec == nil || spec.ExtraPodSpec == nil || spec.ExtraPodSpec.MainContainer == nil {
		return nil
	}
	return spec.ExtraPodSpec.MainContainer
}

func BetaMainContainer(spec *v1beta1.DynamoComponentDeploymentSharedSpec) *corev1.Container {
	if spec == nil || spec.PodTemplate == nil {
		return nil
	}
	return FindContainerByName(spec.PodTemplate.Spec.Containers, v1beta1.MainContainerName)
}
