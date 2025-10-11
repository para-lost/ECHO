"""
AuraFace Face Identity Metric Package

This package provides face identity similarity evaluation using the AuraFace model.
It includes both a direct API and a Flux Kontext compatible wrapper.
"""

from face_identity_metric import FaceIdentityMetric
from flux_kontext_wrapper import FluxKontextFaceMetric, evaluate_face_identity

__version__ = "1.0.0"
__all__ = [
    "FaceIdentityMetric",
    "FluxKontextFaceMetric", 
    "evaluate_face_identity"
]

# Convenience function for quick evaluation
def quick_evaluate(input_images, output_images, threshold=0.5):
    """
    Quick evaluation function for simple use cases.
    
    Args:
        input_images: List of input image paths
        output_images: List of output image paths
        threshold: Similarity threshold (default: 0.5)
    
    Returns:
        Maximum similarity score (0.0 to 1.0)
    """
    return evaluate_face_identity(input_images, output_images, threshold)['score']
