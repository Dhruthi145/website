"""
Module: feature_fusion.py
Step 9: Feature Fusion
Combines image features, object detection results, and encoded preferences.
"""
import numpy as np


def encode_detections(detections: list, max_objects: int = 10) -> np.ndarray:
    """Encode detection confidences into fixed-length vector."""
    vec = np.zeros(max_objects, dtype=np.float32)
    for i, det in enumerate(detections[:max_objects]):
        vec[i] = det.get('confidence', 0.0)
    return vec


def fuse_features(image_features: np.ndarray,
                  detections: list,
                  encoded_prefs: np.ndarray) -> np.ndarray:
    """
    Fuse all feature sources into a single representation.
    - image_features: (2048,) ResNet features
    - detections: list of YOLO detection dicts
    - encoded_prefs: (26,) preference vector
    Returns: normalized concatenated feature vector
    """
    detection_vec = encode_detections(detections)

    # Normalize image features using L2 norm
    norm = np.linalg.norm(image_features)
    if norm > 0:
        img_feat_norm = image_features / norm
    else:
        img_feat_norm = image_features

    fused = np.concatenate([img_feat_norm, detection_vec, encoded_prefs])
    return fused
