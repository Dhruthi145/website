"""
Module: image_preprocessor.py
Step 5: Image Preprocessing
- Resize to 224x224
- Normalize pixel values
- Apply Gaussian Blur
- Apply Canny Edge Detection
"""
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray) -> dict:
    """Preprocess a single frame."""
    # Resize to 224x224
    resized = cv2.resize(frame, (224, 224))

    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Canny Edge Detection for structural guidance
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    return {
        'original': resized,
        'normalized': normalized,
        'blurred': blurred,
        'edges': edges,
        'raw': frame
    }


def preprocess_frames(frames: list) -> list:
    """Preprocess a list of frames."""
    return [preprocess_frame(f) for f in frames]
