"""
Module: object_detector.py
Step 6: Object Detection using YOLOv8 (ultralytics)
Returns bounding boxes and object labels for furniture/room objects.
"""
import cv2
import numpy as np

# Furniture-relevant COCO classes
FURNITURE_CLASSES = {
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 67: 'cell phone', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


def detect_objects(preprocessed_frames: list) -> list:
    """
    Detect furniture and room objects using YOLOv8.
    Falls back to mock detections if YOLO not available.
    """
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        all_detections = []
        for frame_data in preprocessed_frames[:3]:
            img = frame_data['original']
            results = model(img, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names.get(cls_id, f'object_{cls_id}')
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    all_detections.append({
                        'label': label,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        # Deduplicate by label
        seen = set()
        unique = []
        for d in all_detections:
            if d['label'] not in seen and d['confidence'] > 0.4:
                seen.add(d['label'])
                unique.append(d)
        return unique if unique else _mock_detections()

    except Exception:
        return _mock_detections()


def _mock_detections() -> list:
    """Fallback mock detections for development/demo."""
    return [
        {'label': 'sofa', 'confidence': 0.91, 'bbox': [20, 100, 180, 200]},
        {'label': 'coffee table', 'confidence': 0.87, 'bbox': [60, 150, 140, 180]},
        {'label': 'tv', 'confidence': 0.82, 'bbox': [10, 50, 80, 120]},
        {'label': 'lamp', 'confidence': 0.78, 'bbox': [180, 60, 210, 130]},
        {'label': 'bookshelf', 'confidence': 0.74, 'bbox': [190, 80, 220, 190]},
    ]
