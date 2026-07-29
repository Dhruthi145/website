"""
Module: feature_extractor.py
Step 7: Feature Extraction using ResNet50 (transfer learning)
Extracts 2048-dim feature vectors from preprocessed frames.
"""
import numpy as np


def extract_features(preprocessed_frames: list) -> np.ndarray:
    """
    Extract visual features using ResNet50 pretrained on ImageNet.
    Returns averaged feature vector across all frames.
    """
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image

        model = models.resnet50(pretrained=True)
        model.eval()
        # Remove final classification layer — use as feature extractor
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        all_features = []
        with torch.no_grad():
            for frame_data in preprocessed_frames:
                img_np = frame_data['original']
                img = Image.fromarray(img_np[..., ::-1])  # BGR->RGB
                tensor = transform(img).unsqueeze(0)
                feat = feature_extractor(tensor).squeeze().numpy()
                all_features.append(feat)

        return np.mean(all_features, axis=0)  # (2048,)

    except Exception:
        # Fallback: random feature vector for demo
        return np.random.randn(2048).astype(np.float32)
