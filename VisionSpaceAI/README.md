# VisionSpace AI — AI Interior Design Recommendation System

A full-stack AI system that transforms room videos into photorealistic interior redesigns using an 11-step AI pipeline.

## Architecture

```
Video Upload → Frame Extraction → Frame Selection → Preprocessing →
Object Detection → Feature Extraction → Preference Encoding →
Feature Fusion → Prompt Generation → Stable Diffusion img2img → Gallery Output
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run server
python app.py
```

Open `http://localhost:5000` in your browser.

## Module Overview

| Module | Step | Description |
|--------|------|-------------|
| `video_processor.py` | 3–4 | OpenCV frame extraction + SSIM deduplication |
| `image_preprocessor.py` | 5 | Resize, normalize, Gaussian Blur, Canny edges |
| `object_detector.py` | 6 | YOLOv8 furniture detection (falls back gracefully) |
| `feature_extractor.py` | 7 | ResNet50 transfer learning, 2048-dim vectors |
| `preference_encoder.py` | 8 | One-hot encoding + min-max budget scaling |
| `feature_fusion.py` | 9 | NumPy feature concatenation + L2 normalization |
| `prompt_generator.py` | 10 | Dynamic structured prompt builder |
| `design_generator.py` | 11 | Stable Diffusion img2img (with graceful mock fallback) |

## GPU Acceleration

For best results, use a CUDA-compatible GPU. Stable Diffusion will automatically use `float16` on CUDA and `float32` on CPU.

## Notes on Stable Diffusion

The system uses `runwayml/stable-diffusion-v1-5` by default. On first run it will download ~4GB of model weights. You can swap the model ID for any compatible img2img checkpoint.

If Stable Diffusion is unavailable (no GPU, model not downloaded), the system falls back to a color-grading + bilateral filter effect for demo purposes.
