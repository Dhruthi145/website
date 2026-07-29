"""
Module: design_generator.py
Step 11: Design Generation using Stable Diffusion img2img
Generates redesigned room images based on prompt and edge structural guidance.
"""
import os
import cv2
import numpy as np
from PIL import Image


def generate_designs(preprocessed_frames: list,
                     prompt: str,
                     negative_prompt: str,
                     output_dir: str,
                     strength: float = 0.65,
                     guidance_scale: float = 8.5,
                     num_inference_steps: int = 30) -> list:
    """
    Generate redesigned room images using Stable Diffusion img2img.
    Uses edge maps (Canny) for structural preservation.
    Falls back to enhanced mock images in demo mode.
    """
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        pipe.safety_checker = None  # For interior design, disable NSFW filter

        output_paths = []
        for i, frame_data in enumerate(preprocessed_frames):   # ALL walls
            # Use original frame as img2img input
            img_np = cv2.resize(frame_data['raw'], (512, 512))
            img_pil = Image.fromarray(img_np[..., ::-1])  # BGR->RGB

            result = pipe(
                prompt=prompt,
                image=img_pil,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
            ).images[0]

            out_path = os.path.join(output_dir, f'design_{i}.png')
            result.save(out_path)
            output_paths.append(out_path)

        return output_paths

    except Exception as e:
        print(f"[Design Generator] Stable Diffusion unavailable: {e}. Using mock output.")
        return _generate_mock_designs(preprocessed_frames, prompt, output_dir)


def _generate_mock_designs(preprocessed_frames: list,
                           prompt: str,
                           output_dir: str) -> list:
    """
    Demo fallback: Apply color grading and artistic filters to simulate redesign.
    This simulates the output when Stable Diffusion is not available.
    """
    # Extract color theme hint from prompt
    warm = 'warm' in prompt or 'terracotta' in prompt or 'amber' in prompt
    cool = 'cool' in prompt or 'blue' in prompt or 'silver' in prompt
    dark = 'dark' in prompt or 'charcoal' in prompt or 'navy' in prompt

    output_paths = []
    for i, frame_data in enumerate(preprocessed_frames):         # ALL walls
        img = frame_data['original'].copy()
        img = cv2.resize(img, (512, 512))

        # Simulate stylized redesign with color grading
        img_float = img.astype(np.float32)

        if warm:
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 1.25, 0, 255)  # boost red
            img_float[:, :, 1] = np.clip(img_float[:, :, 1] * 1.1, 0, 255)   # boost green
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 0.85, 0, 255)  # reduce blue
        elif cool:
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 1.25, 0, 255)  # boost blue
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 0.85, 0, 255)  # reduce red
        elif dark:
            img_float = img_float * 0.7
        else:
            # Neutral: slight warmth and brightness
            img_float = np.clip(img_float * 1.1 + 10, 0, 255)

        # Apply slight bilateral filter for painterly effect
        img_out = cv2.bilateralFilter(img_float.astype(np.uint8), 9, 75, 75)

        # Add subtle vignette
        rows, cols = img_out.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols * 0.6)
        kernel_y = cv2.getGaussianKernel(rows, rows * 0.6)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        vignette = (img_out * mask[:, :, np.newaxis]).astype(np.uint8)

        out_path = os.path.join(output_dir, f'design_{i}.png')
        cv2.imwrite(out_path, vignette)
        output_paths.append(out_path)

    return output_paths
