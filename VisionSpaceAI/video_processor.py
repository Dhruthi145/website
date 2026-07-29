"""
Module: video_processor.py
Step 3: Frame Extraction
Step 4: Scene/Wall Detection — groups frames into distinct viewpoints (walls/sides)
        and picks the single sharpest, most representative frame per group.

Logic:
  - Sample frames at regular intervals
  - Compute pairwise SSIM between consecutive frames
  - When SSIM drops below a scene-change threshold → new wall/view segment begins
  - Within each segment, select the sharpest frame (highest Laplacian variance)
  - Result: exactly N frames where N = number of detected distinct walls/views
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ── Tunable constants ────────────────────────────────────────────────────────
SAMPLE_INTERVAL   = 15      # sample every N video frames
SCENE_THRESHOLD   = 0.72    # SSIM below this → new wall/view detected
MIN_SEGMENT_FRAMES = 3      # ignore segments shorter than this (transitions)
MAX_WALLS         = 12      # safety cap on output count
# ────────────────────────────────────────────────────────────────────────────


def _sharpness(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper frame."""
    gray = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _hist_diff(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Normalised histogram difference [0..1] as a secondary scene-change signal.
    Used together with SSIM for more robust boundary detection.
    """
    h1 = cv2.calcHist([cv2.resize(f1, (128, 128))], [0, 1, 2],
                      None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([cv2.resize(f2, (128, 128))], [0, 1, 2],
                      None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
    return 1.0 - float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def _ssim_score(f1: np.ndarray, f2: np.ndarray) -> float:
    g1 = cv2.cvtColor(cv2.resize(f1, (256, 256)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(f2, (256, 256)), cv2.COLOR_BGR2GRAY)
    return float(ssim(g1, g2))


def extract_frames(video_path: str, sample_interval: int = SAMPLE_INTERVAL) -> list:
    """Step 3 — Extract frames at regular intervals. Returns raw frame list."""
    cap = cv2.VideoCapture(video_path)
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def select_frames(frames: list,
                  scene_threshold: float = SCENE_THRESHOLD,
                  max_walls: int = MAX_WALLS) -> list:
    """
    Step 4 — Detect distinct wall/view segments via SSIM + histogram diff.
    Returns one best (sharpest) frame per detected segment.

    Example: a 360° room walkthrough touching 4 walls → returns 4 frames.
    """
    if not frames:
        return []

    if len(frames) == 1:
        return frames

    # ── 1. Compute scene-change score between every consecutive pair ──────
    scores = []
    for i in range(1, len(frames)):
        s = _ssim_score(frames[i - 1], frames[i])
        h = _hist_diff(frames[i - 1], frames[i])
        # Combine: treat as scene change when SSIM is low OR hist diff is high
        change_score = (1.0 - s) * 0.6 + h * 0.4
        scores.append(change_score)

    # ── 2. Adaptive threshold: mean + 0.8·std of change scores ───────────
    arr = np.array(scores)
    adaptive_thresh = float(arr.mean() + 0.8 * arr.std())
    # Also honour the explicit scene_threshold (converted to change-score space)
    effective_thresh = min(adaptive_thresh, 1.0 - scene_threshold)

    # ── 3. Split frames into segments at detected boundaries ──────────────
    segments: list[list] = []
    current_seg = [frames[0]]

    for i, change in enumerate(scores):
        if change >= effective_thresh:
            # Scene change detected → close current segment, open new one
            segments.append(current_seg)
            current_seg = [frames[i + 1]]
        else:
            current_seg.append(frames[i + 1])

    segments.append(current_seg)  # close final segment

    # ── 4. Drop very short segments (likely panning transitions) ──────────
    segments = [s for s in segments if len(s) >= MIN_SEGMENT_FRAMES]

    # ── 5. Fallback: if no boundaries found treat the whole video as 1 view ─
    if not segments:
        segments = [frames]

    # ── 6. Pick the sharpest frame from each segment ──────────────────────
    best_frames = []
    for seg in segments[:max_walls]:
        best = max(seg, key=_sharpness)
        best_frames.append(best)

    return best_frames


def get_wall_count(frames: list) -> int:
    """Convenience helper — returns how many distinct walls were detected."""
    return len(frames)
