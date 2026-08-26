"""Corner-region watermark detector for Gemini-generated images.

Gemini stamps a visible 4-point sparkle watermark near the bottom-right
corner. Its exact pixel size/margin isn't a documented constant (an initial
guess based on a public write-up did not match real samples), so instead of
hardcoding one guessed position, this searches a generous proportional
corner region at several candidate scales and picks the best-matching
location via gradient-domain template matching. A confidence score gates
whether a match is trusted — anything below threshold is treated as a
detection failure, which routes the image to failed/ instead of being
edited blindly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Primary search: tight windows around the watermark's empirically measured
# positions, with jitter to absorb resolution/version differences. Narrow on
# purpose — real image content (hands, furniture edges) can otherwise
# out-score the faint, semi-transparent watermark if the search window is
# too wide. The watermark is NOT a fixed fraction of canvas size: measured
# samples show ~5.8% inset / ~2.8% size on a 2048px image but ~9.5% inset /
# ~7.7% size on a 1024px image (smaller canvases get a relatively bigger,
# more-inset mark so it stays legible) — so two anchors are tried, not one.
_ANCHORS = (
    (0.045, 0.028),  # margin_fraction, jitter_fraction — ~2048px-class placement
    (0.095, 0.035),  # margin_fraction, jitter_fraction — ~1024px-class placement
)
_PRIMARY_ACCEPT_THRESHOLD = 0.40

# Fallback search: the whole bottom-right corner, used only when no primary
# window finds anything convincing (e.g. an unfamiliar resolution/version).
# Held to a stricter bar since a wide search is more prone to false matches.
_CORNER_FRACTION = 0.20
_FALLBACK_ACCEPT_THRESHOLD = 0.55

_MIN_ROI_PX = 72

# Candidate watermark sizes, as a fraction of min(width, height), covering
# both the ~2.3-3.2% (2048px-class) and ~7-9% (1024px-class) cases measured.
# Deliberately excludes very small fractions (<2.5%): real measurements never
# showed anything that small, and tiny windows spuriously out-score correct
# larger matches by finding noise that happens to correlate well.
_SIZE_FRACTIONS = (0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "models" / "watermark_template.png"


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original image pixels
    confidence: float


def _synthetic_sparkle(canvas: int) -> Image.Image:
    """Astroid-curve approximation of Gemini's 4-point sparkle silhouette.

    Used only when the user hasn't supplied a real crop of the watermark at
    models/watermark_template.png. A real crop from the user's own image
    (used purely as a local matching template, never redistributed) will
    always match better than this approximation.
    """
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    cx = cy = canvas / 2
    a = canvas * 0.46
    points: List[Tuple[float, float]] = []
    steps = 240
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x = cx + a * (math.cos(t) ** 3)
        y = cy + a * (math.sin(t) ** 3)
        points.append((x, y))
    draw.polygon(points, fill=255)
    return img


def _load_template(template_path: Path) -> Image.Image:
    if template_path.exists():
        return Image.open(template_path).convert("L")
    return _synthetic_sparkle(128)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = mag.max()
    if peak > 0:
        mag = mag / peak * 255.0
    return mag.astype(np.uint8)


def _best_match_in_roi(
    gray_image: Image.Image,
    template_img: Image.Image,
    rx0: int,
    ry0: int,
    rx1: int,
    ry1: int,
) -> Optional[Detection]:
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(gray_image.width, rx1), min(gray_image.height, ry1)
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return None

    region = np.asarray(gray_image.crop((rx0, ry0, rx1, ry1)))
    region_grad = _gradient_magnitude(region)
    min_dim = min(gray_image.width, gray_image.height)

    best: Optional[Detection] = None
    for frac in _SIZE_FRACTIONS:
        size = max(8, int(round(min_dim * frac)))
        if size > region_grad.shape[0] or size > region_grad.shape[1]:
            continue
        template = np.asarray(template_img.resize((size, size), Image.LANCZOS), dtype=np.uint8)
        template_grad = _gradient_magnitude(template)

        result = cv2.matchTemplate(region_grad, template_grad, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if best is None or max_val > best.confidence:
            x1, y1 = rx0 + max_loc[0], ry0 + max_loc[1]
            best = Detection(box=(x1, y1, x1 + size, y1 + size), confidence=float(max_val))
    return best


def detect_watermark(image: Image.Image, template_path: Path = DEFAULT_TEMPLATE_PATH) -> Optional[Detection]:
    """Locate the Gemini watermark near the bottom-right corner.

    Tries tight windows around each empirically measured anchor position
    first (least prone to false positives from busy image content), then
    falls back to a wider corner search under a stricter bar. Returns None
    if nothing clears its threshold, signalling a detection failure to the
    caller.
    """
    width, height = image.size
    gray_image = image.convert("L")
    min_dim = min(width, height)
    template_img = _load_template(template_path)
    max_size = max(8, int(round(min_dim * _SIZE_FRACTIONS[-1])))

    best_primary: Optional[Detection] = None
    for margin_fraction, jitter_fraction in _ANCHORS:
        jitter = int(min_dim * jitter_fraction)
        anchor_x = width - int(min_dim * margin_fraction)
        anchor_y = height - int(min_dim * margin_fraction)
        candidate = _best_match_in_roi(
            gray_image,
            template_img,
            anchor_x - max_size - jitter,
            anchor_y - max_size - jitter,
            anchor_x + jitter,
            anchor_y + jitter,
        )
        if candidate is not None and (best_primary is None or candidate.confidence > best_primary.confidence):
            best_primary = candidate

    if best_primary is not None and best_primary.confidence >= _PRIMARY_ACCEPT_THRESHOLD:
        return best_primary

    roi_w = max(_MIN_ROI_PX, int(width * _CORNER_FRACTION))
    roi_h = max(_MIN_ROI_PX, int(height * _CORNER_FRACTION))
    fallback = _best_match_in_roi(gray_image, template_img, width - roi_w, height - roi_h, width, height)
    if fallback is not None and fallback.confidence >= _FALLBACK_ACCEPT_THRESHOLD:
        return fallback

    return None
