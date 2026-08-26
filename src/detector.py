"""Deterministic corner-region watermark detector for Gemini-generated images.

Gemini places its visible sparkle watermark in the bottom-right corner at a
fixed pixel size/margin that depends on output resolution (not a percentage
of image size). These numbers come from public reverse-engineering of the
watermark's alpha-blending behavior — see README "Attribution & Sources".
Because the position is deterministic, detection only needs to (a) locate the
best-aligned box near that expected corner and (b) score its confidence, so a
non-watermarked or oddly-cropped image is routed to failed/ instead of being
edited blindly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

_LARGE_SIZE, _LARGE_MARGIN = 96, 64   # both dimensions > 1024px
_SMALL_SIZE, _SMALL_MARGIN = 48, 32   # either dimension <= 1024px

# How far (in px) to slide the candidate box while looking for the best
# alignment, to absorb small margin/scaling variations between Gemini versions.
_SEARCH_JITTER = 6

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "models" / "watermark_template.png"


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original image pixels
    confidence: float


def _expected_geometry(width: int, height: int) -> Tuple[int, int]:
    if width > 1024 and height > 1024:
        return _LARGE_SIZE, _LARGE_MARGIN
    return _SMALL_SIZE, _SMALL_MARGIN


def _synthetic_sparkle(canvas: int) -> Image.Image:
    """Fallback template: a 4-point sparkle silhouette.

    Used only when the user hasn't supplied a real crop of the watermark at
    models/watermark_template.png. A real crop from the user's own image
    (used purely as a local matching template, never redistributed) will
    always match better than this approximation.
    """
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    cx = cy = canvas / 2
    outer, inner = canvas * 0.48, canvas * 0.14
    points = []
    for i in range(8):
        r = outer if i % 2 == 0 else inner
        angle = (math.pi / 4) * i - math.pi / 2
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=255)
    return img


def _load_template(size: int, template_path: Path) -> np.ndarray:
    if template_path.exists():
        tpl = Image.open(template_path).convert("L")
    else:
        tpl = _synthetic_sparkle(128)
    tpl = tpl.resize((size, size), Image.LANCZOS)
    return np.asarray(tpl, dtype=np.uint8)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = mag.max()
    if peak > 0:
        mag = mag / peak * 255.0
    return mag.astype(np.uint8)


def detect_watermark(
    image: Image.Image,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    confidence_threshold: float = 0.28,
) -> Optional[Detection]:
    """Locate the Gemini watermark using its known corner geometry.

    Returns None if no candidate region scores above `confidence_threshold`,
    signalling the caller to treat this image as a detection failure.
    """
    width, height = image.size
    size, margin = _expected_geometry(width, height)

    pad = _SEARCH_JITTER
    sx1 = max(0, width - margin - size - pad)
    sy1 = max(0, height - margin - size - pad)
    sx2 = min(width, width - margin + pad)
    sy2 = min(height, height - margin + pad)

    if sx2 - sx1 < size or sy2 - sy1 < size:
        sx1, sy1 = max(0, width - size), max(0, height - size)
        sx2, sy2 = width, height

    region = np.asarray(image.convert("L").crop((sx1, sy1, sx2, sy2)))
    if region.size == 0 or region.shape[0] < size or region.shape[1] < size:
        return None

    region_grad = _gradient_magnitude(region)
    template_grad = _gradient_magnitude(_load_template(size, template_path))

    result = cv2.matchTemplate(region_grad, template_grad, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < confidence_threshold:
        return None

    x1, y1 = sx1 + max_loc[0], sy1 + max_loc[1]
    return Detection(box=(x1, y1, x1 + size, y1 + size), confidence=float(max_val))
