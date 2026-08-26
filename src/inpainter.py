"""Local LaMa-based inpainting wrapper with Apple Silicon (MPS) support.

Uses the `simple-lama-inpainting` package (Apache-2.0), which wraps the
LaMa (advimman/lama, Apache-2.0) TorchScript checkpoint. Everything runs
on-device; the only network access is a one-time model-weight download on
first run (same as installing any pip package with pretrained weights) —
no image data is ever sent anywhere.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

_lama_instance = None
_active_device: Optional[str] = None


def _build_lama() -> Tuple[object, str]:
    import torch
    from simple_lama_inpainting import SimpleLama

    candidates = ["cpu"]
    if torch.backends.mps.is_available():
        candidates.insert(0, "mps")

    last_error: Optional[Exception] = None
    for device_name in candidates:
        try:
            lama = SimpleLama(device=torch.device(device_name))
            # Smoke-test so MPS ops LaMa's Fourier convolutions don't support
            # (e.g. some FFT kernels) fail fast here, not mid-batch.
            probe = Image.new("RGB", (64, 64), (128, 128, 128))
            mask = Image.new("L", (64, 64), 0)
            mask.paste(255, (16, 16, 48, 48))
            lama(probe, mask)
            return lama, device_name
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall back
            last_error = exc
            continue
    raise RuntimeError(f"Could not initialize LaMa on any device ({candidates}): {last_error}")


def get_inpainter():
    global _lama_instance, _active_device
    if _lama_instance is None:
        _lama_instance, _active_device = _build_lama()
    return _lama_instance


def device_name() -> str:
    get_inpainter()
    assert _active_device is not None
    return _active_device


def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    lama = get_inpainter()
    result = lama(image.convert("RGB"), mask.convert("L"))
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result))
    # simple-lama-inpainting internally pads to a multiple of 8 for the model
    # but does not crop the output back — do it here so resolution/aspect
    # ratio always exactly match the original (no silent cropping/padding).
    if result.size != image.size:
        result = result.crop((0, 0, image.width, image.height))
    return result
