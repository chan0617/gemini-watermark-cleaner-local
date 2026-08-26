"""IO helpers: format-preserving save, content hashing, mask construction."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Tuple

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def iter_input_images(input_dir: Path) -> Iterator[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("."):
            yield path


def file_hash(path: Path) -> str:
    """Content hash used to tell an untouched file from a re-dropped/edited one."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def build_mask(size: Tuple[int, int], box: Tuple[int, int, int, int], padding: int = 6) -> Image.Image:
    width, height = size
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    mask = Image.new("L", size, 0)
    mask.paste(255, (x1, y1, x2, y2))
    return mask


def save_result(image: Image.Image, dest_path: Path, original_path: Path) -> None:
    """Save at the same resolution/aspect ratio and (best-effort) same format as the original."""
    ext = original_path.suffix.lower()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        image.convert("RGB").save(dest_path, format="JPEG", quality=95, subsampling=0)
    elif ext == ".webp":
        image.save(dest_path, format="WEBP", quality=95)
    else:
        image.save(dest_path, format="PNG")
