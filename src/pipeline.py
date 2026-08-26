"""Batch pipeline: input/ -> detect -> inpaint -> output/ or failed/.

The original file in input/ is never modified or deleted, regardless of
outcome — success writes a new file to output/, failure copies (not moves)
the original into failed/.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from . import utils
from .detector import detect_watermark
from .inpainter import inpaint
from .state import ProcessedState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
FAILED_DIR = PROJECT_ROOT / "failed"
STATE_PATH = PROJECT_ROOT / ".state" / "processed.json"


@dataclass
class RunSummary:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0


def _output_path(source: Path) -> Path:
    return OUTPUT_DIR / f"{source.stem}_clean{source.suffix.lower()}"


def _fail(source: Path, state: ProcessedState, content_hash: str) -> None:
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, FAILED_DIR / source.name)
    state.mark(source.name, content_hash, "failed")


def process_one(source: Path, state: ProcessedState) -> str:
    content_hash = utils.file_hash(source)
    if state.is_done(source.name, content_hash):
        return "skipped"

    try:
        image = utils.load_image(source)
    except Exception:
        _fail(source, state, content_hash)
        return "failed"

    detection = detect_watermark(image)
    if detection is None:
        _fail(source, state, content_hash)
        return "failed"

    try:
        mask = utils.build_mask(image.size, detection.box)
        result = inpaint(image, mask)
        utils.save_result(result, _output_path(source), source)
    except Exception:
        _fail(source, state, content_hash)
        return "failed"

    state.mark(source.name, content_hash, "success")
    return "success"


def run_once() -> RunSummary:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    state = ProcessedState(STATE_PATH)
    summary = RunSummary()
    for source in utils.iter_input_images(INPUT_DIR):
        summary.total += 1
        outcome = process_one(source, state)
        if outcome == "success":
            summary.success += 1
        elif outcome == "failed":
            summary.failed += 1
        else:
            summary.skipped += 1
    return summary
