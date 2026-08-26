"""Persistent record of which input files have already been handled.

Keyed by filename + content hash so that (a) re-running the batch, and
(b) Watch Mode's polling loop, never reprocess a file that already produced
an output/ or failed/ result — unless the user replaces it with different
bytes under the same name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class ProcessedState:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._data: Dict[str, dict] = {}
        if state_path.exists():
            try:
                self._data = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    def is_done(self, key: str, content_hash: str) -> bool:
        entry = self._data.get(key)
        return bool(entry) and entry.get("hash") == content_hash

    def mark(self, key: str, content_hash: str, status: str) -> None:
        self._data[key] = {"hash": content_hash, "status": status}
        self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
