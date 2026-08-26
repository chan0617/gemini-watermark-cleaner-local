"""Minimal local web UI for managing input/output/failed folders.

This binds to 127.0.0.1 only — nothing leaves the machine. It's a thin,
optional convenience layer on top of the folder-based pipeline in
src/pipeline.py: the underlying automation still works purely via folders
and start.command even if this page is never opened.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from src import utils
from src.pipeline import FAILED_DIR, INPUT_DIR, OUTPUT_DIR, run_once

app = Flask(__name__)

_FOLDERS = {"input": INPUT_DIR, "output": OUTPUT_DIR, "failed": FAILED_DIR}


def _safe_path(folder: str, filename: str) -> Path:
    if folder not in _FOLDERS:
        raise ValueError("invalid folder")
    base = _FOLDERS[folder].resolve()
    name = secure_filename(filename)
    if not name:
        raise ValueError("invalid filename")
    path = (base / name).resolve()
    if path.parent != base:
        raise ValueError("invalid path")
    return path


def _list_folder(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


@app.get("/")
def index():
    return _PAGE


@app.get("/api/files")
def api_files():
    return jsonify(
        {
            "input": _list_folder(INPUT_DIR),
            "output": _list_folder(OUTPUT_DIR),
            "failed": _list_folder(FAILED_DIR),
        }
    )


@app.get("/files/<folder>/<path:filename>")
def serve_file(folder: str, filename: str):
    try:
        path = _safe_path(folder, filename)
    except ValueError:
        return "invalid path", 400
    if not path.exists():
        return "not found", 404
    return send_from_directory(_FOLDERS[folder], path.name)


@app.post("/api/upload")
def api_upload():
    saved = []
    for f in request.files.getlist("files"):
        if not f.filename:
            continue
        name = secure_filename(f.filename)
        ext = Path(name).suffix.lower()
        if not name or ext not in utils.SUPPORTED_EXTENSIONS:
            continue
        dest = INPUT_DIR / name
        counter = 1
        while dest.exists():
            dest = INPUT_DIR / f"{Path(name).stem}_{counter}{ext}"
            counter += 1
        f.save(dest)
        saved.append(dest.name)
    return jsonify({"saved": saved})


@app.post("/api/delete")
def api_delete():
    data = request.get_json(force=True, silent=True) or {}
    try:
        path = _safe_path(data.get("folder", ""), data.get("filename", ""))
    except ValueError:
        return jsonify({"error": "invalid path"}), 400
    if path.exists():
        path.unlink()
        return jsonify({"deleted": True})
    return jsonify({"deleted": False}), 404


@app.post("/api/process")
def api_process():
    summary = run_once()
    return jsonify(
        {
            "total": summary.total,
            "success": summary.success,
            "failed": summary.failed,
            "skipped": summary.skipped,
        }
    )


_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemini Watermark Cleaner</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif;
    background: #f5f5f7; color: #1d1d1f;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #6e6e73; font-size: 13px; margin-bottom: 20px; }
  .dropzone {
    border: 2px dashed #b0b0b5; border-radius: 12px; padding: 28px; text-align: center;
    color: #6e6e73; cursor: pointer; background: #fff; transition: border-color .15s, background .15s;
    margin-bottom: 14px;
  }
  .dropzone.drag { border-color: #0071e3; background: #eef6ff; }
  .actions { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
  button.primary {
    background: #0071e3; color: #fff; border: none; border-radius: 8px; padding: 10px 18px;
    font-size: 14px; cursor: pointer;
  }
  button.primary:disabled { background: #a9a9ac; cursor: default; }
  .status { font-size: 13px; color: #6e6e73; }
  .columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .col { background: #fff; border-radius: 12px; padding: 14px; min-height: 200px; }
  .col h2 { font-size: 14px; margin: 0 0 10px; display: flex; justify-content: space-between; }
  .col h2 span.count { color: #6e6e73; font-weight: normal; }
  .item {
    display: flex; align-items: center; gap: 8px; padding: 6px; border-radius: 8px;
  }
  .item:hover { background: #f5f5f7; }
  .item img { width: 40px; height: 40px; object-fit: cover; border-radius: 6px; background: #eee; }
  .item .name { flex: 1; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .item a.dl { font-size: 12px; color: #0071e3; text-decoration: none; margin-right: 4px; }
  .item button.del {
    border: none; background: transparent; color: #ff3b30; font-size: 16px; cursor: pointer; line-height: 1;
  }
  .empty { color: #b0b0b5; font-size: 12.5px; padding: 6px; }
</style>
</head>
<body>
  <h1>Gemini Watermark Cleaner</h1>
  <div class="sub">모든 처리는 이 컴퓨터에서만 실행됩니다. 외부 업로드 없음.</div>

  <div class="dropzone" id="dropzone">
    이미지를 여기로 드래그하거나 클릭해서 선택하세요 (PNG / JPG / JPEG / WEBP)
    <input type="file" id="fileInput" multiple accept=".png,.jpg,.jpeg,.webp" style="display:none">
  </div>

  <div class="actions">
    <button class="primary" id="processBtn">지금 정리하기</button>
    <span class="status" id="statusText"></span>
  </div>

  <div class="columns">
    <div class="col">
      <h2>대기중 (input) <span class="count" id="count-input">0</span></h2>
      <div id="list-input"></div>
    </div>
    <div class="col">
      <h2>완료 (output) <span class="count" id="count-output">0</span></h2>
      <div id="list-output"></div>
    </div>
    <div class="col">
      <h2>실패 (failed) <span class="count" id="count-failed">0</span></h2>
      <div id="list-failed"></div>
    </div>
  </div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const statusText = document.getElementById('statusText');

dropzone.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('drag'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('drag'); })
);
dropzone.addEventListener('drop', e => uploadFiles(e.dataTransfer.files));
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));

async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  statusText.textContent = '업로드 중...';
  await fetch('/api/upload', { method: 'POST', body: fd });
  statusText.textContent = '업로드 완료';
  fileInput.value = '';
  refresh();
}

processBtn.addEventListener('click', async () => {
  processBtn.disabled = true;
  statusText.textContent = '처리 중... (이미지 수에 따라 시간이 걸릴 수 있습니다)';
  try {
    const res = await fetch('/api/process', { method: 'POST' });
    const s = await res.json();
    statusText.textContent = `총 ${s.total}장 · 성공 ${s.success}장 · 실패 ${s.failed}장` +
      (s.skipped ? ` · 건너뜀 ${s.skipped}장` : '');
  } finally {
    processBtn.disabled = false;
    refresh();
  }
});

async function deleteFile(folder, filename) {
  await fetch('/api/delete', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder, filename }),
  });
  refresh();
}

function renderList(folder, files) {
  const el = document.getElementById(`list-${folder}`);
  document.getElementById(`count-${folder}`).textContent = files.length;
  if (!files.length) { el.innerHTML = '<div class="empty">비어 있음</div>'; return; }
  el.innerHTML = files.map(name => `
    <div class="item">
      <img src="/files/${folder}/${encodeURIComponent(name)}" loading="lazy">
      <div class="name" title="${name}">${name}</div>
      ${folder === 'output' ? `<a class="dl" href="/files/${folder}/${encodeURIComponent(name)}" download>저장</a>` : ''}
      <button class="del" title="삭제" onclick="deleteFile('${folder}', '${name.replace(/'/g, "\\'")}')">×</button>
    </div>
  `).join('');
}

async function refresh() {
  const res = await fetch('/api/files');
  const data = await res.json();
  renderList('input', data.input);
  renderList('output', data.output);
  renderList('failed', data.failed);
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
