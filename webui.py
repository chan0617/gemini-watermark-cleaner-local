"""Minimal local web UI for managing input/output/failed folders.

This binds to 127.0.0.1 only — nothing leaves the machine. It's a thin,
optional convenience layer on top of the folder-based pipeline in
src/pipeline.py: the underlying automation still works purely via folders
and start.command even if this page is never opened.
"""
from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import List

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

from src import utils
from src.inpainter import inpaint
from src.pipeline import FAILED_DIR, INPUT_DIR, OUTPUT_DIR, STATE_PATH, run_once
from src.state import ProcessedState

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


def _list_folder(folder: Path) -> List[dict]:
    if not folder.exists():
        return []
    entries = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
    entries.sort(key=lambda p: p.name)
    return [{"name": p.name, "mtime": p.stat().st_mtime} for p in entries]


@app.get("/")
def index():
    resp = app.make_response(_PAGE)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/files")
def api_files():
    resp = jsonify(
        {
            "input": _list_folder(INPUT_DIR),
            "output": _list_folder(OUTPUT_DIR),
            "failed": _list_folder(FAILED_DIR),
        }
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/files/<folder>/<path:filename>")
def serve_file(folder: str, filename: str):
    try:
        path = _safe_path(folder, filename)
    except ValueError:
        return "invalid path", 400
    if not path.exists():
        return "not found", 404
    return send_from_directory(_FOLDERS[folder], path.name)


@app.get("/api/download_all")
def api_download_all():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in _list_folder(OUTPUT_DIR):
            zf.write(OUTPUT_DIR / name, arcname=name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="cleaned_images.zip")


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


@app.post("/api/manual_process")
def api_manual_process():
    """Inpaint a user-brushed mask directly, bypassing auto-detection."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        src_path = _safe_path(data.get("folder", "input"), data.get("filename", ""))
    except ValueError:
        return jsonify({"error": "invalid path"}), 400
    if not src_path.exists():
        return jsonify({"error": "not found"}), 404

    mask_data_url = data.get("mask", "")
    try:
        _, b64data = mask_data_url.split(",", 1)
        mask_bytes = base64.b64decode(b64data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception:
        return jsonify({"error": "invalid mask"}), 400

    image = utils.load_image(src_path)
    mask_img = mask_img.resize(image.size, Image.NEAREST).point(lambda p: 255 if p > 20 else 0)
    if not mask_img.getbbox():
        return jsonify({"error": "empty mask"}), 400

    try:
        result = inpaint(image, mask_img)
        dest = OUTPUT_DIR / f"{src_path.stem}_clean{src_path.suffix.lower()}"
        utils.save_result(result, dest, src_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    state = ProcessedState(STATE_PATH)
    state.mark(src_path.name, utils.file_hash(src_path), "success")

    stale = FAILED_DIR / src_path.name
    if stale.exists() and stale != src_path:
        stale.unlink()

    return jsonify({"saved": dest.name})


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
  .actions { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; flex-wrap: wrap; }
  button.primary {
    background: #0071e3; color: #fff; border: none; border-radius: 8px; padding: 10px 18px;
    font-size: 14px; cursor: pointer;
  }
  button.secondary {
    background: #fff; color: #0071e3; border: 1px solid #0071e3; border-radius: 8px; padding: 9px 16px;
    font-size: 14px; cursor: pointer;
  }
  button.primary:disabled, button.secondary:disabled { opacity: 0.5; cursor: default; }
  .status { font-size: 13px; color: #6e6e73; }
  .columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .col { background: #fff; border-radius: 12px; padding: 14px; min-height: 200px; }
  .col h2 { font-size: 14px; margin: 0 0 10px; display: flex; justify-content: space-between; }
  .col h2 span.count { color: #6e6e73; font-weight: normal; }
  .item {
    display: flex; align-items: center; gap: 6px; padding: 6px; border-radius: 8px;
  }
  .item:hover { background: #f5f5f7; }
  .item img { width: 40px; height: 40px; object-fit: cover; border-radius: 6px; background: #eee; cursor: pointer; }
  .item .name { flex: 1; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .item a.dl, .item button.link {
    font-size: 12px; color: #0071e3; text-decoration: none; margin-right: 2px; background: none; border: none; cursor: pointer; padding: 0;
  }
  .item button.del {
    border: none; background: transparent; color: #ff3b30; font-size: 16px; cursor: pointer; line-height: 1;
  }
  .empty { color: #b0b0b5; font-size: 12.5px; padding: 6px; }

  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55);
    align-items: center; justify-content: center; z-index: 1000;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: #fff; border-radius: 14px; padding: 18px; max-width: 92vw; max-height: 92vh;
    display: flex; flex-direction: column; gap: 10px;
  }
  .modal h3 { margin: 0; font-size: 15px; }
  .modal .hint { font-size: 12.5px; color: #6e6e73; margin: 0; }
  .canvas-wrap { position: relative; line-height: 0; cursor: crosshair; }
  .canvas-wrap canvas { position: absolute; top: 0; left: 0; max-width: 100%; }
  .canvas-wrap .sizer { visibility: hidden; display: block; }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
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
    <button class="primary" id="processBtn">워터마크 지우기</button>
    <button class="secondary" id="downloadAllBtn">전체 다운로드 (zip)</button>
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

  <div class="modal-overlay" id="modalOverlay">
    <div class="modal">
      <h3 id="modalTitle">직접 선택으로 지우기</h3>
      <p class="hint">지우고 싶은 부분을 브러시(고정 크기)로 문질러 칠한 뒤 "지우기 실행"을 누르세요.</p>
      <div class="canvas-wrap" id="canvasWrap">
        <img class="sizer" id="modalImg">
        <canvas id="baseCanvas"></canvas>
        <canvas id="maskCanvas"></canvas>
      </div>
      <div class="modal-actions">
        <button class="secondary" id="clearMaskBtn">초기화</button>
        <button class="secondary" id="cancelModalBtn">취소</button>
        <button class="primary" id="applyMaskBtn">지우기 실행</button>
      </div>
    </div>
  </div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const downloadAllBtn = document.getElementById('downloadAllBtn');
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
  statusText.textContent = '워터마크 지우는 중... (이미지 수에 따라 시간이 걸릴 수 있습니다)';
  try {
    const res = await fetch('/api/process', { method: 'POST' });
    const s = await res.json();
    statusText.textContent = `총 ${s.total}장 · 성공 ${s.success}장 · 실패 ${s.failed}장` +
      (s.skipped ? ` · 이미 처리되어 건너뜀 ${s.skipped}장` : '');
  } finally {
    processBtn.disabled = false;
    refresh();
  }
});

downloadAllBtn.addEventListener('click', () => {
  window.location.href = '/api/download_all';
});

async function deleteFile(folder, filename, itemEl) {
  // Remove from view immediately — don't wait on the network round-trip to
  // confirm something that's already decided, and don't let a missed poll
  // tick make a deleted file look like it's still there.
  if (itemEl) { itemEl.style.opacity = '0.3'; itemEl.style.pointerEvents = 'none'; }
  try {
    const res = await fetch('/api/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder, filename }),
    });
    if (!res.ok) throw new Error('delete failed');
    if (itemEl) itemEl.remove();
  } catch (e) {
    statusText.textContent = `삭제 실패: ${filename}`;
    if (itemEl) { itemEl.style.opacity = '1'; itemEl.style.pointerEvents = 'auto'; }
  }
  updateCount(folder);
  refresh();
}

function updateCount(folder) {
  const el = document.getElementById(`list-${folder}`);
  document.getElementById(`count-${folder}`).textContent = el.querySelectorAll('.item').length;
}

function renderList(folder, files) {
  const el = document.getElementById(`list-${folder}`);
  document.getElementById(`count-${folder}`).textContent = files.length;
  if (!files.length) { el.innerHTML = '<div class="empty">비어 있음</div>'; return; }
  el.innerHTML = files.map(f => {
    const name = f.name;
    const escaped = name.replace(/'/g, "\\'");
    const src = `/files/${folder}/${encodeURIComponent(name)}?v=${f.mtime}`;
    return `
    <div class="item">
      <img src="${src}" loading="lazy" onclick="openManualEditor('${folder}', '${escaped}')">
      <div class="name" title="${name}">${name}</div>
      ${folder !== 'output' ? `<button class="link" onclick="openManualEditor('${folder}', '${escaped}')">직접선택</button>` : ''}
      ${folder === 'output' ? `<a class="dl" href="${src}" download>저장</a>` : ''}
      <button class="del" title="삭제" onclick="deleteFile('${folder}', '${escaped}', this.closest('.item'))">×</button>
    </div>
  `;
  }).join('');
}

async function refresh() {
  const res = await fetch('/api/files', { cache: 'no-store' });
  const data = await res.json();
  renderList('input', data.input);
  renderList('output', data.output);
  renderList('failed', data.failed);
}

refresh();
setInterval(refresh, 3000);

// ---- Manual brush editor ----
const BRUSH_RADIUS = 16; // fixed on-screen radius in canvas pixels — not user-adjustable, per request
const modalOverlay = document.getElementById('modalOverlay');
const modalImg = document.getElementById('modalImg');
const baseCanvas = document.getElementById('baseCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const canvasWrap = document.getElementById('canvasWrap');
const baseCtx = baseCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
let currentFolder = null, currentFilename = null, painting = false;

function openManualEditor(folder, filename) {
  currentFolder = folder;
  currentFilename = filename;
  const img = new Image();
  img.onload = () => {
    const maxW = Math.min(820, window.innerWidth * 0.86);
    const maxH = window.innerHeight * 0.62;
    const scale = Math.min(maxW / img.width, maxH / img.height, 1);
    const dw = Math.round(img.width * scale), dh = Math.round(img.height * scale);
    [modalImg, baseCanvas, maskCanvas].forEach(el => { el.width = dw; el.height = dh; });
    modalImg.style.width = dw + 'px'; modalImg.style.height = dh + 'px';
    canvasWrap.style.width = dw + 'px'; canvasWrap.style.height = dh + 'px';
    baseCtx.drawImage(img, 0, 0, dw, dh);
    maskCtx.clearRect(0, 0, dw, dh);
    modalOverlay.classList.add('open');
  };
  img.src = `/files/${folder}/${encodeURIComponent(filename)}`;
}

function maskPos(evt) {
  const rect = maskCanvas.getBoundingClientRect();
  const scaleX = maskCanvas.width / rect.width, scaleY = maskCanvas.height / rect.height;
  return { x: (evt.clientX - rect.left) * scaleX, y: (evt.clientY - rect.top) * scaleY };
}

function paintAt(x, y) {
  maskCtx.fillStyle = 'rgba(255,60,60,0.55)';
  maskCtx.beginPath();
  maskCtx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
  maskCtx.fill();
}

maskCanvas.addEventListener('mousedown', e => { painting = true; const p = maskPos(e); paintAt(p.x, p.y); });
maskCanvas.addEventListener('mousemove', e => { if (painting) { const p = maskPos(e); paintAt(p.x, p.y); } });
window.addEventListener('mouseup', () => { painting = false; });

document.getElementById('clearMaskBtn').addEventListener('click', () => {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
});
document.getElementById('cancelModalBtn').addEventListener('click', () => {
  modalOverlay.classList.remove('open');
});

document.getElementById('applyMaskBtn').addEventListener('click', async () => {
  // build a plain black/white mask (white = erase) at the same size as the display canvas;
  // the server scales it up to the original image resolution.
  const w = maskCanvas.width, h = maskCanvas.height;
  const src = maskCtx.getImageData(0, 0, w, h).data;
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const outCtx = out.getContext('2d');
  const outData = outCtx.createImageData(w, h);
  for (let i = 0; i < src.length; i += 4) {
    const painted = src[i + 3] > 0 ? 255 : 0;
    outData.data[i] = outData.data[i + 1] = outData.data[i + 2] = painted;
    outData.data[i + 3] = 255;
  }
  outCtx.putImageData(outData, 0, 0);
  const maskDataUrl = out.toDataURL('image/png');

  const btn = document.getElementById('applyMaskBtn');
  btn.disabled = true;
  statusText.textContent = '직접 선택한 영역 지우는 중...';
  try {
    const res = await fetch('/api/manual_process', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder: currentFolder, filename: currentFilename, mask: maskDataUrl }),
    });
    const result = await res.json();
    statusText.textContent = result.error ? `실패: ${result.error}` : `완료: ${result.saved}`;
    modalOverlay.classList.remove('open');
    refresh();
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""
