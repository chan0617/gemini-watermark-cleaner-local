/*
 * UI orchestration for the fully client-side Gemini watermark cleaner.
 * Everything here runs in the browser: no image ever leaves the machine.
 * The only network fetch is the one-time LaMa ONNX model download (a
 * public, non-personal asset), same idea as the local Python version
 * downloading its model weights once on first run.
 */

const SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".webp"];
const BRUSH_RADIUS = 16; // fixed, not user-adjustable — matches the local app's manual mode

let items = []; // {id, file, name, ext, status, bitmap, resultCanvas, confidence}
let nextId = 1;
let session = null;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const processBtn = document.getElementById("processBtn");
const downloadAllBtn = document.getElementById("downloadAllBtn");
const statusText = document.getElementById("statusText");
const modelStatus = document.getElementById("modelStatus");

function extOf(name) {
  const m = name.toLowerCase().match(/\.[a-z0-9]+$/);
  return m ? m[0] : "";
}

function mimeFor(ext) {
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  return "image/png";
}

async function loadBitmapFromFile(file) {
  return await createImageBitmap(file);
}

function addFiles(fileList) {
  for (const file of fileList) {
    const ext = extOf(file.name);
    if (!SUPPORTED_EXT.includes(ext)) continue;
    items.push({
      id: nextId++,
      file,
      name: file.name,
      ext,
      status: "pending",
      bitmap: null,
      resultCanvas: null,
      confidence: null,
    });
  }
  render();
}

dropzone.addEventListener("click", () => fileInput.click());
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });

async function ensureSession() {
  if (session) return session;
  processBtn.disabled = true;
  session = await inpaintModule.loadSession((msg) => { modelStatus.textContent = msg; });
  processBtn.disabled = false;
  modelStatus.textContent = "모델 준비 완료";
  return session;
}

async function buildMaskCanvasFromBox(width, height, box, padding) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(255,0,0,1)";
  const [x1, y1, x2, y2] = box;
  ctx.fillRect(
    Math.max(0, x1 - padding), Math.max(0, y1 - padding),
    Math.min(width, x2 + padding) - Math.max(0, x1 - padding),
    Math.min(height, y2 + padding) - Math.max(0, y1 - padding)
  );
  return canvas;
}

async function processOne(item) {
  item.status = "processing";
  render();
  try {
    if (!item.bitmap) item.bitmap = await loadBitmapFromFile(item.file);
    const bitmap = item.bitmap;

    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = bitmap.width; srcCanvas.height = bitmap.height;
    srcCanvas.getContext("2d").drawImage(bitmap, 0, 0);
    const imageData = srcCanvas.getContext("2d").getImageData(0, 0, bitmap.width, bitmap.height);

    const detection = await detectWatermark(imageData);
    if (!detection) {
      item.status = "failed";
      item.error = "워터마크를 찾지 못함 (자동 탐지 실패 — \"직접선택\"으로 수동 처리 가능)";
      render();
      return;
    }

    const sess = await ensureSession();
    const destCanvas = document.createElement("canvas");
    destCanvas.width = bitmap.width; destCanvas.height = bitmap.height;
    destCanvas.getContext("2d").drawImage(bitmap, 0, 0);

    const maskCanvas = await buildMaskCanvasFromBox(bitmap.width, bitmap.height, detection.box, 6);
    await inpaintModule.inpaintRegion(sess, bitmap, destCanvas, detection.box, maskCanvas);

    item.resultCanvas = destCanvas;
    item.confidence = detection.confidence;
    item.status = "done";
    item.error = null;
  } catch (err) {
    console.error(err);
    item.status = "failed";
    item.error = `오류: ${err.message || err}`;
  }
  render();
}

// Skips confidence-based detection entirely and erases the geometrically
// expected corner position — a fallback for when detection itself is the
// unreliable part, or a quick way to test whether the erase/fill step
// works independent of where the box comes from.
async function applyFixedPosition(item) {
  item.status = "processing";
  render();
  try {
    if (!item.bitmap) item.bitmap = await loadBitmapFromFile(item.file);
    const bitmap = item.bitmap;
    const box = getDefaultAnchorBox(bitmap.width, bitmap.height);

    const sess = await ensureSession();
    const destCanvas = document.createElement("canvas");
    destCanvas.width = bitmap.width; destCanvas.height = bitmap.height;
    destCanvas.getContext("2d").drawImage(bitmap, 0, 0);

    const maskCanvas = await buildMaskCanvasFromBox(bitmap.width, bitmap.height, box, 10);
    await inpaintModule.inpaintRegion(sess, bitmap, destCanvas, box, maskCanvas);

    item.resultCanvas = destCanvas;
    item.confidence = null;
    item.status = "done";
    item.error = null;
  } catch (err) {
    console.error(err);
    item.status = "failed";
    item.error = `오류: ${err.message || err}`;
  }
  render();
}

processBtn.addEventListener("click", async () => {
  processBtn.disabled = true;
  const pending = items.filter((it) => it.status === "pending" || it.status === "failed");
  let success = 0, failed = 0;
  for (let i = 0; i < pending.length; i++) {
    statusText.textContent = `워터마크 지우는 중... (${i + 1}/${pending.length})`;
    await processOne(pending[i]);
    if (pending[i].status === "done") success++; else failed++;
  }
  statusText.textContent = `완료: 총 ${pending.length}장 · 성공 ${success}장 · 실패 ${failed}장`;
  processBtn.disabled = false;
});

downloadAllBtn.addEventListener("click", async () => {
  const done = items.filter((it) => it.status === "done");
  if (!done.length) { statusText.textContent = "다운로드할 완료 이미지가 없습니다."; return; }
  statusText.textContent = "zip 파일 만드는 중...";
  const zip = new JSZip();
  for (const item of done) {
    const blob = await canvasToBlob(item.resultCanvas, item.ext);
    const stem = item.name.replace(/\.[^.]+$/, "");
    zip.file(`${stem}_clean${item.ext}`, blob);
  }
  const zipBlob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url; a.download = "cleaned_images.zip";
  a.click();
  URL.revokeObjectURL(url);
  statusText.textContent = "다운로드 완료";
});

function canvasToBlob(canvas, ext) {
  return new Promise((resolve) => canvas.toBlob(resolve, mimeFor(ext), 0.95));
}

function removeItem(id) {
  items = items.filter((it) => it.id !== id);
  render();
}

async function downloadOne(item) {
  const blob = await canvasToBlob(item.resultCanvas, item.ext);
  const stem = item.name.replace(/\.[^.]+$/, "");
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `${stem}_clean${item.ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function render() {
  const groups = { pending: [], done: [], failed: [] };
  for (const it of items) {
    const key = it.status === "processing" ? "pending" : it.status;
    (groups[key] || groups.pending).push(it);
  }
  renderList("input", groups.pending);
  renderList("output", groups.done);
  renderList("failed", groups.failed);
}

function thumbUrl(item) {
  if (item.status === "done" && item.resultCanvas) return item.resultCanvas.toDataURL("image/jpeg", 0.6);
  return null;
}

function renderList(folder, list) {
  const el = document.getElementById(`list-${folder}`);
  document.getElementById(`count-${folder}`).textContent = list.length;
  if (!list.length) { el.innerHTML = '<div class="empty">비어 있음</div>'; return; }
  el.innerHTML = "";
  for (const item of list) {
    const div = document.createElement("div");
    div.className = "item";

    const img = document.createElement("img");
    if (item.status === "done") {
      img.src = thumbUrl(item);
    } else if (item.bitmap) {
      const c = document.createElement("canvas");
      c.width = 40; c.height = 40;
      c.getContext("2d").drawImage(item.bitmap, 0, 0, 40, 40);
      img.src = c.toDataURL();
    } else {
      loadBitmapFromFile(item.file).then((bmp) => {
        item.bitmap = bmp;
        const c = document.createElement("canvas");
        c.width = 40; c.height = 40;
        c.getContext("2d").drawImage(bmp, 0, 0, 40, 40);
        img.src = c.toDataURL();
      });
    }
    img.title = item.status === "processing" ? "처리 중..." : "";
    img.onclick = () => openManualEditor(item);
    div.appendChild(img);

    const name = document.createElement("div");
    name.className = "name";
    if (item.status === "processing") {
      name.textContent = `${item.name} (처리 중...)`;
      name.title = item.name;
    } else if (item.status === "failed" && item.error) {
      name.textContent = `${item.name} — ${item.error}`;
      name.title = item.error;
    } else {
      name.textContent = item.name;
      name.title = item.name;
    }
    div.appendChild(name);

    if (folder !== "output") {
      const manualBtn = document.createElement("button");
      manualBtn.className = "link";
      manualBtn.textContent = "직접선택";
      manualBtn.onclick = () => openManualEditor(item);
      div.appendChild(manualBtn);

      const fixedBtn = document.createElement("button");
      fixedBtn.className = "link";
      fixedBtn.textContent = "고정위치";
      fixedBtn.title = "탐지를 건너뛰고 우측 하단 예상 위치를 바로 지웁니다";
      fixedBtn.onclick = () => applyFixedPosition(item);
      div.appendChild(fixedBtn);
    }
    if (folder === "output") {
      const dl = document.createElement("button");
      dl.className = "link";
      dl.textContent = "저장";
      dl.onclick = () => downloadOne(item);
      div.appendChild(dl);
    }

    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "×";
    del.title = "삭제";
    del.onclick = () => removeItem(item.id);
    div.appendChild(del);

    el.appendChild(div);
  }
}

// ---- Manual brush editor ----
const modalOverlay = document.getElementById("modalOverlay");
const modalImg = document.getElementById("modalImg");
const baseCanvas = document.getElementById("baseCanvas");
const maskDrawCanvas = document.getElementById("maskCanvas");
const canvasWrap = document.getElementById("canvasWrap");
const baseCtx = baseCanvas.getContext("2d");
const maskCtx = maskDrawCanvas.getContext("2d");
let painting = false;
let currentItem = null;
let displayScale = 1;

async function openManualEditor(item) {
  currentItem = item;
  if (!item.bitmap) item.bitmap = await loadBitmapFromFile(item.file);
  const bitmap = item.bitmap;
  const maxW = Math.min(820, window.innerWidth * 0.86);
  const maxH = window.innerHeight * 0.62;
  displayScale = Math.min(maxW / bitmap.width, maxH / bitmap.height, 1);
  const dw = Math.round(bitmap.width * displayScale), dh = Math.round(bitmap.height * displayScale);
  [baseCanvas, maskDrawCanvas].forEach((c) => { c.width = dw; c.height = dh; });
  canvasWrap.style.width = dw + "px"; canvasWrap.style.height = dh + "px";
  baseCtx.drawImage(item.status === "done" ? item.resultCanvas : bitmap, 0, 0, dw, dh);
  maskCtx.clearRect(0, 0, dw, dh);
  modalOverlay.classList.add("open");
}

function maskPos(evt) {
  const rect = maskDrawCanvas.getBoundingClientRect();
  const scaleX = maskDrawCanvas.width / rect.width, scaleY = maskDrawCanvas.height / rect.height;
  return { x: (evt.clientX - rect.left) * scaleX, y: (evt.clientY - rect.top) * scaleY };
}
function paintAt(x, y) {
  maskCtx.fillStyle = "rgba(255,60,60,0.55)";
  maskCtx.beginPath();
  maskCtx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
  maskCtx.fill();
}
maskDrawCanvas.addEventListener("mousedown", (e) => { painting = true; const p = maskPos(e); paintAt(p.x, p.y); });
maskDrawCanvas.addEventListener("mousemove", (e) => { if (painting) { const p = maskPos(e); paintAt(p.x, p.y); } });
window.addEventListener("mouseup", () => { painting = false; });

document.getElementById("clearMaskBtn").addEventListener("click", () => {
  maskCtx.clearRect(0, 0, maskDrawCanvas.width, maskDrawCanvas.height);
});
document.getElementById("cancelModalBtn").addEventListener("click", () => {
  modalOverlay.classList.remove("open");
});

document.getElementById("applyMaskBtn").addEventListener("click", async () => {
  const btn = document.getElementById("applyMaskBtn");
  btn.disabled = true;
  statusText.textContent = "직접 선택한 영역 지우는 중...";
  try {
    const item = currentItem;
    const bitmap = item.bitmap;

    // Scale the drawn mask up to full image resolution.
    const fullMask = document.createElement("canvas");
    fullMask.width = bitmap.width; fullMask.height = bitmap.height;
    fullMask.getContext("2d").drawImage(maskDrawCanvas, 0, 0, bitmap.width, bitmap.height);

    const destCanvas = document.createElement("canvas");
    destCanvas.width = bitmap.width; destCanvas.height = bitmap.height;
    destCanvas.getContext("2d").drawImage(item.status === "done" ? item.resultCanvas : bitmap, 0, 0);

    // Bounding box of the painted area, in full-resolution coordinates.
    const mctx = fullMask.getContext("2d");
    const data = mctx.getImageData(0, 0, fullMask.width, fullMask.height).data;
    let minX = fullMask.width, minY = fullMask.height, maxX = 0, maxY = 0, found = false;
    for (let y = 0; y < fullMask.height; y += 4) {
      for (let x = 0; x < fullMask.width; x += 4) {
        const a = data[(y * fullMask.width + x) * 4 + 3];
        if (a > 10) {
          found = true;
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
    }
    if (!found) { statusText.textContent = "칠한 영역이 없습니다."; btn.disabled = false; return; }

    const sess = await ensureSession();
    await inpaintModule.inpaintRegion(sess, bitmap, destCanvas, [minX, minY, maxX, maxY], fullMask);

    item.resultCanvas = destCanvas;
    item.status = "done";
    statusText.textContent = `완료: ${item.name}`;
    modalOverlay.classList.remove("open");
    render();
  } catch (err) {
    console.error(err);
    statusText.textContent = "직접 선택 처리 실패";
  } finally {
    btn.disabled = false;
  }
});

render();
ensureSession().catch((e) => { modelStatus.textContent = "모델 로딩 실패: " + e.message; });
