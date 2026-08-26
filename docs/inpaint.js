/*
 * Client-side LaMa inpainting via ONNX Runtime Web.
 *
 * Model: Carve/LaMa-ONNX (lama_fp32.onnx), a fork of advimman/lama re-worked
 * to avoid the FFT ops PyTorch's ONNX exporter can't handle. Apache-2.0.
 * See README.md "Attribution & Sources".
 *
 * The model takes a FIXED 512x512 image + mask. To keep full-resolution
 * output for the whole photo, only a padded patch around the watermark box
 * is cropped, resized to 512x512, inpainted, resized back, and composited
 * (with a small feathered edge) into the original-resolution image — the
 * rest of the photo is never touched or resampled.
 */

const MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx";
const MODEL_INPUT_SIZE = 512;
const PATCH_PADDING_MULTIPLIER = 2.5; // how much surrounding context to give the model
const MIN_PATCH_PADDING_PX = 80;
const FEATHER_PX = 10;

let _sessionPromise = null;

function loadSession(onProgress) {
  if (_sessionPromise) return _sessionPromise;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  _sessionPromise = (async () => {
    onProgress && onProgress("모델 다운로드 중... (최초 1회, 약 200MB)");
    const resp = await fetch(MODEL_URL);
    if (!resp.ok) throw new Error(`모델 다운로드 실패: ${resp.status}`);
    const total = Number(resp.headers.get("content-length")) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total && onProgress) {
        const pct = ((received / total) * 100).toFixed(0);
        onProgress(`모델 다운로드 중... ${pct}% (최초 1회만)`);
      }
    }
    const modelBuffer = await new Blob(chunks).arrayBuffer();
    onProgress && onProgress("모델 로딩 중...");
    // WebGPU crashes mid-inference on this model's patched Fourier-conv ops
    // ("[Add] .../ffc/convg2g/Add failed") — confirmed via a real user
    // report, not just theory. That failure happens during kernel
    // execution, after the session is already committed to WebGPU, so
    // onnxruntime-web's own provider fallback never kicks in. wasm is what
    // was actually verified end-to-end (via CPUExecutionProvider in
    // Python) to produce correct output, so it's the only provider used.
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
    });
    onProgress && onProgress("준비 완료");
    return session;
  })();
  return _sessionPromise;
}

function canvasFromImageBitmapRegion(bitmap, x, y, w, h, outW, outH) {
  const canvas = document.createElement("canvas");
  canvas.width = outW; canvas.height = outH;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, x, y, w, h, 0, 0, outW, outH);
  return canvas;
}

function canvasToCHWFloat(canvas) {
  const ctx = canvas.getContext("2d");
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const out = new Float32Array(3 * width * height);
  const plane = width * height;
  for (let p = 0, i = 0; i < data.length; i += 4, p++) {
    out[p] = data[i] / 255;
    out[plane + p] = data[i + 1] / 255;
    out[2 * plane + p] = data[i + 2] / 255;
  }
  return out;
}

function maskCanvasToFloat(canvas) {
  const ctx = canvas.getContext("2d");
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const out = new Float32Array(width * height);
  for (let p = 0, i = 0; i < data.length; i += 4, p++) {
    out[p] = data[i + 3] > 10 ? 1 : 0; // alpha-channel painted mask
  }
  return out;
}

function outputTensorToCanvas(tensorData, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(width, height);
  const plane = width * height;
  for (let p = 0, i = 0; i < imgData.data.length; i += 4, p++) {
    imgData.data[i] = Math.max(0, Math.min(255, tensorData[p]));
    imgData.data[i + 1] = Math.max(0, Math.min(255, tensorData[plane + p]));
    imgData.data[i + 2] = Math.max(0, Math.min(255, tensorData[2 * plane + p]));
    imgData.data[i + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

/**
 * Inpaint `box` (in the source image's own pixel coords) within `bitmap`,
 * compositing the result onto `destCanvas` (already drawn with the full
 * original image at full resolution). Does not touch pixels outside the
 * padded patch.
 */
async function inpaintRegion(session, bitmap, destCanvas, box, maskCanvasFull) {
  const [bx1, by1, bx2, by2] = box;
  const bw = bx2 - bx1, bh = by2 - by1;
  const pad = Math.max(MIN_PATCH_PADDING_PX, Math.round(Math.max(bw, bh) * PATCH_PADDING_MULTIPLIER));

  const px1 = Math.max(0, bx1 - pad);
  const py1 = Math.max(0, by1 - pad);
  const px2 = Math.min(bitmap.width, bx2 + pad);
  const py2 = Math.min(bitmap.height, by2 + pad);
  const pw = px2 - px1, ph = py2 - py1;

  const imgPatchCanvas = canvasFromImageBitmapRegion(bitmap, px1, py1, pw, ph, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  // Build the 512x512 mask by cropping+resizing the same patch region from the full mask canvas.
  const maskPatchCanvas = document.createElement("canvas");
  maskPatchCanvas.width = MODEL_INPUT_SIZE; maskPatchCanvas.height = MODEL_INPUT_SIZE;
  const mctx = maskPatchCanvas.getContext("2d");
  mctx.drawImage(maskCanvasFull, px1, py1, pw, ph, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  const imageTensorData = canvasToCHWFloat(imgPatchCanvas);
  const maskFlat = maskCanvasToFloat(maskPatchCanvas);

  const imageTensor = new ort.Tensor("float32", imageTensorData, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const maskTensor = new ort.Tensor("float32", maskFlat, [1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);

  const results = await session.run({ image: imageTensor, mask: maskTensor });
  const outputName = session.outputNames[0];
  const outTensor = results[outputName];
  const resultCanvas512 = outputTensorToCanvas(outTensor.data, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  // Resize the inpainted 512x512 patch back to its native patch size.
  const resultPatchCanvas = document.createElement("canvas");
  resultPatchCanvas.width = pw; resultPatchCanvas.height = ph;
  const rctx = resultPatchCanvas.getContext("2d");
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = "high";
  rctx.drawImage(resultCanvas512, 0, 0, pw, ph);

  // Feathered composite: only the masked area (plus small padding) actually
  // changes; blend its edge softly into the untouched surrounding pixels.
  // The blur must be applied while drawing FROM a separate source canvas —
  // drawing a canvas onto itself with a filter active is undefined
  // behavior and was silently producing a blank/broken alpha mask, so
  // every "successful" result quietly changed nothing.
  const maskCropCanvas = document.createElement("canvas");
  maskCropCanvas.width = pw; maskCropCanvas.height = ph;
  maskCropCanvas.getContext("2d").drawImage(maskCanvasFull, px1, py1, pw, ph, 0, 0, pw, ph);

  const featherCanvas = document.createElement("canvas");
  featherCanvas.width = pw; featherCanvas.height = ph;
  const fctx = featherCanvas.getContext("2d");
  fctx.filter = `blur(${FEATHER_PX}px)`;
  fctx.drawImage(maskCropCanvas, 0, 0);
  fctx.filter = "none";
  const alphaMask = fctx.getImageData(0, 0, pw, ph);

  const resultData = rctx.getImageData(0, 0, pw, ph);
  const destCtx = destCanvas.getContext("2d");
  const baseData = destCtx.getImageData(px1, py1, pw, ph);
  for (let i = 0; i < baseData.data.length; i += 4) {
    const a = alphaMask.data[i + 3] / 255;
    for (let c = 0; c < 3; c++) {
      baseData.data[i + c] = resultData.data[i + c] * a + baseData.data[i + c] * (1 - a);
    }
  }
  destCtx.putImageData(baseData, px1, py1);
}

window.inpaintModule = { loadSession, inpaintRegion };
