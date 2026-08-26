/*
 * Client-side port of src/detector.py — same anchors/thresholds, ported so
 * detection can run entirely in the browser (no server involved).
 * See README.md "Attribution & Sources" for where the watermark geometry
 * assumptions come from.
 */

const ANCHORS = [
  { margin: 0.045, jitter: 0.028 }, // ~2048px-class placement
  { margin: 0.095, jitter: 0.035 }, // ~1024px-class placement
];
const PRIMARY_ACCEPT_THRESHOLD = 0.40;
const CORNER_FRACTION = 0.20;
const FALLBACK_ACCEPT_THRESHOLD = 0.55;
const MIN_ROI_PX = 72;
// Fewer candidates than the Python version's 10, and a coarser stride —
// naive JS template matching has no FFT/integral-image speedup here, so
// this keeps a 2048px photo from taking tens of seconds (or hanging the
// tab) on the largest anchor/scale combinations.
const SIZE_FRACTIONS = [0.03, 0.045, 0.06, 0.08, 0.095];
const SEARCH_STRIDE = 4;

function toGray(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return { gray, width, height };
}

function sobelMagnitude(gray, width, height) {
  const mag = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const gx =
        -gray[i - width - 1] + gray[i - width + 1] +
        -2 * gray[i - 1] + 2 * gray[i + 1] +
        -gray[i + width - 1] + gray[i + width + 1];
      const gy =
        -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1] +
        gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
      mag[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  let max = 0;
  for (let i = 0; i < mag.length; i++) if (mag[i] > max) max = mag[i];
  if (max > 0) for (let i = 0; i < mag.length; i++) mag[i] = (mag[i] / max) * 255;
  return mag;
}

function gradientMagnitudeFromImageData(imageData) {
  const { gray, width, height } = toGray(imageData);
  return { mag: sobelMagnitude(gray, width, height), width, height };
}

// Resize a Float32Array "image" (grayscale) to a new size via canvas (bilinear).
function resizeGray(src, srcW, srcH, dstW, dstH) {
  const c1 = document.createElement("canvas");
  c1.width = srcW; c1.height = srcH;
  const ctx1 = c1.getContext("2d");
  const id = ctx1.createImageData(srcW, srcH);
  for (let i = 0; i < src.length; i++) {
    const v = Math.max(0, Math.min(255, src[i]));
    id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  ctx1.putImageData(id, 0, 0);

  const c2 = document.createElement("canvas");
  c2.width = dstW; c2.height = dstH;
  const ctx2 = c2.getContext("2d");
  ctx2.imageSmoothingEnabled = true;
  ctx2.imageSmoothingQuality = "high";
  ctx2.drawImage(c1, 0, 0, dstW, dstH);
  const outData = ctx2.getImageData(0, 0, dstW, dstH).data;
  const out = new Float32Array(dstW * dstH);
  for (let p = 0; p < out.length; p++) out[p] = outData[p * 4];
  return out;
}

function ncc(region, rw, template, tw, th, ox, oy) {
  const n = tw * th;
  let regionSum = 0, tplSum = 0;
  for (let y = 0; y < th; y++) {
    const rowBase = (oy + y) * rw + ox;
    const tRowBase = y * tw;
    for (let x = 0; x < tw; x++) {
      regionSum += region[rowBase + x];
      tplSum += template[tRowBase + x];
    }
  }
  const regionMean = regionSum / n, tplMean = tplSum / n;
  let num = 0, denomA = 0, denomB = 0;
  for (let y = 0; y < th; y++) {
    const rowBase = (oy + y) * rw + ox;
    const tRowBase = y * tw;
    for (let x = 0; x < tw; x++) {
      const rv = region[rowBase + x] - regionMean;
      const tv = template[tRowBase + x] - tplMean;
      num += rv * tv;
      denomA += rv * rv;
      denomB += tv * tv;
    }
  }
  const denom = Math.sqrt(denomA * denomB);
  return denom > 0 ? num / denom : 0;
}

// Slide `template` (tw x th) over `region` (rw x rh), return best {x, y, score}.
function matchTemplate(region, rw, rh, template, tw, th, stride) {
  let best = { x: 0, y: 0, score: -Infinity };
  for (let oy = 0; oy <= rh - th; oy += stride) {
    for (let ox = 0; ox <= rw - tw; ox += stride) {
      const score = ncc(region, rw, template, tw, th, ox, oy);
      if (score > best.score) best = { x: ox, y: oy, score };
    }
  }
  return best;
}

// A 4-point sparkle (astroid curve) — same fallback shape as detector.py's
// _synthetic_sparkle, used when no better template is available.
function drawSyntheticSparkle(canvasSize) {
  const canvas = document.createElement("canvas");
  canvas.width = canvasSize; canvas.height = canvasSize;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvasSize, canvasSize);
  ctx.fillStyle = "white";
  const cx = canvasSize / 2, cy = canvasSize / 2, a = canvasSize * 0.46;
  ctx.beginPath();
  const steps = 240;
  for (let i = 0; i < steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    const x = cx + a * Math.pow(Math.cos(t), 3);
    const y = cy + a * Math.pow(Math.sin(t), 3);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  const imgData = ctx.getImageData(0, 0, canvasSize, canvasSize);
  const gray = new Float32Array(canvasSize * canvasSize);
  for (let i = 0; i < gray.length; i++) gray[i] = imgData.data[i * 4];
  return gray;
}

// A real crop of the watermark matches far better than the synthetic
// approximation — the difference between honestly failing on a hard
// background and confidently locking onto the wrong spot. Persisted in
// localStorage (this browser's equivalent of a cache file) so it survives
// page reloads, and auto-bootstrapped from the first high-confidence hit
// so accuracy quietly improves with no user action needed.
const TEMPLATE_STORAGE_KEY = "gwc_watermark_template_v1";
const BOOTSTRAP_CONFIDENCE_THRESHOLD = 0.6;

function loadImageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

let _templateCache = null; // { gray: Float32Array, size: number }
async function getBaseTemplate() {
  if (_templateCache) return _templateCache;
  const stored = localStorage.getItem(TEMPLATE_STORAGE_KEY);
  if (stored) {
    try {
      const img = await loadImageFromDataUrl(stored);
      const canvas = document.createElement("canvas");
      canvas.width = img.width; canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height).data;
      const gray = new Float32Array(img.width * img.height);
      for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4];
      _templateCache = { gray, size: img.width };
      return _templateCache;
    } catch (e) {
      console.warn("stored watermark template failed to load, using synthetic", e);
    }
  }
  _templateCache = { gray: drawSyntheticSparkle(128), size: 128 };
  return _templateCache;
}

// Only bootstraps once — a confident-enough match is spatially accurate,
// so the very first one seeds the cache for every image after it.
function maybeBootstrapTemplate(imageData, box) {
  if (localStorage.getItem(TEMPLATE_STORAGE_KEY)) return;
  const pad = 8;
  const [x1, y1, x2, y2] = box;
  const cx1 = Math.max(0, x1 - pad), cy1 = Math.max(0, y1 - pad);
  const cx2 = Math.min(imageData.width, x2 + pad), cy2 = Math.min(imageData.height, y2 + pad);
  const w = cx2 - cx1, h = cy2 - cy1;
  if (w <= 0 || h <= 0) return;

  const full = document.createElement("canvas");
  full.width = imageData.width; full.height = imageData.height;
  full.getContext("2d").putImageData(imageData, 0, 0);
  const crop = document.createElement("canvas");
  crop.width = w; crop.height = h;
  crop.getContext("2d").drawImage(full, cx1, cy1, w, h, 0, 0, w, h);
  try {
    localStorage.setItem(TEMPLATE_STORAGE_KEY, crop.toDataURL("image/png"));
    _templateCache = null; // force reload from storage next call
  } catch (e) {
    console.warn("could not persist watermark template", e);
  }
}

const _yield = () => new Promise((r) => setTimeout(r, 0));

async function bestMatchInRoiAsync(fullMag, fullW, fullH, rx0, ry0, rx1, ry1, minDim) {
  rx0 = Math.max(0, rx0); ry0 = Math.max(0, ry0);
  rx1 = Math.min(fullW, rx1); ry1 = Math.min(fullH, ry1);
  const rw = rx1 - rx0, rh = ry1 - ry0;
  if (rw < 8 || rh < 8) return null;

  const region = new Float32Array(rw * rh);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      region[y * rw + x] = fullMag[(ry0 + y) * fullW + (rx0 + x)];
    }
  }

  const { gray: baseTemplate, size: baseSize } = await getBaseTemplate();
  let best = null;
  for (const frac of SIZE_FRACTIONS) {
    const size = Math.max(8, Math.round(minDim * frac));
    if (size > rw || size > rh) continue;
    const template = resizeGray(baseTemplate, baseSize, baseSize, size, size);
    const match = matchTemplate(region, rw, rh, template, size, size, SEARCH_STRIDE);
    if (!best || match.score > best.confidence) {
      best = {
        box: [rx0 + match.x, ry0 + match.y, rx0 + match.x + size, ry0 + match.y + size],
        confidence: match.score,
      };
    }
    await _yield(); // keep the tab responsive on large photos
  }
  return best;
}

/**
 * Detect the Gemini watermark in an ImageData. Returns {box:[x1,y1,x2,y2], confidence} or null.
 * Async so it periodically yields to the browser instead of freezing the tab
 * on large photos — this is plain JS template matching with no FFT/SIMD
 * speedup, so a 2048px image is genuinely a few hundred million operations.
 */
async function detectWatermark(imageData) {
  const width = imageData.width, height = imageData.height;
  const { mag } = gradientMagnitudeFromImageData(imageData);
  const minDim = Math.min(width, height);
  const maxSize = Math.max(8, Math.round(minDim * SIZE_FRACTIONS[SIZE_FRACTIONS.length - 1]));

  let bestPrimary = null;
  for (const { margin, jitter: jitterFrac } of ANCHORS) {
    const jitter = Math.round(minDim * jitterFrac);
    const anchorX = width - Math.round(minDim * margin);
    const anchorY = height - Math.round(minDim * margin);
    const candidate = await bestMatchInRoiAsync(
      mag, width, height,
      anchorX - maxSize - jitter, anchorY - maxSize - jitter,
      anchorX + jitter, anchorY + jitter,
      minDim
    );
    if (candidate && (!bestPrimary || candidate.confidence > bestPrimary.confidence)) {
      bestPrimary = candidate;
    }
  }
  if (bestPrimary && bestPrimary.confidence >= PRIMARY_ACCEPT_THRESHOLD) {
    if (bestPrimary.confidence >= BOOTSTRAP_CONFIDENCE_THRESHOLD) maybeBootstrapTemplate(imageData, bestPrimary.box);
    return bestPrimary;
  }

  const roiW = Math.max(MIN_ROI_PX, Math.round(width * CORNER_FRACTION));
  const roiH = Math.max(MIN_ROI_PX, Math.round(height * CORNER_FRACTION));
  const fallback = await bestMatchInRoiAsync(mag, width, height, width - roiW, height - roiH, width, height, minDim);
  if (fallback && fallback.confidence >= FALLBACK_ACCEPT_THRESHOLD) {
    if (fallback.confidence >= BOOTSTRAP_CONFIDENCE_THRESHOLD) maybeBootstrapTemplate(imageData, fallback.box);
    return fallback;
  }

  return null;
}
