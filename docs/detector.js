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
const SIZE_FRACTIONS = [0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095];
const SEARCH_STRIDE = 2; // coarser sliding step keeps this fast in plain JS

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

let _templateCache = null;
function getBaseTemplate() {
  if (!_templateCache) _templateCache = drawSyntheticSparkle(128);
  return _templateCache;
}

function bestMatchInRoi(fullMag, fullW, fullH, rx0, ry0, rx1, ry1, minDim) {
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

  const baseTemplate = getBaseTemplate();
  let best = null;
  for (const frac of SIZE_FRACTIONS) {
    const size = Math.max(8, Math.round(minDim * frac));
    if (size > rw || size > rh) continue;
    const template = resizeGray(baseTemplate, 128, 128, size, size);
    const match = matchTemplate(region, rw, rh, template, size, size, SEARCH_STRIDE);
    if (!best || match.score > best.confidence) {
      best = {
        box: [rx0 + match.x, ry0 + match.y, rx0 + match.x + size, ry0 + match.y + size],
        confidence: match.score,
      };
    }
  }
  return best;
}

/**
 * Detect the Gemini watermark in an ImageData. Returns {box:[x1,y1,x2,y2], confidence} or null.
 */
function detectWatermark(imageData) {
  const width = imageData.width, height = imageData.height;
  const { mag } = gradientMagnitudeFromImageData(imageData);
  const minDim = Math.min(width, height);
  const maxSize = Math.max(8, Math.round(minDim * SIZE_FRACTIONS[SIZE_FRACTIONS.length - 1]));

  let bestPrimary = null;
  for (const { margin, jitter: jitterFrac } of ANCHORS) {
    const jitter = Math.round(minDim * jitterFrac);
    const anchorX = width - Math.round(minDim * margin);
    const anchorY = height - Math.round(minDim * margin);
    const candidate = bestMatchInRoi(
      mag, width, height,
      anchorX - maxSize - jitter, anchorY - maxSize - jitter,
      anchorX + jitter, anchorY + jitter,
      minDim
    );
    if (candidate && (!bestPrimary || candidate.confidence > bestPrimary.confidence)) {
      bestPrimary = candidate;
    }
  }
  if (bestPrimary && bestPrimary.confidence >= PRIMARY_ACCEPT_THRESHOLD) return bestPrimary;

  const roiW = Math.max(MIN_ROI_PX, Math.round(width * CORNER_FRACTION));
  const roiH = Math.max(MIN_ROI_PX, Math.round(height * CORNER_FRACTION));
  const fallback = bestMatchInRoi(mag, width, height, width - roiW, height - roiH, width, height, minDim);
  if (fallback && fallback.confidence >= FALLBACK_ACCEPT_THRESHOLD) return fallback;

  return null;
}
