// Caption OCR on the phone: collectors.ocr_crop / ocr_variants (bit exact with OpenCV) and a reader standing in for
// the PC's EasyOCR reader (collectors.create_ocr_reader), built on Tesseract.js (LSTM, Vietnamese model).
//
// Frames are RGBA ImageData-like {width, height, data}; Python's frames are BGR, the channel order is translated
// here. A grey image is {width, height, data: Uint8Array(width * height)} (what ocrCrop returns).
// ocrCrop reproduces OpenCV bit for bit: the crop with numpy's slicing, cv2.resize INTER_AREA (light.js) or
// INTER_CUBIC 2x (OpenCV's fixed-point taps, and the float32 SIMD rounding its x86 build uses on full 8-value
// groups), cvtColor BGR2GRAY (OpenCV's 15-bit fixed point), CLAHE (light.js Clahe) and the inversion.
// Golden test: mobile/tests/ocr.test.js (tools/golden_ocr.py).
//
// createOcr() is the only thing that downloads: Tesseract.js, its WebAssembly core and vie.traineddata come from
// vendor/tesseract/ on this site (tools/build_mobile.py, SHA-256 pinned), never from a CDN; importing this module
// fetches nothing.

import { Clahe, resizeArea, roundHalfEven } from "./light.js";

const f32 = Math.fround;

// ------------------------------------------------------------------ collectors.ocr_crop / ocr_variants

// Python's slice bounds for frame[start:stop] on an axis of `size` (start >= 0 here): a negative stop counts
// from the end, both are clipped to the axis.
function sliceBounds(start, stop, size) {
  const first = Math.min(start, size);
  const last = stop < 0 ? Math.max(0, stop + size) : Math.min(stop, size);
  return [first, Math.max(first, last)];
}

// cvtColor(COLOR_BGR2GRAY) of an 8-bit image with `stride` channels per pixel in R, G, B order (OpenCV 4's
// RGB2Gray<uchar>: 15-bit coefficients, rounded).
function toGrey(data, count, stride) {
  const grey = new Uint8Array(count);
  for (let i = 0, p = 0; i < count; i++, p += stride) {
    grey[i] = (data[p] * 9798 + data[p + 1] * 19235 + data[p + 2] * 3735 + 16384) >> 15;
  }
  return grey;
}

// OpenCV's interpolateCubic (A = -0.75) in float32, as saturate_cast<short>(coefficient * 2048).
function cubicTaps(x) {
  const A = f32(-0.75);
  const x1 = f32(x + 1), y = f32(1 - x);
  const c0 = f32(f32(f32(f32(f32(f32(A * x1) - f32(5 * A)) * x1) + f32(8 * A)) * x1) - f32(4 * A));
  const c1 = f32(f32(f32(f32(f32(f32(A + 2) * x) - f32(A + 3)) * x) * x) + 1);
  const c2 = f32(f32(f32(f32(f32(f32(A + 2) * y) - f32(A + 3)) * y) * y) + 1);
  const c3 = f32(f32(f32(1 - c0) - c1) - c2);
  return [c0, c1, c2, c3].map((c) => roundHalfEven(f32(c * 2048)));
}

// Source index and 4 taps of each destination index (cv::resize's tables for INTER_CUBIC).
function cubicTable(sourceSize, destinationSize) {
  const scale = 1 / (destinationSize / sourceSize);
  const index = new Int32Array(destinationSize * 4), taps = new Int32Array(destinationSize * 4);
  for (let d = 0; d < destinationSize; d++) {
    const position = f32((d + 0.5) * scale - 0.5);
    const s = Math.floor(position);
    const weights = cubicTaps(f32(position - s));
    for (let k = 0; k < 4; k++) {
      index[d * 4 + k] = Math.min(Math.max(s - 1 + k, 0), sourceSize - 1); // BORDER_REPLICATE
      taps[d * 4 + k] = weights[k];
    }
  }
  return { index, taps };
}

// cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) of the first 3 channels of an interleaved
// 8-bit image (`stride` channels per pixel) -> Uint8Array(2w * 2h * 3). Horizontal taps in int, vertical taps as
// OpenCV's VResizeCubicVec_32s8u (SSE baseline: float32 multiply-adds, rounded half to even) on every full group of
// 8 values of a row and the integer path ((sum + 2^21) >> 22) on the rest.
function resizeCubic2x(data, width, height, stride) {
  const dstWidth = width * 2, dstHeight = height * 2, rowLength = dstWidth * 3;
  const tx = cubicTable(width, dstWidth), ty = cubicTable(height, dstHeight);
  const rows = new Int32Array(height * rowLength);
  for (let y = 0; y < height; y++) {
    const source = y * width * stride, target = y * rowLength;
    for (let dx = 0; dx < dstWidth; dx++) {
      const i = dx * 4;
      const p0 = source + tx.index[i] * stride, p1 = source + tx.index[i + 1] * stride;
      const p2 = source + tx.index[i + 2] * stride, p3 = source + tx.index[i + 3] * stride;
      const a0 = tx.taps[i], a1 = tx.taps[i + 1], a2 = tx.taps[i + 2], a3 = tx.taps[i + 3];
      for (let c = 0; c < 3; c++) {
        rows[target + dx * 3 + c] = data[p0 + c] * a0 + data[p1 + c] * a1 + data[p2 + c] * a2 + data[p3 + c] * a3;
      }
    }
  }
  const out = new Uint8Array(dstHeight * rowLength);
  const vectorEnd = rowLength - (rowLength % 8);
  const scale = 2 ** -22; // 1 / (INTER_RESIZE_COEF_SCALE * INTER_RESIZE_COEF_SCALE), exact in float32
  for (let dy = 0; dy < dstHeight; dy++) {
    const i = dy * 4;
    const r0 = ty.index[i] * rowLength, r1 = ty.index[i + 1] * rowLength;
    const r2 = ty.index[i + 2] * rowLength, r3 = ty.index[i + 3] * rowLength;
    const b0 = ty.taps[i], b1 = ty.taps[i + 1], b2 = ty.taps[i + 2], b3 = ty.taps[i + 3];
    const f0 = b0 * scale, f1 = b1 * scale, f2 = b2 * scale, f3 = b3 * scale;
    const target = dy * rowLength;
    let x = 0;
    for (; x < vectorEnd; x++) {
      let sum = f32(rows[r3 + x] * f3);
      sum = f32(f32(rows[r2 + x] * f2) + sum);
      sum = f32(f32(rows[r1 + x] * f1) + sum);
      sum = f32(f32(rows[r0 + x] * f0) + sum);
      const value = roundHalfEven(sum);
      out[target + x] = value < 0 ? 0 : value > 255 ? 255 : value;
    }
    for (; x < rowLength; x++) {
      const sum = rows[r0 + x] * b0 + rows[r1 + x] * b1 + rows[r2 + x] * b2 + rows[r3 + x] * b3;
      const value = Math.floor((sum + 2097152) / 4194304);
      out[target + x] = value < 0 ? 0 : value > 255 ? 255 : value;
    }
  }
  return out;
}

// collectors.ocr_crop(frame, roi): the text region (x, y, width, height; null = the whole frame) as a grey image,
// at most 1280 wide (INTER_AREA) and doubled (INTER_CUBIC) when lower than 160 rows. Throws like OpenCV when the
// region leaves nothing of the frame.
export function ocrCrop(frame, roi = null) {
  const { width, height, data } = frame;
  if (!data || data.length !== width * height * 4) throw new Error("ocrCrop: the frame must be an RGBA image.");
  let x0 = 0, y0 = 0, x1 = width, y1 = height;
  if (roi && roi.length) { // Python: `if roi:` (None and an empty tuple mean the whole frame)
    const [x, y, w, h] = roi;
    if (![x, y, w, h].every(Number.isInteger)) throw new TypeError("ocrCrop: the region must be 4 integers.");
    [y0, y1] = sliceBounds(Math.max(0, y), y + h, height);
    [x0, x1] = sliceBounds(Math.max(0, x), x + w, width);
  }
  const cropWidth = x1 - x0, cropHeight = y1 - y0;
  let outWidth = cropWidth, outHeight = cropHeight;
  if (cropWidth > 1280) outWidth = 1280, outHeight = Math.trunc(cropHeight * 1280 / cropWidth);
  else if (cropHeight < 160) outWidth = cropWidth * 2, outHeight = cropHeight * 2;
  if (!(outWidth > 0 && outHeight > 0)) throw new Error("ocrCrop: the text region holds no pixel of the frame.");
  let crop = data, stride = 4;
  if (cropWidth !== width || cropHeight !== height) {
    crop = new Uint8Array(cropWidth * cropHeight * 4);
    for (let y = 0; y < cropHeight; y++) {
      const start = ((y0 + y) * width + x0) * 4;
      crop.set(data.subarray(start, start + cropWidth * 4), y * cropWidth * 4);
    }
  }
  if (cropWidth > 1280) {
    crop = resizeArea(crop, cropWidth, cropHeight, 4, outWidth, outHeight);
    stride = 3;
  } else if (cropHeight < 160) {
    crop = resizeCubic2x(crop, cropWidth, cropHeight, 4);
    stride = 3;
  }
  return { width: outWidth, height: outHeight, data: toGrey(crop, outWidth * outHeight, stride) };
}

let clahe = null;

// collectors.ocr_variants(frame, roi): [grey, CLAHE (clip 3, 8x8 tiles), inverted grey].
export function ocrVariants(frame, roi = null) {
  const grey = ocrCrop(frame, roi);
  clahe ??= new Clahe(3.0, 8, 8);
  const contrast = { width: grey.width, height: grey.height, data: clahe.apply(grey.data, grey.width, grey.height) };
  const inverted = new Uint8Array(grey.data.length);
  for (let i = 0; i < inverted.length; i++) inverted[i] = 255 - grey.data[i];
  return [grey, contrast, { width: grey.width, height: grey.height, data: inverted }];
}

// ------------------------------------------------------------------ the reader (Tesseract.js)
//
// EasyOCR reads grey crops of any polarity. Tesseract's layout analysis wants dark letters alone on white: a
// light bar inside darker video, or white letters on a dark badge, look to it like one big picture and it reads
// nothing. So an image is read as letters only, once as dark text and once as light text (lettersOf): Otsu's
// split of the grey values (split again on the text's side while that side is mostly badge), the ink's connected
// components that touch the border (the video around the caption), span the image's height (a badge's outline)
// or cover a fifth of it (a bar) removed. The surer reading wins: letters weighted by confidence above 50, so
// texture read as words counts against. An empty or unsure winner is tried one split deeper (white letters on a
// light badge, a hand touching a letter). Tesseract reads in sparse-text mode; a region left with only short,
// unsure or no boxes is read again as one line, since its layout analysis never takes a lone letter (a
// fingerspelled caption) for a line. Noise: words below MIN_WORD_CONFIDENCE or MIN_WORD_HEIGHT, punctuation
// alone or at a box's ends, boxes of at most 2 letters that are unsure, much lower than the caption, or alone
// without holding the image's letters.
// Measured on 164 caption crops against EasyOCR on the PC (tools/golden_ocr.py; mobile/tests/ocr.test.js logs
// the comparison): tuned on 92, checked on 72 held out.

export const OCR_LANGUAGE = "vie";
export const OCR_PAGE_MODE = 11;        // Tesseract PSM: sparse text (lines in any place of the region)
export const OCR_LINE_MODE = 7;         // PSM for one line: a box from detect(), or the retry of a lone letter
export const OCR_PAD = 16;              // white border around the image (Tesseract misses text touching an edge)
export const MIN_WORD_CONFIDENCE = 30;  // Tesseract confidence (0..100) below which a word is noise
export const SHORT_CONFIDENCE = 70;     // a box of at most 2 letters or digits needs this (noise is short)
export const SHORT_HEIGHT = 0.6;        // ... and at least this x the tallest box's height
export const LONE_LETTERS = 0.6;        // short boxes alone: share of the letters' ink they must hold,
export const LONE_HEIGHT = 20;          // and pixels one of them must be high
export const SURE_WORD = 80;            // a reading whose words are all this sure needs no deeper split
export const MAX_INK = 0.35;            // share of the image above which one side of a split cannot be letters
export const MIN_LETTERS = 0.01;        // ... and below which what is left of it after cleaning is no caption
export const MIN_WORD_HEIGHT = 10;      // pixels: smaller "words" are texture (ocrCrop doubles low regions)
export const MAX_SPLITS = 4;            // Otsu splits at most (lettersOf)
export const WORD_GAP = 0.5;            // a gap wider than this x the line height splits a line, as EasyOCR does
export const BOX_MARGIN = 0.25;         // read(image, {boxes}): each box is read with this x its height around it

// A grey copy of the image for Tesseract: grey images as they are, RGBA images through BGR2GRAY.
function greyOf(image) {
  const { width, height, data } = image || {};
  if (!data || !(width > 0) || !(height > 0)) throw new Error("OCR: the image must be {width, height, data}.");
  if (data.length === width * height) return { width, height, data };
  if (data.length === width * height * 4) return { width, height, data: toGrey(data, width * height, 4) };
  throw new Error("OCR: the image must be grey (width x height bytes) or RGBA.");
}

// Otsu's threshold of the histogram bins first..last (values <= threshold form the darker class).
function otsu(hist, first = 0, last = 255) {
  let count = 0, sum = 0;
  for (let i = first; i <= last; i++) count += hist[i], sum += i * hist[i];
  let below = 0, belowSum = 0, best = -1, threshold = first;
  for (let t = first; t < last; t++) {
    below += hist[t];
    belowSum += t * hist[t];
    const above = count - below;
    if (!below || !above) continue;
    const difference = belowSum / below - (sum - belowSum) / above;
    const between = below * above * difference * difference;
    if (between > best) best = between, threshold = t;
  }
  return threshold;
}

// Letters only, black on white ({width, height, data, ink}: `ink` letter pixels): the ink's 8-connected
// components without those touching the border, spanning 90 % of the height or covering a fifth of the image.
function letters(grey, mask) {
  const { width, height } = grey;
  const seen = new Uint8Array(mask.length), stack = new Int32Array(mask.length);
  const out = new Uint8Array(mask.length).fill(255);
  const component = [];
  let ink = 0;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    let top = 0, x0 = width, x1 = -1, y0 = height, y1 = -1;
    component.length = 0;
    stack[top++] = start;
    seen[start] = 1;
    while (top) {
      const p = stack[--top], x = p % width, y = (p - x) / width;
      component.push(p);
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      for (let ny = Math.max(0, y - 1); ny <= Math.min(height - 1, y + 1); ny++) {
        for (let nx = Math.max(0, x - 1); nx <= Math.min(width - 1, x + 1); nx++) {
          const q = ny * width + nx;
          if (mask[q] && !seen[q]) { seen[q] = 1; stack[top++] = q; }
        }
      }
    }
    if (x0 === 0 || y0 === 0 || x1 === width - 1 || y1 === height - 1) continue;
    if (y1 - y0 + 1 > 0.9 * height || component.length > 0.2 * width * height) continue;
    for (const p of component) out[p] = 0;
    ink += component.length;
  }
  return { width, height, data: out, ink };
}

// The letters of dark or light text: Otsu's split of the grey values, split again on the text's side while
// that side covers more than MAX_INK of the image and leaves almost no letters (white letters on a light badge
// in darker video: the badge goes with the letters until the second or third split), then `deeper` more times.
function lettersOf(grey, polarity, deeper = 0) {
  const { data } = grey, count = data.length;
  const hist = new Int32Array(256);
  for (let i = 0; i < count; i++) hist[data[i]]++;
  const inkCount = (t) => {
    let total = 0;
    if (polarity === "dark") for (let v = 0; v <= t; v++) total += hist[v];
    else for (let v = t + 1; v < 256; v++) total += hist[v];
    return total;
  };
  const split = (t) => (polarity === "dark" ? otsu(hist, 0, t) : otsu(hist, t + 1, 255));
  const mask = (t) => {
    const out = new Uint8Array(count);
    if (polarity === "dark") for (let i = 0; i < count; i++) out[i] = data[i] <= t;
    else for (let i = 0; i < count; i++) out[i] = data[i] > t;
    return out;
  };
  let threshold = otsu(hist), result = letters(grey, mask(threshold));
  for (let level = 1; level < MAX_SPLITS && result.ink < MIN_LETTERS * count && inkCount(threshold) > MAX_INK * count;
    level++) {
    threshold = split(threshold);
    result = letters(grey, mask(threshold));
  }
  for (let i = 0; i < deeper; i++) {
    threshold = split(threshold);
    result = letters(grey, mask(threshold));
  }
  return result;
}

// PGM bytes (Leptonica reads them without a codec) of the image on a white border of OCR_PAD pixels.
function pgm(grey) {
  const { width, height, data } = grey, pad = OCR_PAD;
  const paddedWidth = width + 2 * pad, paddedHeight = height + 2 * pad;
  const header = new TextEncoder().encode(`P5\n${paddedWidth} ${paddedHeight}\n255\n`);
  const bytes = new Uint8Array(header.length + paddedWidth * paddedHeight);
  bytes.set(header);
  const pixels = bytes.subarray(header.length);
  pixels.fill(255);
  for (let y = 0; y < height; y++) pixels.set(data.subarray(y * width, (y + 1) * width), (y + pad) * paddedWidth + pad);
  return bytes;
}

function crop(grey, left, top, right, bottom) {
  const width = right - left, height = bottom - top, data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    data.set(grey.data.subarray((top + y) * grey.width + left, (top + y) * grey.width + right), y * width);
  }
  return { width, height, data };
}

const LETTER = /[\p{Lu}\p{Ll}\p{Lt}\p{Lo}\p{N}]/gu; // letters and digits (not modifier letters such as ˆ)
const letterCount = (text) => (text.match(LETTER) || []).length;
const box4 = (x0, y0, x1, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];

// Words of a Tesseract result in image pixels ({text, confidence, x0, y0, x1, y1}), noise left out (unsure or
// tiny words); `words.unsure`: some word with letters (noise included) is less sure than SURE_WORD.
function wordsOf(blocks, left = 0, top = 0) {
  const words = [];
  words.unsure = false;
  for (const block of blocks || []) {
    for (const paragraph of block.paragraphs || []) {
      for (const line of paragraph.lines || []) {
        for (const word of line.words || []) {
          const text = word.text.normalize("NFC").trim();
          if (!text || word.bbox.y1 - word.bbox.y0 < MIN_WORD_HEIGHT || word.bbox.x1 <= word.bbox.x0) continue;
          if (letterCount(text) && word.confidence < SURE_WORD) words.unsure = true;
          if (word.confidence < MIN_WORD_CONFIDENCE) continue;
          words.push({ text, confidence: word.confidence, x0: word.bbox.x0 - OCR_PAD + left,
            y0: word.bbox.y0 - OCR_PAD + top, x1: word.bbox.x1 - OCR_PAD + left, y1: word.bbox.y1 - OCR_PAD + top });
        }
      }
    }
  }
  return words;
}

const meanConfidence = (words) => words.reduce((sum, word) => sum + word.confidence, 0) / words.length;
const isShort = (text) => letterCount(text) <= 2;
const keepBox = (words) => letterCount(words.map((word) => word.text).join("")) > 2
  || meanConfidence(words) >= SHORT_CONFIDENCE;

// Words -> EasyOCR-like detections: visual lines top to bottom (a word overlapping a line by half the height of
// the lower of the two joins it), each split where a gap exceeds WORD_GAP x the line's height;
// a box's text is its words joined by spaces, its confidence their mean (0..1). Short unsure boxes are noise, and
// short boxes much lower than the tallest one (a button, a fold read as a letter beside the caption).
function detectionsOf(words, width, height) {
  const lines = [];
  for (const word of [...words].sort((a, b) => (a.y0 + a.y1) - (b.y0 + b.y1))) {
    const line = lines.at(-1);
    const overlap = line ? Math.min(line.bottom, word.y1) - Math.max(line.top, word.y0) : 0;
    if (line && overlap >= 0.5 * Math.min(line.bottom - line.top, word.y1 - word.y0)) {
      line.words.push(word);
      line.top = Math.min(line.top, word.y0);
      line.bottom = Math.max(line.bottom, word.y1);
    } else lines.push({ words: [word], top: word.y0, bottom: word.y1 });
  }
  const clampX = (x) => Math.min(Math.max(x, 0), width), clampY = (y) => Math.min(Math.max(y, 0), height);
  const detections = [];
  for (const line of lines) {
    const sorted = line.words.sort((a, b) => a.x0 - b.x0), gap = WORD_GAP * (line.bottom - line.top);
    const groups = [[sorted[0]]];
    for (const word of sorted.slice(1)) {
      if (word.x0 - groups.at(-1).at(-1).x1 > gap) groups.push([word]);
      else groups.at(-1).push(word);
    }
    for (let group of groups) {
      // Punctuation stays between words ("Xin chào / Hello"), not alone or at a box's ends.
      const first = group.findIndex((w) => letterCount(w.text)), last = group.findLastIndex((w) => letterCount(w.text));
      if (first < 0) continue;
      group = group.slice(first, last + 1);
      if (!keepBox(group)) continue;
      const x0 = clampX(Math.min(...group.map((w) => w.x0))), x1 = clampX(Math.max(...group.map((w) => w.x1)));
      const y0 = clampY(Math.min(...group.map((w) => w.y0))), y1 = clampY(Math.max(...group.map((w) => w.y1)));
      detections.push([box4(x0, y0, x1, y1), group.map((w) => w.text).join(" "), meanConfidence(group) / 100]);
    }
  }
  const tallest = Math.max(0, ...detections.map(([box]) => box[2][1] - box[0][1]));
  return detections.filter(([box, text]) => !isShort(text) || box[2][1] - box[0][1] >= SHORT_HEIGHT * tallest);
}

// A reading of one letters image (lettersOf; words offset by left, top): its detections and how much sure text
// they hold (letters weighted by their box's confidence above 50). Boxes of at most 2 letters alone (a
// fingerspelled letter, or texture read as one) only count when one is LONE_HEIGHT high and most of the image's
// letters lie in them.
function readingOf(image, words, left, top) {
  const detections = detectionsOf(words, Infinity, Infinity);
  if (detections.length && detections.every(([, text]) => isShort(text))) {
    let inside = 0;
    for (const [box] of detections) {
      const margin = (box[2][1] - box[0][1]) * 0.3;
      const x0 = Math.max(0, Math.floor(box[0][0] - left - margin)), x1 = Math.min(image.width, Math.ceil(box[2][0] - left + margin));
      const y0 = Math.max(0, Math.floor(box[0][1] - top - margin)), y1 = Math.min(image.height, Math.ceil(box[2][1] - top + margin));
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) inside += image.data[y * image.width + x] === 0;
    }
    const high = detections.some(([box]) => box[2][1] - box[0][1] >= LONE_HEIGHT);
    if (!high || inside < LONE_LETTERS * image.ink) return { words: [], detections: [], score: 0 };
  }
  const score = detections.reduce((sum, [, text, confidence]) => sum + (confidence * 100 - 50) * letterCount(text), 0);
  return { words, detections, score };
}

// createOcr(options) -> Promise<{read(image, {boxes}), detect(image), close()}>: a Tesseract worker with the
// Vietnamese model, loaded from vendor/tesseract/ under `base` (the app's folder). Images are grey planes
// ({width, height, data: Uint8Array(width * height)}, what ocrCrop / ocrVariants return) or RGBA images.
// Calls run one at a time, in order.
//   read(image)          -> [[box (4 points [x, y] in image pixels), text, confidence 0..1], ...] like EasyOCR's
//                           readtext(image, detail=1, paragraph=False): one entry per text line (a line with a wide
//                           gap gives one entry per part, left to right), lines top to bottom; [] = no text.
//   detect(image)        -> [box, ...]: the boxes read(image) finds (EasyOCR's detect: where text is; [] = none).
//   read(image, {boxes}) -> like EasyOCR's recognize(image, boxes): each box read as one line, one entry
//                           [box, text, confidence] per box that holds text, with the box as given. On the very
//                           image detect() just measured, its own reading is returned without reading again.
//   close()              -> frees the worker; later calls reject.
// `onProgress(message)` receives Tesseract's messages ({status, progress}: loading, then each recognition).
export async function createOcr({ base = new URL("../", import.meta.url), onProgress = null } = {}) {
  const folder = new URL("vendor/tesseract/", base).href;
  const { default: Tesseract } = await import(new URL("tesseract.esm.min.js", folder).href);
  const worker = await Tesseract.createWorker(OCR_LANGUAGE, Tesseract.OEM.LSTM_ONLY, {
    workerPath: new URL("worker.min.js", folder).href,
    corePath: folder, // a folder: the worker takes tesseract-core-simd-lstm.wasm.js (or -lstm.wasm.js without SIMD)
    langPath: folder, // vie.traineddata.gz
    cacheMethod: "none", // the service worker keeps vendor/tesseract/: no second copy in IndexedDB
    workerBlobURL: false,
    logger: (message) => onProgress?.(message),
  });
  await worker.setParameters({ tessedit_pageseg_mode: String(OCR_PAGE_MODE), user_defined_dpi: "300" });
  let mode = OCR_PAGE_MODE, closed = false, queue = Promise.resolve();
  const measured = new WeakMap(); // image -> {boxes, detections} of the last detect()
  const run = (job) => {
    const next = queue.then(() => {
      if (closed) throw new Error("OCR: the reader is closed.");
      return job();
    });
    queue = next.catch(() => {});
    return next;
  };

  // The surest reading ({words, detections, score}) of `grey` in page segmentation mode `psm`; words are offset by
  // (left, top). Dark and light letters first; one split deeper when the better one is empty or unsure.
  async function bestReading(grey, psm, left = 0, top = 0) {
    if (psm !== mode) {
      await worker.setParameters({ tessedit_pageseg_mode: String(psm) });
      mode = psm;
    }
    let best = null, unsure = false;
    const tryImage = async (image) => {
      const { data } = await worker.recognize(pgm(image), {}, { text: false, blocks: true });
      const words = wordsOf(data.blocks, left, top), reading = readingOf(image, words, left, top);
      if (!best || reading.score > best.score) best = reading, unsure = words.unsure;
    };
    for (const polarity of ["dark", "light"]) await tryImage(lettersOf(grey, polarity));
    if (!best.detections.length || unsure) {
      for (const polarity of ["dark", "light"]) await tryImage(lettersOf(grey, polarity, 1));
    }
    return best;
  }

  // A region holding only short boxes, none or only unsure ones is read again as one line: Tesseract's layout
  // analysis does not take a lone letter for a text line. The surer reading wins.
  async function readAll(grey) {
    let reading = await bestReading(grey, OCR_PAGE_MODE);
    if (reading.detections.every(([, text, confidence]) => isShort(text) || confidence * 100 < SURE_WORD)) {
      const line = await bestReading(grey, OCR_LINE_MODE);
      if (line.score > reading.score) reading = line;
    }
    return detectionsOf(reading.words, grey.width, grey.height);
  }

  async function readBoxes(grey, boxes) {
    const detections = [];
    for (const box of boxes) {
      const xs = box.map((point) => point[0]), ys = box.map((point) => point[1]);
      const margin = Math.round((Math.max(...ys) - Math.min(...ys)) * BOX_MARGIN);
      const left = Math.max(0, Math.floor(Math.min(...xs)) - margin);
      const top = Math.max(0, Math.floor(Math.min(...ys)) - margin);
      const right = Math.min(grey.width, Math.ceil(Math.max(...xs)) + margin);
      const bottom = Math.min(grey.height, Math.ceil(Math.max(...ys)) + margin);
      if (right - left < 2 || bottom - top < 2) continue;
      let words = (await bestReading(crop(grey, left, top, right, bottom), OCR_LINE_MODE, left, top)).words
        .sort((a, b) => a.x0 - b.x0);
      const first = words.findIndex((w) => letterCount(w.text)), last = words.findLastIndex((w) => letterCount(w.text));
      words = first < 0 ? [] : words.slice(first, last + 1);
      if (!words.length || !keepBox(words)) continue;
      detections.push([box.map((point) => [point[0], point[1]]), words.map((word) => word.text).join(" "),
        meanConfidence(words) / 100]);
    }
    return detections;
  }

  return {
    read(image, { boxes = null } = {}) {
      const grey = greyOf(image);
      return run(async () => {
        if (!boxes) return readAll(grey);
        const known = measured.get(image);
        if (known && JSON.stringify(known.boxes) === JSON.stringify(boxes)) return known.detections;
        return readBoxes(grey, boxes);
      });
    },
    detect(image) {
      const grey = greyOf(image);
      return run(async () => {
        const detections = await readAll(grey);
        const boxes = detections.map((detection) => detection[0]);
        measured.set(image, { boxes, detections });
        return boxes;
      });
    },
    async close() {
      if (closed) return;
      await run(async () => { closed = true; });
      await worker.terminate();
    },
  };
}
