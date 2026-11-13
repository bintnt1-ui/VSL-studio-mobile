// Port of camera_enhancement.py: correct webcam exposure and local contrast before preview and detection.
//
// Images are RGBA ImageData-like objects {data: Uint8ClampedArray | Uint8Array, width, height} (what a canvas
// gives); Python works on BGR frames, so channel order is translated here and the result is the same picture.
// The OpenCV 8-bit routines camera_enhancement.py relies on are reproduced bit for bit (cv2.resize INTER_AREA,
// cvtColor BGR<->Lab, createCLAHE(1.6, (8, 8)).apply, LUT, addWeighted), float32 steps with Math.fround, so the
// phone corrects a frame exactly like the PC (golden test: mobile/tests/light.test.js).

const f32 = Math.fround;
const DBL_EPSILON = 2.220446049250313e-16;

// cvRound / np.round / Python round(): nearest integer, ties to even. Below 2^51, adding and subtracting
// 1.5 x 2^52 rounds a double exactly that way (no branches in the per-pixel loops).
const ROUNDER = 6755399441055744;
export function roundHalfEven(value) {
  if (Math.abs(value) < 2251799813685248) return (value + ROUNDER) - ROUNDER;
  const rounded = Math.round(value);
  return rounded - value === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}

// saturate_cast<uchar>(float): round half to even, then clamp to 0..255.
function saturate8(value) {
  const rounded = roundHalfEven(value);
  return rounded < 0 ? 0 : rounded > 255 ? 255 : rounded;
}

// ------------------------------------------------------------------ cv2.resize(..., INTER_AREA)

// OpenCV's computeResizeAreaTab: (destination index, source index, float32 weight) of each overlap.
function areaTable(sourceSize, destinationSize, scale) {
  const dst = [], src = [], weight = [];
  for (let dx = 0; dx < destinationSize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, sourceSize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, sourceSize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) {
      dst.push(dx); src.push(sx1 - 1); weight.push(f32((sx1 - fsx1) / cellWidth));
    }
    for (let sx = sx1; sx < sx2; sx++) {
      dst.push(dx); src.push(sx); weight.push(f32(1.0 / cellWidth));
    }
    if (fsx2 - sx2 > 1e-3) {
      dst.push(dx); src.push(sx2); weight.push(f32(Math.min(Math.min(fsx2 - sx2, 1.0), cellWidth) / cellWidth));
    }
  }
  return { dst: Int32Array.from(dst), src: Int32Array.from(src), weight: Float32Array.from(weight) };
}

const areaTables = new Map(); // "w,h,dw,dh" -> tables, so a video stream builds them once

// cv2.resize(image, (dstWidth, dstHeight), interpolation=cv2.INTER_AREA) of the first 3 channels of an
// interleaved 8-bit image with `stride` channels per pixel (4 for RGBA). Returns Uint8Array(dstWidth*dstHeight*3)
// in the source's channel order. Downscaling is exact (integer factors: ResizeAreaFast, 2x2 rounds halves up
// like OpenCV's SIMD path; other factors: the float32 area weights). Enlarging (never needed for camera frames)
// falls back to OpenCV's area-mode bilinear weights computed exactly (OpenCV's SIMD rounding is not reproduced).
export function resizeArea(data, width, height, stride, dstWidth, dstHeight, out = null) {
  const dst = out && out.length === dstWidth * dstHeight * 3 ? out : new Uint8Array(dstWidth * dstHeight * 3);
  if (dstWidth === width && dstHeight === height) {
    for (let i = 0, j = 0; j < dst.length; i += stride, j += 3) {
      dst[j] = data[i]; dst[j + 1] = data[i + 1]; dst[j + 2] = data[i + 2];
    }
    return dst;
  }
  const scaleX = 1 / (dstWidth / width), scaleY = 1 / (dstHeight / height);
  if (scaleX >= 1 && scaleY >= 1) {
    const integerX = roundHalfEven(scaleX), integerY = roundHalfEven(scaleY);
    if (Math.abs(scaleX - integerX) < DBL_EPSILON && Math.abs(scaleY - integerY) < DBL_EPSILON) {
      return resizeAreaFast(data, width, stride, dstWidth, dstHeight, integerX, integerY, dst);
    }
    const key = `${width},${height},${dstWidth},${dstHeight}`;
    let tables = areaTables.get(key);
    if (!tables) {
      if (areaTables.size > 16) areaTables.clear();
      tables = { x: areaTable(width, dstWidth, scaleX), y: areaTable(height, dstHeight, scaleY) };
      areaTables.set(key, tables);
    }
    return resizeAreaGeneric(data, width, stride, dstWidth, dstHeight, tables, dst);
  }
  return resizeAreaUp(data, width, height, stride, dstWidth, dstHeight, dst);
}

function resizeAreaFast(data, width, stride, dstWidth, dstHeight, scaleX, scaleY, dst) {
  const rowStride = width * stride;
  if (scaleX === 2 && scaleY === 2) { // ResizeAreaFastVec: (a + b + c + d + 2) >> 2
    for (let dy = 0, o = 0; dy < dstHeight; dy++) {
      const top = 2 * dy * rowStride, bottom = top + rowStride;
      for (let dx = 0; dx < dstWidth; dx++, o += 3) {
        const p = top + 2 * dx * stride, q = bottom + 2 * dx * stride;
        for (let c = 0; c < 3; c++) {
          dst[o + c] = (data[p + c] + data[p + stride + c] + data[q + c] + data[q + stride + c] + 2) >> 2;
        }
      }
    }
    return dst;
  }
  const scale = f32(1 / (scaleX * scaleY));
  for (let dy = 0, o = 0; dy < dstHeight; dy++) {
    const top = dy * scaleY * rowStride;
    for (let dx = 0; dx < dstWidth; dx++, o += 3) {
      const left = top + dx * scaleX * stride;
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let sy = 0, row = left + c; sy < scaleY; sy++, row += rowStride) {
          for (let sx = 0, p = row; sx < scaleX; sx++, p += stride) sum += data[p];
        }
        dst[o + c] = saturate8(f32(sum * scale));
      }
    }
  }
  return dst;
}

const areaBuffers = { buf: new Float32Array(0), sum: new Float32Array(0) };

function resizeAreaGeneric(data, width, stride, dstWidth, dstHeight, tables, dst) {
  const size = dstWidth * 3;
  if (areaBuffers.buf.length < size) {
    areaBuffers.buf = new Float32Array(size);
    areaBuffers.sum = new Float32Array(size);
  }
  const { buf, sum } = areaBuffers; // Float32Array stores round every value to float32 like OpenCV's WT=float
  const xd = tables.x.dst, xs = tables.x.src, xw = tables.x.weight;
  const yd = tables.y.dst, ys = tables.y.src, yw = tables.y.weight;
  const rowStride = width * stride;
  let k = 0;
  for (let dy = 0; dy < dstHeight; dy++) {
    sum.fill(0, 0, size);
    for (; k < yd.length && yd[k] === dy; k++) {
      const beta = yw[k], row = ys[k] * rowStride;
      buf.fill(0, 0, size);
      for (let t = 0; t < xd.length; t++) {
        const alpha = xw[t], p = row + xs[t] * stride, o = xd[t] * 3;
        buf[o] = buf[o] + f32(data[p] * alpha);
        buf[o + 1] = buf[o + 1] + f32(data[p + 1] * alpha);
        buf[o + 2] = buf[o + 2] + f32(data[p + 2] * alpha);
      }
      for (let i = 0; i < size; i++) sum[i] = sum[i] + f32(beta * buf[i]);
    }
    const base = dy * size;
    for (let i = 0; i < size; i++) dst[base + i] = saturate8(sum[i]);
  }
  return dst;
}

// INTER_AREA when enlarging: bilinear with OpenCV's area-mode weights (fixed point, 11 bits per axis).
function resizeAreaUp(data, width, height, stride, dstWidth, dstHeight, dst) {
  const weights = (sourceSize, destinationSize) => {
    const invScale = destinationSize / sourceSize, scale = 1 / invScale;
    const index = new Int32Array(destinationSize), next = new Int32Array(destinationSize);
    const alpha = new Int32Array(destinationSize);
    for (let d = 0; d < destinationSize; d++) {
      let s = Math.floor(d * scale);
      let fraction = f32((d + 1) - (s + 1) * invScale);
      fraction = fraction <= 0 ? 0 : f32(fraction - Math.floor(fraction));
      if (s >= sourceSize - 1) { s = sourceSize - 1; fraction = 0; }
      index[d] = s;
      next[d] = Math.min(s + 1, sourceSize - 1);
      alpha[d] = roundHalfEven(fraction * 2048);
    }
    return { index, next, alpha };
  };
  const wx = weights(width, dstWidth), wy = weights(height, dstHeight);
  const rowStride = width * stride;
  for (let dy = 0, o = 0; dy < dstHeight; dy++) {
    const r0 = wy.index[dy] * rowStride, r1 = wy.next[dy] * rowStride, b1 = wy.alpha[dy], b0 = 2048 - b1;
    for (let dx = 0; dx < dstWidth; dx++, o += 3) {
      const c0 = wx.index[dx] * stride, c1 = wx.next[dx] * stride, a1 = wx.alpha[dx], a0 = 2048 - a1;
      for (let c = 0; c < 3; c++) {
        const top = data[r0 + c0 + c] * a0 + data[r0 + c1 + c] * a1;
        const bottom = data[r1 + c0 + c] * a0 + data[r1 + c1 + c] * a1;
        dst[o + c] = Math.min(255, Math.max(0, (top * b0 + bottom * b1 + (1 << 21)) >> 22));
      }
    }
  }
  return dst;
}

// ------------------------------------------------------------------ cvtColor BGR <-> Lab (8 bit, bit exact)
// OpenCV's RGB2Lab_b and Lab2RGBinteger (color_lab.cpp): fixed-point arithmetic over tables built in float32
// (softfloat) and float64 (softdouble).

const LAB_SHIFT = 12, LAB_SHIFT2 = 15, BASE = 1 << 14, MIN_AB = -8145;
const D65 = [0.950456, 1.0, 1.088754];
const SRGB_TO_XYZ = [0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227];
const XYZ_TO_SRGB = [3.240479, -1.53715, -0.498535, -0.969256, 1.875991, 0.041556, 0.055648, -0.204043, 1.057311];
const GAMMA_THRESHOLD = 809 / 20000, GAMMA_LOW_SCALE = 323 / 25, GAMMA_POWER = 12 / 5, GAMMA_X_SHIFT = 11 / 200;
const INV_GAMMA_THRESHOLD = GAMMA_THRESHOLD / GAMMA_LOW_SCALE;

function applyGamma(x) { // softfloat in, softdouble maths, softfloat out
  return f32(x <= GAMMA_THRESHOLD ? x / GAMMA_LOW_SCALE : Math.pow((x + GAMMA_X_SHIFT) / (1 + GAMMA_X_SHIFT), GAMMA_POWER));
}

function applyInvGamma(x) {
  return f32(x <= INV_GAMMA_THRESHOLD ? x * GAMMA_LOW_SCALE
    : Math.pow(x, 1 / GAMMA_POWER) * (1 + GAMMA_X_SHIFT) - GAMMA_X_SHIFT);
}

// OpenCV's softfloat cbrt (x > 0, float32): Turkowski's rational approximation in float64 on the mantissa
// scaled to [0.125, 1), truncated (not rounded) to float32. Rounding instead changes 2 of the 8-bit Lab
// cube-root table's entries and 17645 of the 2^24 colours (checked against cv2.cvtColor on every colour).
const cbrtView = new DataView(new ArrayBuffer(4));
function softCbrt(x) {
  cbrtView.setFloat32(0, x);
  const bits = cbrtView.getUint32(0);
  const exponent = ((bits >>> 23) & 255) - 127;
  let shift = exponent % 3;
  if (shift >= 0) shift -= 3;
  const fr = (1 + (bits & 0x7fffff) / 8388608) * 2 ** shift;
  const root = ((((45.2548339756803022511987494 * fr + 192.2798368355061050458134625) * fr
    + 119.1654824285581628956914143) * fr + 13.43250139086239872172837314) * fr + 0.1636161226585754240958355063)
    / ((((14.80884093219134573786480845 * fr + 151.9714051044435648658557668) * fr
      + 168.5254414101568283957668343) * fr + 33.9905941350215598754191872) * fr + 1.0);
  let truncated = f32(root); // root in [0.5, 1): step down one float32 when rounding went up
  if (truncated > root) {
    cbrtView.setFloat32(0, truncated);
    cbrtView.setUint32(0, cbrtView.getUint32(0) - 1);
    truncated = cbrtView.getFloat32(0);
  }
  return truncated * 2 ** ((exponent - shift) / 3);
}

// to Lab: sRGB gamma (x 2040), cube root (x 32768), XYZ coefficients (x 4096) per (X|Y|Z) x (R|G|B).
const GAMMA_TABLE = new Int32Array(256);
const CBRT_TABLE = new Int32Array(256 * 3 / 2 * 8);
const TO_XYZ = new Int32Array(9);
// from Lab: L -> (y, f(y)) x 16384, f(x|z) -> x|z, linear -> sRGB byte, sRGB rows (R|G|B) x (X|Y|Z) x 4096.
const L_TO_Y = new Int32Array(512);
const AB_TO_XZ = new Int32Array(BASE * 9 / 4);
const INV_GAMMA_TABLE = new Uint8Array(4096);
const FROM_XYZ = new Int32Array(9);

(function buildLabTables() {
  const intScale = f32(255 * 8);
  for (let i = 0; i < 256; i++) GAMMA_TABLE[i] = roundHalfEven(f32(intScale * applyGamma(f32(i / 255))));
  const lthresh = f32(216 / 24389), lscale = f32(841 / 108), lbias = f32(16 / 116);
  const cbTabScale = f32(1 / f32(255 * 8));
  for (let i = 0; i < CBRT_TABLE.length; i++) {
    const x = f32(cbTabScale * i);
    const value = x < lthresh ? f32(x * lscale + lbias) : softCbrt(x); // mulAdd: one rounding
    CBRT_TABLE[i] = roundHalfEven(f32(32768 * value));
  }
  for (let i = 0; i < 4096; i++) INV_GAMMA_TABLE[i] = roundHalfEven(f32(255 * applyInvGamma(i / 4096)));
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      TO_XYZ[row * 3 + column] = roundHalfEven(4096 * SRGB_TO_XYZ[row * 3 + column] / D65[row]);
      FROM_XYZ[row * 3 + column] = roundHalfEven(4096 * XYZ_TO_SRGB[row * 3 + column] * D65[column]);
    }
  }
  for (let i = 0; i < 256; i++) {
    let y, ify;
    if (i <= 20) {
      y = roundHalfEven(f32(f32(i * BASE * 20 * 9) / f32(17 * 29 * 29 * 29)));
      ify = roundHalfEven(f32(BASE * f32(f32(16 / 116) + f32(f32(i * 5) / f32(3 * 17 * 29)))));
    } else {
      const fy = f32(f32(f32(i * 100 * BASE) / f32(255 * 116)) + f32(f32(16 * BASE) / f32(116)));
      ify = roundHalfEven(fy);
      y = roundHalfEven(f32(f32(f32(fy * fy) * fy) / f32(BASE * BASE)));
    }
    L_TO_Y[i * 2] = y;
    L_TO_Y[i * 2 + 1] = ify;
  }
  const offset = Math.trunc(Math.trunc(Math.trunc(BASE * 16 / 116) * 108) / 841);
  for (let i = MIN_AB; i < BASE * 9 / 4 + MIN_AB; i++) {
    AB_TO_XZ[i - MIN_AB] = i <= 3390 ? Math.trunc(i * 108 / 841) - offset
      : Math.trunc(Math.trunc(Math.trunc(i * i / BASE) * i) / BASE);
  }
})();

// Lab lightness (OpenCV 8-bit scale 0..255) of one sRGB pixel.
function lightness(r, g, b) {
  const R = GAMMA_TABLE[r], G = GAMMA_TABLE[g], B = GAMMA_TABLE[b];
  const fY = CBRT_TABLE[(R * TO_XYZ[3] + G * TO_XYZ[4] + B * TO_XYZ[5] + 2048) >> LAB_SHIFT];
  const L = (296 * fY - 1336934 + 16384) >> LAB_SHIFT2;
  return L < 0 ? 0 : L > 255 ? 255 : L;
}

// cv2.cvtColor(BGR -> LAB) of an interleaved image (`stride` channels per pixel, R, G, B first): fills the
// L, a, b planes (Uint8Array(width*height) each).
export function toLab(data, count, stride, L, A, B) {
  const x0 = TO_XYZ[0], x1 = TO_XYZ[1], x2 = TO_XYZ[2], y0 = TO_XYZ[3], y1 = TO_XYZ[4], y2 = TO_XYZ[5];
  const z0 = TO_XYZ[6], z1 = TO_XYZ[7], z2 = TO_XYZ[8];
  for (let i = 0, p = 0; i < count; i++, p += stride) {
    const r = GAMMA_TABLE[data[p]], g = GAMMA_TABLE[data[p + 1]], b = GAMMA_TABLE[data[p + 2]];
    const fX = CBRT_TABLE[(r * x0 + g * x1 + b * x2 + 2048) >> LAB_SHIFT];
    const fY = CBRT_TABLE[(r * y0 + g * y1 + b * y2 + 2048) >> LAB_SHIFT];
    const fZ = CBRT_TABLE[(r * z0 + g * z1 + b * z2 + 2048) >> LAB_SHIFT];
    const l = (296 * fY - 1336934 + 16384) >> LAB_SHIFT2;
    const a = (500 * (fX - fY) + 4194304 + 16384) >> LAB_SHIFT2;
    const bb = (200 * (fY - fZ) + 4194304 + 16384) >> LAB_SHIFT2;
    L[i] = l < 0 ? 0 : l > 255 ? 255 : l;
    A[i] = a < 0 ? 0 : a > 255 ? 255 : a;
    B[i] = bb < 0 ? 0 : bb > 255 ? 255 : bb;
  }
}

// cv2.cvtColor(LAB -> BGR) of L, a, b planes into an interleaved image (`stride` channels, R, G, B first; a 4th
// channel, if any, is left as it is).
export function fromLab(L, A, B, count, out, stride) {
  const r0 = FROM_XYZ[0], r1 = FROM_XYZ[1], r2 = FROM_XYZ[2], g0 = FROM_XYZ[3], g1 = FROM_XYZ[4], g2 = FROM_XYZ[5];
  const b0 = FROM_XYZ[6], b1 = FROM_XYZ[7], b2 = FROM_XYZ[8];
  for (let i = 0, p = 0; i < count; i++, p += stride) {
    const l = L[i] * 2;
    const y = L_TO_Y[l], ify = L_TO_Y[l + 1];
    const adiv = ((5 * A[i] * 53687 + 128) >> 13) - 4194;
    const bdiv = ((B[i] * 41943 + 16) >> 9) - 10484;
    const x = AB_TO_XZ[ify + adiv - MIN_AB], z = AB_TO_XZ[ify - bdiv - MIN_AB];
    let r = (r0 * x + r1 * y + r2 * z + 8192) >> 14;
    let g = (g0 * x + g1 * y + g2 * z + 8192) >> 14;
    let b = (b0 * x + b1 * y + b2 * z + 8192) >> 14;
    out[p] = INV_GAMMA_TABLE[r < 0 ? 0 : r > 4095 ? 4095 : r];
    out[p + 1] = INV_GAMMA_TABLE[g < 0 ? 0 : g > 4095 ? 4095 : g];
    out[p + 2] = INV_GAMMA_TABLE[b < 0 ? 0 : b > 4095 ? 4095 : b];
  }
}

// ------------------------------------------------------------------ CLAHE (cv2.createCLAHE(...).apply, 8 bit)

export class Clahe {
  constructor(clipLimit = 40.0, tilesX = 8, tilesY = 8) {
    this.clipLimit = clipLimit;
    this.tilesX = tilesX;
    this.tilesY = tilesY;
    this._cache = null;
  }

  // One-channel 8-bit plane (width x height) -> new plane (or `out`).
  apply(src, width, height, out = null) {
    const { tilesX, tilesY } = this;
    const dst = out || new Uint8Array(width * height);
    let plane = src, planeWidth = width, planeHeight = height;
    if (width % tilesX !== 0 || height % tilesY !== 0) {
      // copyMakeBorder(BORDER_REFLECT_101) on the bottom / right; OpenCV pads both axes then.
      planeWidth = width + tilesX - (width % tilesX);
      planeHeight = height + tilesY - (height % tilesY);
      plane = new Uint8Array(planeWidth * planeHeight);
      for (let y = 0; y < planeHeight; y++) {
        const sy = reflect101(y, height);
        for (let x = 0; x < planeWidth; x++) plane[y * planeWidth + x] = src[sy * width + reflect101(x, width)];
      }
    }
    const tileWidth = planeWidth / tilesX, tileHeight = planeHeight / tilesY;
    const tileArea = tileWidth * tileHeight;
    const lutScale = f32(255 / tileArea);
    let clipLimit = 0;
    if (this.clipLimit > 0) clipLimit = Math.max(Math.trunc(this.clipLimit * tileArea / 256), 1);
    const luts = new Uint8Array(tilesX * tilesY * 256);
    const hist = new Int32Array(256);
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        hist.fill(0);
        for (let y = ty * tileHeight, end = y + tileHeight; y < end; y++) {
          for (let p = y * planeWidth + tx * tileWidth, stop = p + tileWidth; p < stop; p++) hist[plane[p]]++;
        }
        if (clipLimit > 0) {
          let clipped = 0;
          for (let i = 0; i < 256; i++) {
            if (hist[i] > clipLimit) { clipped += hist[i] - clipLimit; hist[i] = clipLimit; }
          }
          const batch = Math.trunc(clipped / 256);
          let residual = clipped - batch * 256;
          for (let i = 0; i < 256; i++) hist[i] += batch;
          if (residual !== 0) {
            const step = Math.max(Math.trunc(256 / residual), 1);
            for (let i = 0; i < 256 && residual > 0; i += step, residual--) hist[i]++;
          }
        }
        const lut = (ty * tilesX + tx) * 256;
        for (let i = 0, total = 0; i < 256; i++) {
          total += hist[i];
          luts[lut + i] = saturate8(f32(total * lutScale));
        }
      }
    }
    // Bilinear blend of the 4 nearest tiles' maps, all in float32 like OpenCV.
    const cache = this._interpolation(width, height, tileWidth, tileHeight);
    const { ind1, ind2, xa, xa1, ty1, ty2, ya, ya1 } = cache;
    for (let y = 0; y < height; y++) {
      const plane1 = ty1[y], plane2 = ty2[y], wy = ya[y], wy1 = ya1[y];
      for (let x = 0, p = y * width; x < width; x++, p++) {
        const v = src[p], i1 = ind1[x] + v, i2 = ind2[x] + v, w = xa[x], w1 = xa1[x];
        const top = f32(f32(luts[plane1 + i1] * w1) + f32(luts[plane1 + i2] * w));
        const bottom = f32(f32(luts[plane2 + i1] * w1) + f32(luts[plane2 + i2] * w));
        dst[p] = (f32(f32(top * wy1) + f32(bottom * wy)) + ROUNDER) - ROUNDER; // in 0..255: a blend of LUT bytes
      }
    }
    return dst;
  }

  _interpolation(width, height, tileWidth, tileHeight) {
    const key = `${width},${height}`;
    if (this._cache && this._cache.key === key) return this._cache;
    const { tilesX, tilesY } = this;
    const ind1 = new Int32Array(width), ind2 = new Int32Array(width);
    const xa = new Float32Array(width), xa1 = new Float32Array(width);
    const invWidth = f32(1 / tileWidth);
    for (let x = 0; x < width; x++) {
      const txf = f32(f32(x * invWidth) - 0.5);
      const tx1 = Math.floor(txf);
      xa[x] = f32(txf - tx1);
      xa1[x] = f32(1 - xa[x]);
      ind1[x] = Math.max(tx1, 0) * 256;
      ind2[x] = Math.min(tx1 + 1, tilesX - 1) * 256;
    }
    const ty1 = new Int32Array(height), ty2 = new Int32Array(height);
    const ya = new Float32Array(height), ya1 = new Float32Array(height);
    const invHeight = f32(1 / tileHeight);
    for (let y = 0; y < height; y++) {
      const tyf = f32(f32(y * invHeight) - 0.5);
      const t1 = Math.floor(tyf);
      ya[y] = f32(tyf - t1);
      ya1[y] = f32(1 - ya[y]);
      ty1[y] = Math.max(t1, 0) * tilesX * 256;
      ty2[y] = Math.min(t1 + 1, tilesY - 1) * tilesX * 256;
    }
    this._cache = { key, ind1, ind2, xa, xa1, ty1, ty2, ya, ya1 };
    return this._cache;
  }
}

function reflect101(index, size) {
  if (size === 1) return 0;
  while (index < 0 || index >= size) index = index < 0 ? -index : 2 * size - index - 2;
  return index;
}

// ------------------------------------------------------------------ numpy helpers

// np.percentile(values, q) ("linear") of uint8 values given as a 256-bin histogram with `count` values.
function percentileFromHistogram(hist, count, q) {
  const virtual = (count - 1) * (q / 100);
  const previous = Math.floor(virtual), next = Math.min(previous + 1, count - 1);
  const gamma = virtual - previous;
  const a = orderStatistic(hist, previous), b = orderStatistic(hist, next);
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

function orderStatistic(hist, rank) {
  for (let value = 0, seen = 0; value < 256; value++) {
    seen += hist[value];
    if (seen > rank) return value;
  }
  return 255;
}

const clip = (value, low, high) => Math.min(Math.max(value, low), high);

// ------------------------------------------------------------------ AdaptiveLightCorrection

// Correct webcam exposure and local contrast before preview and detection.
export class AdaptiveLightCorrection {
  constructor() {
    this.autoEnabled = true;
    this.exposureBiasEv = 0.0;
    this.autoExposureEv = null;
    this.effectiveExposureEv = 0.0;
    this._clahe = new Clahe(1.6, 8, 8);
    this.advice = "Đang đo ánh sáng...";
    this._planes = null;
    this._sample = new Uint8Array(96 * 72 * 3);
  }

  reset() {
    this.autoExposureEv = null;
  }

  // RGBA image -> corrected RGBA image (same size; alpha copied). `out` (ImageData-like of the same size) is
  // filled instead of allocating, e.g. to draw straight back onto a canvas.
  process(image, out = null) {
    const { data, width, height } = image || {};
    if (!data || !(width > 0) || !(height > 0) || data.length !== width * height * 4) {
      throw new Error("Camera frame must be a non-empty uint8 BGR image.");
    }
    const sample = resizeArea(data, width, height, 4, 96, 72, this._sample);
    const hist = new Int32Array(256);
    let clippedCount = 0;
    for (let p = 0; p < sample.length; p += 3) {
      const value = lightness(sample[p], sample[p + 1], sample[p + 2]);
      hist[value]++;
      if (value >= 250) clippedCount++;
    }
    const count = 96 * 72;
    const p10 = percentileFromHistogram(hist, count, 10);
    const median = percentileFromHistogram(hist, count, 50);
    const p90 = percentileFromHistogram(hist, count, 90);
    this._updateAdvice(p10, median, p90, clippedCount / count);

    if (this.autoEnabled) {
      const targetEv = clip(Math.log2(116.0 / Math.max(median, 24.0)), -1.25, 1.25);
      if (this.autoExposureEv === null) this.autoExposureEv = targetEv;
      else this.autoExposureEv += 0.12 * (targetEv - this.autoExposureEv);
    } else {
      this.autoExposureEv = 0.0;
    }
    this.effectiveExposureEv = clip((this.autoExposureEv || 0.0) + this.exposureBiasEv, -2.0, 2.0);
    const lut = brightnessLut(2.0 ** this.effectiveExposureEv);

    const pixels = width * height;
    if (!this._planes || this._planes.L.length !== pixels) {
      this._planes = {
        L: new Uint8Array(pixels), A: new Uint8Array(pixels), B: new Uint8Array(pixels),
        exposed: new Uint8Array(pixels), local: new Uint8Array(pixels),
      };
    }
    const { L, A, B, exposed, local } = this._planes;
    toLab(data, pixels, 4, L, A, B);
    for (let i = 0; i < pixels; i++) exposed[i] = lut[L[i]];
    this._clahe.apply(exposed, width, height, local);
    const blend = blendTable(); // cv2.addWeighted(exposed, 0.72, local, 0.28, 0.0)
    for (let i = 0; i < pixels; i++) L[i] = blend[exposed[i] * 256 + local[i]];
    const target = out && out.data && out.data.length === data.length ? out : {
      data: new Uint8ClampedArray(data.length), width, height,
    };
    fromLab(L, A, B, pixels, target.data, 4);
    for (let p = 3; p < data.length; p += 4) target.data[p] = data[p];
    return target;
  }

  _updateAdvice(p10, median, p90, clippedFraction) {
    let advice;
    if (clippedFraction >= 0.025) {
      const percentage = roundHalfEven(clippedFraction * 100);
      advice = `⚠ Cháy sáng (${percentage}%). Không thể khôi phục vùng trắng đã mất; `
        + "đặt đèn lệch 30–45° và tránh chiếu thẳng vào ống kính.";
    } else if (p90 >= 250.0) {
      advice = "⚠ Vùng sáng rất gắt; xoay đèn lệch 30–45° và tránh chiếu thẳng vào ống kính.";
    } else if (p10 < 34.0 && p90 > 175.0) {
      advice = "⚠ Ánh sáng chênh lệch mạnh; đưa đèn dịu ra trước mặt, "
        + "tránh để cửa sổ hoặc đèn sáng phía sau.";
    } else if (median < 55.0) {
      advice = "Ánh sáng yếu; tăng đèn dịu phía trước hoặc tăng Bù sáng.";
    } else if (median > 205.0) {
      advice = "Khung hình quá sáng; giảm Bù sáng hoặc giảm phơi sáng của camera.";
    } else {
      advice = "Ánh sáng ổn; đã cân bằng vùng tối và nén vùng sáng.";
    }
    if (advice !== this.advice) this.advice = advice;
  }
}

// cv2.addWeighted(a, 0.72, b, 0.28, 0.0) of every byte pair: float32 like OpenCV's vector path (v_muladd, fused
// on FMA CPUs): fma(a, alpha, round(b*beta)); a*alpha is exact in float64, so the sum is rounded once.
let blendCache = null;
export function blendTable() {
  if (blendCache) return blendCache;
  const alpha = f32(0.72), beta = f32(0.28);
  blendCache = new Uint8Array(256 * 256);
  for (let a = 0; a < 256; a++) {
    for (let b = 0; b < 256; b++) blendCache[a * 256 + b] = saturate8(f32(a * alpha + f32(b * beta)));
  }
  return blendCache;
}

// The brightness curve of camera_enhancement.py (numpy float32): gain, soft highlight compression, gamma 0.96,
// truncated to uint8.
export function brightnessLut(gain) {
  const lut = new Uint8Array(256);
  const g = f32(gain);
  for (let i = 0; i < 256; i++) {
    let value = f32(i * g);
    if (value > 190.0) {
      const excess = Math.max(f32(value - 190.0), 0);
      value = f32(190.0 + f32(65.0 * f32(1.0 - f32(Math.exp(f32(-excess / 65.0))))));
    }
    const powered = f32(255.0 * f32(Math.pow(f32(value / 255.0), f32(0.96))));
    lut[i] = Math.trunc(clip(powered, 0.0, 255.0));
  }
  return lut;
}
