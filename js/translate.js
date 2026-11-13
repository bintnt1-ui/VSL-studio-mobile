// Sign recognition (port of translate.py): nearest-template matching with dynamic time warping (DTW).
//
// Every stored sample becomes a template, together with its mirror image, so a sign matches whichever hand
// performs it. Every sequence (stored sample or live window) is brought to a common form first: true-shape
// coordinates (x and z rescaled to the frame height), everything in seconds from per-frame timestamps (never
// frame counts), short dropouts bridged and jitter smoothed in time, resting hands ignored, only the signing
// part kept and resampled evenly in time. A coarse pass picks the closest template parts, then DTW whose path
// may skip the start and end of either sequence gives the final distance.
//
// Numerics follow numpy exactly: float32 arrays are computed with float32 rounding (Math.fround) where numpy
// works in float32 (cleaning, features), float64 elsewhere; np.gradient / np.interp / np.median / np.argmin /
// np.linspace / round-half-even semantics are replicated.
//
// Shapes: a frame is a Float32Array(KEYPOINT_SIZE); a sequence is an array of frames (a {data, count, width}
// block is accepted too); times are seconds (number[] / Float64Array). Spans are [first, last) frame pairs.
// Python keyword arguments: either positional in Python's order or one options object in camelCase (snake_case
// keys are accepted too), e.g. predict(window, null, true, aspect, times) or predict(window, {live: true, aspect,
// times}); undefined / null mean the Python default (`live: null` = decided by the length, as live=None).
// Per-frame pairs are flat: SignActivity.arms[i * 2 + side].

import { FACE_POINTS_USED, KEYPOINT_SIZE, mirrorKeypoints } from "./layout.js";

const f32 = Math.fround;

export const FRAME_COUNT = 30;                  // app_paths.FRAME_COUNT: frames of a stored sample
export const MIN_TEMPLATES_PER_LABEL = 1;
export const MIN_HAND_COVERAGE = 0.45;          // kept for callers; live windows need a long enough burst instead
export const MIN_TEMPLATE_HAND_COVERAGE = 0.40;
// Time, never frame counts: webcams deliver 8 to 30 frames per second and the rate changes while signing.
export const JOIN_SECONDS = 0.24;               // a hand lost (blur) or lowered this long is still the same sign
export const SMOOTH_SECONDS = 0.07;             // landmark jitter is averaged over this radius
export const MIN_ACTIVE_SECONDS = 0.3;          // shortest burst of signing worth recognising...
export const MIN_ACTIVE_FRAMES = 3;             // ... with at least this many frames
export const GAP_FRAMES = 3;                    // stored samples (frames far apart): up to this many missing frames too
export const RESAMPLED = 20;                    // frames (evenly spaced in time) of the signing part compared by DTW
export const COARSE = 8;                        // frames of the quick first pass
export const CANDIDATES = 128;                  // template parts kept after the quick pass
export const RELAX = 0.2;                       // share of either sequence DTW may skip at the start and at the end
// Decision thresholds, calibrated on own, cross-signer and re-filmed samples (tools/evaluate_translation.py).
export const MIN_MARGIN = 0.10;
export const MAX_DISTANCE = 0.30;
export const MIN_CONFIDENCE = 0.5;
export const MARGIN_PER_DISTANCE = 2.0;         // a farther best match needs a clearer lead over the next sign:
export const CLOSE_DISTANCE = 0.047;            // required = max(MIN_MARGIN, MARGIN_PER_DISTANCE x (distance - CLOSE))
export const MODALITY_WEIGHTS = Object.freeze({ hands: 0.72, pose: 0.18, face: 0.10 });
export const DEFAULT_ASPECT = 4 / 3;            // webcam frames are 640 x 480 unless a sample records its size
export const DEFAULT_FRAME_SECONDS = 1 / 15;    // samples that do not record their duration: about 15 fps
export const REST_HEIGHT = 1.1;                 // a wrist this many shoulder widths below the shoulders is lowered
export const REST_TRAVEL = 0.3;                 // a hand lowered the whole time and moving less than this is not used
export const REST_SPEED = 0.5;                  // a lowered wrist slower than this (shoulder widths / s) is resting:
export const REST_WINDOW_SECONDS = 0.15;        // ... by smoothed speed, or by how far its raw track went over this
export const MOVE_WINDOW_SECONDS = 0.2;         // SignActivity.moved: how far each wrist went over this last stretch
export const TIME_TOLERANCE = 1e-6;             // seconds: limits from the frame spacing must not depend on rounding
export const TRIM_REST = true;                  // resting moments do not count as signing
export const SPLIT_SECONDS = 0.5;               // hands lowered this long between two performances of a clip
export const MIN_PART_FRAMES = 4;
export const CAPTURE_MARGIN_SECONDS = 0.15;     // kept around a live sign saved as a sample
export const CANDIDATE_LABELS = 3;              // nearest signs reported with every prediction

export const POSE_SLICE = [0, 132];
export const FACE_SLICE = [132, 1536];
export const LEFT_HAND_SLICE = [1536, 1599];
export const RIGHT_HAND_SLICE = [1599, 1662];
export const PARTS = [[POSE_SLICE, 4], [FACE_SLICE, 3], [LEFT_HAND_SLICE, 3], [RIGHT_HAND_SLICE, 3]];

export const FINGER_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
export const HAND_BONES = FINGER_CHAINS.flatMap((chain) => chain.slice(0, -1).map((start, i) => [start, chain[i + 1]]));
export const ANGLE_TRIPLETS = FINGER_CHAINS.flatMap(
  (chain) => chain.slice(1, -1).map((joint, i) => [chain[i], joint, chain[i + 2]]));
export const FINGERTIPS = [4, 8, 12, 16, 20];
// Distances that separate look-alike handshapes (thumb across vs beside the fist, fingers spread vs together,
// tips touching the thumb).
export const HAND_DISTANCE_PAIRS = [
  ...FINGERTIPS.flatMap((a, i) => FINGERTIPS.slice(i + 1).map((b) => [a, b])),
  [4, 5], [4, 9], [4, 13], [4, 17],
  ...FINGERTIPS.map((tip) => [0, tip]),
];
export const HAND_GROUPS = Object.freeze({   // name: [dimensions, share of the hand score]
  shape: [21 * 3, 0.30],
  bones: [HAND_BONES.length * 3, 0.22],
  angles: [ANGLE_TRIPLETS.length, 0.20],
  distances: [HAND_DISTANCE_PAIRS.length, 0.25],
  position: [4, 0.03],
});
export const HAND_FEATURE_WEIGHTS = Float64Array.from(
  Object.values(HAND_GROUPS).flatMap(([size, share]) => new Array(size).fill(share / size)));
export const HAND_DIMS = HAND_FEATURE_WEIGHTS.length;     // 161 values per hand and frame
export const POSE_POINTS = [0, 13, 14, 15, 16];           // nose, elbows, wrists relative to the shoulders
export const ARM_WRISTS = [15, 16];                       // pose wrists (tracked even when a hand is blurred)
export const POSE_WEIGHTS = new Float64Array(POSE_POINTS.length * 2).fill(1.0 / (POSE_POINTS.length * 2));
export const FACE_WEIGHTS = new Float64Array(5).fill(1.0 / 5);
export const MISSING_HAND_DISTANCE = 0.6;   // a frame where a hand is seen on one side only, or on neither
export const HAND_MISMATCH_PENALTY = 0.18;
export const HAND_LENGTH_PER_PALM = 1.31;   // median wrist -> middle knuckle length / knuckle line |5-17|

const POSE_DIMS = POSE_WEIGHTS.length;
const FACE_DIMS = FACE_WEIGHTS.length;
const BONES_AT = 63, ANGLES_AT = BONES_AT + HAND_BONES.length * 3, DISTANCES_AT = ANGLES_AT + ANGLE_TRIPLETS.length;
const POSITION_AT = DISTANCES_AT + HAND_DISTANCE_PAIRS.length;
const F32_005 = f32(0.05), F32_0025 = f32(0.025), F32_1E6 = f32(1e-6), F32_1E3 = f32(1e-3), F32_PALM = f32(HAND_LENGTH_PER_PALM);
const QUIET_DISTANCE = REST_SPEED * REST_WINDOW_SECONDS;
// The face columns recognition reads (gesture_features): the recogniser cleans only these of the face.
const USED_FACE_COLUMNS = Int32Array.from(FACE_POINTS_USED.flatMap((p) => [0, 1, 2].map((a) => FACE_SLICE[0] + 3 * p + a)));

// ------------------------------------------------------------------ numpy helpers

function range(start, end) {
  const out = new Int32Array(end - start);
  for (let i = 0; i < out.length; i++) out[i] = start + i;
  return out;
}

// np.median of float64 values (mean of the two middle values when even).
function median64(values) {
  const sorted = Float64Array.from(values).sort();
  const n = sorted.length, half = n >> 1;
  return n % 2 ? sorted[half] : (sorted[half - 1] + sorted[half]) / 2;
}

// np.median of a float32 array: the mean of the two middle values is computed in float32.
function median32(values) {
  const sorted = Float32Array.from(values).sort();
  const n = sorted.length, half = n >> 1;
  return n % 2 ? sorted[half] : f32(f32(sorted[half - 1] + sorted[half]) / 2);
}

// np.linspace(start, stop, num) (endpoint included, exact stop).
export function linspace(start, stop, num) {
  const out = new Float64Array(num);
  const div = num - 1;
  const delta = stop - start;
  if (div > 0) {
    const step = delta / div;
    for (let i = 0; i < num; i++) out[i] = (step === 0 ? (i / div) * delta : i * step) + start;
    out[num - 1] = stop;
  } else if (num === 1) {
    out[0] = 0 * delta + start;
  }
  return out;
}

// np.interp for one x on increasing xp (holds the end values), as numpy's C loop computes it.
function interp(x, xp, fp, n) {
  if (x > xp[n - 1]) return fp[n - 1];
  if (x < xp[0]) return fp[0];
  if (x === xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;   // largest j with xp[j] <= x
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  const j = lo;
  if (x === xp[j]) return fp[j];
  const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  let result = slope * (x - xp[j]) + fp[j];
  if (Number.isNaN(result)) {
    result = slope * (x - xp[j + 1]) + fp[j + 1];
    if (Number.isNaN(result) && fp[j] === fp[j + 1]) result = fp[j];
  }
  return result;
}

// np.gradient(f, x) of one column (n >= 2): central differences inside, one-sided at the ends; numpy uses the
// uniform-spacing formula when every step of x is exactly equal.
function gradient(f, x, n, out) {
  let uniform = true;
  const first = x[1] - x[0];
  for (let i = 2; i < n; i++) if (x[i] - x[i - 1] !== first) { uniform = false; break; }
  if (uniform) {
    for (let i = 1; i < n - 1; i++) out[i] = (f[i + 1] - f[i - 1]) / (2.0 * first);
    out[0] = (f[1] - f[0]) / first;
    out[n - 1] = (f[n - 1] - f[n - 2]) / first;
    return out;
  }
  for (let i = 1; i < n - 1; i++) {
    const dx1 = x[i] - x[i - 1], dx2 = x[i + 1] - x[i];
    const a = -dx2 / (dx1 * (dx1 + dx2));
    const b = (dx2 - dx1) / (dx1 * dx2);
    const c = dx1 / (dx2 * (dx1 + dx2));
    out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
  }
  out[0] = (f[1] - f[0]) / (x[1] - x[0]);
  out[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
  return out;
}

// Python's round(): half to even.
export function roundHalfEven(value) {
  const floor = Math.floor(value), diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

// Python keyword arguments (see the header): `args` after the required ones, `names` in Python's order.
function isOptions(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) && !ArrayBuffer.isView(value)
    && !(value instanceof Set) && !(value instanceof Map);
}

function keywords(args, names, defaults) {
  const out = { ...defaults };
  const given = (name, value) => {
    if (value !== undefined && (value !== null || defaults[name] === null)) out[name] = value;
  };
  if (args.length === 1 && isOptions(args[0])) {
    for (const [key, value] of Object.entries(args[0])) {
      const name = key.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
      if (!(name in defaults)) throw new TypeError(`unexpected keyword argument ${key}`);
      given(name, value);
    }
    return out;
  }
  names.forEach((name, i) => given(name, args[i]));
  return out;
}

// np.argsort(values, kind="stable") (NaN last).
function argsortStable(values) {
  const order = Array.from({ length: values.length }, (_, i) => i);
  order.sort((a, b) => {
    const x = values[a], y = values[b];
    if (x < y) return -1;
    if (x > y) return 1;
    const xn = Number.isNaN(x), yn = Number.isNaN(y);
    if (xn !== yn) return xn ? 1 : -1;
    return a - b;
  });
  return order;
}

function norm2(dx, dy) {        // float32 Euclidean length (numpy float32 arithmetic)
  return f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy))));
}

function norm3(dx, dy, dz) {
  return f32(Math.sqrt(f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz))));
}

// ------------------------------------------------------------------ frames

// A sequence as a float32 block {data, count, width} (a copy), NaN and infinities set to 0 (_as_frames).
export function asFrames(sequence) {
  if (sequence && sequence.data && typeof sequence.count === "number") {
    const data = Float32Array.from(sequence.data);
    for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) data[i] = 0;
    return { data, count: sequence.count, width: sequence.width };
  }
  const count = sequence ? sequence.length : 0;
  const width = count ? sequence[0].length : KEYPOINT_SIZE;
  const data = new Float32Array(count * width);
  for (let i = 0; i < count; i++) {
    const row = sequence[i];
    if (row.length !== width) throw new Error(`frame ${i} has ${row.length} values, not ${width}`);
    data.set(row, i * width);
  }
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) data[i] = 0;
  return { data, count, width };
}

// Per-frame views of a block (a sequence: array of Float32Array frames).
export function toRows(block) {
  const out = [];
  for (let i = 0; i < block.count; i++) out.push(block.data.subarray(i * block.width, (i + 1) * block.width));
  return out;
}

// Capture time of every frame in seconds (evenly spaced when unknown), strictly increasing.
export function frameTimes(count, times = null, frameSeconds = DEFAULT_FRAME_SECONDS) {
  const out = new Float64Array(count);
  if (times == null) {
    for (let i = 0; i < count; i++) out[i] = i * frameSeconds;
    return out;
  }
  if (times.length !== count) throw new Error(`${times.length} timestamps for ${count} frames`);
  let running = -Infinity;
  for (let i = 0; i < count; i++) {
    let t = Number(times[i]);
    if (Number.isNaN(t)) t = 0;
    else if (t === Infinity) t = Number.MAX_VALUE;     // np.nan_to_num
    else if (t === -Infinity) t = -Number.MAX_VALUE;
    const step = i * 1e-6;
    const shifted = t - step;
    if (i === 0 || shifted > running) running = shifted;   // each stamp at least 1e-6 after the one before
    const adjusted = running + step;
    out[i] = adjusted - t < 5e-7 ? t : adjusted;            // stamps that already increase stay exactly as given
  }
  return out;
}

function spacingOf(times) {
  if (times.length < 2) return DEFAULT_FRAME_SECONDS;
  const steps = new Float64Array(times.length - 1);
  for (let i = 1; i < times.length; i++) steps[i - 1] = times[i] - times[i - 1];
  return median64(steps);
}

// Longest time between two frames that still belong to one sign: the hand lost or lowered for at most
// JOIN_SECONDS, at any frame rate; stored samples also bridge up to GAP_FRAMES missing frames.
function gapLimit(spacing, stored = false) {
  const limit = JOIN_SECONDS + spacing;
  return stored ? Math.max(limit, (GAP_FRAMES + 1) * spacing) : limit;
}

function scaleAspect(block, aspect) {
  const { data, count, width } = block;
  const scale = f32(aspect);   // numpy: a float32 array times a Python float works in float32
  for (const [[start, end], columns] of PARTS) {
    for (let i = 0; i < count; i++) {
      const o = i * width;
      for (let c = start; c < end; c += columns) {
        data[o + c] = data[o + c] * scale;
        data[o + c + 2] = data[o + c + 2] * scale;   // MediaPipe's z shares x's scale
      }
    }
  }
  return block;
}

// Normalised landmarks (x / width, y / height) with x and z rescaled to the frame height.
export function isotropic(sequence, aspect = DEFAULT_ASPECT) {
  return toRows(scaleAspect(asFrames(sequence), aspect));
}

function presentMask(data, count, width, start, end) {
  const mask = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const o = i * width;
    for (let c = start; c < end; c++) if (Math.abs(data[o + c]) > 1e-6) { mask[i] = 1; break; }
  }
  return mask;
}

// Linear interpolation (in time) across detection dropouts lasting at most `limit` seconds (in place on the
// float32 `points` of `n` columns); returns the new mask.
function bridge(points, n, mask, times, limit) {
  const out = Uint8Array.from(mask);
  let before = -1;
  for (let after = 0; after < mask.length; after++) {
    if (!mask[after]) continue;
    if (before >= 0 && after - before > 1 && times[after] - times[before] <= limit + TIME_TOLERANCE) {
      const span = times[after] - times[before];
      for (let k = before + 1; k < after; k++) {
        const w = (times[k] - times[before]) / span;
        for (let c = 0; c < n; c++) points[k * n + c] = (1 - w) * points[before * n + c] + w * points[after * n + c];
        out[k] = 1;
      }
    }
    before = after;
  }
  return out;
}

// Mean over the detected frames within `radius` seconds (float32 sums, as numpy's float32 matrix product).
function smoothInto(points, n, mask, times, radius, target, width, columns) {
  const count = mask.length;
  for (let i = 0; i < count; i++) {
    const o = i * width;
    if (!mask[i]) {
      for (let c = 0; c < n; c++) target[o + columns[c]] = points[i * n + c];
      continue;
    }
    let lo = i, hi = i;
    while (lo > 0 && Math.abs(times[i] - times[lo - 1]) <= radius) lo--;
    while (hi < count - 1 && Math.abs(times[i] - times[hi + 1]) <= radius) hi++;
    let near = 0;
    for (let j = lo; j <= hi; j++) near += mask[j];
    for (let c = 0; c < n; c++) {
      let sum = 0;
      for (let j = lo; j <= hi; j++) if (mask[j]) sum = f32(sum + points[j * n + c]);
      target[o + columns[c]] = sum / near;
    }
  }
}

// (shoulder line height, shoulder width) over the sequence, or null without shoulders.
function shouldersOf(values, count, width) {
  const centers = [], widths = [];
  for (let i = 0; i < count; i++) {
    const o = i * width;
    const x11 = values[o + 44], y11 = values[o + 45], x12 = values[o + 48], y12 = values[o + 49];
    const seen = (Math.abs(x11) > 1e-6 || Math.abs(y11) > 1e-6) && (Math.abs(x12) > 1e-6 || Math.abs(y12) > 1e-6);
    if (!seen) continue;
    centers.push(f32(f32(y11 + y12) * 0.5));
    widths.push(norm2(f32(x11 - x12), f32(y11 - y12)));
  }
  if (!centers.length) return null;
  const center = median32(centers), size = median32(widths);
  return size >= 1e-3 ? [center, size] : null;
}

// A hand hanging below the chest the whole time and barely moving: the signer is not using it.
function resting(values, count, width, start, mask, shoulders) {
  if (!shoulders) return false;
  const [center, size] = shoulders;
  let any = false, lowest = Infinity;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    if (!mask[i]) continue;
    any = true;
    const x = values[i * width + start], y = values[i * width + start + 1];
    const height = f32(f32(y - center) / size);
    if (height < lowest) lowest = height;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!any) return false;
  const travel = norm2(f32(maxX - minX), f32(maxY - minY)) / size;   // a numpy float32 scalar / float: float64
  return lowest > REST_HEIGHT && travel < REST_TRAVEL;
}

function foundOf(mask) {
  const found = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) found.push(i);
  return found;
}

// Per frame, in shoulder widths: wrist height below the shoulders (NaN where the hand is not seen), wrist speed
// and downward speed (per second, within each stretch of frames the hand is seen in), and how far the wrist is
// from where it was exactly MOVE_WINDOW_SECONDS earlier (interpolated). Speeds and displacements are 0 unseen.
function handMotion(values, count, width, start, mask, times, shoulders) {
  const height = new Float64Array(count).fill(NaN);
  const speed = new Float64Array(count), descent = new Float64Array(count), moved = new Float64Array(count);
  const found = foundOf(mask);
  if (!shoulders || !found.length) return { height, speed, descent, moved };
  const [center, size] = shoulders;
  const n = found.length;
  const xs = new Float64Array(n), ys = new Float64Array(n), stamps = new Float64Array(n);
  const level = center / size;
  for (let k = 0; k < n; k++) {
    const o = found[k] * width + start;
    xs[k] = values[o] / size;
    ys[k] = values[o + 1] / size;
    stamps[k] = times[found[k]];
    height[found[k]] = ys[k] - level;
  }
  if (n > 1) {
    const vx = new Float64Array(n), vy = new Float64Array(n);
    let first = 0;
    for (let k = 1; k <= n; k++) {
      if (k < n && found[k] - found[k - 1] === 1) continue;
      const length = k - first;   // stretch [first, k) of consecutive frames
      if (length > 1) {
        gradient(xs.subarray(first, k), stamps.subarray(first, k), length, vx.subarray(first, k));
        gradient(ys.subarray(first, k), stamps.subarray(first, k), length, vy.subarray(first, k));
      }
      first = k;
    }
    for (let k = 0; k < n; k++) {
      const i = found[k];
      speed[i] = Math.sqrt(vx[k] * vx[k] + vy[k] * vy[k]);
      descent[i] = vy[k];   // image y grows downwards
      const before = stamps[k] - MOVE_WINDOW_SECONDS;   // np.interp holds the first position before the hand was seen
      const dx = xs[k] - interp(before, stamps, xs, n), dy = ys[k] - interp(before, stamps, ys, n);
      moved[i] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return { height, speed, descent, moved };
}

// Frames where the raw wrist (bridged, not smoothed; `wrist`: x, y per frame) moved less than REST_SPEED x
// REST_WINDOW_SECONDS over the last REST_WINDOW_SECONDS: exact in time at any frame rate.
function quiet(wrist, mask, times, size) {
  const out = new Uint8Array(mask.length);
  const found = foundOf(mask);
  const n = found.length;
  if (n < 2) return out;
  const xs = new Float64Array(n), ys = new Float64Array(n), stamps = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    xs[k] = wrist[found[k] * 2] / size;
    ys[k] = wrist[found[k] * 2 + 1] / size;
    stamps[k] = times[found[k]];
  }
  for (let k = 0; k < n; k++) {
    const before = stamps[k] - REST_WINDOW_SECONDS;
    const dx = xs[k] - interp(before, stamps, xs, n), dy = ys[k] - interp(before, stamps, ys, n);
    out[found[k]] = Math.sqrt(dx * dx + dy * dy) < QUIET_DISTANCE ? 1 : 0;
  }
  return out;
}

// The cleaning shared by templates, live windows and sign activity (_clean). `light`: leave the face as detected
// (only hand activity is needed). `stored`: a stored sample (see gapLimit) rather than camera frames.
// `faceColumns`: clean only these face columns (the recogniser reads 27 of the 1404; the others stay 0).
// Returns {values (Float32Array count x width), count, width, masks (4 x Uint8Array), active, speed, times,
// height, descent, moved, join, arms (Float64Array count x 2, NaN unseen), side (Int32Array, -1 none)}.
function clean(sequence, aspect, times = null, { light = false, stored = false, faceColumns = null } = {}) {
  const block = scaleAspect(asFrames(sequence), aspect);
  const { data: values, count, width } = block;
  const stamps = frameTimes(count, times);
  const spacing = spacingOf(stamps);
  const join = gapLimit(spacing, stored);
  const radius = Math.max(SMOOTH_SECONDS, 1.01 * spacing);   // at least the neighbouring frames
  const cleaned = new Float32Array(count * width);
  const masks = [];
  const wrists = [null, null, null, null];   // bridged (unsmoothed) wrist x, y of each hand
  PARTS.forEach(([[start, end]], index) => {
    const mask = presentMask(values, count, width, start, end);
    if (light && index === 1) {
      for (let i = 0; i < count; i++) {
        cleaned.set(values.subarray(i * width + start, i * width + end), i * width + start);
      }
      masks.push(mask);
      return;
    }
    const columns = index === 1 && faceColumns ? faceColumns : range(start, end);
    const n = columns.length;
    const points = new Float32Array(count * n);
    for (let i = 0; i < count; i++) for (let c = 0; c < n; c++) points[i * n + c] = values[i * width + columns[c]];
    const bridged = bridge(points, n, mask, stamps, join);
    if (index >= 2) {
      const wrist = new Float32Array(count * 2);
      for (let i = 0; i < count; i++) { wrist[i * 2] = points[i * n]; wrist[i * 2 + 1] = points[i * n + 1]; }
      wrists[index] = wrist;
    }
    smoothInto(points, n, bridged, stamps, radius, cleaned, width, columns);
    masks.push(bridged);
  });
  const shoulders = shouldersOf(cleaned, count, width);
  const active = new Uint8Array(count);
  const speed = new Float64Array(count), moved = new Float64Array(count);
  const highest = new Float64Array(count).fill(Infinity), descent = new Float64Array(count);
  const arms = new Float64Array(count * 2).fill(NaN), side = new Int32Array(count).fill(-1);
  if (shoulders) {
    const [center, size] = shoulders;
    for (let i = 0; i < count; i++) {
      if (!masks[0][i]) continue;
      for (let s = 0; s < 2; s++) arms[i * 2 + s] = f32(f32(cleaned[i * width + ARM_WRISTS[s] * 4 + 1] - center) / size);
    }
  }
  for (const index of [2, 3]) {   // left and right hand
    const start = PARTS[index][0][0], end = PARTS[index][0][1];
    if (resting(cleaned, count, width, start, masks[index], shoulders)) {
      for (let i = 0; i < count; i++) cleaned.fill(0, i * width + start, i * width + end);
      masks[index] = new Uint8Array(count);
      continue;
    }
    const mask = masks[index];
    const motion = handMotion(cleaned, count, width, start, mask, stamps, shoulders);
    const still = TRIM_REST && shoulders ? quiet(wrists[index], mask, stamps, shoulders[1]) : new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      if (TRIM_REST && shoulders && motion.speed[i] < REST_SPEED) still[i] = 1;
      const lowered = motion.height[i] > REST_HEIGHT && still[i];
      const signing = mask[i] && !lowered;
      const higher = signing && motion.height[i] < highest[i];
      if (signing) active[i] = 1;
      const handSpeed = mask[i] ? motion.speed[i] : 0.0;
      if (handSpeed > speed[i]) speed[i] = handSpeed;
      const handMoved = signing ? motion.moved[i] : 0.0;
      if (handMoved > moved[i]) moved[i] = handMoved;
      if (higher) {
        highest[i] = motion.height[i];
        descent[i] = motion.descent[i];
        if (masks[0][i]) {   // the arm of this hand: its pose wrist is the nearest
          const o = i * width;
          const x = cleaned[o + start], y = cleaned[o + start + 1];
          const near = ARM_WRISTS.map((joint) => norm2(f32(x - cleaned[o + joint * 4]), f32(y - cleaned[o + joint * 4 + 1])));
          side[i] = near[1] < near[0] ? 1 : 0;
        }
      }
    }
  }
  return { values: cleaned, count, width, masks, active, speed, times: stamps, height: highest, descent, moved,
    join, arms, side };
}

// [first, last) runs of true frames; runs less than `join` seconds apart are one run.
function runsOf(mask, times, join) {
  const runs = [];
  let start = -1, previous = -1;
  for (let index = 0; index < mask.length; index++) {
    if (!mask[index]) continue;
    if (start < 0) {
      start = previous = index;
      continue;
    }
    if (times[index] - times[previous] > join + TIME_TOLERANCE) {
      runs.push([start, previous + 1]);
      start = index;
    }
    previous = index;
  }
  if (start >= 0) runs.push([start, previous + 1]);
  return runs;
}

function liveSpan(cleaned) {
  const runs = runsOf(cleaned.active, cleaned.times, cleaned.join);
  if (!runs.length) return null;
  const [first, last] = runs[runs.length - 1];
  if (last - first < MIN_ACTIVE_FRAMES || cleaned.times[last - 1] - cleaned.times[first] < MIN_ACTIVE_SECONDS) return null;
  return [first, last];
}

function wholeSpan(cleaned) {
  const runs = runsOf(cleaned.active, cleaned.times, cleaned.join);
  return runs.length ? [runs[0][0], runs[runs.length - 1][1]] : null;
}

// The whole sign, plus each performance when the clip shows it more than once (hands lowered between).
function templateSpans(cleaned) {
  const whole = wholeSpan(cleaned);
  if (!whole) return [];
  const spans = [whole];
  const runs = runsOf(cleaned.active, cleaned.times, Math.max(SPLIT_SECONDS, cleaned.join));
  if (runs.length > 1) for (const run of runs) if (run[1] - run[0] >= MIN_PART_FRAMES) spans.push(run);
  return spans;
}

// `length` frames evenly spaced in time between frames `first` and `last - 1` of `values` (columns `columns`),
// written into `out` (float32, `length` x `width`). _resample.
function resampleColumns(values, width, columns, mask, times, first, last, length, out) {
  const count = last - first, n = columns.length;
  if (count === 1) {
    if (mask[first]) {
      for (let t = 0; t < length; t++) for (let c = 0; c < n; c++) out[t * width + columns[c]] = values[first * width + columns[c]];
    }
    return;
  }
  if (count < 1) throw new Error("nothing to resample");
  const targets = linspace(times[first], times[last - 1], length);
  for (let t = 0; t < length; t++) {
    const target = targets[t];
    let lo = first, hi = last;   // searchsorted(side="right"): first index with times > target
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= target) lo = mid + 1; else hi = mid;
    }
    const upper = Math.min(Math.max(lo - first, 1), count - 1) + first;
    const lower = upper - 1;
    const gap = times[upper] - times[lower];
    let weight = gap > 0 ? (target - times[lower]) / gap : 0.0;
    weight = Math.min(Math.max(weight, 0.0), 1.0);
    const o = t * width;
    if (mask[lower] && mask[upper]) {
      for (let c = 0; c < n; c++) {
        const col = columns[c];
        out[o + col] = (1 - weight) * values[lower * width + col] + weight * values[upper * width + col];
      }
    } else {
      const nearest = weight < 0.5 ? lower : upper;
      if (mask[nearest]) for (let c = 0; c < n; c++) out[o + columns[c]] = values[nearest * width + columns[c]];
    }
  }
}

function resampledBlock(cleaned, span, length, faceColumns = null) {
  const { values, width, masks, times } = cleaned;
  const out = new Float32Array(length * width);
  PARTS.forEach(([[start, end]], index) => {
    const columns = index === 1 && faceColumns ? faceColumns : range(start, end);
    resampleColumns(values, width, columns, masks[index], times, span[0], span[1], length, out);
  });
  return { data: out, count: length, width };
}

// Where a live stream signs (sign_activity's result): active (Uint8Array: frames where a hand signs), speed
// (fastest wrist, shoulder widths / s), times (seconds), runs ([first, last) bursts of signing, gaps up to `join`
// included), join (seconds, gapLimit), height (highest signing wrist, shoulder widths below the shoulders;
// Infinity: none), descent (downward speed of that wrist), moved (farthest a signing wrist went over the last
// MOVE_WINDOW_SECONDS), spacing (median frame interval), arms (Float64Array frames x 2, flat: arms[i * 2 + s] =
// height of the ARM_WRISTS pose landmark s, NaN unseen), side (Int32Array: which arm belongs to the highest
// signing hand, -1 none). Float64Array measures, frozen.
export class SignActivity {
  constructor(active, speed, times, runs, join, height, descent, moved, spacing, arms = null, side = null) {
    Object.assign(this, { active, speed, times, runs, join, height, descent, moved, spacing, arms, side });
    Object.freeze(this);
  }
}

// Where a live stream signs: the same definition the recogniser uses for templates and windows.
export function signActivity(sequence, aspect = DEFAULT_ASPECT, times = null) {
  const c = clean(sequence, aspect, times, { light: true });
  return new SignActivity(c.active, c.speed, c.times, runsOf(c.active, c.times, c.join), c.join, c.height, c.descent,
    c.moved, spacingOf(c.times), c.arms, c.side);
}

// A burst long enough to be a sign (MIN_ACTIVE_FRAMES frames and MIN_ACTIVE_SECONDS).
export function longEnough(activity, span) {
  const [first, last] = span;
  return last - first >= MIN_ACTIVE_FRAMES && activity.times[last - 1] - activity.times[first] >= MIN_ACTIVE_SECONDS;
}

// Signing part of a sequence, cleaned and resampled to `length` frames (array of Float32Array); null without a
// signing hand. normalise(sequence, live = false, length = RESAMPLED, aspect = DEFAULT_ASPECT, times = null).
export function normalise(sequence, ...args) {
  const { live, length, aspect, times } = keywords(args, ["live", "length", "aspect", "times"],
    { live: false, length: RESAMPLED, aspect: DEFAULT_ASPECT, times: null });
  const c = clean(sequence, aspect, times);
  const span = live ? liveSpan(c) : wholeSpan(c);
  return span ? toRows(resampledBlock(c, span, length)) : null;
}

// [first, last) frames of the most recent burst of signing in a live window, or null.
export function latestSign(sequence, aspect = DEFAULT_ASPECT, times = null) {
  return liveSpan(clean(sequence, aspect, times, { faceColumns: USED_FACE_COLUMNS }));
}

// A live sign (default: the latest burst) as a stored sample: [`length` frames evenly spaced in time, in
// MediaPipe coordinates (array of Float32Array); its duration in seconds], or null.
// captureSample(sequence, aspect = DEFAULT_ASPECT, length = FRAME_COUNT, times = null, span = null).
export function captureSample(sequence, ...args) {
  const { aspect, length, times, span: given } = keywords(args, ["aspect", "length", "times", "span"],
    { aspect: DEFAULT_ASPECT, length: FRAME_COUNT, times: null, span: null });
  const block = asFrames(sequence);
  const stamps = frameTimes(block.count, times);
  const span = (given && given.length ? given : null)   // Python: `span or ...` (an empty tuple is false)
    || liveSpan(clean(block, aspect, stamps, { faceColumns: USED_FACE_COLUMNS }));
  if (!span) return null;
  const start = stamps[span[0]] - CAPTURE_MARGIN_SECONDS;
  const end = stamps[span[1] - 1] + CAPTURE_MARGIN_SECONDS;
  const keep = [];
  for (let i = 0; i < block.count; i++) if (stamps[i] >= start && stamps[i] <= end) keep.push(i);
  const kept = new Float32Array(keep.length * block.width);
  keep.forEach((i, k) => kept.set(block.data.subarray(i * block.width, (i + 1) * block.width), k * block.width));
  const keptTimes = Float64Array.from(keep, (i) => stamps[i]);
  const frames = resampleSequence({ data: kept, count: keep.length, width: block.width }, keptTimes, length);
  return [frames, keptTimes[keptTimes.length - 1] - keptTimes[0]];
}

// `length` frames evenly spaced in time over a whole recording (MediaPipe coordinates, missing parts kept).
export function resampleSequence(sequence, times, length = FRAME_COUNT) {
  const block = asFrames(sequence);
  const stamps = frameTimes(block.count, times);
  const out = new Float32Array(length * block.width);
  for (const [[start, end]] of PARTS) {
    const mask = presentMask(block.data, block.count, block.width, start, end);
    resampleColumns(block.data, block.width, range(start, end), mask, stamps, 0, block.count, length, out);
  }
  return toRows({ data: out, count: length, width: block.width });
}

// ------------------------------------------------------------------ features

// Per-frame descriptors (float32 values, numpy float32 arithmetic): hands (count x 2 x HAND_DIMS), handValid
// (count x 2), pose (count x 10), poseValid, face (count x 5), faceValid.
function computeFeatures(block) {
  const { data, count, width } = block;
  const hands = new Float32Array(count * 2 * HAND_DIMS), handValid = new Uint8Array(count * 2);
  const pose = new Float32Array(count * POSE_DIMS), poseValid = new Uint8Array(count);
  const face = new Float32Array(count * FACE_DIMS), faceValid = new Uint8Array(count);
  const local = new Float64Array(63);
  for (let f = 0; f < count; f++) {
    const o = f * width;
    const x11 = data[o + 44], y11 = data[o + 45], x12 = data[o + 48], y12 = data[o + 49];
    const shouldersValid = Math.abs(x11) > 1e-6 || Math.abs(y11) > 1e-6 || Math.abs(x12) > 1e-6 || Math.abs(y12) > 1e-6;
    const cx = f32(f32(x11 + x12) * 0.5), cy = f32(f32(y11 + y12) * 0.5);
    const shoulderWidth = Math.max(norm2(f32(x11 - x12), f32(y11 - y12)), F32_005);
    let faceSeen = false;
    for (let c = FACE_SLICE[0]; c < FACE_SLICE[1]; c++) if (Math.abs(data[o + c]) > 1e-6) { faceSeen = true; break; }
    faceValid[f] = faceSeen ? 1 : 0;
    poseValid[f] = shouldersValid ? 1 : 0;
    const noseX = data[o + FACE_SLICE[0] + 3], noseY = data[o + FACE_SLICE[0] + 4];   // face landmark 1
    for (let slot = 0; slot < 2; slot++) {
      const base = o + (slot ? RIGHT_HAND_SLICE[0] : LEFT_HAND_SLICE[0]);
      let seen = false;
      for (let c = 0; c < 63; c++) if (Math.abs(data[base + c]) > 1e-6) { seen = true; break; }
      handValid[f * 2 + slot] = seen ? 1 : 0;
      const out = (f * 2 + slot) * HAND_DIMS;
      // Hand size from the rigid wrist -> middle knuckle length (in knuckle-line widths).
      const palm = Math.max(f32(norm3(f32(data[base + 27] - data[base]), f32(data[base + 28] - data[base + 1]),
        f32(data[base + 29] - data[base + 2])) / F32_PALM), F32_0025);
      for (let k = 0; k < 21; k++) {
        for (let a = 0; a < 3; a++) {
          const value = f32(f32(data[base + k * 3 + a] - data[base + a]) / palm);
          local[k * 3 + a] = value < -4 ? -4 : value > 4 ? 4 : value;
          hands[out + k * 3 + a] = local[k * 3 + a];
        }
      }
      HAND_BONES.forEach(([start, end], b) => {
        for (let a = 0; a < 3; a++) hands[out + BONES_AT + b * 3 + a] = f32(local[end * 3 + a] - local[start * 3 + a]);
      });
      ANGLE_TRIPLETS.forEach(([start, joint, end], t) => {
        const a0 = f32(local[joint * 3] - local[start * 3]), a1 = f32(local[joint * 3 + 1] - local[start * 3 + 1]);
        const a2 = f32(local[joint * 3 + 2] - local[start * 3 + 2]);
        const b0 = f32(local[end * 3] - local[joint * 3]), b1 = f32(local[end * 3 + 1] - local[joint * 3 + 1]);
        const b2 = f32(local[end * 3 + 2] - local[joint * 3 + 2]);
        const denominator = Math.max(f32(norm3(a0, a1, a2) * norm3(b0, b1, b2)), F32_1E6);
        const dot = f32(f32(f32(a0 * b0) + f32(a1 * b1)) + f32(a2 * b2));
        const cosine = f32(dot / denominator);
        hands[out + ANGLES_AT + t] = cosine < -1 ? -1 : cosine > 1 ? 1 : cosine;
      });
      HAND_DISTANCE_PAIRS.forEach(([p, q], d) => {
        hands[out + DISTANCES_AT + d] = norm3(f32(local[p * 3] - local[q * 3]), f32(local[p * 3 + 1] - local[q * 3 + 1]),
          f32(local[p * 3 + 2] - local[q * 3 + 2]));
      });
      const px = data[base], py = data[base + 1];
      const position = [
        f32(f32(px - cx) / shoulderWidth), f32(f32(py - cy) / shoulderWidth),
        f32(f32(f32(px - noseX) / shoulderWidth) * (faceSeen ? 1 : 0)),
        f32(f32(f32(py - noseY) / shoulderWidth) * (faceSeen ? 1 : 0)),
      ];
      for (let p = 0; p < 4; p++) hands[out + POSITION_AT + p] = position[p] < -3 ? -3 : position[p] > 3 ? 3 : position[p];
    }
    POSE_POINTS.forEach((point, p) => {
      const rx = f32(f32(data[o + point * 4] - cx) / shoulderWidth), ry = f32(f32(data[o + point * 4 + 1] - cy) / shoulderWidth);
      pose[f * POSE_DIMS + p * 2] = f32((rx < -4 ? -4 : rx > 4 ? 4 : rx) * (shouldersValid ? 1 : 0));
      pose[f * POSE_DIMS + p * 2 + 1] = f32((ry < -4 ? -4 : ry > 4 ? 4 : ry) * (shouldersValid ? 1 : 0));
    });
    const F = (point, axis) => data[o + FACE_SLICE[0] + point * 3 + axis];
    const eye = Math.max(norm2(f32(F(33, 0) - F(263, 0)), f32(F(33, 1) - F(263, 1))), F32_1E3);
    const mouthX = f32(f32(F(13, 0) + F(14, 0)) * 0.5), mouthY = f32(f32(F(13, 1) + F(14, 1)) * 0.5);
    const measures = [
      norm2(f32(F(61, 0) - F(291, 0)), f32(F(61, 1) - F(291, 1))),
      norm2(f32(F(13, 0) - F(14, 0)), f32(F(13, 1) - F(14, 1))),
      norm2(f32(F(105, 0) - F(33, 0)), f32(F(105, 1) - F(33, 1))),
      norm2(f32(F(334, 0) - F(263, 0)), f32(F(334, 1) - F(263, 1))),
      norm2(f32(F(1, 0) - mouthX), f32(F(1, 1) - mouthY)),
    ];
    measures.forEach((value, m) => {
      const ratio = f32(value / eye);
      face[f * FACE_DIMS + m] = f32((ratio < 0 ? 0 : ratio > 4 ? 4 : ratio) * (faceSeen ? 1 : 0));
    });
  }
  return { count, hands, handValid, pose, poseValid, face, faceValid };
}

// Per-frame descriptors of a sequence: {count, hands: Float64Array(count x 2 x HAND_DIMS), handValid:
// Uint8Array(count x 2), pose: Float64Array(count x 10), poseValid, face: Float64Array(count x 5), faceValid}.
export function gestureFeatures(sequence) {
  const features = computeFeatures(asFrames(sequence));
  return {
    count: features.count,
    hands: Float64Array.from(features.hands), handValid: features.handValid,
    pose: Float64Array.from(features.pose), poseValid: features.poseValid,
    face: Float64Array.from(features.face), faceValid: features.faceValid,
  };
}

// Share of frames where a hand is seen.
export function handCoverage(features) {
  const { count, handValid } = features;
  if (!count) return NaN;   // np.mean of nothing
  let seen = 0;
  for (let f = 0; f < count; f++) if (handValid[f * 2] || handValid[f * 2 + 1]) seen++;
  return seen / count;
}

// handCoverage(gestureFeatures(frames)) without computing the features.
function handCoverageOfFrames(frames) {
  if (!frames.length) return NaN;
  let seen = 0;
  for (const frame of frames) {
    for (let c = LEFT_HAND_SLICE[0]; c < RIGHT_HAND_SLICE[1]; c++) {
      const v = f32(frame[c]);
      if (Number.isFinite(v) && Math.abs(v) > 1e-6) { seen++; break; }
    }
  }
  return seen / frames.length;
}

// ------------------------------------------------------------------ matching

const dtwBuffers = { cost: new Float64Array(0), length: new Float64Array(0) };

function dtwWork(size) {
  if (dtwBuffers.cost.length < size) {
    dtwBuffers.cost = new Float64Array(size);
    dtwBuffers.length = new Float64Array(size);
  }
  return dtwBuffers;
}

// DTW over one (rows x columns) grid starting at `offset`; `skipRows` / `skipColumns` > 0: relaxed ends.
function dtwOne(grid, offset, rows, columns, band, skipRows, skipColumns, relaxed) {
  const W = columns + 1;
  const { cost, length } = dtwWork((rows + 1) * W);
  cost.fill(Infinity, 0, (rows + 1) * W);
  length.fill(0, 0, (rows + 1) * W);
  if (relaxed) {
    for (let r = 0; r <= skipRows; r++) cost[r * W] = 0.0;
    for (let c = 0; c <= skipColumns; c++) cost[c] = 0.0;
  } else {
    cost[0] = 0.0;
  }
  for (let row = 1; row <= rows; row++) {
    const top = Math.min(columns, row + band);
    for (let column = Math.max(1, row - band); column <= top; column++) {
      const diagonal = (row - 1) * W + column - 1, up = diagonal + 1, left = row * W + column - 1;
      let best = cost[diagonal], from = diagonal;   // np.argmin: the first of ties
      if (cost[up] < best) { best = cost[up]; from = up; }
      if (cost[left] < best) { best = cost[left]; from = left; }
      cost[row * W + column] = best + grid[offset + (row - 1) * columns + column - 1];
      length[row * W + column] = length[from] + 1.0;
    }
  }
  if (!relaxed) return cost[rows * W + columns] / Math.max(length[rows * W + columns], 1.0);
  let result = Infinity;
  const end = (row, column) => {
    if (row < 1 || column < 1) return;
    const value = cost[row * W + column] / Math.max(length[row * W + column], 1.0);
    if (value < result || Number.isNaN(value)) result = Number.isNaN(result) ? result : value;
  };
  for (let skip = 0; skip <= skipRows; skip++) end(rows - skip, columns);
  for (let skip = 1; skip <= skipColumns; skip++) end(rows, columns - skip);
  return result;
}

function dtwPlain(grid, offset, rows, columns) {
  const band = Math.max(Math.abs(rows - columns), Math.floor(Math.max(rows, columns) / 5));
  return dtwOne(grid, offset, rows, columns, band, 0, 0, false);
}

function dtwRelaxedOne(grid, offset, rows, columns, relax) {
  const skipRows = roundHalfEven(relax * rows), skipColumns = roundHalfEven(relax * columns);
  const band = Math.max(Math.abs(rows - columns), Math.max(skipRows, skipColumns) + 2,
    Math.floor(Math.max(rows, columns) * 3 / 10));
  return dtwOne(grid, offset, rows, columns, band, skipRows, skipColumns, true);
}

// Grids as a flat array with shape [batch, rows, columns] (a typed array with .shape, or nested arrays).
function gridBatch(grids) {
  if (ArrayBuffer.isView(grids) && grids.shape) return { data: grids, shape: grids.shape };
  const batch = grids.length, rows = batch ? grids[0].length : 0, columns = rows ? grids[0][0].length : 0;
  const data = new Float64Array(batch * rows * columns);
  let k = 0;
  for (const grid of grids) for (const row of grid) for (const value of row) data[k++] = value;
  return { data, shape: [batch, rows, columns] };
}

// Length-normalised DTW cost for a batch of (rows x columns) distance grids.
export function dtwBatch(grids) {
  const { data, shape: [batch, rows, columns] } = gridBatch(grids);
  const out = new Float64Array(batch);
  for (let b = 0; b < batch; b++) out[b] = dtwPlain(data, b * rows * columns, rows, columns);
  return out;
}

// DTW whose path may skip up to `relax` of either sequence at the start and at the end (how the hands rise into
// view and come down differs between signers; the sign itself must still be matched, cost per path step).
export function dtwRelaxed(grids, relax = RELAX) {
  const { data, shape: [batch, rows, columns] } = gridBatch(grids);
  const out = new Float64Array(batch);
  for (let b = 0; b < batch; b++) out[b] = dtwRelaxedOne(data, b * rows * columns, rows, columns, relax);
  return out;
}

// One oriented template part at one temporal resolution: float32 features, hands kept only where seen.
function bankPart(features, first, frames) {
  let seen = 0;
  for (let k = first * 2; k < (first + frames) * 2; k++) seen += features.handValid[k];
  const hands = new Float32Array(seen * HAND_DIMS), handIndex = new Int32Array(frames * 2).fill(-1);
  let at = 0;
  for (let f = 0; f < frames; f++) {
    for (let slot = 0; slot < 2; slot++) {
      const source = (first + f) * 2 + slot;
      if (!features.handValid[source]) continue;
      hands.set(features.hands.subarray(source * HAND_DIMS, (source + 1) * HAND_DIMS), at);
      handIndex[f * 2 + slot] = at;
      at += HAND_DIMS;
    }
  }
  let poseSeen = 0, faceSeen = 0;
  for (let f = first; f < first + frames; f++) { poseSeen += features.poseValid[f]; faceSeen += features.faceValid[f]; }
  return {
    frames, hands, handIndex,
    pose: features.pose.slice(first * POSE_DIMS, (first + frames) * POSE_DIMS), poseOk: poseSeen / frames >= 0.5,
    face: features.face.slice(first * FACE_DIMS, (first + frames) * FACE_DIMS), faceOk: faceSeen / frames >= 0.5,
  };
}

// The query's features weighted as numpy does (current * weights, float64).
function weighted(features) {
  const { count } = features;
  const hands = new Float64Array(count * 2 * HAND_DIMS);
  for (let k = 0; k < count * 2; k++) {
    for (let d = 0; d < HAND_DIMS; d++) hands[k * HAND_DIMS + d] = features.hands[k * HAND_DIMS + d] * HAND_FEATURE_WEIGHTS[d];
  }
  const pose = new Float64Array(count * POSE_DIMS), face = new Float64Array(count * FACE_DIMS);
  for (let k = 0; k < pose.length; k++) pose[k] = features.pose[k] * POSE_WEIGHTS[k % POSE_DIMS];
  for (let k = 0; k < face.length; k++) face[k] = features.face[k] * FACE_WEIGHTS[k % FACE_DIMS];
  let poseSeen = 0, faceSeen = 0;
  for (let f = 0; f < count; f++) { poseSeen += features.poseValid[f]; faceSeen += features.faceValid[f]; }
  return { count, hands, handValid: features.handValid, pose, face, poseOk: poseSeen / count >= 0.5,
    faceOk: faceSeen / count >= 0.5 };
}

// Frame distances (rows: query frames, columns: template frames) of one part into `grid`. `sameRecording`:
// frames where neither side shows a hand agree; otherwise such a frame costs MISSING_HAND_DISTANCE.
// The part's rows are weighted once per call in float64 (numpy's bank rows: features x weights), then every
// distance is a sequential sum of |now - then| from 0, as scipy's cityblock adds them.
let weightedPart = { hands: new Float64Array(0), pose: new Float64Array(0), face: new Float64Array(0) };

function weighPart(part) {
  if (weightedPart.hands.length < part.hands.length) weightedPart.hands = new Float64Array(part.hands.length);
  if (weightedPart.pose.length < part.pose.length) weightedPart.pose = new Float64Array(part.pose.length);
  if (weightedPart.face.length < part.face.length) weightedPart.face = new Float64Array(part.face.length);
  const { hands, pose, face } = weightedPart;
  const source = part.hands, W = HAND_FEATURE_WEIGHTS;
  for (let k = 0; k < source.length; k += HAND_DIMS) {
    for (let d = 0; d < HAND_DIMS; d++) hands[k + d] = source[k + d] * W[d];
  }
  for (let k = 0; k < part.pose.length; k++) pose[k] = part.pose[k] * POSE_WEIGHTS[k % POSE_DIMS];
  for (let k = 0; k < part.face.length; k++) face[k] = part.face[k] * FACE_WEIGHTS[k % FACE_DIMS];
  return weightedPart;
}

// Sum of |a[ai + d] - b[bi + d]| over the HAND_DIMS values of a hand.
function handCityblock(a, ai, b, bi) {
  let s = 0;
  for (let d = 0; d < HAND_DIMS; d++) s += Math.abs(a[ai + d] - b[bi + d]);
  return s;
}

function partDistances(current, part, sameRecording, grid) {
  const rows = current.count, frames = part.frames;
  const weightPose = current.poseOk && part.poseOk ? MODALITY_WEIGHTS.pose : 0.0;
  const weightFace = current.faceOk && part.faceOk ? MODALITY_WEIGHTS.face : 0.0;
  const total = MODALITY_WEIGHTS.hands + weightPose + weightFace;
  const { hands: then, pose: thenPose, face: thenFace } = weighPart(part);
  const now = current.hands, nowPose = current.pose, nowFace = current.face, handIndex = part.handIndex;
  for (let i = 0; i < rows; i++) {
    const now0 = current.handValid[i * 2], now1 = current.handValid[i * 2 + 1];
    const base0 = i * 2 * HAND_DIMS, base1 = base0 + HAND_DIMS;
    for (let j = 0; j < frames; j++) {
      const at0 = handIndex[j * 2], at1 = handIndex[j * 2 + 1];
      const then0 = at0 >= 0, then1 = at1 >= 0;
      let sum = 0, shared = 0;
      if (now0 && then0) {
        sum += handCityblock(now, base0, then, at0);
        shared++;
      }
      if (now1 && then1) {
        sum += handCityblock(now, base1, then, at1);
        shared++;
      }
      let hands = sum / Math.max(shared, 1);
      let unknown = shared === 0;
      if (sameRecording) unknown = unknown && (now0 || now1 || then0 || then1);
      if (unknown) hands = MISSING_HAND_DISTANCE;
      hands += HAND_MISMATCH_PENALTY * ((((now0 ? 1 : 0) ^ (then0 ? 1 : 0)) + ((now1 ? 1 : 0) ^ (then1 ? 1 : 0))) / 2);
      let value = MODALITY_WEIGHTS.hands * hands;
      if (weightPose) {
        let s = 0;
        const a = i * POSE_DIMS, b = j * POSE_DIMS;
        for (let d = 0; d < POSE_DIMS; d++) s += Math.abs(nowPose[a + d] - thenPose[b + d]);
        value += weightPose * s;
      }
      if (weightFace) {
        let s = 0;
        const a = i * FACE_DIMS, b = j * FACE_DIMS;
        for (let d = 0; d < FACE_DIMS; d++) s += Math.abs(nowFace[a + d] - thenFace[b + d]);
        value += weightFace * s;
      }
      grid[i * frames + j] = value / total;
    }
  }
  return grid;
}

// Frame distances of a query (gestureFeatures) to every part of a bank: Float64Array(parts x rows x frames).
// `bank`: list of parts from preparedParts (`fine` or `coarse` of each).
function frameDistances(features, bank, sameRecording = false) {
  const current = weighted(features.hands instanceof Float32Array ? features : features);
  if (!bank.length) return new Float64Array(0);
  const size = current.count * bank[0].frames;
  const out = new Float64Array(bank.length * size);
  bank.forEach((part, p) => partDistances(current, part, sameRecording, out.subarray(p * size, (p + 1) * size)));
  return out;
}

export class Template {
  // label, display, source (sample id), keypoints (frames), features (gestureFeatures, computed when first
  // read if not given), aspect (frame width / height), frameSeconds (time between two stored frames).
  constructor(label, display, source, keypoints, features = null, aspect = DEFAULT_ASPECT,
    frameSeconds = DEFAULT_FRAME_SECONDS) {
    this.label = label;
    this.display = display;
    this.source = source;
    this.keypoints = keypoints;
    this.aspect = aspect;
    this.frameSeconds = frameSeconds;
    // gestureFeatures of the keypoints, computed when first read (recognition never reads them). The keypoints
    // are not needed once the template is prepared: setting them to null (releaseKeypoints) frees the memory.
    let cached = features;
    Object.defineProperty(this, "features", {
      enumerable: true,
      get: () => (cached ??= gestureFeatures(this.keypoints ?? [])),
    });
  }
}

export class Prediction {
  // candidates: [[label, display, distance], ...] nearest signs, best first.
  constructor(label, display, confidence, distance, margin, bestLabel = null, candidates = []) {
    this.label = label;
    this.display = display;
    this.confidence = confidence;
    this.distance = distance;
    this.margin = margin;
    this.bestLabel = bestLabel;
    this.candidates = candidates;
    Object.freeze(this);
  }
}

export const REJECTED = new Prediction(null, null, 0.0, Infinity, 0.0);

// Lead over the next sign a best match at `distance` needs to be accepted (confidence 0.5).
export function requiredMargin(distance, minMargin = MIN_MARGIN) {
  return Math.max(minMargin, MARGIN_PER_DISTANCE * (distance - CLOSE_DISTANCE));
}

// [{fine, coarse}] bank parts of every part of a template, in both orientations (_prepare).
export function prepareTemplate(template) {
  const times = frameTimes(template.keypoints.length, null, template.frameSeconds);
  const pieces = [];
  for (const values of [template.keypoints, mirrorKeypoints(template.keypoints)]) {
    const cleaned = clean(values, template.aspect, times, { stored: true, faceColumns: USED_FACE_COLUMNS });
    for (const span of templateSpans(cleaned)) {
      pieces.push(resampledBlock(cleaned, span, RESAMPLED, USED_FACE_COLUMNS), resampledBlock(cleaned, span, COARSE, USED_FACE_COLUMNS));
    }
  }
  if (!pieces.length) return [];
  // one call: a frame's features ignore other frames
  const width = pieces[0].width;
  const total = pieces.reduce((sum, piece) => sum + piece.count, 0);
  const data = new Float32Array(total * width);
  let at = 0;
  for (const piece of pieces) { data.set(piece.data, at * width); at += piece.count; }
  const features = computeFeatures({ data, count: total, width });
  const parts = [];
  at = 0;
  for (let k = 0; k < pieces.length; k += 2) {
    const fine = bankPart(features, at, pieces[k].count);
    at += pieces[k].count;
    const coarse = bankPart(features, at, pieces[k + 1].count);
    at += pieces[k + 1].count;
    parts.push({ fine, coarse });
  }
  return parts;
}

// str.casefold for the sort of the labels (lower case plus the full case foldings that differ from it).
function casefold(text) {
  return text.toLowerCase().replace(/[\u00df\u1e9e]/g, "ss").replace(/\u03c2/g, "\u03c3")
    .replace(/\ufb01/g, "fi").replace(/\ufb02/g, "fl");
}

function casefoldCompare(a, b) {
  const x = casefold(a), y = casefold(b);
  return x < y ? -1 : x > y ? 1 : 0;
}

export class Recognizer {
  // templates: Template[]; labels: sorted (str.casefold); displays: Map label -> display (the last template's).
  // `prepared`: prepareTemplate results of the first templates (reused, not prepared again). Only the prepared
  // parts are used after construction: the templates' keypoints may be set to null.
  constructor(templates, prepared = null) {
    this.templates = [...templates];
    this.labels = [...new Set(this.templates.map((t) => t.label))].sort(casefoldCompare);
    this.displays = new Map();
    for (const template of this.templates) this.displays.set(template.label, template.display);
    const reused = (prepared || []).slice(0, this.templates.length);
    this._prepared = reused.concat(this.templates.slice(reused.length).map(prepareTemplate));
    const labelIndex = new Map(this.labels.map((label, i) => [label, i]));
    this._templateLabel = Int32Array.from(this.templates, (t) => labelIndex.get(t.label));
    this._build();
  }

  _build() {
    this._fine = [];
    this._coarse = [];
    const owners = [];
    this._prepared.forEach((parts, index) => {
      for (const { fine, coarse } of parts) {
        this._fine.push(fine);
        this._coarse.push(coarse);
        owners.push(index);
      }
    });
    this._owner = Int32Array.from(owners);
  }

  get length() {
    return this.templates.length;
  }

  // Recognizer of every usable sample (bundle.js / store.js samples).
  static fromSamples(samples) {
    return new Recognizer(loadTemplates(samples));
  }

  // The same, preparing the templates in slices so the page stays responsive; onProgress(done, total).
  static async create(templates, { prepared = null, onProgress = null, slice = 16 } = {}) {
    const ready = (prepared || []).slice(0, templates.length);
    for (let i = ready.length; i < templates.length; i++) {
      ready.push(prepareTemplate(templates[i]));
      if ((i + 1) % slice === 0) {
        onProgress?.(i + 1, templates.length);
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    onProgress?.(templates.length, templates.length);
    return new Recognizer(templates, ready);
  }

  // A recognizer that also knows `templates`; the existing ones are not prepared again.
  extended(templates) {
    return new Recognizer([...this.templates, ...templates], this._prepared);
  }

  // Drops every template's keypoints (~200 KB each): predict / extended only use the prepared parts.
  releaseKeypoints() {
    for (const template of this.templates) template.keypoints = null;
    return this;
  }

  // Distance from a sequence to every template, best part and orientation (Infinity where not compared):
  // Float64Array. templateDistances(sequence, live = false, aspect = DEFAULT_ASPECT, times = null,
  // sameRecording = false, stored = false). `times`: capture time of every frame (evenly spaced when omitted).
  // `sameRecording`: the templates may be copies of this very recording (near-duplicate check), so frames where
  // neither shows a hand agree. `stored`: the sequence is a stored sample, cleaned like the templates (gapLimit).
  templateDistances(sequence, ...args) {
    const { live, aspect, times, sameRecording, stored } = keywords(args,
      ["live", "aspect", "times", "sameRecording", "stored"],
      { live: false, aspect: DEFAULT_ASPECT, times: null, sameRecording: false, stored: false });
    const best = new Float64Array(this.templates.length).fill(Infinity);
    if (!this._fine.length) return best;
    const cleaned = clean(sequence, aspect, times, { stored, faceColumns: USED_FACE_COLUMNS });
    const span = live ? liveSpan(cleaned) : wholeSpan(cleaned);
    if (!span) return best;
    const coarse = weighted(computeFeatures(resampledBlock(cleaned, span, COARSE, USED_FACE_COLUMNS)));
    const rough = new Float64Array(this._coarse.length);
    let grid = new Float64Array(COARSE * COARSE);
    this._coarse.forEach((part, p) => {
      if (grid.length < coarse.count * part.frames) grid = new Float64Array(coarse.count * part.frames);
      partDistances(coarse, part, sameRecording, grid);
      rough[p] = dtwPlain(grid, 0, coarse.count, part.frames);
    });
    const picks = argsortStable(rough).slice(0, CANDIDATES);
    const fine = weighted(computeFeatures(resampledBlock(cleaned, span, RESAMPLED, USED_FACE_COLUMNS)));
    grid = new Float64Array(RESAMPLED * RESAMPLED);
    for (const p of picks) {
      const part = this._fine[p];
      if (grid.length < fine.count * part.frames) grid = new Float64Array(fine.count * part.frames);
      partDistances(fine, part, sameRecording, grid);
      const exact = dtwRelaxedOne(grid, 0, fine.count, part.frames, RELAX);
      const owner = this._owner[p];
      if (exact < best[owner] || Number.isNaN(exact)) best[owner] = exact;
    }
    return best;
  }

  // Best sign for a sequence (Prediction); `live` (default null: any length but FRAME_COUNT) uses the latest
  // burst only; `exclude`: template indices left out (Set or array, negative from the end); `stored`: the
  // sequence is a stored sample (evaluation). predict(sequence, exclude = null, live = null,
  // aspect = DEFAULT_ASPECT, times = null, stored = false).
  predict(sequence, ...args) {
    const { exclude, live: given, aspect, times, stored } = keywords(args,
      ["exclude", "live", "aspect", "times", "stored"],
      { exclude: null, live: null, aspect: DEFAULT_ASPECT, times: null, stored: false });
    if (!this.templates.length) return REJECTED;
    const block = asFrames(sequence);
    const live = given ?? block.count !== FRAME_COUNT;
    let distances = this.templateDistances(block, { live, aspect, times, stored });
    const excluded = exclude ? [...exclude] : [];
    if (excluded.length) {
      distances = Float64Array.from(distances);
      for (let index of excluded) {
        if (index < 0) index += distances.length;
        if (!(index >= 0 && index < distances.length)) throw new RangeError(`template ${index} does not exist`);
        distances[index] = Infinity;
      }
    }
    if (!distances.some(Number.isFinite)) return REJECTED;
    return this.decide(distances);
  }

  // Prediction from the distance to every template (_decide).
  decide(distances) {
    const scores = new Float64Array(this.labels.length).fill(Infinity);
    for (let t = 0; t < distances.length; t++) {
      const label = this._templateLabel[t], value = distances[t];
      if (value < scores[label] || Number.isNaN(value)) scores[label] = value;
    }
    const order = argsortStable(scores).filter((index) => Number.isFinite(scores[index]));
    if (!order.length) return REJECTED;
    const bestDistance = scores[order[0]], bestLabel = this.labels[order[0]];
    let margin = 1.0;
    if (order.length > 1) {
      const runnerUp = scores[order[1]];
      margin = Math.max(0.0, runnerUp - bestDistance) / Math.max(runnerUp, 1e-6);
    }
    const required = requiredMargin(bestDistance);
    let confidence = required > 0 ? Math.min(1.0, margin / (2 * required)) : 1.0;
    if (bestDistance > MAX_DISTANCE) confidence = 0.0;
    const accepted = confidence >= MIN_CONFIDENCE;
    const candidates = order.slice(0, CANDIDATE_LABELS).map(
      (index) => Object.freeze([this.labels[index], this.displays.get(this.labels[index]), scores[index]]));
    return new Prediction(accepted ? bestLabel : null, accepted ? (this.displays.get(bestLabel) ?? null) : null,
      confidence, bestDistance, margin, bestLabel, Object.freeze(candidates));
  }
}

// ------------------------------------------------------------------ samples

// Python float(): whitespace, sign, digits with single underscores, inf / infinity / nan; NaN when invalid
// (callers treat NaN as "not a number" like Python's ValueError).
function pythonFloat(text) {
  if (typeof text === "number") return text;
  if (typeof text !== "string") return NaN;
  const value = text.trim();
  if (/^[+-]?(inf|infinity)$/i.test(value)) return value.startsWith("-") ? -Infinity : Infinity;
  if (/^[+-]?nan$/i.test(value)) return NaN;
  if (!/^[+-]?((\d(_?\d)*)(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$/.test(value)) return undefined;
  return Number(value.replace(/_/g, ""));
}

function pythonInt(text) {
  if (typeof text === "number") return Number.isFinite(text) ? Math.trunc(text) : undefined;
  if (typeof text !== "string") return undefined;
  const value = text.trim();
  if (!/^[+-]?\d(_?\d)*$/.test(value)) return undefined;
  return Number.parseInt(value.replace(/_/g, ""), 10);
}

// [frame width / height, seconds between stored frames] of a sample, from its metadata (strings).
export function sampleGeometry(metadata) {
  let aspect = DEFAULT_ASPECT;
  const size = metadata.video_size || metadata.frame_size || "";
  const parts = String(size).toLowerCase().split("x");
  if (parts.length === 2) {
    const [width, height] = parts.map(pythonFloat);
    if (width !== undefined && height !== undefined && width > 0 && height > 0) aspect = width / height;
  }
  let frames = metadata.frames ? pythonInt(metadata.frames) : FRAME_COUNT;
  if (frames === undefined) frames = FRAME_COUNT;
  const spans = [
    () => [metadata.end, metadata.start],   // video samples: frames from start to end
    () => [metadata.duration, "0"],         // webcam / translation samples
  ];
  for (const span of spans) {   // the first usable one
    const [a, b] = span();
    if (a == null || b == null) continue;
    const end = pythonFloat(a), start = pythonFloat(b);
    if (end === undefined || start === undefined) continue;
    const duration = end - start;
    if (duration > 0 && frames > 1) return [aspect, duration / (frames - 1)];
  }
  return [aspect, DEFAULT_FRAME_SECONDS];
}

// Every sample whose hands are visible often enough (hand coverage of its frames at least
// MIN_TEMPLATE_HAND_COVERAGE), grouped by sign: Map label -> Template[] (usable_templates). Samples as bundle.js /
// store.js give them: {id, label, display, keypoints: Float32Array[], aspect, seconds} (aspect / seconds missing:
// sampleGeometry(sample.metadata), else the defaults). Samples without keypoints are skipped.
export function templatesByLabel(samples) {
  const byLabel = new Map();
  for (const sample of samples) {
    const values = sample.keypoints;
    if (!values || !values.length) continue;
    if (!(handCoverageOfFrames(values) >= MIN_TEMPLATE_HAND_COVERAGE)) continue;
    let { aspect, seconds } = sample;
    if (aspect == null || seconds == null) {
      const [measured, interval] = sample.metadata ? sampleGeometry(sample.metadata) : [DEFAULT_ASPECT, DEFAULT_FRAME_SECONDS];
      aspect ??= measured;
      seconds ??= interval;
    }
    const template = new Template(sample.label, sample.display ?? sample.label, sample.id, values, null, aspect, seconds);
    if (!byLabel.has(sample.label)) byLabel.set(sample.label, []);
    byLabel.get(sample.label).push(template);
  }
  return byLabel;
}

// The usable samples as templates: templatesByLabel flattened (signs in order of appearance), Template[].
export function usableTemplates(samples) {
  return [...templatesByLabel(samples).values()].flat();
}

// [ready templates, missing: plain object display -> usable samples] (_split).
function split(byLabel) {
  const ready = [], missing = {};
  for (const templates of byLabel.values()) {
    if (templates.length >= MIN_TEMPLATES_PER_LABEL) ready.push(...templates);
    else missing[templates[0].display] = templates.length;
  }
  return [ready, missing];
}

// Templates for every sign with at least MIN_TEMPLATES_PER_LABEL usable samples.
export function loadTemplates(samples) {
  return split(templatesByLabel(samples))[0];
}

// Signs that exist but have too few usable samples to be recognised yet: plain object display -> count.
export function underSampled(samples) {
  return split(templatesByLabel(samples))[1];
}

let cache = null;

// What a cached recognizer depends on: every sample's id, sign, name, content key, geometry and length.
function signature(samples) {
  return samples.map((s) => [s.id, s.label, s.display, s.key ?? "", s.aspect, s.seconds, s.keypoints?.length ?? 0]
    .join("\u0001")).join("\u0002");
}

function cached(stamp, keepKeypoints) {
  return cache && cache.stamp === stamp && (!keepKeypoints || !cache.released) ? cache : null;
}

// {recognizer, missing (underSampled: plain object display -> count)}, rebuilt only when the samples changed
// (ids, signs, names, content keys, geometry). `keepKeypoints: false` releases the templates' keypoints once
// prepared (the decoded bundle is ~110 MB of frames; recognition never reads them again).
export function loadRecognizer(samples, { keepKeypoints = true } = {}) {
  const stamp = signature(samples);
  const hit = cached(stamp, keepKeypoints);
  if (hit) return { recognizer: hit.recognizer, missing: hit.missing };
  const [ready, missing] = split(templatesByLabel(samples));
  const recognizer = new Recognizer(ready);
  if (!keepKeypoints) recognizer.releaseKeypoints();
  cache = { stamp, recognizer, missing, released: !keepKeypoints };
  return { recognizer, missing };
}

// loadRecognizer preparing the templates in slices (Recognizer.create) so the thread can answer between them;
// onProgress(done, total).
export async function loadRecognizerAsync(samples, { onProgress = null, keepKeypoints = true } = {}) {
  const stamp = signature(samples);
  const hit = cached(stamp, keepKeypoints);
  if (hit) return { recognizer: hit.recognizer, missing: hit.missing };
  const [ready, missing] = split(templatesByLabel(samples));
  const recognizer = await Recognizer.create(ready, { onProgress });
  if (!keepKeypoints) recognizer.releaseKeypoints();
  cache = { stamp, recognizer, missing, released: !keepKeypoints };
  return { recognizer, missing };
}

// _resample of a (count x k) float32 block `points` (flat): Float32Array(length x k).
function resampleRows(points, k, mask, times, length) {
  const out = new Float32Array(length * k);
  resampleColumns(points, k, range(0, k), mask, times, 0, mask.length, length, out);
  return out;
}

// _bridge of a (count x k) float32 block: [bridged copy, mask].
function bridgeRows(points, k, mask, times, limit) {
  const copy = Float32Array.from(points);
  return [copy, bridge(copy, k, mask, times, limit)];
}

// Internals for tests and the other ports (names of translate.py's private functions).
export const _internal = {
  clean, runs: runsOf, liveSpan, wholeSpan, templateSpans, resampled: resampledBlock, spacing: spacingOf, gapLimit,
  frameDistances, computeFeatures, median32, median64, interp, gradient, argsortStable, USED_FACE_COLUMNS,
  resample: resampleRows, bridge: bridgeRows, handMotion, quiet, shoulders: shouldersOf, keywords, weighted,
  partDistances, dtwPlain, dtwRelaxedOne,
};
