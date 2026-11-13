// YuNet face detection + SFace embeddings on the phone (port of identity.FaceEngine), in ONNX Runtime Web.
//
// identity.py runs the models through OpenCV (cv2.FaceDetectorYN / cv2.FaceRecognizerSF); this module
// reproduces OpenCV's pre- and post-processing around the same ONNX models: the DETECT_WIDTH downscale
// (cv2.resize INTER_AREA, light.js), zero padding to a multiple of 32, the BGR blob, the decode of YuNet's
// multi-stride outputs, NMSBoxes on integer boxes (top-k 20, IoU 0.3), FaceEngine.detect's rescale and clipping;
// FaceRecognizerSF::alignCrop (getSimilarityTransformMatrix to the 112x112 reference + warpAffine INTER_LINEAR in
// OpenCV's 5-bit fixed point, bit exact) and feature (RGB blob, 0..255), normalised like FaceEngine.embed.
// The phone uses SFace int8 (face_recognition_sface_2021dec_int8.onnx, 10 MB instead of 37 MB); its
// identity scores stay on the fp32 model's scale (tools/golden_faces.py faces_identity). Both model files are
// adjusted in memory for ONNX Runtime, numerically unchanged: YuNet's declared 640x640 input is freed
// (freeYuNetInput), SFace's weights stop being graph inputs (withoutInitializerInputs) and its activations become
// uint8 (unsignedActivations: wasm has SIMD kernels for uint8 x int8 only; ~5x faster, identical embeddings).
//
// Frames are RGBA ImageData-like {width, height, data: Uint8ClampedArray}; Python's frames are BGR, the channel
// order is translated here. Models and ONNX Runtime are only fetched by FaceEngine.create() (pipeline.js imports
// this module when a profile or an enrollment first needs it). Golden test: mobile/tests/faces.test.js.

import { DETECTOR_MODEL, DETECT_WIDTH, Face, MIN_DETECTION_SCORE } from "./lock.js";
import { resizeArea, roundHalfEven } from "./light.js";

export { DETECTOR_MODEL, DETECT_WIDTH, MIN_DETECTION_SCORE };
export const RECOGNIZER_MODEL = "face_recognition_sface_2021dec_int8.onnx"; // the PC uses the fp32 model
export const NMS_THRESHOLD = 0.3; // cv2.FaceDetectorYN.create(..., nms_threshold=0.3, top_k=20)
export const TOP_K = 20;
export const STRIDES = Object.freeze([8, 16, 32]);
export const DIVISOR = 32; // YuNet's input is zero padded to a multiple of this
export const ALIGNED_SIZE = 112;
// SFace's reference landmarks (right eye, left eye, nose tip, mouth corners) in the 112x112 crop, and their mean.
export const REFERENCE_POINTS = Object.freeze([38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493,
  92.3655, 70.7299, 92.2041].map(Math.fround));
export const REFERENCE_MEAN = Object.freeze([56.0262, 71.9008].map(Math.fround));
export const OUTPUT_NAMES = Object.freeze(["cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
  "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"]);

const f32 = Math.fround;

function checkFrame(frame) {
  const { width, height, data } = frame || {};
  if (!data || !(width > 0) || !(height > 0) || data.length !== width * height * 4) {
    throw new Error("Face engine: the frame must be a non-empty RGBA image {width, height, data}.");
  }
}

// ------------------------------------------------------------------ detection

// FaceEngine.detect's input: the frame scaled to DETECT_WIDTH wide when wider (INTER_AREA), zero padded to a
// multiple of DIVISOR, as cv2.dnn.blobFromImage gives it (NCHW float32, B, G, R planes, 0..255).
// Returns {blob, width, height (scaled image), padWidth, padHeight, scale, small (RGB bytes, or null at scale 1)}.
export function detectionInput(frame, buffers = null) {
  checkFrame(frame);
  const { width, height, data } = frame;
  const scale = Math.min(1.0, DETECT_WIDTH / width);
  let pixels = data, stride = 4, smallWidth = width, smallHeight = height, small = null;
  if (scale !== 1.0) {
    smallWidth = roundHalfEven(width * scale);
    smallHeight = roundHalfEven(height * scale);
    const reuse = buffers?.small && buffers.small.length === smallWidth * smallHeight * 3 ? buffers.small : null;
    small = resizeArea(data, width, height, 4, smallWidth, smallHeight, reuse);
    if (buffers) buffers.small = small;
    pixels = small;
    stride = 3;
  }
  const padWidth = (Math.trunc((smallWidth - 1) / DIVISOR) + 1) * DIVISOR;
  const padHeight = (Math.trunc((smallHeight - 1) / DIVISOR) + 1) * DIVISOR;
  const plane = padWidth * padHeight;
  let blob = buffers?.blob;
  if (!blob || blob.length !== 3 * plane) {
    blob = new Float32Array(3 * plane);
    if (buffers) buffers.blob = blob;
  } else {
    blob.fill(0);
  }
  for (let y = 0; y < smallHeight; y++) {
    let p = y * smallWidth * stride, o = y * padWidth;
    for (let x = 0; x < smallWidth; x++, p += stride, o++) {
      blob[o] = pixels[p + 2]; // B
      blob[plane + o] = pixels[p + 1]; // G
      blob[2 * plane + o] = pixels[p]; // R
    }
  }
  return { blob, width: smallWidth, height: smallHeight, padWidth, padHeight, scale, small };
}

// FaceDetectorYN's postProcess: YuNet outputs ({cls_8: Float32Array, ...}) of a padWidth x padHeight input ->
// rows (Float32Array(15): x, y, w, h, right eye, left eye, nose tip, right and left mouth corner, score) in input
// pixels, best first after NMS (one candidate or none: as decoded). float32 arithmetic like OpenCV.
export function decodeDetections(outputs, padWidth, padHeight, scoreThreshold = MIN_DETECTION_SCORE,
  nmsThreshold = NMS_THRESHOLD, topK = TOP_K) {
  const threshold = f32(scoreThreshold);
  const faces = [];
  for (const stride of STRIDES) {
    const cols = Math.trunc(padWidth / stride), rows = Math.trunc(padHeight / stride);
    const cls = outputs[`cls_${stride}`], obj = outputs[`obj_${stride}`];
    const bbox = outputs[`bbox_${stride}`], kps = outputs[`kps_${stride}`];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const clsScore = Math.max(Math.min(cls[idx], 1), 0), objScore = Math.max(Math.min(obj[idx], 1), 0);
        const score = f32(Math.sqrt(f32(clsScore * objScore)));
        if (score < threshold) continue;
        const b = idx * 4, k = idx * 10;
        const cx = f32(f32(c + bbox[b]) * stride), cy = f32(f32(r + bbox[b + 1]) * stride);
        const w = f32(f32(Math.exp(bbox[b + 2])) * stride), h = f32(f32(Math.exp(bbox[b + 3])) * stride);
        const face = new Float32Array(15);
        face[0] = cx - w / 2; face[1] = cy - h / 2; face[2] = w; face[3] = h; // stores round to float32
        for (let n = 0; n < 5; n++) {
          face[4 + 2 * n] = f32(kps[k + 2 * n] + c) * stride;
          face[5 + 2 * n] = f32(kps[k + 2 * n + 1] + r) * stride;
        }
        face[14] = score;
        faces.push(face);
      }
    }
  }
  if (faces.length <= 1) return faces;
  const boxes = faces.map((face) => [Math.trunc(face[0]), Math.trunc(face[1]), Math.trunc(face[2]), Math.trunc(face[3])]);
  return nmsBoxes(boxes, faces.map((face) => face[14]), scoreThreshold, nmsThreshold, topK).map((i) => faces[i]);
}

// 1 - Jaccard distance of two integer [x, y, w, h] boxes (cv::Rect), as float32 like OpenCV's rectOverlap.
function rectOverlap(a, b) {
  const areaA = a[2] * a[3], areaB = b[2] * b[3];
  if (areaA + areaB <= 0) return 1;
  const width = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const both = width > 0 && height > 0 && a[2] > 0 && a[3] > 0 && b[2] > 0 && b[3] > 0 ? width * height : 0;
  return f32(1 - f32(1.0 - both / (areaA + areaB - both)));
}

// cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, eta = 1, topK): indices kept, best first
// (scores above the threshold, stable sort by score, the topK best, greedy suppression above nmsThreshold).
export function nmsBoxes(boxes, scores, scoreThreshold, nmsThreshold, topK = 0) {
  const threshold = f32(scoreThreshold), limit = f32(nmsThreshold);
  let order = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] > threshold) order.push(i);
  order.sort((i, j) => scores[j] - scores[i]); // stable: ties keep their order
  if (topK > 0 && topK < order.length) order = order.slice(0, topK);
  const kept = [];
  for (const index of order) {
    let keep = true;
    for (let k = 0; k < kept.length && keep; k++) keep = rectOverlap(boxes[index], boxes[kept[k]]) <= limit;
    if (keep) kept.push(index);
  }
  return kept;
}

// FaceEngine.detect after the detector: rows in detector-input pixels -> Face list in frame pixels (landmarks and
// box divided by the downscale in float32; the box clipped to the frame, faces narrower than 5 px dropped).
export function facesFromRows(rows, scale, width, height) {
  const divisor = f32(scale);
  const faces = [];
  for (const row of rows) {
    const full = Float32Array.from(row);
    for (let i = 0; i < 14; i++) full[i] = full[i] / divisor;
    const left = full[0] > 0 ? full[0] : 0, top = full[1] > 0 ? full[1] : 0; // Python max(0.0, v)
    const rightEdge = f32(full[0] + full[2]), bottomEdge = f32(full[1] + full[3]);
    const right = rightEdge < width ? rightEdge : width, bottom = bottomEdge < height ? bottomEdge : height;
    const x = Math.trunc(left), y = Math.trunc(top), w = Math.trunc(right - left), h = Math.trunc(bottom - top);
    if (w > 4 && h > 4) faces.push(new Face([x, y, w, h], full, row[14]));
  }
  return faces;
}

// ------------------------------------------------------------------ alignment (FaceRecognizerSF::alignCrop)

// getSimilarityTransformMatrix: the similarity (rotation, uniform scale, shift) taking the face's 5 landmarks
// (row[4..13], float32) closest to REFERENCE_POINTS (Umeyama). OpenCV takes U diag(1, sign det A) V^T and
// s0 + d s1 from the SVD of the 2x2 cross-covariance A; for 2x2 that is the rotation by
// atan2(A10 - A01, A00 + A11) and |(A00 + A11, A10 - A01)| (same values up to rounding). Returns Float64Array(6)
// [m00, m01, m02, m10, m11, m12] mapping frame -> crop.
export function similarityTransform(row) {
  const mean0 = f32(f32(f32(f32(f32(row[4] + row[6]) + row[8]) + row[10]) + row[12]) / 5);
  const mean1 = f32(f32(f32(f32(f32(row[5] + row[7]) + row[9]) + row[11]) + row[13]) / 5);
  let a00 = 0, a01 = 0, a10 = 0, a11 = 0, var1 = 0, var2 = 0;
  for (let i = 0; i < 5; i++) {
    const s0 = f32(row[4 + 2 * i] - mean0), s1 = f32(row[5 + 2 * i] - mean1);
    const d0 = f32(REFERENCE_POINTS[2 * i] - REFERENCE_MEAN[0]), d1 = f32(REFERENCE_POINTS[2 * i + 1] - REFERENCE_MEAN[1]);
    a00 += f32(d0 * s0); a01 += f32(d0 * s1); a10 += f32(d1 * s0); a11 += f32(d1 * s1); // float products
    var1 += f32(s0 * s0); var2 += f32(s1 * s1);
  }
  a00 /= 5; a01 /= 5; a10 /= 5; a11 /= 5; var1 /= 5; var2 /= 5;
  const along = a00 + a11, across = a10 - a01;
  const norm = Math.hypot(along, across);
  const cos = norm > 0 ? along / norm : 1, sin = norm > 0 ? across / norm : 0;
  const scale = 1.0 / (var1 + var2) * norm;
  const shift0 = cos * mean0 - sin * mean1, shift1 = sin * mean0 + cos * mean1;
  return Float64Array.of(cos * scale, -sin * scale, REFERENCE_MEAN[0] - scale * shift0,
    sin * scale, cos * scale, REFERENCE_MEAN[1] - scale * shift1);
}

// cv2.warpAffine(src, matrix, (dstWidth, dstHeight), INTER_LINEAR, BORDER_CONSTANT 0) on 8-bit interleaved
// images (`stride` channels per pixel, the first 3 warped; a 4th output channel is set to 255): OpenCV's fixed
// point (matrix inverted in float64, coordinates in 1/1024 px rounded to 1/32 px, 15-bit bilinear weights).
export function warpAffine(src, width, height, stride, matrix, dstWidth, dstHeight, out = null, outStride = 4) {
  const dst = out || new Uint8ClampedArray(dstWidth * dstHeight * outStride);
  let [m0, m1, m2, m3, m4, m5] = matrix;
  let det = m0 * m4 - m1 * m3;
  det = det !== 0 ? 1 / det : 0;
  const a11 = m4 * det, a22 = m0 * det;
  m0 = a11; m1 *= -det; m3 *= -det; m4 = a22;
  const b1 = -m0 * m2 - m1 * m5, b2 = -m3 * m2 - m4 * m5;
  m2 = b1; m5 = b2;
  const cvRoundInt = (value) => Math.min(Math.max(roundHalfEven(value), -2147483648), 2147483647);
  const short = (value) => (value < -32768 ? -32768 : value > 32767 ? 32767 : value);
  const adelta = new Int32Array(dstWidth), bdelta = new Int32Array(dstWidth);
  for (let x = 0; x < dstWidth; x++) {
    adelta[x] = cvRoundInt(m0 * x * 1024);
    bdelta[x] = cvRoundInt(m3 * x * 1024);
  }
  const rowStride = width * stride;
  for (let y = 0, o = 0; y < dstHeight; y++) {
    const X0 = cvRoundInt((m1 * y + m2) * 1024) + 16, Y0 = cvRoundInt((m4 * y + m5) * 1024) + 16;
    for (let x = 0; x < dstWidth; x++, o += outStride) {
      const X = (X0 + adelta[x]) >> 5, Y = (Y0 + bdelta[x]) >> 5;
      const sx = short(X >> 5), sy = short(Y >> 5), fx = X & 31, fy = Y & 31;
      if (outStride === 4) dst[o + 3] = 255;
      if (sx >= width || sx + 1 < 0 || sy >= height || sy + 1 < 0) {
        dst[o] = 0; dst[o + 1] = 0; dst[o + 2] = 0;
        continue;
      }
      const w00 = (32 - fy) * (32 - fx) * 32, w01 = (32 - fy) * fx * 32, w10 = fy * (32 - fx) * 32, w11 = fy * fx * 32;
      const inX0 = sx >= 0 && sx < width, inX1 = sx + 1 >= 0 && sx + 1 < width;
      const inY0 = sy >= 0 && sy < height, inY1 = sy + 1 >= 0 && sy + 1 < height;
      const p00 = inY0 && inX0 ? sy * rowStride + sx * stride : -1;
      const p01 = inY0 && inX1 ? sy * rowStride + (sx + 1) * stride : -1;
      const p10 = inY1 && inX0 ? (sy + 1) * rowStride + sx * stride : -1;
      const p11 = inY1 && inX1 ? (sy + 1) * rowStride + (sx + 1) * stride : -1;
      for (let c = 0; c < 3; c++) {
        const sum = (p00 >= 0 ? src[p00 + c] * w00 : 0) + (p01 >= 0 ? src[p01 + c] * w01 : 0)
          + (p10 >= 0 ? src[p10 + c] * w10 : 0) + (p11 >= 0 ? src[p11 + c] * w11 : 0);
        const value = (sum + 16384) >> 15;
        dst[o + c] = value > 255 ? 255 : value;
      }
    }
  }
  return dst;
}

// FaceRecognizerSF::alignCrop: the face warped onto SFace's 112x112 reference, RGBA (alpha 255).
export function alignCrop(frame, row, out = null) {
  checkFrame(frame);
  const data = warpAffine(frame.data, frame.width, frame.height, 4, similarityTransform(row), ALIGNED_SIZE,
    ALIGNED_SIZE, out?.data ?? null, 4);
  return out ?? { width: ALIGNED_SIZE, height: ALIGNED_SIZE, data };
}

// FaceRecognizerSF::feature's blob of an aligned crop (RGBA or RGB bytes, 112x112): NCHW float32 R, G, B planes
// (blobFromImage swapRB=true of the BGR crop), 0..255.
export function embeddingInput(aligned, blob = null) {
  const data = aligned.data ?? aligned;
  const pixels = ALIGNED_SIZE * ALIGNED_SIZE;
  const stride = data.length === pixels * 4 ? 4 : data.length === pixels * 3 ? 3 : 0;
  if (!stride) throw new Error("Face engine: the aligned crop must be 112x112 RGBA or RGB.");
  const out = blob && blob.length === 3 * pixels ? blob : new Float32Array(3 * pixels);
  for (let i = 0, p = 0; i < pixels; i++, p += stride) {
    out[i] = data[p];
    out[pixels + i] = data[p + 1];
    out[2 * pixels + i] = data[p + 2];
  }
  return out;
}

// FaceEngine.embed's normalisation: feature / max(|feature|, 1e-6), float32.
export function normalizeFeature(feature) {
  let sum = 0;
  for (let i = 0; i < feature.length; i++) sum += feature[i] * feature[i];
  const norm = f32(Math.max(f32(Math.sqrt(sum)), 1e-6));
  const out = new Float32Array(feature.length);
  for (let i = 0; i < feature.length; i++) out[i] = feature[i] / norm;
  return out;
}

// frame[y0:y1, x0:x1].copy() (bounds clipped to the frame like a numpy slice's stop) as an RGBA image.
export function cropImage(frame, box) {
  checkFrame(frame);
  const clip = (value, size) => Math.min(Math.max(Math.trunc(value), 0), size);
  const x0 = clip(box[0], frame.width), y0 = clip(box[1], frame.height);
  const x1 = Math.max(x0, clip(box[2], frame.width)), y1 = Math.max(y0, clip(box[3], frame.height));
  const width = x1 - x0, height = y1 - y0;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const start = ((y0 + y) * frame.width + x0) * 4;
    data.set(frame.data.subarray(start, start + width * 4), y * width * 4);
  }
  return { width, height, data };
}

// ------------------------------------------------------------------ YuNet with any input size

// The YuNet file declares a fixed 1x3x640x640 input (and 640x640 output / intermediate shapes); its graph works
// at any size (Reshape to [1, -1, k], Resize by scales), and OpenCV simply feeds the padded frame. ONNX Runtime
// checks declared shapes, so the declarations are freed in the protobuf: input height / width and the outputs'
// anchor counts become symbolic, the intermediate value_info entries are dropped (re-inferred). Weights untouched.

function readVarint(bytes, position) {
  let value = 0, scale = 1, byte;
  do {
    if (position >= bytes.length) throw new Error("Face engine: truncated model file.");
    byte = bytes[position++];
    value += (byte & 0x7f) * scale;
    scale *= 128;
  } while (byte & 0x80);
  return [value, position];
}

function varintBytes(value) {
  const out = [];
  while (value >= 128) {
    out.push((value % 128) | 0x80);
    value = Math.floor(value / 128);
  }
  out.push(value);
  return out;
}

// Rewrites a protobuf message: `edit(field, payload)` for length-delimited fields returns a new payload
// (Uint8Array), null to drop the field, or undefined to keep it as it is. Other fields are copied unchanged.
function rewriteMessage(bytes, edit) {
  const chunks = [];
  let position = 0, changed = false;
  while (position < bytes.length) {
    const start = position;
    let key;
    [key, position] = readVarint(bytes, position);
    const field = Math.floor(key / 8), wire = key % 8;
    if (wire === 0) [, position] = readVarint(bytes, position);
    else if (wire === 1) position += 8;
    else if (wire === 5) position += 4;
    else if (wire === 2) {
      let length;
      [length, position] = readVarint(bytes, position);
      const payload = bytes.subarray(position, position + length);
      position += length;
      const replaced = edit(field, payload);
      if (replaced === null) { changed = true; continue; }
      if (replaced !== undefined) {
        chunks.push(encodeField(field, wire, replaced));
        changed = true;
        continue;
      }
    } else {
      throw new Error(`Face engine: unsupported protobuf wire type ${wire}.`);
    }
    chunks.push(bytes.subarray(start, position));
  }
  return changed ? joinBytes(chunks) : bytes;
}

const textEncoder = new TextEncoder();

// ValueInfoProto with the tensor dimensions at `indices` made symbolic (`names[i]`).
function freeDimensions(valueInfo, indices, names) {
  return rewriteMessage(valueInfo, (field, type) => (field !== 2 ? undefined : rewriteMessage(type, (f, tensor) => (
    f !== 1 ? undefined : rewriteMessage(tensor, (g, shape) => {
      if (g !== 2) return undefined;
      let index = -1;
      return rewriteMessage(shape, (h, dimension) => {
        if (h !== 1) return undefined;
        index++;
        const at = indices.indexOf(index);
        if (at < 0) return undefined;
        const name = textEncoder.encode(names[at]);
        return Uint8Array.from([0x12, ...varintBytes(name.length), ...name]); // Dimension.dim_param
      });
    })))));
}

// The YuNet model bytes with a free input size (see above).
export function freeYuNetInput(model) {
  let outputs = 0;
  return rewriteMessage(model, (field, graph) => (field !== 7 ? undefined : rewriteMessage(graph, (f, payload) => {
    if (f === 11) return freeDimensions(payload, [2, 3], ["height", "width"]); // graph input
    if (f === 12) return freeDimensions(payload, [1], [`anchors_${outputs++}`]); // graph outputs
    if (f === 13) return null; // value_info
    return undefined;
  })));
}

const textDecoder = new TextDecoder();

// First length-delimited field `wanted` of a message as text ("" when absent).
function stringField(bytes, wanted) {
  let found = "";
  rewriteMessage(bytes, (field, payload) => {
    if (field === wanted && !found) found = textDecoder.decode(payload);
    return undefined;
  });
  return found;
}

// The model bytes without the graph inputs that are initializers (the SFace file lists its 145 weights as
// inputs, an old exporter's habit): ONNX Runtime then treats them as constants (no warnings; constant folding).
// Same as onnxruntime/tools/python/remove_initializer_from_input.py.
export function withoutInitializerInputs(model) {
  return rewriteMessage(model, (field, graph) => {
    if (field !== 7) return undefined;
    const initializers = new Set();
    rewriteMessage(graph, (f, payload) => {
      if (f === 5) initializers.add(stringField(payload, 8)); // TensorProto.name
      return undefined;
    });
    return rewriteMessage(graph, (f, payload) => (f === 11 && initializers.has(stringField(payload, 1)) ? null : undefined));
  });
}

// Fields of a protobuf message: [{field, wire, value (varint / fixed bytes) or payload (length-delimited)}].
function parseFields(bytes) {
  const out = [];
  let position = 0;
  while (position < bytes.length) {
    let key;
    [key, position] = readVarint(bytes, position);
    const field = Math.floor(key / 8), wire = key % 8;
    if (wire === 0) {
      const start = position;
      [, position] = readVarint(bytes, position);
      out.push({ field, wire, payload: bytes.subarray(start, position) });
    } else if (wire === 1 || wire === 5) {
      const length = wire === 1 ? 8 : 4;
      out.push({ field, wire, payload: bytes.subarray(position, position + length) });
      position += length;
    } else if (wire === 2) {
      let length;
      [length, position] = readVarint(bytes, position);
      out.push({ field, wire, payload: bytes.subarray(position, position + length) });
      position += length;
    } else {
      throw new Error(`Face engine: unsupported protobuf wire type ${wire}.`);
    }
  }
  return out;
}

// int32 value of a varint (int32 fields encode negatives as 10-byte varints: the low 32 bits count).
function varintInt32(bytes, position) {
  let low = 0, scale = 1, byte, count = 0;
  do {
    byte = bytes[position++];
    if (count++ < 5) low += (byte & 0x7f) * scale;
    scale *= 128;
  } while (byte & 0x80);
  return [(low % 4294967296) | 0, position];
}

const INT8 = 3, UINT8 = 2; // TensorProto.DataType

// The int8 values of a TensorProto (raw_data or int32_data, packed or not), or null if it is not int8.
function int8Values(fields) {
  const type = fields.find((f) => f.field === 2);
  if (!type || readVarint(type.payload, 0)[0] !== INT8) return null;
  const raw = fields.find((f) => f.field === 9);
  if (raw) return Int8Array.from(new Int8Array(raw.payload.buffer, raw.payload.byteOffset, raw.payload.length));
  const values = [];
  for (const f of fields) {
    if (f.field !== 5) continue;
    for (let position = 0; position < f.payload.length;) {
      let value;
      [value, position] = varintInt32(f.payload, position);
      values.push(value);
    }
  }
  return Int8Array.from(values);
}

function encodeField(field, wire, payload) {
  const head = varintBytes(field * 8 + wire);
  if (wire === 2) head.push(...varintBytes(payload.length));
  const out = new Uint8Array(head.length + payload.length);
  out.set(head);
  out.set(payload, head.length);
  return out;
}

function joinBytes(chunks) {
  const out = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

// The same TensorProto as uint8 holding value + 128 (raw_data).
function shiftedTensor(fields, values) {
  const chunks = [];
  for (const f of fields) {
    if (f.field === 2 || f.field === 5 || f.field === 9) continue;
    chunks.push(encodeField(f.field, f.wire, f.payload));
  }
  chunks.push(encodeField(2, 0, varintBytes(UINT8)));
  chunks.push(encodeField(9, 2, Uint8Array.from(values, (value) => value + 128)));
  return joinBytes(chunks);
}

// SFace int8 with uint8 activations: ONNX Runtime Web's wasm build multiplies uint8 x int8 with SIMD but
// int8 x int8 with a scalar loop (SFace int8 took ~340 ms per face on a desktop, now about a fifth). Every
// activation zero point (and the constant operands of QLinearAdd / QLinearMul, which share the activation type)
// becomes uint8 holding value + 128: (q + 128) - (zero point + 128) is the same number, so every integer product,
// requantisation and dequantised value is unchanged (the test checks the embeddings are identical). Weights stay
// int8. Returns the model unchanged when its graph is not laid out as expected.
export function unsignedActivations(model) {
  return rewriteMessage(model, (field, graphBytes) => {
    if (field !== 7) return undefined;
    const graph = parseFields(graphBytes);
    const tensors = new Map();
    for (const f of graph) if (f.field === 5) tensors.set(stringField(f.payload, 8), parseFields(f.payload));
    const activation = new Set(), constant = new Set(), weights = new Set();
    for (const f of graph) {
      if (f.field !== 1) continue;
      const node = parseFields(f.payload);
      const op = textDecoder.decode(node.find((n) => n.field === 4)?.payload ?? new Uint8Array());
      const inputs = node.filter((n) => n.field === 1).map((n) => textDecoder.decode(n.payload));
      if (op === "QuantizeLinear" || op === "DequantizeLinear") {
        if (inputs[2] === undefined) return undefined; // default uint8 zero point: leave the model alone
        (tensors.has(inputs[0]) ? weights : activation).add(inputs[2]);
      } else if (op === "QLinearConv" || op === "QLinearMatMul") {
        activation.add(inputs[2]); activation.add(inputs[7]); weights.add(inputs[5]);
      } else if (op === "QLinearAdd" || op === "QLinearMul") {
        activation.add(inputs[2]); activation.add(inputs[7]);
        if (tensors.has(inputs[3])) { constant.add(inputs[3]); constant.add(inputs[5]); } else activation.add(inputs[5]);
      }
    }
    const shifted = new Set([...activation, ...constant]);
    if (!shifted.size) return undefined; // not a quantized model
    for (const name of shifted) {
      if (weights.has(name) || !tensors.has(name) || !int8Values(tensors.get(name))) return undefined;
    }
    return joinBytes(graph.flatMap((f) => {
      if (f.field === 13) return []; // value_info (int8 activations): re-inferred
      if (f.field === 5) {
        const fields = tensors.get(stringField(f.payload, 8));
        if (shifted.has(stringField(f.payload, 8))) return [encodeField(5, 2, shiftedTensor(fields, int8Values(fields)))];
      }
      return [encodeField(f.field, f.wire, f.payload)];
    }));
  });
}

// ------------------------------------------------------------------ the engine

const ortModules = new Map(); // base URL -> Promise of the ONNX Runtime Web module

function loadOrt(base) {
  const key = String(base);
  if (!ortModules.has(key)) {
    ortModules.set(key, (async () => {
      const ort = await import(new URL("vendor/ort/ort.wasm.min.mjs", base).href);
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", base).href;
      ort.env.wasm.numThreads = 1; // no cross-origin isolation needed; the camera loop is sequential anyway
      return ort;
    })().catch((error) => {
      ortModules.delete(key);
      throw error;
    }));
  }
  return ortModules.get(key);
}

async function fetchModel(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Không tải được mô hình khuôn mặt ${url} (${response.status}).`);
  return new Uint8Array(await response.arrayBuffer());
}

// YuNet face detection + SFace embeddings (identity.FaceEngine). One model run at a time: calls are queued.
export class FaceEngine {
  // Loads ONNX Runtime Web (vendor/ort) and both models (models/) under `base` (the app folder).
  static async create(base = new URL("../", import.meta.url), options = {}) {
    const ort = await loadOrt(base);
    const sessionOptions = { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...options.session };
    const [detectorBytes, recognizerBytes] = await Promise.all([
      fetchModel(new URL(`models/${DETECTOR_MODEL}`, base).href),
      fetchModel(new URL(`models/${options.recognizerModel ?? RECOGNIZER_MODEL}`, base).href),
    ]);
    const detector = await ort.InferenceSession.create(freeYuNetInput(detectorBytes), sessionOptions);
    let recognizerModel = withoutInitializerInputs(recognizerBytes);
    if (options.unsignedActivations !== false) recognizerModel = unsignedActivations(recognizerModel);
    const recognizer = await ort.InferenceSession.create(recognizerModel, sessionOptions);
    return new FaceEngine(ort, detector, recognizer);
  }

  constructor(ort, detector, recognizer) {
    this.ort = ort;
    this.detector = detector;
    this.recognizer = recognizer;
    this._queue = Promise.resolve();
    this._detectBuffers = {};
    this._embedBlob = new Float32Array(3 * ALIGNED_SIZE * ALIGNED_SIZE);
  }

  // Runs `task` after every earlier model call has finished (ORT sessions take one run at a time).
  _exclusive(task) {
    const result = this._queue.then(task);
    this._queue = result.catch(() => {});
    return result;
  }

  // YuNet outputs ({name: Float32Array}) for a detectionInput (exposed for tests).
  async detectorOutputs(input) {
    const tensor = new this.ort.Tensor("float32", input.blob, [1, 3, input.padHeight, input.padWidth]);
    const results = await this.detector.run({ [this.detector.inputNames[0]]: tensor });
    const outputs = {};
    for (const name of OUTPUT_NAMES) outputs[name] = results[name].data;
    return outputs;
  }

  // Faces of an RGBA frame (Face list, frame pixels, best first). The frame is read when the call's turn comes:
  // keep its pixels until the promise settles (the camera loop awaits every call).
  detect(frame) {
    return this._exclusive(async () => {
      const input = detectionInput(frame, this._detectBuffers);
      const rows = decodeDetections(await this.detectorOutputs(input), input.padWidth, input.padHeight);
      return facesFromRows(rows, input.scale, frame.width, frame.height);
    });
  }

  // The face aligned for SFace: 112x112 RGBA image.
  async aligned(frame, face) {
    return alignCrop(frame, face.row);
  }

  // Unit-length SFace embedding (Float32Array(128)) of the face, from `aligned` when given.
  embed(frame, face, aligned = null) {
    let crop;
    try {
      crop = aligned ?? alignCrop(frame, face.row); // now: the frame may change while earlier calls run
    } catch (error) {
      return Promise.reject(error);
    }
    return this._exclusive(async () => {
      const blob = embeddingInput(crop, this._embedBlob);
      const tensor = new this.ort.Tensor("float32", blob, [1, 3, ALIGNED_SIZE, ALIGNED_SIZE]);
      const results = await this.recognizer.run({ [this.recognizer.inputNames[0]]: tensor });
      return normalizeFeature(results[this.recognizer.outputNames[0]].data);
    });
  }

  // Copy of frame[y0:y1, x0:x1] (Enrollment's avatar).
  crop(frame, box) {
    return cropImage(frame, box);
  }

  async release() {
    await this._queue;
    await Promise.all([this.detector.release?.(), this.recognizer.release?.()]);
  }
}
