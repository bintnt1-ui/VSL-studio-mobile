// The built-in samples (tools/build_mobile.py bundle): data/index.json + data/keypoints.bin (float16, the frame
// values recognition reads). Expanded here to full KEYPOINT_SIZE frames, zeros elsewhere.

import { KEYPOINT_SIZE } from "./layout.js";

const HALF_EXPONENT = new Float32Array(32);
for (let e = 0; e < 32; e++) HALF_EXPONENT[e] = 2 ** (e - 15);

export function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * HALF_EXPONENT[exponent] * (1 + fraction / 1024);
}

// Map from the bundle's frame values to KEYPOINT_SIZE positions: pose 0..131, the used face points, both hands.
export function frameColumns(facePoints) {
  const columns = [];
  for (let i = 0; i < 132; i++) columns.push(i);
  for (const point of facePoints) for (let axis = 0; axis < 3; axis++) columns.push(132 + 3 * point + axis);
  for (let i = 1536; i < 1662; i++) columns.push(i);
  return Int32Array.from(columns);
}

// index: the parsed index.json; buffer: keypoints.bin as an ArrayBuffer, or null for the samples without their
// keypoints (the data store's lists need no coordinates: ~110 MB less). Returns
// { labels: [{label, display}], samples: [{id, label, display, keypoints: Float32Array[] | null, aspect, seconds, source, key,
// mobile (phone sample id) | null, media ([media id, first frame, last frame] of a video segment) | null}], vocabulary,
// sha256, retired, imported }.
export function decodeBundle(index, buffer) {
  const columns = frameColumns(index.facePoints);
  const width = index.frameValues;
  if (columns.length !== width) throw new Error("bundle: frame layout does not match");
  const halves = buffer ? new Uint16Array(buffer) : null;
  if (halves && halves.length !== index.frames * width) throw new Error("bundle: keypoints.bin has the wrong size");
  const samples = index.samples.map((sample) => {
    let keypoints = null;
    if (halves) {
      keypoints = [];
      for (let f = 0; f < sample.frames; f++) {
        const frame = new Float32Array(KEYPOINT_SIZE);
        const base = (sample.offset + f) * width;
        for (let c = 0; c < width; c++) frame[columns[c]] = halfToFloat(halves[base + c]);
        keypoints.push(frame);
      }
    }
    const { label, display } = index.labels[sample.label];
    return { id: sample.id, label, display, keypoints, aspect: sample.aspect, seconds: sample.seconds,
      source: sample.source, key: sample.key, mobile: sample.mobile ?? null, media: sample.media ?? null,
      own: sample.own ?? null, // own: the sample's own display when it differs from its sign's
      prov: sample.prov ?? [] }; // seg: / src: provenance keys of a collected video sample
  });
  // retired: phone sample ids the PC deleted or refused (the phone drops its copies); imported: the phone's sync files
  // the PC took in (their changes are in this bundle); sha256: the bundle's identity.
  return { labels: index.labels, samples, vocabulary: index.vocabulary ?? [], sha256: index.sha256,
    retired: index.retired ?? [], imported: index.imported ?? [],
    // tombstone keys of deleted video segments / whole videos (sources.js deletedBefore / sourceDeleted)
    segments: index.segments ?? [], sources: index.sources ?? [] };
}

export async function loadBundle(base = "data/", { keypoints = true } = {}) {
  const [index, buffer] = await Promise.all([
    fetch(`${base}index.json`).then((r) => { if (!r.ok) throw new Error(`index.json: ${r.status}`); return r.json(); }),
    keypoints
      ? fetch(`${base}keypoints.bin`).then((r) => { if (!r.ok) throw new Error(`keypoints.bin: ${r.status}`); return r.arrayBuffer(); })
      : Promise.resolve(null),
  ]);
  return decodeBundle(index, buffer);
}
