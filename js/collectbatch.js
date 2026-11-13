// Importing one-sign videos whose sign name is known (QIPEDC clips, labelled files) and the checks around saving a
// collected segment: the phone's port of collectors.import_labelled / sample_metadata / near_duplicate /
// saved_segments / deleted_before and dataset's provenance keys (source_deleted, deleted_segments, provenance_keys).
// Pure logic: everything that reads a video, runs Holistic, asks the recognizer or writes the data store is an injected
// effect, so the collector UI drives it with videocollect.js, recognition.js and store.js. nearDuplicateIn runs in
// recognition.worker.js ("nearDuplicate" request), on the recognizer it already holds.
//
// No DOM and no vendored library is imported here: the recognition worker loads this module too.

import { frameColumns, halfToFloat } from "./bundle.js";
import { KEYPOINT_SIZE } from "./layout.js";
import { canonicalLabel } from "./naming.js";
import { overlaps, Proposal, singleSignSpan } from "./segmentation.js";
import {
  DEFAULT_ASPECT, DEFAULT_FRAME_SECONDS, FRAME_COUNT, MIN_TEMPLATE_HAND_COVERAGE, Recognizer, Template, frameTimes,
  gestureFeatures, handCoverage, sampleGeometry,
} from "./translate.js";

export const NEAR_DUPLICATE_DISTANCE = 0.05; // same recording published twice (QIPEDC Nam/Trung clips): < 0.04; two signs: > 0.08
export const COLLECTOR_VERSION = "2";
export const SINGLE_SIGN_REASON = "video một ký hiệu"; // scan_video's reason for a one-sign clip's proposal

// import_labelled's outcomes, exactly as the PC shows them.
export const OUTCOMES = Object.freeze({
  unnamed: "chưa có tên",
  deleted: "đã bị xóa trước đây",
  exists: "đã có",
  noHands: "không thấy tay",
  unclear: "tay không rõ",
  duplicate: "trùng bản quay đã có",
  saved: "đã lưu",
});

// collectors.Cancelled: thrown when the scan or the segment effect gives nothing back (null). The effects built on
// videocollect.js throw videocollect's own Cancelled instead; pass that class as importLabelled's `Cancelled` option to
// get one class for both.
export class Cancelled extends Error {
  constructor(message = "Đã hủy.") {
    super(message);
    this.name = "Cancelled";
  }
}

// ------------------------------------------------------------------ Python formatting and parsing

// f"{value:.{digits}f}": the exact binary value rounded half to even (toFixed rounds exact ties up, e.g. 0.125 ->
// "0.13" where Python writes "0.12"), "-" kept for negative values that round to zero, nan / inf as Python writes them.
export function pythonFixed(value, digits) {
  const x = Number(value);
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const high = view.getUint32(0), low = view.getUint32(4);
  const biased = (high >>> 20) & 0x7ff;
  let mantissa = (BigInt(high & 0xfffff) << 32n) | BigInt(low);
  let exponent = -1074;
  if (biased) {
    mantissa |= 1n << 52n;
    exponent = biased - 1075;
  }
  const scale = 10n ** BigInt(digits);
  let units;
  if (exponent >= 0) {
    units = (mantissa << BigInt(exponent)) * scale;
  } else {
    const numerator = mantissa * scale, denominator = 1n << BigInt(-exponent);
    units = numerator / denominator;
    const twice = 2n * (numerator % denominator);
    if (twice > denominator || (twice === denominator && units % 2n === 1n)) units += 1n;
  }
  let text = units.toString().padStart(digits + 1, "0");
  if (digits > 0) text = `${text.slice(0, -digits)}.${text.slice(-digits)}`;
  return (high >>> 31 ? "-" : "") + text;
}

// round(): half to even.
function roundHalfEven(value) {
  const floor = Math.floor(value), diff = value - floor;
  if (diff !== 0.5) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}

// float(text) (undefined where Python raises ValueError / TypeError).
function pythonFloat(text) {
  if (typeof text === "number") return text;
  if (typeof text !== "string") return undefined;
  const value = text.trim();
  if (/^[+-]?(inf|infinity)$/i.test(value)) return value.startsWith("-") ? -Infinity : Infinity;
  if (/^[+-]?nan$/i.test(value)) return NaN;
  if (!/^[+-]?((\d(_?\d)*)(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$/.test(value)) return undefined;
  return Number(value.replace(/_/g, ""));
}

// int(text) of a string or an int (undefined where Python raises ValueError).
function pythonInt(text) {
  if (typeof text === "number") return Number.isFinite(text) ? Math.trunc(text) : undefined;
  if (typeof text !== "string") return undefined;
  const value = text.trim();
  if (!/^[+-]?\d(_?\d)*$/.test(value)) return undefined;
  return Number.parseInt(value.replace(/_/g, ""), 10);
}

// str(value) as metadata.txt writes it (dataset._metadata_text).
function pythonStr(value) {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  return String(value);
}

// Python truthiness of a metadata value (None, "", 0, empty list / dict are false; NaN is true).
function truthy(value) {
  if (value === null || value === undefined || value === false || value === "" || value === 0) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object" && !ArrayBuffer.isView(value)) return Object.keys(value).length > 0;
  return true;
}

function regionText(region) {
  return Array.from(region, (value) => String(Math.trunc(Number(value)))).join(",");
}

// ------------------------------------------------------------------ metadata

// collectors.sample_metadata: provenance of a collected sample, never a local path. item: a source item
// {mediaId, title, origin, platform, variant, details} (media_id is read too); segment: {first, last, coverage, region};
// size: [width, height]; signerRegion: the Holistic crop (default: segment.region). Every value is a string, as
// metadata.txt stores it (start_frame "10", not 10), in the PC's key order.
export function sampleMetadataFor(item, segment, fps, size, labelFrom, textRegion = null, ocrText = "",
  ocrConfidence = null, signerRegion = null) {
  const mediaId = String(item.mediaId ?? item.media_id);
  const [width, height] = size || [0, 0];
  const metadata = {
    source: mediaId.startsWith("qipedc:") ? "qipedc" : "video",
    media_id: mediaId,
    origin: pythonStr(item.origin),
    platform: pythonStr(item.platform),
    title: pythonStr(item.title),
    start: pythonFixed(segment.first / fps, 3),
    end: pythonFixed(segment.last / fps, 3),
    start_frame: pythonStr(segment.first),
    end_frame: pythonStr(segment.last),
    fps: pythonFixed(fps, 3),
    video_size: `${pythonStr(width)}x${pythonStr(height)}`,
    label_from: labelFrom,
    hand_coverage: pythonFixed(segment.coverage, 2),
    orientation: "original",
    collector: COLLECTOR_VERSION,
  };
  if (truthy(item.variant)) metadata.variant = pythonStr(item.variant);
  const region = signerRegion ?? segment.region;
  if (truthy(region)) metadata.signer_region = regionText(region);
  if (truthy(textRegion)) metadata.text_region = regionText(textRegion);
  if (truthy(ocrText)) metadata.ocr_text = pythonStr(ocrText);
  if (ocrConfidence !== null && ocrConfidence !== undefined) metadata.ocr_confidence = pythonFixed(pythonFloat(ocrConfidence), 2);
  for (const [key, value] of Object.entries(item.details ?? {})) {
    if (truthy(value) && !Object.hasOwn(metadata, key)) metadata[key] = pythonStr(value);
  }
  return metadata;
}

// ------------------------------------------------------------------ scan of a one-sign clip

// scan_video(..., label=label): the clip holds one sign, so its only proposal is the span from the first to the last
// activity (segmentation.single_sign_span), named `label`. scan: videocollect.scanVideo's result ({frames, signals,
// fps, total}); returns it with `proposals` replaced (empty when no hand moved) and no captions.
export function labelledScan(scan, label) {
  const span = singleSignSpan(scan.frames, scan.signals.active, scan.fps, scan.total);
  const proposals = span ? [new Proposal(span[0], span[1], label, "", 0.0, SINGLE_SIGN_REASON)] : [];
  return { ...scan, captions: [], proposals };
}

// ------------------------------------------------------------------ import_labelled

function isError(error, name) {
  for (let proto = error; proto; proto = Object.getPrototypeOf(proto)) {
    if (proto.name === name || proto.constructor?.name === name) return true;
  }
  return false;
}

// collectors.import_labelled: find the sign of one video whose sign name is known, extract it and save it. Never
// brings back what the user deleted, never saves a second copy of an item, and skips a regional variant that is the
// same recording as a sample already saved. Resolves to [outcome (OUTCOMES), what save returned or null].
//
// item: {mediaId, title, origin, platform, label, variant, details}. effects (each may be async):
//   sourceDeleted(mediaId) -> bool            dataset.source_deleted (a whole-video import of it was deleted)
//   savedSegments(mediaId) -> array           collectors.saved_segments (any entry: "đã có")
//   open(item) -> clip {fps, width, height}   video_sources.fetch + probe_video
//   scan(clip, label) -> {proposals: [{start, end}]} | null   scan_video(label=...) (labelledScan(scanVideo(clip)))
//   deletedBefore(mediaId, first, last, fps) -> bool          collectors.deleted_before
//   compute(clip, first, last) -> {first, last, keypoints, preview} | null   compute_segment
//   coverage(keypoints) -> number             optional; default handCoverage(gestureFeatures(keypoints))
//   nearDuplicate(label, keypoints, aspect, frameSeconds) -> id | null (or bool)   collectors.near_duplicate
//   save(label, keypoints, {display, metadata, preview, aspect, seconds}) -> saved   collectors.save_segment;
//     throws DuplicateSampleError ("đã có") / PreviouslyDeletedError ("đã bị xóa trước đây"), recognised by class name
//   close(clip)                               optional; called once the clip is no longer needed
// options: progress(fraction, text) (-1: no fraction, the PC's texts), Cancelled: the class thrown when scan /
// compute give null (default: this module's Cancelled).
export async function importLabelled(item, effects, { progress = null, Cancelled: CancelledError = Cancelled } = {}) {
  if (!item.label) return [OUTCOMES.unnamed, null];
  const mediaId = item.mediaId ?? item.media_id;
  if (await effects.sourceDeleted(mediaId)) return [OUTCOMES.deleted, null];
  if ((await effects.savedSegments(mediaId))?.length) return [OUTCOMES.exists, null];
  progress?.(-1.0, `Đang tải “${item.title}”…`);
  const clip = await effects.open(item);
  try {
    const { fps, width, height } = clip;
    progress?.(-1.0, `Đang tìm động tác trong “${item.title}”…`);
    const scan = await effects.scan(clip, item.label);
    if (scan === null || scan === undefined) throw new CancelledError("Đã hủy.");
    if (!scan.proposals?.length) return [OUTCOMES.noHands, null];
    const proposal = scan.proposals[0];
    if (await effects.deletedBefore(mediaId, proposal.start, proposal.end, fps)) return [OUTCOMES.deleted, null];
    const computed = await effects.compute(clip, proposal.start, proposal.end);
    if (computed === null || computed === undefined) throw new CancelledError("Đã hủy.");
    const coverage = effects.coverage ? await effects.coverage(computed.keypoints)
      : handCoverage(gestureFeatures(computed.keypoints));
    const segment = { ...computed, coverage };
    if (coverage < MIN_TEMPLATE_HAND_COVERAGE) return [OUTCOMES.unclear, null];
    const seconds = fps ? (segment.last - segment.first) / fps / (FRAME_COUNT - 1) : null;
    const duplicate = await effects.nearDuplicate(item.label, segment.keypoints, height ? width / height : null, seconds);
    if (duplicate !== null && duplicate !== undefined && duplicate !== false) return [OUTCOMES.duplicate, null];
    const labelFrom = String(mediaId).startsWith("qipedc:") ? "dictionary" : "folder";
    const metadata = sampleMetadataFor(item, segment, fps, [width, height], labelFrom);
    const [aspect, interval] = sampleGeometry(metadata); // what the PC reads back from this metadata
    try {
      const saved = await effects.save(item.label, segment.keypoints, {
        display: item.label, metadata, preview: segment.preview ?? null, aspect, seconds: interval,
      });
      return [OUTCOMES.saved, saved];
    } catch (error) {
      if (isError(error, "DuplicateSampleError")) return [OUTCOMES.exists, null];
      if (isError(error, "PreviouslyDeletedError")) return [OUTCOMES.deleted, null];
      throw error;
    }
  } finally {
    await effects.close?.(clip);
  }
}

// ------------------------------------------------------------------ near duplicates (recognition worker)

function frames(values) {
  if (!(values instanceof Float32Array)) return values;
  const out = [];
  for (let i = 0; i < values.length / KEYPOINT_SIZE; i++) out.push(values.subarray(i * KEYPOINT_SIZE, (i + 1) * KEYPOINT_SIZE));
  return out;
}

// The stored samples of sign `label` that `recognizer` does not hold (too few hand frames to be a template), which
// collectors.near_duplicate compares too: from the worker's catalog ({builtin: [{id, label}], phone: [{id, label,
// keypoints, aspect, seconds, metadata}]}) and the bundle ({index, buffer}: the float16 keypoints of built-in ones).
// Returns [{id, label, display, keypoints, aspect, seconds}].
export function samplesOutside(recognizer, label, catalog, bundle = null) {
  let folder;
  try {
    folder = canonicalLabel(label);
  } catch {
    return [];
  }
  const known = new Set(recognizer.templates.filter((t) => t.label === folder).map((t) => t.source));
  const out = [];
  const builtin = (catalog?.builtin ?? []).filter((entry) => entry.label === folder && !known.has(entry.id));
  if (builtin.length && bundle) {
    const byId = new Map(bundle.index.samples.map((sample) => [sample.id, sample]));
    const columns = frameColumns(bundle.index.facePoints);
    const width = bundle.index.frameValues;
    for (const entry of builtin) {
      const sample = byId.get(entry.id);
      if (!sample) continue;
      const halves = new Uint16Array(bundle.buffer, sample.offset * width * 2, sample.frames * width);
      const keypoints = [];
      for (let f = 0; f < sample.frames; f++) {
        const frame = new Float32Array(KEYPOINT_SIZE);
        for (let c = 0; c < width; c++) frame[columns[c]] = halfToFloat(halves[f * width + c]);
        keypoints.push(frame);
      }
      out.push({ id: entry.id, label: folder, display: entry.display ?? folder, keypoints, aspect: sample.aspect,
        seconds: sample.seconds });
    }
  }
  for (const entry of catalog?.phone ?? []) {
    if (entry.label !== folder || known.has(entry.id) || !entry.keypoints?.length) continue;
    out.push({ ...entry, label: folder, keypoints: frames(entry.keypoints) });
  }
  return out;
}

// collectors.near_duplicate's comparison: {sources, distances} of every stored sample of sign `label` — the
// recognizer's templates of that sign (prepared parts reused: their keypoints may be released) and `outside` samples
// (samplesOutside) — to `keypoints`, the frames of a new recording; same_recording and stored semantics. aspect: the
// new recording's frame width / height (null: 4:3), frameSeconds: time between its frames (null: 1/15 s).
export function nearDuplicateDistances(recognizer, label, keypoints, { aspect = null, frameSeconds = null, outside = [] } = {}) {
  let folder;
  try {
    folder = canonicalLabel(label);
  } catch {
    return { sources: [], distances: new Float64Array(0) };
  }
  const templates = [], prepared = [];
  recognizer.templates.forEach((template, index) => {
    if (template.label !== folder) return;
    templates.push(template);
    prepared.push(recognizer._prepared[index]);
  });
  for (const sample of outside) {
    let { aspect: sampleAspect, seconds } = sample;
    if (sampleAspect == null || seconds == null) {
      const [measured, interval] = sample.metadata ? sampleGeometry(sample.metadata) : [DEFAULT_ASPECT, DEFAULT_FRAME_SECONDS];
      sampleAspect ??= measured;
      seconds ??= interval;
    }
    templates.push(new Template(folder, sample.display ?? folder, sample.id, sample.keypoints, null, sampleAspect, seconds));
  }
  if (!templates.length) return { sources: [], distances: new Float64Array(0) };
  const sequence = frames(keypoints);
  const times = frameTimes(sequence.length, null, frameSeconds || DEFAULT_FRAME_SECONDS);
  const distances = new Recognizer(templates, prepared).templateDistances(sequence, {
    aspect: aspect || DEFAULT_ASPECT, times, sameRecording: true, stored: true,
  });
  return { sources: templates.map((t) => t.source), distances };
}

// collectors.near_duplicate: the id of a stored sample of sign `label` that is the same recording as `keypoints`
// (best distance below `threshold`), or null.
export function nearDuplicateIn(recognizer, label, keypoints, { aspect = null, frameSeconds = null, outside = [],
  threshold = NEAR_DUPLICATE_DISTANCE } = {}) {
  const { sources, distances } = nearDuplicateDistances(recognizer, label, keypoints, { aspect, frameSeconds, outside });
  if (!sources.length) return null;
  let best = 0; // np.argmin: the first minimum, or the first NaN
  for (let i = 0; i < distances.length; i++) {
    if (Number.isNaN(distances[i])) {
      best = i;
      break;
    }
    if (distances[i] < distances[best]) best = i;
  }
  return distances[best] < threshold ? sources[best] : null;
}

// ------------------------------------------------------------------ what was collected / deleted

// dataset._source_hash: sha256(media id)[:20], lower-case hex (tombstone keys never reveal URLs or file names).
export async function sourceHash(mediaId) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(String(mediaId)));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 20);
}

// dataset.provenance_keys: tombstone keys of where a sample came from ("mobile:<id>", "seg:<h>@<start_ms>-<end_ms>",
// "src:<h>" for whole one-sign videos imported automatically: label_from dictionary / folder).
export async function provenanceKeys(metadata) {
  const keys = [];
  const mobileId = pythonStr(metadata.mobile_id || "").trim().toLowerCase();
  if (mobileId && /^[0-9a-f]+$/.test(mobileId)) keys.push(`mobile:${mobileId}`);
  const mediaId = pythonStr(metadata.media_id || "").trim();
  if (!mediaId) return keys;
  const source = await sourceHash(mediaId);
  const start = pythonFloat(metadata.start), end = pythonFloat(metadata.end);
  if (Number.isFinite(start) && Number.isFinite(end)) {
    keys.push(`seg:${source}@${roundHalfEven(start * 1000)}-${roundHalfEven(end * 1000)}`);
  }
  if (metadata.label_from === "dictionary" || metadata.label_from === "folder") keys.push(`src:${source}`);
  return keys;
}

// dataset.deleted_segments over tombstone keys (array, Set or object): [[start, end]] seconds of deleted samples
// collected from the video whose sourceHash is `hash`, sorted.
export function deletedSegmentsIn(keys, hash) {
  const prefix = `seg:${hash}@`;
  const list = keys instanceof Set || Array.isArray(keys) ? [...keys] : Object.keys(keys ?? {});
  const segments = [];
  for (const key of list) {
    if (!key.startsWith(prefix)) continue;
    const text = key.slice(prefix.length), dash = text.indexOf("-");
    if (dash < 0) continue;
    const start = pythonInt(text.slice(0, dash)), end = pythonInt(text.slice(dash + 1));
    if (start === undefined || end === undefined) continue;
    segments.push([start / 1000, end / 1000]);
  }
  return segments.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

// dataset.source_deleted over tombstone keys: a whole-video import of this source was deleted.
export function sourceDeletedIn(keys, hash) {
  const key = `src:${hash}`;
  if (keys instanceof Set) return keys.has(key);
  if (Array.isArray(keys)) return keys.includes(key);
  return Object.hasOwn(keys ?? {}, key);
}

// collectors.deleted_before: the user deleted a sample of (most of) frames [first, last] of this video;
// segments: deletedSegmentsIn's [[start, end]] seconds.
export function deletedBeforeIn(segments, first, last, fps) {
  return segments.some(([start, end]) => overlaps(first, last, roundHalfEven(start * fps), roundHalfEven(end * fps)));
}

// collectors.saved_segments over the data store's samples (store.samples(): built-in ones carry `media` [media id,
// first frame, last frame], phone ones their metadata): [[first, last, display, id]] of this video, sorted by frames.
export function savedSegmentsIn(samples, mediaId) {
  const out = [];
  for (const sample of samples) {
    let first, last;
    if (sample.media) {
      if (sample.media[0] !== mediaId) continue;
      [, first, last] = sample.media;
    } else {
      const metadata = sample.metadata ?? {};
      if (!metadata.media_id || metadata.media_id !== mediaId) continue;
      first = pythonInt(metadata.start_frame);
      last = pythonInt(metadata.end_frame);
      if (first === undefined || last === undefined) continue;
    }
    out.push([first, last, sample.display || sample.label, sample.id]);
  }
  return out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}
