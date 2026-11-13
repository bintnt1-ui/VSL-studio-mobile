// Collecting signs from a video on the phone (the file part of the PC's multi-source collector: collectors.py
// scan_video / compute_segment / sample_metadata, video_sources.file_item). A video from the phone's gallery is
// scanned with Holistic (hand activity -> proposals, segmentation.js); the user names each proposal and saves it:
// FRAME_COUNT evenly spaced frames of the segment through a fresh tracker, whole-frame keypoints, the annotated
// middle frame as its picture. No captions are read on the phone (no OCR): every proposal is named by hand.

import { createHolistic, FACE_TESSELATION } from "./holistic.js";
import { drawDense } from "./draw.js";
import * as segmentation from "./segmentation.js";

export const FRAME_COUNT = 30;
export const PROCESS_WIDTH = 1280; // wider frames are downscaled before Holistic; keypoints are normalised anyway
export const STORED_WIDTH = 640; // the annotated picture is kept at the width dataset stores it
export const SHORT_CLIP_SECONDS = 8.0; // dictionary-style clips are scanned frame by frame
export const COLLECTOR_VERSION = "2";
export const THUMB_HEIGHT = 96; // filmstrip thumbnails (collectors.THUMB_HEIGHT)
export const PLATFORM = "Tệp trên máy";
const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120];

export class Cancelled extends Error {}

// video_sources.file_fingerprint: sha1(size + first 4 MB [+ last MB when over 5 MB])[:16] — the same media id as
// the PC gives the same file, so both know it is one video.
export async function fileFingerprint(file) {
  const parts = [new TextEncoder().encode(String(file.size)), new Uint8Array(await file.slice(0, 4 * 1024 * 1024).arrayBuffer())];
  if (file.size > 5 * 1024 * 1024) parts.push(new Uint8Array(await file.slice(file.size - 1024 * 1024).arrayBuffer()));
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const buffer = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    buffer.set(part, offset);
    offset += part.length;
  }
  const digest = await crypto.subtle.digest("SHA-1", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

// Python round(): half to even (np.round of linspace).
function roundHalfEven(value) {
  const floor = Math.floor(value), diff = value - floor;
  if (Math.abs(diff - 0.5) > 1e-9) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}

export function linspaceIndices(first, last, count = FRAME_COUNT) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(roundHalfEven(count === 1 ? first : first + ((last - first) * i) / (count - 1)));
  return out;
}

function makeCanvas(width, height) {
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(width, height) : document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function toBlob(canvas, type = "image/jpeg", quality = 0.85) {
  return canvas.convertToBlob ? canvas.convertToBlob({ type, quality }) : new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

// One video file: frame-accurate access by seeking (frame i is shown at (i + 0.5) / fps).
// <video> -> its latest queued seek (VideoClip.frame).
const SEEKS = new WeakMap();

export class VideoClip {
  constructor(file, video) {
    this.file = file;
    this.video = video;
    this.url = null;
    this.fps = 30;
    this.frameCount = 0;
    this.width = 0;
    this.height = 0;
    this.mediaId = null;
    this.canvas = null;
  }

  get title() {
    return this.file.name.replace(/\.[^.]+$/, "");
  }

  async open() {
    this.close();
    const video = this.video;
    await (SEEKS.get(video) ?? Promise.resolve()).catch(() => {}); // a closed clip's queued seeks end first
    this.closed = false;
    this.url = URL.createObjectURL(this.file);
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = this.url;
    await new Promise((resolve, reject) => {
      const ok = () => { cleanup(); resolve(); };
      const fail = () => { cleanup(); reject(new Error("Không mở được video này (định dạng không hỗ trợ trên máy?).")); };
      const cleanup = () => { video.removeEventListener("loadeddata", ok); video.removeEventListener("error", fail); };
      video.addEventListener("loadeddata", ok);
      video.addEventListener("error", fail);
    });
    if (!video.videoWidth || !Number.isFinite(video.duration)) throw new Error("Video không có hình hoặc không rõ độ dài.");
    this.mediaId = `file:${await fileFingerprint(this.file)}`;
    this.fps = await this._measureFps();
    const scale = Math.min(1, PROCESS_WIDTH / video.videoWidth);
    this.width = video.videoWidth;
    this.height = video.videoHeight;
    this.processWidth = Math.round(video.videoWidth * scale);
    this.processHeight = Math.max(1, Math.round(video.videoHeight * scale));
    this.frameCount = Math.max(1, Math.round(video.duration * this.fps)); // the container's frame count, as OpenCV reports it
    this.canvas = makeCanvas(this.processWidth, this.processHeight);
    return this;
  }

  // Browsers do not tell a video's frame rate: measured from the frames presented while it plays a moment.
  async _measureFps() {
    const video = this.video;
    if (!video.requestVideoFrameCallback) return 30;
    const times = [];
    const done = new Promise((resolve) => {
      const step = (now, meta) => {
        times.push(meta.mediaTime);
        if (times.length < 14 && !video.ended) video.requestVideoFrameCallback(step);
        else resolve();
      };
      video.requestVideoFrameCallback(step);
      setTimeout(resolve, 2500);
    });
    video.playbackRate = 1;
    await video.play().catch(() => {});
    await done;
    video.pause();
    const deltas = [];
    for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) deltas.push(times[i] - times[i - 1]);
    if (deltas.length < 3) return 30;
    deltas.sort((a, b) => a - b);
    const measured = 1 / deltas[deltas.length >> 1];
    const near = COMMON_FPS.find((fps) => Math.abs(fps - measured) / fps < 0.03);
    return near ?? Math.round(measured * 100) / 100;
  }

  close() {
    this.closed = true; // its frames still queued resolve without touching the video (the next clip's by then)
    this.cancelSeek?.();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
    this.video.removeAttribute("src");
    this.video.load();
  }

  // The pixels of `region` ([x, y, w, h] in video pixels) of the frame shown now, at the video's own resolution (the
  // text region is read from full-size frames, like the PC's OCR).
  regionImage(region) {
    const x0 = Math.max(0, Math.round(region[0])), y0 = Math.max(0, Math.round(region[1]));
    const x1 = Math.min(this.width, Math.round(region[0] + region[2])), y1 = Math.min(this.height, Math.round(region[1] + region[3]));
    const [x, y, w, h] = [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)];
    this.regionCanvas ??= makeCanvas(Math.max(1, w), Math.max(1, h));
    if (this.regionCanvas.width !== w || this.regionCanvas.height !== h) {
      this.regionCanvas.width = Math.max(1, w);
      this.regionCanvas.height = Math.max(1, h);
    }
    const context = this.regionCanvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(this.video, x, y, w, h, 0, 0, w, h);
    return context.getImageData(0, 0, Math.max(1, w), Math.max(1, h));
  }

  // collectors.region_box for this clip's processing canvas: [x, y, w, h] video pixels -> [x0, y0, x1, y1] or null.
  crop(region) {
    return regionBox(region, this.processWidth / this.width, this.processWidth, this.processHeight);
  }

  // Seek to frame `index` and draw it (downscaled to PROCESS_WIDTH) into this.canvas. Every seek of a <video> runs
  // after the previous one (whichever clip asked: one element serves every clip opened in it), so a seek of the
  // timeline never lands in the middle of a scan's or a segment's frame.
  frame(index) {
    const run = (SEEKS.get(this.video) ?? Promise.resolve()).catch(() => {}).then(() => this._frame(index));
    SEEKS.set(this.video, run);
    return run;
  }

  // The frame the stage should show: while a seek runs, only the latest index asked for is sought next (dragging the
  // timeline never piles up seeks). Resolves once the video shows the latest one.
  show(index) {
    this.wanted = index;
    this.showing ??= (async () => {
      try {
        let shown;
        do {
          shown = this.wanted;
          await this.frame(shown);
        } while (shown !== this.wanted && !this.closed);
      } catch {
        // closed meanwhile: nothing to show
      } finally {
        this.showing = null;
      }
    })();
    return this.showing;
  }

  async _frame(index) {
    const video = this.video;
    // A closed clip has no frames: its work (30 frames, a scan) stops instead of taking the last frame drawn again.
    if (this.closed) throw new Cancelled("Video đã đóng.");
    const target = Math.min(video.duration - 1e-3, (index + 0.5) / this.fps);
    await new Promise((resolve) => {
      let finished = false;
      const finish = () => {
        if (finished) return;
        finished = true;
        video.removeEventListener("seeked", seeked);
        this.cancelSeek = null;
        resolve();
      };
      const seeked = () => finish(); // the frame at the new position is ready to draw
      video.addEventListener("seeked", seeked);
      this.cancelSeek = finish; // close() ends the wait at once (the element is emptied: no "seeked" comes)
      setTimeout(finish, 3000);
      video.currentTime = target;
    });
    if (this.closed) throw new Cancelled("Video đã đóng.");
    const context = this.canvas.getContext("2d");
    context.drawImage(video, 0, 0, this.processWidth, this.processHeight);
    return this.canvas;
  }
}

// collectors.region_box: (x, y, w, h) in video pixels -> clamped [x0, y0, x1, y1] in a frame scaled by `scale`, or
// null when there is no region or it is smaller than 16 pixels.
export function regionBox(region, scale, width, height) {
  if (!region) return null;
  const [x, y, w, h] = region;
  const clip = (value, low, high) => Math.min(Math.max(value, low), high);
  const x0 = Math.trunc(clip(roundHalfEven(x * scale), 0, width - 1));
  const y0 = Math.trunc(clip(roundHalfEven(y * scale), 0, height - 1));
  const x1 = Math.trunc(clip(roundHalfEven((x + w) * scale), x0 + 1, width));
  const y1 = Math.trunc(clip(roundHalfEven((y + h) * scale), y0 + 1, height));
  if (x1 - x0 < 16 || y1 - y0 < 16) return null;
  return [x0, y0, x1, y1];
}

// collectors.read_caption: sign-name candidates read in the text region (an ImageData of the region itself) and the raw
// text; `thorough` also tries the contrast variants until one matches a known sign name and combines their readings.
export async function readCaption(ocr, image, vocabulary = [], thorough = false) {
  const { ocrCrop, ocrVariants } = await import("./ocr.js");
  // The phone's reader already tries dark and light text, so the inverted variant would only count the grey
  // reading twice in the vote: grey and contrast are read.
  const images = thorough ? ocrVariants(image, null).slice(0, 2) : [ocrCrop(image, null)];
  let boxes = null;
  if (images.length > 1 && typeof ocr.detect === "function") {
    try { // locate the text once on the grey image; the variants only change how it is read
      boxes = await ocr.detect(images[0]);
    } catch {
      boxes = null;
    }
    if (boxes && Array.isArray(boxes) && !boxes.length) return [[], ""];
  }
  const readings = [];
  for (const picture of images) {
    const [candidates, raw, known] = await readOnce(ocr, picture, vocabulary, boxes);
    if (known || (candidates.length && candidates[0][1] >= segmentation.CONFIDENT_READING)) return [candidates, raw];
    readings.push([candidates, raw]);
  }
  if (readings.length === 1) return readings[0];
  const [combined] = segmentation.voteCaptions(readings);
  return [combined, readings[0][1] || (readings.find(([, raw]) => raw)?.[1] ?? "")];
}

// collectors._read_once: detections -> candidates (a tolerant vocabulary match first), raw text, whether it is known.
async function readOnce(ocr, image, vocabulary, boxes) {
  const detections = boxes && boxes.length ? await ocr.read(image, { boxes }) : await ocr.read(image);
  let candidates = segmentation.captionCandidates(detections, vocabulary);
  const [known, , confidence] = segmentation.matchOcrLabel(detections, vocabulary);
  if (known) candidates = [[known, confidence], ...candidates.filter(([text]) => text !== known)];
  const raw = segmentation.captionLines(detections).map(([text]) => text).join(" / ");
  return [candidates, raw, known != null];
}

// collectors.scan_video: one pass over the video — hand activity (Holistic over SCAN_FPS frames a second, every frame of
// a short clip, inside the signer region when one is marked) and, with a text region and an OCR engine, the caption
// (read when the region changes and has settled, at most every MAX_OCR_GAP_SECONDS). With `label` the video holds one
// sign: the proposal is the span from the first to the last activity. progress(fraction, text); cancelled() stops it.
export async function scanVideo(clip, { progress = null, cancelled = null, region = null, textRegion = null, ocr = null,
  vocabulary = [], label = null } = {}) {
  const fps = clip.fps, total = clip.frameCount;
  const short = total / fps <= SHORT_CLIP_SECONDS;
  const step = short ? 1 : Math.max(1, roundHalfEven(fps / segmentation.SCAN_FPS));
  const useOcr = Boolean(ocr && textRegion && !label);
  const ocrStep = useOcr ? Math.max(1, roundHalfEven(fps / segmentation.OCR_FPS)) : 0;
  const ocrGap = useOcr ? roundHalfEven(segmentation.MAX_OCR_GAP_SECONDS * fps) : 0;
  const crop = clip.crop(region);
  const holistic = await createHolistic();
  const rows = [], frames = [], readings = [];
  let lastSignature = null, previousSignature = null, lastReading = ["", 0.0, ""], lastOcr = -1e9, changeFrom = null, ocrRuns = 0;
  try {
    for (let index = 0; index < total; index++) {
      const needPose = index % step === 0;
      const needText = useOcr && index % ocrStep === 0;
      if (!needPose && !needText) continue;
      if (cancelled?.()) throw new Cancelled("Đã dừng quét.");
      const canvas = await clip.frame(index);
      if (needPose) {
        const result = holistic.run(canvas, canvas.width, canvas.height, index / fps, crop);
        rows.push(result.keypoints);
        frames.push(index);
      }
      if (needText) {
        const image = clip.regionImage(textRegion);
        const signature = segmentation.regionSignature(image, [0, 0, image.width, image.height]);
        const changed = segmentation.differs(signature, lastSignature, segmentation.REGION_CHANGE);
        const settled = !segmentation.differs(signature, previousSignature, segmentation.REGION_STILL);
        if (changed && changeFrom === null) changeFrom = readings.length;
        if (changed && (settled || index - lastOcr >= ocrGap)) {
          const [candidates, raw] = await readCaption(ocr, image, vocabulary, true);
          lastReading = candidates.length ? [candidates[0][0], candidates[0][1], raw] : ["", 0.0, raw];
          // The caption on screen changed when the region did, not when it settled.
          for (let position = changeFrom; position < readings.length; position++) readings[position] = [readings[position][0], ...lastReading];
          lastSignature = signature;
          lastOcr = index;
          changeFrom = null;
          ocrRuns++;
        } else if (!changed) {
          changeFrom = null; // back to the caption already read (a hand passed by)
        }
        previousSignature = signature;
        readings.push([index, ...lastReading]);
      }
      if (progress && index % Math.max(step * 5, 15) === 0) {
        progress(index / Math.max(total, 1), `Đang quét video… ${(index / fps).toFixed(1)}/${(total / fps).toFixed(1)} s`
          + (useOcr ? ` · đọc chữ ${ocrRuns} lần` : "") + (useOcr && lastReading[0] ? ` · chữ: ${lastReading[0]}` : ""));
      }
    }
  } finally {
    holistic.close();
  }
  const signals = segmentation.activity(rows);
  const spans = segmentation.activeSpans(frames, signals.active, fps, total);
  let captions = [], proposals;
  if (label) {
    const span = segmentation.singleSignSpan(frames, signals.active, fps, total);
    proposals = span ? [new segmentation.Proposal(span[0], span[1], label, "", 0.0, "video một ký hiệu")] : [];
  } else {
    if (useOcr) captions = segmentation.captionIntervals(readings.map(([frame, name, confidence]) => [frame, name, confidence]), total);
    proposals = segmentation.propose(total, fps, spans, useOcr ? captions : null);
  }
  progress?.(1, `Quét xong: ${proposals.length} đề xuất.`);
  return { total, fps, frames, signals, spans, readings, captions, proposals,
    // the activity curve the timeline draws (ScanResult.activity_curve): 1 where a hand signs, per video frame
    activityCurve: activityCurve(frames, signals.active, total, step) };
}

function activityCurve(frames, active, total, step) {
  const curve = new Float32Array(total);
  frames.forEach((frame, i) => {
    if (!active[i]) return;
    for (let f = frame; f < Math.min(total, frame + step); f++) curve[f] = 1;
  });
  return curve;
}

// collectors.compute_segment: a fresh tracker over FRAME_COUNT evenly spaced frames of [first, last] (in order), inside
// the signer region when one is marked; whole-frame keypoints, the annotated middle frame (STORED_WIDTH wide, dense
// landmarks) as a JPEG, and with `thumbnails` the 30 annotated frames THUMB_HEIGHT tall (the filmstrip of exactly what
// is stored).
export async function computeSegment(clip, first, last, { progress = null, cancelled = null, region = null, thumbnails = false } = {}) {
  if (last - first + 1 < segmentation.MIN_SEGMENT_FRAMES) {
    throw new Error(`Đoạn quá ngắn; cần ít nhất ${segmentation.MIN_SEGMENT_FRAMES} frame.`);
  }
  const indices = linspaceIndices(first, last);
  const crop = clip.crop(region);
  const holistic = await createHolistic();
  const keypoints = [], thumbs = [];
  let preview = null;
  try {
    for (let count = 0; count < indices.length; count++) {
      if (cancelled?.()) throw new Cancelled("Đã hủy.");
      const canvas = await clip.frame(indices[count]);
      const result = holistic.run(canvas, canvas.width, canvas.height, indices[count] / clip.fps, crop);
      keypoints.push(new Float32Array(result.keypoints));
      const annotate = (width, height) => {
        const picture = makeCanvas(width, height);
        const context = picture.getContext("2d");
        context.drawImage(canvas, 0, 0, width, height);
        const scale = width / canvas.width;
        const place = crop ? crop.map((value) => value * scale) : null;
        drawDense(context, result.drawing, width, height, place, FACE_TESSELATION);
        return picture;
      };
      if (thumbnails) { // drawn now, from this frame (the clip's canvas is the next frame's after an await)
        const height = THUMB_HEIGHT;
        thumbs.push(annotate(Math.max(1, Math.round((canvas.width * height) / canvas.height)), height));
      }
      if (count === indices.length >> 1) {
        const width = Math.min(STORED_WIDTH, canvas.width);
        preview = await toBlob(annotate(width, Math.max(1, Math.round((canvas.height * width) / canvas.width))));
      }
      progress?.((count + 1) / indices.length);
    }
  } finally {
    holistic.close();
  }
  return { first, last, indices, keypoints, preview, thumbnails: thumbs, region };
}

// collectors.sample_metadata for a phone video file (never a local path: the file name only).
export function sampleMetadata(clip, segment, coverage, labelFrom = "manual") {
  const fps = clip.fps;
  return {
    source: "video", media_id: clip.mediaId, origin: clip.file.name, platform: PLATFORM, title: clip.title,
    start: (segment.first / fps).toFixed(3), end: (segment.last / fps).toFixed(3),
    start_frame: String(segment.first), end_frame: String(segment.last), fps: fps.toFixed(3),
    video_size: `${clip.width}x${clip.height}`, label_from: labelFrom, hand_coverage: coverage.toFixed(2),
    orientation: "original", collector: COLLECTOR_VERSION,
  };
}

// Play [first, last] of the clip into `canvas` (the stage) once; resolves when done or stopped.
export async function playSegment(clip, canvas, first, last, stopped = () => false) {
  const video = clip.video;
  const context = canvas.getContext("2d");
  canvas.width = clip.width;
  canvas.height = clip.height;
  await clip.frame(first);
  const end = (last + 1) / clip.fps;
  await video.play().catch(() => {});
  await new Promise((resolve) => {
    const draw = () => {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      if (stopped() || video.paused || video.ended || video.currentTime >= end) {
        video.pause();
        resolve();
        return;
      }
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(draw);
      else requestAnimationFrame(draw);
    };
    draw();
  });
}

export async function showFrame(clip, canvas, index) {
  await clip.show(index);
  if (clip.closed) return;
  canvas.width = clip.width;
  canvas.height = clip.height;
  canvas.getContext("2d").drawImage(clip.video, 0, 0, canvas.width, canvas.height);
}
