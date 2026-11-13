// Face identity tracking: keep Holistic on the registered signer (port of the tracking part of identity.py:
// Face, identity_score, _iou, crop_around, steady_crop, Region, PersonLock, Enrollment, prepare_region).
//
// While a profile is logged in, PersonLock finds that person among all faces of each frame (YuNet boxes +
// SFace embeddings, faces.js) and says where Holistic should look (Region: crop with other faces masked);
// people.js then checks that the skeleton Holistic found is really theirs. Profiles (passwords, embeddings,
// avatar) are store.js's; biometric data never leaves the phone.
//
// Conventions: a face box is [x, y, width, height] in frame pixels, a crop is [x0, y0, x1, y1]; a frame is any
// object with `width` and `height` (canvas, ImageData, video frame, or {width, height}). The face engine
// (faces.js, ONNX Runtime Web) provides embed(frame, face) -> Promise<Float32Array(128)> unit vector, and for
// enrollment aligned(frame, face) -> crop (or a Promise of it) and optionally crop(frame, [x0, y0, x1, y1]) for
// the avatar. So PersonLock.locate and Enrollment.offer are async; one call at a time per object (the camera
// loop awaits each frame). Everything else is synchronous and exactly identity.py's logic.

export const DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx";
export const RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx";

export const MATCH_THRESHOLD = 0.363; // SFace cosine similarity for "same person" (OpenCV Zoo)
// A partly covered face exactly where the verified person just was; different people score below ~0.15
// against an enrollment (measured), the same person with a hand over part of the face drops towards ~0.3.
export const CONTINUITY_THRESHOLD = 0.25;
export const DETECT_WIDTH = 320; // faces are detected on a downscaled copy
export const MIN_DETECTION_SCORE = 0.8;
export const ENROLL_FRAMES = 120;
export const ENROLL_BINS = Object.freeze(["left", "center", "right"]);
export const BIN_NAMES = Object.freeze({ left: "Trái", center: "Giữa", right: "Phải" });
export const FRAMES_PER_BIN = Math.floor(ENROLL_FRAMES / ENROLL_BINS.length);
export const YAW_CENTER = 0.12; // |nose offset| / eye distance below this counts as looking straight
export const YAW_LIMIT = 0.65; // beyond this the face is too sideways to be useful
export const MIN_FACE_FRACTION = 0.10; // face width / frame width while enrolling
export const MIN_ENROLL_SCORE = 0.9;
export const VERIFY_EVERY = 5; // frames between identity checks of the tracked face...
export const CROWDED_VERIFY_EVERY = 2; // ... while other people were seen within CROWD_SECONDS
export const CROWD_SECONDS = 2.0;
export const FAST_IOU = 0.8; // a face followed by position must overlap its previous box this much
export const STRANGER_SCORE = 0.15; // an unrecognised face looking at the camera scoring below this is surely someone else...
export const COVER_MARGIN = 0.2; // ... unless the person's own hand is on it (within this many face-box sizes)
export const STRANGE_CHECKS = 3; // an unrecognised face at the person's place this many checks in a row: let go
export const CROP_LINGER_SECONDS = 1.0; // Holistic keeps its crop this long after the other faces are gone
export const HIDDEN_HOLD_SECONDS = 1.5; // keep the last position while the face is covered by hands
export const ALONE_HOLD_SECONDS = 6.0; // same, when no other face is visible (the body must stay put: people.js)
// Profile rules (store.js keeps the accounts).
export const PASSWORD_ITERATIONS = 200_000;
export const MIN_PASSWORD_LENGTH = 4;
export const MAX_NAME_LENGTH = 40;

const f32 = Math.fround;

// One YuNet detection: box [x, y, w, h] (visible part, frame pixels), row (Float32Array(15): box, right eye,
// left eye, nose, mouth corners, score; frame pixels), score.
export class Face {
  constructor(box, row, score) {
    this.box = box;
    this.row = row instanceof Float32Array ? row : Float32Array.from(row);
    this.score = score;
  }

  get width() {
    return this.box[2];
  }

  // Nose offset from the eyes' midpoint, in eye distances (negative: facing image-left). float32 like numpy.
  yaw() {
    const r = this.row;
    const dx = f32(r[4] - r[6]), dy = f32(r[5] - r[7]);
    const eyeDistance = Math.max(f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy)))), 1.0);
    return (r[8] - f32(r[4] + r[6]) / 2) / eyeDistance;
  }
}

// N x 128 rows (an array of 128-vectors, or one flat array) as one flat Float32Array (numpy float32).
function flatRows(rows) {
  if (rows instanceof Float32Array) return rows;
  if (ArrayBuffer.isView(rows) || (rows.length && typeof rows[0] === "number")) return Float32Array.from(rows);
  const flat = new Float32Array(rows.length * 128);
  rows.forEach((row, index) => flat.set(row, index * 128));
  return flat;
}

// Cosine similarity of a face to a person: mean of the 5 closest enrolled views. `enrolled`: N x 128 (flat
// Float32Array or an array of rows, unit length); `feature`: Float32Array(128). float32 like numpy (dot products
// and the mean are rounded to float32; BLAS sums in another order, so the last bit may differ). NaN when nothing
// is enrolled or a product is NaN.
export function identityScore(enrolled, feature) {
  enrolled = flatRows(enrolled);
  const count = Math.floor(enrolled.length / 128);
  const keep = Math.min(5, count);
  if (!keep) return NaN; // numpy: mean of nothing
  const best = new Float64Array(keep).fill(-Infinity); // the `keep` largest, descending
  for (let row = 0, base = 0; row < count; row++, base += 128) {
    let dot = 0;
    for (let i = 0; i < 128; i++) dot += enrolled[base + i] * feature[i];
    dot = f32(dot);
    if (Number.isNaN(dot)) return NaN; // np.sort puts NaN first once reversed
    if (dot <= best[keep - 1]) continue;
    let j = keep - 1;
    while (j > 0 && best[j - 1] < dot) { best[j] = best[j - 1]; j--; }
    best[j] = dot;
  }
  let sum = 0;
  for (let i = 0; i < keep; i++) sum = f32(sum + best[i]);
  return f32(sum / keep);
}

// Overlap / union of two [x, y, w, h] boxes (0 when both are empty).
export function _iou(first, second) {
  const [ax, ay, aw, ah] = first;
  const [bx, by, bw, bh] = second;
  const width = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx));
  const height = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
  const overlap = width * height;
  const union = aw * ah + bw * bh - overlap;
  return union ? overlap / union : 0.0;
}

// Body-sized window [x0, y0, x1, y1] around a face: wide and tall enough for signing hands.
export function cropAround(box, width, height) {
  const [x, y, w, h] = box;
  const center = x + w / 2;
  const half = Math.max(w * 3.25, width * 0.2);
  return [
    Math.trunc(Math.max(0, center - half)),
    Math.trunc(Math.max(0, y - 1.6 * h)),
    Math.trunc(Math.min(width, center + half)),
    height,
  ];
}

// Keep the previous crop while the face stays well inside it (no per-frame jitter). A side already at the
// frame border needs no margin: a face cut by the border is clipped to it and could never be inside.
export function steadyCrop(previous, box, width, height) {
  const [x, y, w, h] = box;
  const wanted = cropAround(box, width, height);
  if (previous != null) {
    const [x0, y0, x1, y1] = previous;
    // A face that grew or shrank (leaning towards or away from the camera) needs a window of its new size.
    const sameSize = Math.abs((x1 - x0) - (wanted[2] - wanted[0])) <= 0.15 * (wanted[2] - wanted[0]);
    if (sameSize && (x0 <= 0 || x - x0 >= 0.5 * w) && (x1 >= width || x1 - (x + w) >= 0.5 * w)
        && (y0 <= 0 || y - y0 >= 0.5 * h) && x0 <= x && x + w <= x1 && y0 <= y && y1 === height) {
      return previous;
    }
  }
  return wanted;
}

// Where Holistic should look in this frame.
// status: "locked" (the verified face), "hidden" (where it was a moment ago: covered or turned away),
// "lost" (only other people's faces), "searching" (no face seen and the person not held).
// crop: [x0, y0, x1, y1] or null (whole frame); masks: other faces to blank ([x, y, w, h]); target: the
// person's face box (or where it was); others: every other face box; verified: the target was recognised
// by SFace this frame (not followed by position).
export class Region {
  constructor(status, crop, masks, target, others, score, verified = false) {
    this.status = status;
    this.crop = crop;
    this.masks = masks;
    this.target = target;
    this.others = others;
    this.score = score;
    this.verified = verified;
  }

  // Nobody here is known to be the registered person: take no landmarks at all.
  get skip() {
    return this.status === "lost" || this.status === "searching";
  }
}

// numpy's float32 add.reduce of values[start:start + n] (pairwise summation, 8 accumulators per block of 128).
function pairwiseSum32(values, start, n) {
  if (n < 8) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum = f32(sum + values[start + i]);
    return sum;
  }
  if (n <= 128) {
    const r = values.slice(start, start + 8); // Float32Array: every += rounds to float32
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += values[start + i + j];
    let sum = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3])) + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])));
    for (; i < n; i++) sum = f32(sum + values[start + i]);
    return sum;
  }
  let half = Math.floor(n / 2);
  half -= half % 8;
  return f32(pairwiseSum32(values, start, half) + pairwiseSum32(values, start + half, n - half));
}

// Rows of an N x 128 enrollment, each divided by max(norm, 1e-6), as one Float32Array: bit-exact with numpy's
// float32 `enrolled / np.maximum(np.linalg.norm(enrolled, axis=1, keepdims=True), 1e-6)`.
function unitRows(embeddings) {
  const flat = flatRows(embeddings);
  const out = new Float32Array(flat.length);
  const squares = new Float32Array(128);
  const floor = f32(1e-6);
  for (let base = 0; base + 128 <= flat.length; base += 128) {
    for (let i = 0; i < 128; i++) squares[i] = flat[base + i] * flat[base + i];
    const norm = Math.max(f32(Math.sqrt(pairwiseSum32(squares, 0, 128))), floor);
    for (let i = 0; i < 128; i++) out[base + i] = flat[base + i] / norm;
  }
  return out;
}

const defaultClock = () => performance.now() / 1000;

// Follows one enrolled person among all faces in the frame.
// Rules (each guards against a stranger taking over the landmarks):
// * a face is followed by position only from a frame where it was the verified person, with the same faces
//   present, nearly in the same place (FAST_IOU), and at most VERIFY_EVERY frames in a row
//   (CROWDED_VERIFY_EVERY while other people were seen within CROWD_SECONDS); a face that reappears is
//   verified again;
// * a face where the person just was (occlusion, head turned) needs CONTINUITY_THRESHOLD; any other face
//   needs the full MATCH_THRESHOLD;
// * an unrecognised face at the person's position (while they are held) is their own covered, turned or
//   lowered face ("hidden": people.js checks the body) unless surely someone else's (looking at the camera,
//   scoring below STRANGER_SCORE and no hand of the person on it: `coverPoints`, set by people.js from the
//   latest accepted frame); then the frame is skipped, nobody is followed by position until the face is
//   verified again (`strange`), and with someone else in view or seen within CROWD_SECONDS, or for
//   STRANGE_CHECKS checks in a row, the person is released: they must be verified again;
// * while no face is seen at the person's position, their last position is kept for HIDDEN_HOLD_SECONDS
//   (ALONE_HOLD_SECONDS when no face is visible at all) and never refreshed by an unverified face;
//   people.js also requires the body to stay put meanwhile and calls release() when it does not;
// * nobody is followed before the face has been verified, even when no face is visible ("searching");
// * Holistic gets a crop only while other people are visible (or were within CROP_LINGER_SECONDS), and the
//   crop is kept steady, so a signer alone always uses the whole frame.
// Fields as in Python, camelCase: trackBox, lastSeen, tracking, sinceCheck, faceCount, lastScore, crop,
// othersSeen, coverPoints (Float64Array [x0, y0, x1, y1, ...] frame pixels, or null), scores (Map face ->
// score of every face checked this frame), verified, strange.
export class PersonLock {
  constructor(engine, embeddings, threshold = MATCH_THRESHOLD, clock = defaultClock) {
    this.embeddings = unitRows(embeddings);
    this.engine = engine;
    this.threshold = threshold;
    this.clock = clock;
    this.trackBox = null; // where the verified person was last seen
    this.lastSeen = -Infinity;
    this.tracking = false; // the previous frame's target was the verified person
    this.sinceCheck = 0;
    this.faceCount = -1;
    this.lastScore = 0.0;
    this.crop = null;
    this.othersSeen = -Infinity; // when a face other than the person's was last seen
    // The person's hand points (and wrists whose hand was not found) in the latest accepted frame, a moment
    // ago (people.js), in frame pixels: on a face they explain why it is not recognised.
    this.coverPoints = null;
    this.scores = new Map();
    this.verified = false;
    this.strange = 0; // checks in a row with an unrecognised face at the person's place
  }

  static cropAround(box, width, height) {
    return cropAround(box, width, height);
  }

  // The face at the person's last position, if they were there within `within` seconds.
  _atPlace(faces, now, within) {
    if (this.trackBox == null || now - this.lastSeen > within || !faces.length) return null;
    let nearest = faces[0], best = _iou(faces[0].box, this.trackBox);
    for (let i = 1; i < faces.length; i++) {
      const overlap = _iou(faces[i].box, this.trackBox);
      if (overlap > best) { best = overlap; nearest = faces[i]; }
    }
    return best >= 0.3 ? nearest : null;
  }

  // [the person's face or null, its score]. Sets `verified` (recognised by SFace this frame, not followed by
  // position) and `scores` (face -> score of every face checked).
  async _identify(frame, faces, near, now) {
    this.verified = false;
    this.scores = new Map();
    if (!faces.length) {
      this.tracking = false;
      return [null, 0.0];
    }
    const every = now - this.othersSeen <= CROWD_SECONDS ? CROWDED_VERIFY_EVERY : VERIFY_EVERY;
    if (this.tracking && near != null && faces.length === this.faceCount && this.sinceCheck + 1 < every
        && _iou(near.box, this.trackBox) >= FAST_IOU) {
      this.sinceCheck += 1;
      return [near, this.lastScore];
    }
    this.sinceCheck = 0;
    this.faceCount = faces.length;
    this.verified = true;
    const scored = [];
    if (near != null) {
      // The face where the person just was, first: fully recognised, nobody else needs checking (SFace costs
      // a few ms per face, and the person is checked every few frames).
      const score = identityScore(this.embeddings, await this.engine.embed(frame, near));
      this.scores.set(near, score);
      if (score >= this.threshold) {
        this.tracking = true;
        this.lastScore = score;
        return [near, score];
      }
      if (score >= CONTINUITY_THRESHOLD) scored.push([score, near]);
    }
    const widest = faces.slice().sort((a, b) => b.width - a.width).slice(0, 4); // stable, like sorted()
    for (const face of widest) {
      if (face === near) continue;
      const score = identityScore(this.embeddings, await this.engine.embed(frame, face));
      this.scores.set(face, score);
      if (score >= this.threshold) scored.push([score, face]);
    }
    if (!scored.length) {
      this.tracking = this.verified = false;
      return [null, 0.0];
    }
    let best = scored[0];
    for (const item of scored) if (item[0] > best[0]) best = item; // first of ties, like max()
    this.tracking = true;
    this.lastScore = best[0];
    return [best[1], best[0]];
  }

  // An unrecognised face SFace can judge: looking at the camera (|yaw| within YAW_LIMIT), not covered by the
  // person's own hand (`coverPoints` of the latest accepted frame on it) and scoring like a different person
  // (below STRANGER_SCORE). The person's own face turned, looking down or partly covered scores higher, is too
  // sideways to judge, or has their hand on it.
  async _surelySomeoneElse(frame, face) {
    if (Math.abs(face.yaw()) > YAW_LIMIT || this._covers(face)) return false;
    if (!this.scores.has(face)) { // not among the faces checked this frame: check it now
      this.scores.set(face, identityScore(this.embeddings, await this.engine.embed(frame, face)));
    }
    return this.scores.get(face) < STRANGER_SCORE;
  }

  _covers(face) {
    const points = this.coverPoints;
    if (points == null || !points.length) return false;
    const [x, y, w, h] = face.box;
    const left = x - COVER_MARGIN * w, right = x + w + COVER_MARGIN * w;
    const top = y - COVER_MARGIN * h, bottom = y + h + COVER_MARGIN * h;
    for (let i = 0; i + 1 < points.length; i += 2) {
      if (points[i] >= left && points[i] <= right && points[i + 1] >= top && points[i + 1] <= bottom) return true;
    }
    return false;
  }

  // The person is no longer known to be at the held position: nothing is followed until the face is
  // verified again.
  release() {
    this.trackBox = null;
    this.lastSeen = -Infinity;
    this.tracking = false;
    this.crop = null;
    this.strange = 0;
  }

  // Region for this frame (a Promise). `faces`: the frame's Face detections (null: engine.detect(frame)).
  async locate(frame, faces = null) {
    if (faces == null) faces = await this.engine.detect(frame);
    const now = this.clock();
    const crowded = now - this.othersSeen <= CROWD_SECONDS; // someone else was in view a moment ago
    const atPlace = this._atPlace(faces, now, ALONE_HOLD_SECONDS); // whoever is where the person is held
    // Only a face where the person just was may be recognised with the lower CONTINUITY_THRESHOLD.
    const near = now - this.lastSeen <= HIDDEN_HOLD_SECONDS ? atPlace : null;
    const [target, score] = await this._identify(frame, faces, near, now);
    const verified = target != null && this.verified;
    let box, status, ours;
    if (target != null) {
      this.trackBox = target.box;
      this.lastSeen = now;
      this.strange = 0;
      box = target.box;
      status = "locked";
      ours = target;
    } else if (atPlace != null && !this.strange && !(await this._surelySomeoneElse(frame, atPlace))) {
      // Not recognised, but not surely someone else either: the person's own face covered by their hand,
      // turned or looking down. Followed as hidden: people.js checks that the body is theirs.
      box = this.trackBox;
      status = "hidden";
      ours = atPlace;
    } else {
      const hold = faces.length ? HIDDEN_HOLD_SECONDS : ALONE_HOLD_SECONDS;
      // A face surely someone else's where the person was: someone who sat down or leaned in there. Skip the
      // frame rather than risk their landmarks, and follow nobody by position until the face is verified again
      // (`strange`); with someone else around (the one who leaned in) or for STRANGE_CHECKS checks in a row,
      // let go of the person: verified again from scratch.
      if (atPlace != null || this.strange || this.trackBox == null || now - this.lastSeen > hold) {
        if (atPlace != null) {
          this.strange += 1;
          if (crowded || faces.length > 1 || this.strange >= STRANGE_CHECKS) this.release();
        }
        this.crop = null;
        if (faces.length > 1) this.othersSeen = now; // surely someone else (one face may be the person's own)
        const others = faces.map((face) => face.box);
        return new Region(faces.length ? "lost" : "searching", null, [], null, others, score);
      }
      box = this.trackBox; // face covered by hands or turned away
      status = "hidden";
      ours = null;
    }
    const others = faces.filter((face) => face !== ours).map((face) => face.box);
    if (others.length && (faces.length > 1 || status === "locked" || crowded)) {
      // A single unrecognised face away from the held place may be the person's own (looking down): it is
      // masked and followed as someone else, but does not make the scene crowded.
      this.othersSeen = now;
    }
    if (!others.length && (this.crop == null || now - this.othersSeen > CROP_LINGER_SECONDS)) {
      this.crop = null;
      return new Region(status, null, [], box, [], score, verified);
    }
    const crop = this.crop = steadyCrop(this.crop, box, frame.width, frame.height);
    const masks = others.filter((other) => other[0] < crop[2] && other[0] + other[2] > crop[0]
      && other[1] + other[3] > crop[1]);
    return new Region(status, crop, masks, box, others, score, verified);
  }
}

// What Holistic should see for this region (identity.prepare_region), as a description: holistic.js does the
// pixel work. {frame, crop: [x0, y0, x1, y1] or null (whole frame, masks ignored), masks: the face boxes to
// blank, width / height: the view's size, blanks: the blanked rectangles in view pixels [left, top, right,
// bottom) (each face padded by a quarter of its size), blank: false}. people.Tracker.reset passes
// {blank: true, width, height} (a black image: clears MediaPipe's tracking) and adds `stamp` to both.
export function prepareRegion(frame, region) {
  if (region.crop == null) {
    return { frame, crop: null, masks: [], width: frame.width, height: frame.height, blanks: [], blank: false };
  }
  const [x0, y0, x1, y1] = region.crop;
  const blanks = [];
  for (const [x, y, w, h] of region.masks) {
    const padX = Math.trunc(0.25 * w), padY = Math.trunc(0.25 * h);
    const left = Math.max(0, x - padX - x0), top = Math.max(0, y - padY - y0);
    const right = Math.min(x1, x + w + padX) - x0, bottom = Math.min(y1, y + h + padY) - y0;
    if (right > left && bottom > top) blanks.push([left, top, right, bottom]);
  }
  return { frame, crop: region.crop, masks: region.masks, width: x1 - x0, height: y1 - y0, blanks, blank: false };
}

// Collects ENROLL_FRAMES face views spread over looking left, straight and right. The engine: embed(frame, face,
// aligned?) -> (Promise of) Float32Array(128) unit vector; optional aligned(frame, face) -> the aligned face crop
// (kept in `crops`, null without it); optional crop(frame, [x0, y0, x1, y1]) -> the avatar image. `avatarBox`:
// [x0, y0, x1, y1] of the first straight view (+35% margin, clipped to the frame like the numpy slice); `avatar`:
// engine.crop(frame, avatarBox), else null.
export class Enrollment {
  constructor() {
    this.embeddings = [];
    this.yaws = [];
    this.crops = [];
    this.counts = { left: 0, center: 0, right: 0 };
    this.avatar = null;
    this.avatarBox = null;
  }

  get done() {
    return this.embeddings.length >= ENROLL_FRAMES;
  }

  static binFor(yaw) {
    if (Math.abs(yaw) > YAW_LIMIT) return null;
    if (yaw < -YAW_CENTER) return "left";
    if (yaw > YAW_CENTER) return "right";
    return "center";
  }

  instruction() {
    if (this.done) return "Hoàn tất! Đang lưu tài khoản...";
    if (this.counts.center < FRAMES_PER_BIN) return "Nhìn thẳng vào camera, giữ yên một chút.";
    if (this.counts.left < FRAMES_PER_BIN) return "Từ từ quay mặt sang TRÁI.";
    return "Tiếp tục quay mặt từ trái sang PHẢI.";
  }

  progressText() {
    return ENROLL_BINS.map((name) => `${BIN_NAMES[name]} ${this.counts[name]}/${FRAMES_PER_BIN}`).join("  ·  ");
  }

  // Consider one frame; returns (a Promise of) the instruction to show. `frame.width`: the frame's width.
  async offer(engine, frame, faces) {
    if (this.done) return this.instruction();
    if (!faces.length) return "Không thấy khuôn mặt — hãy nhìn vào camera.";
    if (faces.length > 1) return "Chỉ để một người trong khung hình khi đăng ký.";
    const face = faces[0];
    if (face.width < MIN_FACE_FRACTION * frame.width) return "Hãy ngồi gần camera hơn.";
    if (face.score < MIN_ENROLL_SCORE) return "Khuôn mặt chưa rõ — giữ yên và đủ sáng.";
    const position = Enrollment.binFor(face.yaw());
    if (position === null) return "Quay hơi quá — quay lại một chút.";
    if (this.counts[position] < FRAMES_PER_BIN) {
      // An engine without aligned() aligns inside embed(); the crops are then null.
      const crop = typeof engine.aligned === "function" ? await engine.aligned(frame, face) : null;
      this.embeddings.push(await engine.embed(frame, face, crop ?? undefined));
      this.yaws.push(face.yaw());
      this.crops.push(crop);
      this.counts[position] += 1;
      if (position === "center" && this.avatarBox === null) {
        const [x, y, w, h] = face.box;
        const pad = Math.trunc(0.35 * Math.max(w, h));
        this.avatarBox = [Math.max(0, x - pad), Math.max(0, y - pad),
          Math.min(frame.width, x + w + pad), Math.min(frame.height, y + h + pad)];
        this.avatar = typeof engine.crop === "function" ? await engine.crop(frame, this.avatarBox) : null;
      }
    }
    return this.instruction();
  }

  // The same (pipeline.js calls this name): offer is asynchronous because SFace runs in ONNX Runtime Web.
  offerAsync(engine, frame, faces) {
    return this.offer(engine, frame, faces);
  }
}
