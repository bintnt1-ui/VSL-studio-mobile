// Live sign translation from a timestamped camera stream (port of live_translation.py; pure logic: no camera, no UI).
//
// Webcams deliver 8 to 30 frames per second and the rate changes while signing (light, CPU load), so everything
// here is measured in seconds from the moment each frame was captured, never in frames:
//
//   - the stream keeps the last WINDOW_SECONDS of keypoints;
//   - a sign is a burst of hand activity (translate.signActivity). It ends when the hands have been lowered, out of
//     view or resting low and still for longer than the recogniser's pause (translate.JOIN_SECONDS, at any frame
//     rate); a hand lost while high in the frame and not moving down (motion blur) is waited for up to
//     DROPOUT_SECONDS, unless its arm goes down meanwhile (Holistic still tracks the pose wrist when the hand is
//     blurred) or it is found again low (it went down in between: the frames found start the next sign). A hand
//     lost while the pose model still sees its arm up in the picture (wrist inside the frame, visible, at chest
//     level or higher, forearm not pointing down) has not been put down: the burst waits for it as long as the arm
//     stays up (up to ARM_HOLD_SECONDS), wherever the hand is found again, and is recognised both with the hand
//     missing during that gap and with it interpolated across the gap (the more decisive guess is kept). A resting
//     arm below a head-and-shoulders picture is only extrapolated by the pose model: it is never "in the picture";
//     nor is an arm hanging down with its wrist just inside the bottom of a wide picture. The whole burst is then
//     recognised once and the result is final. A far match needs a clear lead (translate.requiredMargin), unless a
//     provisional guess of the same burst was sure (HOLD_CONFIDENCE) of that sign: then the usual
//     translate.MIN_MARGIN lead is enough;
//   - while signing, a provisional guess is shown every PROVISIONAL_SECONDS;
//   - fingerspelling without lowering the hands: once the wrist has stayed still for HOLD_SECONDS, its first
//     HOLD_SECONDS still are recognised on their own (in capture time, however late the frames arrive), then the
//     last HOLD_SECONDS every PROVISIONAL_SECONDS. When a sure guess names another letter than the previous sure
//     guess, the previous letter is added; the last one is added when the hands come down. A single held
//     handshape, or a still moment inside a moving sign, is decided with the whole burst;
//   - a burst starts at the first signing frame after the previous one ended and never takes its frames, even when
//     later frames change how the pause between them is seen; when the frames of several signs arrive at once,
//     each sign is still decided on its own;
//   - the latest finished sign stays available (frozen) to be saved as a sample: the whole burst, or the last
//     letter when it was spelled.
//
// Frames are kept in one preallocated Float32Array (no per-frame allocation besides the copy of the frame itself);
// update() hands translate.signActivity a {data, count, width} block view of the window and the recognizer an array
// of frame views (valid during the predict call only). Per-frame pairs are flat, as in translate.js:
// activity.arms[i * 2 + side], the arms-up mask _up[i * 2 + side].

import { KEYPOINT_SIZE } from "./layout.js";
import {
  ARM_WRISTS, DEFAULT_ASPECT, LEFT_HAND_SLICE, MAX_DISTANCE, MIN_CONFIDENCE, MIN_MARGIN, POSE_SLICE, Prediction,
  REST_HEIGHT, RIGHT_HAND_SLICE, TIME_TOLERANCE, longEnough, requiredMargin, signActivity,
} from "./translate.js";

export const WINDOW_SECONDS = 6.0;          // longest stretch of signing kept whole
export const MAX_FRAMES = 360;              // safety cap on the buffer (60 fps x WINDOW_SECONDS)
export const PROVISIONAL_SECONDS = 0.25;    // between two guesses while signing
export const DROPOUT_SECONDS = 0.6;         // a hand lost while high (motion blur) is waited for this long (unless found again low)...
export const EXIT_HEIGHT = 0.5;             // ... "high": wrist less than this many shoulder widths below the shoulders
export const LOWERING_SPEED = 0.8;          // ... and not moving down faster than this (shoulder widths per second)
export const ARM_HOLD_SECONDS = 1.5;        // a hand lost while its arm stays up in view is waited for up to this long...
export const ARM_VISIBLE = 0.5;             // ... "in view": the pose wrist inside the frame with at least this visibility...
export const ARM_HOLD_HEIGHT = 0.8;         // ... at most this many shoulder widths below the shoulders (chest level: a
                                            // forearm resting on a desk in the picture is lower, about one shoulder width)...
export const ARM_ELBOWS = Object.freeze([13, 14]); // ... with its forearm (from these pose elbows, as ARM_WRISTS) pointing...
export const ARM_DOWN_SINE = 0.5;           // ... less than 30 degrees below the horizontal: an arm hanging down with its
                                            // hand out of the bottom of the picture is not up, a forearm held level is
export const HOLD_SECONDS = 0.6;            // a wrist still this long: the handshape is recognised on its own...
export const HOLD_MOVE = 0.1;               // ... "still": it moved less than this (shoulder widths) over MOVE_WINDOW_SECONDS
export const HOLD_LEAD_SECONDS = 0.9;       // ... when the stillness starts this soon after the hand arrived or was last still
export const HOLD_CONFIDENCE = 0.9;         // a letter needs a guess at least this sure (and a provisional guess, to count as sure)
export const CONTEXT_SECONDS = 0.15;        // frames kept around a sign when it is recognised or saved
export const TOLERANCE = TIME_TOLERANCE;

const W = KEYPOINT_SIZE;
const F32_1E6 = Math.fround(1e-6);

// ------------------------------------------------------------------ small helpers (Python / numpy semantics)

// Python max(a, b) / min(a, b): the first argument unless the second is strictly larger / smaller.
function pyMax(a, b) {
  return b > a ? b : a;
}

function pyMin(a, b) {
  return b < a ? b : a;
}

// numpy's sort order (NaN last): a < b.
function before(a, b) {
  return a < b || (b !== b && a === a); // eslint-disable-line no-self-compare
}

// np.searchsorted(times, value) (side "left"): the first index whose time is not before `value`.
function searchLeft(times, value) {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (before(times[mid], value)) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// np.searchsorted(times, value, side="right"): the first index whose time is after `value`.
function searchRight(times, value) {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (before(value, times[mid])) hi = mid; else lo = mid + 1;
  }
  return lo;
}

// Frames as a {data, count, width} block: a block is returned as it is, an array of frames is copied into one.
function blockOf(values) {
  if (values && values.data && typeof values.count === "number") return values;
  const count = values.length;
  const width = count ? values[0].length : W;
  const data = new Float32Array(count * width);
  for (let i = 0; i < count; i++) data.set(values[i], i * width);
  return { data, count, width };
}

// Frames [start, stop) of a block as an array of per-frame views (no copy).
function frameViews(block, start, stop) {
  const out = new Array(stop - start);
  for (let i = start; i < stop; i++) out[i - start] = block.data.subarray(i * block.width, (i + 1) * block.width);
  return out;
}

// Frames [start, stop) of a block copied into one buffer: an array of per-frame views of that copy.
function frameCopies(block, start, stop) {
  const copy = block.data.slice(start * block.width, stop * block.width);
  return frameViews({ data: copy, width: block.width }, 0, stop - start);
}

// ------------------------------------------------------------------ results

// What to show and what to append: kind "signing" (provisional guess), "hold" (held handshape), "final" (sign
// ended); prediction (translate.Prediction); start / end: capture times of the first / last frame concerned
// (seconds); append: the word belongs in the sentence (added once); sign: "final": number of this sign's
// SignCapture (latestSign()), 0 when none.
export class LiveEvent {
  constructor(kind, prediction, start, end, append = false, sign = 0) {
    this.kind = kind;
    this.prediction = prediction;
    this.start = start;
    this.end = end;
    this.append = append;
    this.sign = sign;
    Object.freeze(this);
  }
}

// A finished sign as the recogniser saw it, frozen when it ended (for "save the sign"): keypoints (array of
// Float32Array frames, a copy: CONTEXT_SECONDS around the sign), times (Float64Array: their capture times),
// span ([first, last) frames of the sign itself), number (1, 2, ...: a new number for every finished sign).
export class SignCapture {
  constructor(keypoints, times, span, number) {
    this.keypoints = Array.isArray(keypoints) ? Object.freeze(keypoints) : keypoints; // a typed array cannot be frozen
    this.times = times;
    this.span = Object.freeze([span[0], span[1]]);
    this.number = number;
    Object.freeze(this);
  }
}

// A sure guess of a held handshape (one per letter).
export class _Letter {
  constructor(prediction, start, end, committed = false) {
    this.prediction = prediction;
    this.start = start;
    this.end = end;
    this.committed = committed;
  }
}

// One burst of signing: start (first signing frame), last (latest signing frame), provisionalAt, committed (labels
// added to the sentence for this burst), letters (_Letter: sure guesses of held handshapes), sure (labels a
// provisional guess of this burst was sure of), gaps ([last frame before, first frame after] of hands lost, arm
// up), stillFrom (first frame of the current still stretch of the wrist), stillLast (latest still frame of this
// stretch or the previous one), stillIgnored (this stretch began too long after a movement), lookedAt (last time
// this stretch was recognised), finished. `fields` overrides any of them (Python's keyword arguments).
export class _Burst {
  constructor(start, last, fields = {}) {
    this.start = start;
    this.last = last;
    this.provisionalAt = -Infinity;
    this.committed = [];
    this.letters = [];
    this.sure = new Set();
    this.gaps = [];
    this.stillFrom = null;
    this.stillLast = null;
    this.stillIgnored = false;
    this.lookedAt = -Infinity;
    this.finished = false;
    Object.assign(this, fields);
  }
}

// ------------------------------------------------------------------ the translator

// Feed frames with add(), call update() after each batch; events say what to show and what to append.
// `recognizer`: anything with predict(sequence, exclude, live, aspect, times) -> translate.Prediction (a
// translate.Recognizer); `recognizer` and `aspect` (frame width / height) may be changed between calls.
export class LiveTranslator {
  constructor(recognizer, aspect = DEFAULT_ASPECT) {
    this.recognizer = recognizer;
    this.aspect = aspect;
    this._capacity = 2 * MAX_FRAMES + 2;
    this._data = new Float32Array(this._capacity * W); // frames [_head, _tail) are the window
    this._stamps = new Float64Array(this._capacity);
    this._head = 0;
    this._tail = 0;
    this._burst = null;
    this._capture = null;
    this._up = new Uint8Array(0); // frames x 2 (flat) of the last update: each arm up in view (_armsUp)
  }

  // Capture times of the frames kept (a copy).
  get _times() {
    return this._stamps.slice(this._head, this._tail);
  }

  // The frames kept, as per-frame views (valid until the next add).
  get _frames() {
    return frameViews({ data: this._data, width: W }, this._head, this._tail);
  }

  // One processed frame (KEYPOINT_SIZE values) and the time it was captured (seconds, increasing).
  add(time, keypoints) {
    if (keypoints.length !== W) throw new Error(`a frame has ${W} values, not ${keypoints.length}`);
    time = Number(time);
    if (this._tail > this._head && time <= this._stamps[this._tail - 1]) {
      time = this._stamps[this._tail - 1] + 1e-6; // a stalled or coarse clock must not reorder frames
    }
    if (this._tail === this._capacity) { // move the window to the front of the buffer
      this._data.copyWithin(0, this._head * W, this._tail * W);
      this._stamps.copyWithin(0, this._head, this._tail);
      this._tail -= this._head;
      this._head = 0;
    }
    this._data.set(keypoints, this._tail * W);
    this._stamps[this._tail] = time;
    this._tail += 1;
    while (this._tail > this._head
      && (this._stamps[this._head] < time - WINDOW_SECONDS || this._tail - this._head > MAX_FRAMES)) {
      this._head += 1;
    }
  }

  // Decide what changed since the last call: LiveEvent[]. `now` (seconds) defaults to the newest frame's time;
  // update(now) or update({now}).
  update(now = null) {
    if (now !== null && typeof now === "object") now = now.now ?? null;
    const count = this._tail - this._head;
    if (!count) return [];
    const times = this._stamps.subarray(this._head, this._tail);
    const values = { data: this._data.subarray(this._head * W, this._tail * W), count, width: W };
    now = now == null ? times[count - 1] : pyMax(Number(now), times[count - 1]);
    const activity = signActivity(values, this.aspect, times);
    this._up = this.constructor._armsUp(values, activity, this.aspect);
    const events = [];
    for (;;) { // several signs may have ended since the last call (frames handed over late)
      let burst = this._burst;
      if (burst === null || burst.finished) {
        burst = this._nextBurst(times, activity);
        if (burst === null) break;
        this._burst = burst;
      }
      const wait = this._extend(times, activity, burst); // first: it may move burst.last
      if (now - burst.last <= wait - activity.spacing) break; // the next frame may still continue the burst
      for (const event of this._finish(values, times, activity, burst)) events.push(event);
    }
    const burst = this._burst;
    if (burst !== null && !burst.finished && now - burst.last <= activity.join) {
      const span = this.constructor._span(times, activity, burst);
      if (span !== null && longEnough(activity, span)) {
        for (const event of this._watch(values, times, activity, burst, span, now)) events.push(event);
      }
    }
    return events;
  }

  // The latest finished sign (frozen when it ended), or null before the first one.
  latestSign() {
    return this._capture;
  }

  // ------------------------------------------------------------------ internals

  // A burst from the first signing frame after the previous burst ended, or null.
  _nextBurst(times, activity) {
    const floor = this._burst !== null ? this._burst.last : -Infinity;
    const limit = floor + TOLERANCE;
    for (let i = 0; i < times.length; i++) {
      if (activity.active[i] && times[i] > limit) return new _Burst(times[i], times[i]);
    }
    return null;
  }

  // Take in the runs that continue the burst (gaps up to the wait); returns the wait. A run that only the
  // lost-high wait reaches (later than the usual pause) continues the burst when the hand is found again high (it
  // was blurred); found again low, the hands went down in between: it is the next sign, and the burst ends now
  // (the usual pause is returned).
  _extend(times, activity, burst) {
    let wait = this._wait(activity, times, burst);
    for (const [first, last] of activity.runs) {
      if (times[last - 1] < burst.start || times[first] > burst.last + wait + TOLERANCE) continue;
      if (times[first] > burst.last + activity.join + TOLERANCE) {
        const held = this._held(activity, times, burst);
        if (held === null && activity.height[first] >= EXIT_HEIGHT) return activity.join;
        if (held !== null) burst.gaps.push([burst.last, times[first]]);
      }
      burst.last = pyMax(burst.last, times[last - 1]);
      wait = this._wait(activity, times, burst);
    }
    return wait;
  }

  // Longest time between two signing frames of the burst: the hands may be inactive for JOIN_SECONDS, or
  // DROPOUT_SECONDS when lost high and not going down (motion blur) while the arm stays up, or as long as the arm
  // stays up in view (up to ARM_HOLD_SECONDS).
  _wait(activity, times, burst) {
    const index = Math.min(searchLeft(times, burst.last), times.length - 1);
    let wait = activity.join;
    if (activity.height[index] < EXIT_HEIGHT && activity.descent[index] < LOWERING_SPEED
      && !this.constructor._armDown(activity, index)) {
      wait = pyMax(activity.join, DROPOUT_SECONDS + activity.spacing);
    }
    const held = this._held(activity, times, burst);
    if (held !== null) wait = pyMax(wait, pyMin(ARM_HOLD_SECONDS, held - burst.last + activity.join));
    return wait;
  }

  // The hand of the burst is lost but its arm stays up in view (_armsUp): capture time of the last such frame of
  // the gap after the burst, or null when the gap has a frame where the arm is down, out of view or unknown. A
  // resting arm out of the frame (webcams that film the head and shoulders) is extrapolated: never "in view".
  _held(activity, times, burst) {
    const n = times.length;
    const index = Math.min(searchLeft(times, burst.last), n - 1);
    if (activity.side == null || activity.side[index] < 0 || index + 1 >= n) return null;
    let stop = index + 1;
    while (stop < n && !activity.active[stop]) stop += 1;
    if (stop === index + 1) return null;
    const side = activity.side[index];
    for (let i = index + 1; i < stop; i++) if (!this._up[i * 2 + side]) return null;
    return times[stop - 1];
  }

  // Uint8Array frames x 2 (flat, translate.ARM_WRISTS): the pose wrist is inside the frame, visible, at chest
  // level or higher (ARM_HOLD_HEIGHT), and the forearm is not pointing down (ARM_DOWN_SINE, true shape: x scaled
  // by `aspect`). `values`: the raw frames ({data, count, width} block or array of frames).
  static _armsUp(values, activity, aspect) {
    const { data, count, width } = blockOf(values);
    const up = new Uint8Array(count * 2);
    if (activity.arms == null) return up;
    const arms = activity.arms;
    const pose = POSE_SLICE[0];
    for (let i = 0; i < count; i++) {
      const o = i * width + pose;
      for (let s = 0; s < 2; s++) {
        const w = o + ARM_WRISTS[s] * 4, e = o + ARM_ELBOWS[s] * 4;
        const x = data[w], y = data[w + 1];
        if (!(x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0)) continue;
        if (!(data[w + 3] >= ARM_VISIBLE) || !(arms[i * 2 + s] < ARM_HOLD_HEIGHT)) continue;
        const dx = Math.fround(x - data[e]) * aspect, dy = Math.fround(y - data[e + 1]);
        if (dy < ARM_DOWN_SINE * Math.hypot(dx, dy)) up[i * 2 + s] = 1; // y grows downwards
      }
    }
    return up;
  }

  // After frame `index`, the pose wrist of the signing arm reached the resting height: the hand was put down, not
  // blurred (a blurred hand's pose wrist stays where the hand is).
  static _armDown(activity, index) {
    if (activity.arms == null || activity.side == null || activity.side[index] < 0) return false;
    const side = activity.side[index], count = activity.arms.length / 2;
    for (let i = index + 1; i < count; i++) if (activity.arms[i * 2 + side] >= REST_HEIGHT) return true;
    return false;
  }

  // Frames of the burst still in the window: [first, last), never before the burst's own start (a run can reach
  // back into the previous sign when later frames change how the pause between them is seen); null when none.
  static _span(times, activity, burst) {
    let firstRun = null, lastRun = null;
    for (const run of activity.runs) {
      if (times[run[1] - 1] >= burst.start && times[run[0]] <= burst.last) {
        firstRun ??= run;
        lastRun = run;
      }
    }
    if (firstRun === null) return null;
    const first = Math.max(firstRun[0], searchLeft(times, burst.start - TOLERANCE));
    const last = lastRun[1];
    return first < last ? [first, last] : null;
  }

  // Frames around a span: [first, last) of the span with CONTEXT_SECONDS on each side.
  static _keep(times, span) {
    const [first, last] = span;
    return [searchLeft(times, times[first] - CONTEXT_SECONDS - TOLERANCE),
      searchRight(times, times[last - 1] + CONTEXT_SECONDS + TOLERANCE)];
  }

  // The recogniser's guess for the frames of `span` with their context (the hands interpolated across `gaps`).
  _predict(values, times, span, gaps = []) {
    const [start, stop] = this.constructor._keep(times, span);
    const block = blockOf(values);
    const stamps = times.subarray(start, stop);
    let window = frameViews(block, start, stop);
    if (gaps.length) window = this.constructor._bridged(window, stamps, gaps);
    return this.recognizer.predict(window, null, false, this.aspect, stamps);
  }

  // The hands lost during `gaps` (their arm stayed up in view), interpolated in time across each gap as the
  // recogniser does for dropouts shorter than JOIN_SECONDS: a copy (array of frames).
  static _bridged(values, times, gaps) {
    const source = blockOf(values);
    const frames = frameCopies(source, 0, source.count);
    const n = times.length;
    for (const [beforeGap, afterGap] of gaps) {
      const first = searchLeft(times, beforeGap - TOLERANCE);
      const last = searchLeft(times, afterGap - TOLERANCE);
      if (first >= n || last >= n || last - first < 2) continue;
      for (const [start, end] of [LEFT_HAND_SLICE, RIGHT_HAND_SLICE]) {
        let previous = -1; // every stretch of missing frames between two seen ones
        for (let b = first; b <= last; b++) {
          const row = frames[b];
          let seen = false;
          for (let c = start; c < end; c++) if (Math.abs(row[c]) > F32_1E6) { seen = true; break; }
          if (!seen) continue;
          const a = previous;
          previous = b;
          if (a < 0 || b - a <= 1) continue;
          const ra = frames[a], span = times[b] - times[a];
          for (let k = a + 1; k < b; k++) {
            const weight = (times[k] - times[a]) / span, rk = frames[k];
            for (let c = start; c < end; c++) rk[c] = (1 - weight) * ra[c] + weight * row[c];
          }
        }
      }
    }
    return frames;
  }

  _finish(values, times, activity, burst) {
    burst.finished = true;
    const span = this.constructor._span(times, activity, burst);
    const sign = span !== null && longEnough(activity, span);
    if (sign) {
      let kept = span;
      if (burst.committed.length && burst.letters.length) { // spelled: the sign just made is the last letter
        const letter = burst.letters[burst.letters.length - 1];
        const held = [Math.max(span[0], searchLeft(times, letter.start - TOLERANCE)),
          Math.min(span[1], searchRight(times, letter.end + TOLERANCE))];
        kept = held[0] < held[1] ? held : span;
      }
      const [start, stop] = this.constructor._keep(times, kept);
      const number = this._capture ? this._capture.number + 1 : 1;
      this._capture = new SignCapture(frameCopies(blockOf(values), start, stop), times.slice(start, stop),
        [kept[0] - start, kept[1] - start], number);
    }
    const made = sign ? this._capture.number : 0;
    if (burst.committed.length) { // spelling: the last held letter, unless it was just added
      const letter = burst.letters.length ? burst.letters[burst.letters.length - 1] : null;
      if (letter === null || letter.committed || letter.prediction.label === burst.committed[burst.committed.length - 1]) {
        return [];
      }
      letter.committed = true;
      burst.committed.push(letter.prediction.label);
      return [new LiveEvent("final", letter.prediction, letter.start, letter.end, true, made)];
    }
    if (!sign) return [];
    const guesses = [this._predict(values, times, span)];
    if (burst.gaps.length) { // the hand was not seen during a gap: as missing, or where it must have been
      guesses.push(this._predict(values, times, span, burst.gaps));
    }
    const settled = guesses.map((guess) => this.constructor._settled(guess, burst));
    // the more decisive guess: an accepted one first, then the larger lead for its distance (ties: the first)
    let prediction = settled[0];
    for (let i = 1; i < settled.length; i++) {
      const guess = settled[i], accepted = guess.label != null, best = prediction.label != null;
      if (accepted !== best ? accepted : this.constructor._lead(guess) > this.constructor._lead(prediction)) {
        prediction = guess;
      }
    }
    if (prediction.label != null) burst.committed.push(prediction.label);
    return [new LiveEvent("final", prediction, burst.start, burst.last, prediction.label != null, made)];
  }

  // A far match the recogniser rejected only for want of a clear lead is accepted when a provisional guess of the
  // same burst was sure of that sign (before a hand was lost, say): the usual MIN_MARGIN lead is enough.
  static _settled(prediction, burst) {
    if (prediction.label == null && burst.sure.has(prediction.bestLabel)
      && prediction.margin >= MIN_MARGIN && prediction.distance <= MAX_DISTANCE) {
      return new Prediction(prediction.bestLabel, prediction.candidates[0][1], MIN_CONFIDENCE, prediction.distance,
        prediction.margin, prediction.bestLabel, prediction.candidates);
    }
    return prediction;
  }

  // How decisive a guess is: its margin over the margin its distance requires (1: just accepted).
  static _lead(prediction) {
    const required = requiredMargin(prediction.distance);
    return Number.isFinite(required) && required > 0 ? prediction.margin / required : 0.0;
  }

  // A held handshape is recognised as soon as it is long enough, then every PROVISIONAL_SECONDS; otherwise the
  // whole burst so far gets a provisional guess every PROVISIONAL_SECONDS.
  _watch(values, times, activity, burst, span, now) {
    const hold = this._hold(times, activity, burst, span);
    if (hold !== null && now - burst.lookedAt >= PROVISIONAL_SECONDS) {
      burst.lookedAt = burst.provisionalAt = now;
      return this._letter(values, times, burst, hold);
    }
    if (hold === null && now - burst.provisionalAt >= PROVISIONAL_SECONDS) {
      burst.provisionalAt = now;
      const prediction = this._predict(values, times, span);
      if (prediction.label != null && prediction.confidence >= HOLD_CONFIDENCE) burst.sure.add(prediction.label);
      return [new LiveEvent("signing", prediction, burst.start, burst.last)];
    }
    return [];
  }

  // [first, last) of a held handshape, or null: at the first look at a still stretch its first HOLD_SECONDS
  // (fixed in capture time, so a camera stall just then cannot mix in the next letter, which changes the
  // handshape without moving the wrist), afterwards the last HOLD_SECONDS.
  _hold(times, activity, burst, span) {
    const [first, last] = span;
    // a frame without a signing hand (lost, lowered) is not a still handshape: a hand found again after a gap near
    // where it was lost would otherwise make the gap look like a held letter
    let stillFirst = first;
    for (let i = last - 1; i >= first; i--) {
      if (activity.moved[i] >= HOLD_MOVE || !activity.active[i]) {
        stillFirst = i + 1;
        break;
      }
    }
    if (stillFirst >= last) return null;
    const start = times[stillFirst];
    if (burst.stillFrom === null || start > burst.stillLast + TOLERANCE) { // a new still stretch
      const leadFrom = burst.stillLast === null ? burst.start : burst.stillLast;
      burst.stillFrom = start;
      burst.lookedAt = -Infinity;
      burst.stillIgnored = start - leadFrom > HOLD_LEAD_SECONDS;
    } else { // the same stretch (landmark smoothing may move its first frame a little)
      burst.stillFrom = pyMin(burst.stillFrom, start);
    }
    burst.stillLast = times[last - 1];
    if (burst.stillIgnored || burst.stillLast - burst.stillFrom < HOLD_SECONDS - TOLERANCE) return null;
    if (burst.lookedAt === -Infinity) { // not looked at yet
      const begin = Math.max(searchLeft(times, burst.stillFrom - TOLERANCE), first);
      const stop = searchRight(times, burst.stillFrom + HOLD_SECONDS + TOLERANCE);
      return [begin, Math.min(Math.max(stop, begin + 1), last)];
    }
    const window = searchLeft(times, burst.stillLast - HOLD_SECONDS - TOLERANCE);
    return [Math.max(window, stillFirst), last];
  }

  // Recognise a held handshape; a sure guess of another letter adds the previous letter (spelling).
  _letter(values, times, burst, hold) {
    const prediction = this._predict(values, times, hold);
    const start = times[hold[0]], end = times[hold[1] - 1];
    const events = [];
    if (prediction.label != null && prediction.confidence >= HOLD_CONFIDENCE) {
      const previous = burst.letters.length ? burst.letters[burst.letters.length - 1] : null;
      if (previous !== null && previous.prediction.label === prediction.label) {
        previous.end = end; // the same letter, still held
      } else {
        if (previous !== null && !previous.committed
          && (!burst.committed.length || burst.committed[burst.committed.length - 1] !== previous.prediction.label)) {
          previous.committed = true;
          burst.committed.push(previous.prediction.label);
          events.push(new LiveEvent("hold", previous.prediction, previous.start, previous.end, true));
        }
        burst.letters.push(new _Letter(prediction, start, end));
      }
    }
    events.push(new LiveEvent("hold", prediction, start, end));
    return events;
  }
}

export const _internal = { searchLeft, searchRight, pyMax, pyMin, blockOf, frameViews };
