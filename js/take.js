// Webcam takes: the take logic of vsl_ui.py (take_span, take_sample, take_frame's size and the take state machine of
// CameraWorker: start_recording, cancel_recording, _record, _next_take, _drop_before, the altered marks).
// A take keeps the frames from PREROLL_SECONDS before its countdown ends until its sign is over; only the sign
// (± translate.CAPTURE_MARGIN_SECONDS) is stored, as FRAME_COUNT frames evenly spaced in time.

import {
  CAPTURE_MARGIN_SECONDS,
  FRAME_COUNT,
  REST_HEIGHT,
  linspace,
  longEnough,
  resampleSequence,
  roundHalfEven,
  signActivity,
} from "./translate.js";

export const COUNTDOWN_SECONDS = 3;
export const PAUSE_BETWEEN_TAKES = 1.5;
// people.judge verdicts that cut or substituted the signer's landmarks (and frames where the person was not
// verified): a take around them is recorded again ("other").
export const ALTERED_REASONS = Object.freeze(["head", "body", "arm", "rest", "hand"]);
export const ALTERED_MEMORY_SECONDS = 30.0;
// A webcam take lasts as long as the sign: only the sign is stored, as FRAME_COUNT frames evenly spaced in time.
export const PREROLL_SECONDS = 0.6;     // a take also keeps the end of the countdown: a sign started a little early stays whole
export const TAKE_WAIT_SECONDS = 4.0;   // no sign this long after the countdown: the take is recorded again
export const TAKE_MAX_SECONDS = 6.0;    // a take never lasts longer than this after the countdown
export const TAKE_END_SECONDS = 1.0;    // the hands stopped signing (lowered, out of view) this long: the take ends. Pauses
                                        // inside one sign are shorter (dictionary clips: at most 0.83 s), two performances longer
export const TAKE_CHECK_SECONDS = 0.2;  // how often a take is analysed (capture time)
export const TAKE_EDGE_SECONDS = 0.35;  // a wrist moving, within this of the first frame kept (or of the end of a take that
export const TAKE_EDGE_MOVE = 0.25;     // reached TAKE_MAX_SECONDS), this much over MOVE_WINDOW_SECONDS (shoulder widths): cut sign
export const TAKE_FRAME_WIDTH = 640;    // display copies kept for a take's stored frames (dataset.IMAGE_MAX_WIDTH)
export const TAKE_GAP_SECONDS = 0.35;   // a longer pause between two frames of the sign (camera stall): the take is recorded again,
export const TAKE_RESTARTS = 3;         // ... until a take fails this many times in a row: then recording stops (controls back)
export const TAKE_FAILURES = Object.freeze({   // why a take is recorded again
  none: "Chưa thấy ký hiệu nào trong lượt này — lượt được ghi lại. Làm ký hiệu ngay khi hết đếm ngược.",
  early: "Tay đã cử động trước khi hết đếm ngược (ký hiệu có thể bị mất đoạn đầu) — lượt được ghi lại. "
    + "Giữ tay yên hoặc để tay xuống cho đến khi hết đếm ngược rồi mới ký.",
  long: `Sau ${TAKE_MAX_SECONDS.toFixed(0)} giây tay vẫn đang chuyển động — lượt được ghi lại. `
    + "Làm ký hiệu một lần rồi hạ tay xuống.",
  stall: "Hình bị khựng giữa lúc ký (camera hoặc máy xử lý chậm) — lượt này được ghi lại.",
  other: "Giữa lúc ký, ứng dụng chưa nhận ra người đã đăng ký hoặc có người khác chen vào sát bên "
    + "(landmark của họ bị loại) — lượt được ghi lại. Nhìn vào camera và ngồi tách khỏi người khác.",
});
export const TAKE_HINTS = Object.freeze({ wait: "đang chờ ký hiệu", signing: "đang ký — xong thì hạ tay xuống" });

const RECENT_FRAMES = 10;   // CameraWorker._recent: capture times of the last frames processed, in every phase

// Python's format(value, ".Nf"): the exact binary value rounded half to even (toFixed breaks exact ties upwards).
function pyFixed(value, digits) {
  if (Number.isNaN(value)) return "nan";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  let text = value.toFixed(digits);
  if (Math.abs(value) < 1e21) {
    const exact = Math.abs(value).toFixed(100);   // exact for every double that can be a tie at <= 100 digits
    const dot = exact.indexOf(".");
    const tail = exact.slice(dot + 1 + digits);
    if (tail[0] === "5" && /^50*$/.test(tail)) {
      const kept = exact.slice(0, dot + 1 + digits).replace(/\.$/, "");
      const last = kept[kept.length - 1];
      if ((last.charCodeAt(0) - 48) % 2 === 0) text = (value < 0 ? "-" : "") + kept;   // even: rounded down
    }
  }
  if (Object.is(value, -0)) text = `-${text}`;
  return text;
}

// np.median of float64 values.
function median(values) {
  const sorted = Float64Array.from(values).sort();
  const n = sorted.length, half = n >> 1;
  return n % 2 ? sorted[half] : (sorted[half - 1] + sorted[half]) / 2;
}

// ndarray.max over the frames where `mask(i)` holds (NaN propagates); -Infinity when none.
function maxWhere(values, count, mask) {
  let best = -Infinity;
  for (let i = 0; i < count; i++) {
    if (!mask(i)) continue;
    const value = values[i];
    if (Number.isNaN(value)) return NaN;
    if (value > best) best = value;
  }
  return best;
}

// np.searchsorted(sorted, value) (side "left"): the first index whose value is not below `value`.
function searchLeft(sorted, value) {
  let low = 0, high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

// Size [width, height] of take_frame's copy of a width x height frame: at most TAKE_FRAME_WIDTH wide.
export function takeFrameSize(width, height) {
  if (width <= TAKE_FRAME_WIDTH) return [width, height];
  return [TAKE_FRAME_WIDTH, roundHalfEven((height * TAKE_FRAME_WIDTH) / width)];
}

// A webcam take as a stored sample: FRAME_COUNT frames evenly spaced in time, whatever the frame rate.
// Returns [keypoints (array of FRAME_COUNT Float32Array), the frames filmed nearest to those times (objects kept as
// they are), recording details {duration, fps} as strings].
export function takeSample(keypoints, frames, stamps) {
  const times = Float64Array.from(stamps, Number);
  const count = times.length;
  const values = resampleSequence(keypoints, times, FRAME_COUNT);
  const targets = linspace(times[0], times[count - 1], FRAME_COUNT);
  const chosen = [];
  for (let t = 0; t < FRAME_COUNT; t++) {   // np.abs(stamps - target).argmin(): the first of ties, NaN first
    let best = 0, nearest = Infinity;
    for (let i = 0; i < count; i++) {
      const distance = Math.abs(times[i] - targets[t]);
      if (Number.isNaN(distance)) { best = i; break; }
      if (distance < nearest) { nearest = distance; best = i; }
    }
    chosen.push(frames[best]);
  }
  const duration = times[count - 1] - times[0];
  const details = { duration: pyFixed(duration, 3), fps: pyFixed((count - 1) / Math.max(duration, 1e-6), 1) };
  return [values, chosen, details];
}

// Where the sign of a take is: [state, [first, last) frames of the sign or null, translate.SignActivity].
// The sign runs from the first burst of signing with a sign's length (longEnough) filmed after the countdown ended
// (`started`) through every burst less than TAKE_END_SECONDS from it (a sign with a pause, a hand lost to motion
// blur). state: "wait" (no sign yet), "signing", "done" (the hands stopped TAKE_END_SECONDS ago: store it), "none"
// (no sign in time), "early" (the sign began before the countdown ended and is cut: the wrist already moves at the
// first frame kept, or only a blurred hand with its arm up is missing there; or a separate burst long enough to be
// a sign came before the countdown ended, right before the sign: getting ready or begun too early; or the only
// sign of the take was over before the countdown ended), "long" (still moving at TAKE_MAX_SECONDS). A brief twitch
// before the countdown ended is left out; a handshape held still at either end (a letter, a number) is not cut.
export function takeSpan(values, stamps, aspect, started, now) {
  const times = stamps instanceof Float64Array ? stamps : Float64Array.from(stamps, Number);
  const count = times.length;
  const activity = signActivity(values, aspect, times);
  const runs = activity.runs;
  const after = searchLeft(times, started);   // the first frame filmed after the countdown ended
  const signs = [];
  runs.forEach(([first, last], index) => {
    if (last > after && longEnough(activity, [Math.max(first, after), last])) signs.push(index);
  });
  const elapsed = now - started;
  if (!signs.length) {
    const beginning = runs.length > 0 && now - times[runs[runs.length - 1][1] - 1] <= activity.join;
    if (elapsed >= TAKE_MAX_SECONDS || (elapsed >= TAKE_WAIT_SECONDS && !beginning)) {
      const overEarly = runs.some((run) => longEnough(activity, run));
      return [overEarly ? "early" : "none", null, activity];
    }
    return ["wait", null, activity];
  }
  let low = signs[0], high = signs[0];
  let ready = false;   // a separate burst long enough to be a sign before the countdown ended, right before the sign
  while (low > 0 && times[runs[low][0]] - times[runs[low - 1][1] - 1] < TAKE_END_SECONDS) {
    if (runs[low - 1][0] < after) {
      ready = longEnough(activity, runs[low - 1]);
      break;
    }
    low -= 1;
  }
  while (high + 1 < runs.length && times[runs[high + 1][0]] - times[runs[high][1] - 1] < TAKE_END_SECONDS) high += 1;
  const span = [runs[low][0], runs[high][1]];
  const ended = now - times[span[1] - 1] >= TAKE_END_SECONDS;
  if (!ended && elapsed < TAKE_MAX_SECONDS) return ["signing", span, activity];
  const first = span[0];
  const side = activity.side != null ? activity.side[first] : -1;
  let blurred = 0 < first && times[first] - times[0] <= activity.join && side >= 0 && activity.arms != null;
  if (blurred) {   // hand missed at the start, its arm up
    for (let i = 0; i < first; i++) {
      if (!(activity.arms[i * 2 + side] < REST_HEIGHT)) { blurred = false; break; }
    }
  }
  const edge = times[first] + TAKE_EDGE_SECONDS;
  const moving = maxWhere(activity.moved, count, (i) => times[i] <= edge) >= TAKE_EDGE_MOVE;
  if (ready || ((first === 0 || blurred) && moving)) return ["early", span, activity];
  const end = times[count - 1] - TAKE_EDGE_SECONDS;
  if (!ended && maxWhere(activity.moved, count, (i) => times[i] >= end) >= TAKE_EDGE_MOVE) {
    return ["long", span, activity];
  }
  return ["done", span, activity];
}

const defaultClock = () => performance.now() / 1000;

// The take state machine of CameraWorker (collect mode). callbacks: progress(take, tenths, total, state) for every
// recording frame that neither stored nor failed a take, sample(keypoints, frames, details), message(text),
// finished(); all called after the state is updated. `clock`: monotonic seconds (countdowns and pauses).
export class TakeRecorder {
  constructor({ clock = defaultClock, progress = null, sample = null, message = null, finished = null } = {}) {
    this.clock = clock;
    this.callbacks = { progress, sample, message, finished };
    this.takesLeft = 0;
    this.takeNumber = 0;
    this.phase = "idle";        // "idle" | "countdown" | "recording"
    this.deadline = 0.0;        // clock time the countdown (or the pause between takes) ends
    this.keypoints = [];        // the frames of the current take (Float32Array copies), their frame objects and times
    this.frames = [];
    this.stamps = [];
    this.recent = [];           // capture times of the last RECENT_FRAMES frames processed, in every phase
    this.restarts = 0;          // takes recorded again in a row (no sign, cut sign, stall, other)
    this.takeStart = 0.0;       // capture time the countdown of the current take ended
    this.takeState = "wait";    // takeSpan state of the current take
    this.checked = -Infinity;   // capture time the current take was last analysed
    this.altered = [];          // capture times of frames whose landmarks were cut or substituted (ALTERED_REASONS)
  }

  // A frame of the registered person was cut or substituted (call before record() sees it).
  markAltered(stamp) {
    this.altered.push(stamp);
    let drop = 0;
    while (drop < this.altered.length && this.altered[drop] < stamp - ALTERED_MEMORY_SECONDS) drop += 1;
    if (drop) this.altered.splice(0, drop);
  }

  alteredWithin(start, end) {
    return this.altered.some((stamp) => start <= stamp && stamp <= end);
  }

  startRecording(takes) {
    this.takesLeft = Math.max(1, Math.trunc(Number(takes)) || 0);
    this.takeNumber = 0;
    this.restarts = 0;
    this.phase = "countdown";
    this.deadline = this.clock() + COUNTDOWN_SECONDS;
    this._clear();
  }

  cancelRecording() {
    this.takesLeft = 0;
    this.restarts = 0;
    this.phase = "idle";
    this._clear();
  }

  _clear() {
    this.keypoints = [];
    this.frames = [];
    this.stamps = [];
  }

  _dropBefore(limit) {
    let drop = 0;
    while (drop < this.stamps.length && this.stamps[drop] < limit) drop += 1;
    if (!drop) return;
    this.keypoints.splice(0, drop);
    this.frames.splice(0, drop);
    this.stamps.splice(0, drop);
  }

  _nextTake() {
    if (this.takesLeft > 0) {
      this.phase = "countdown";
      this.deadline = this.clock() + PAUSE_BETWEEN_TAKES;
    } else {
      this.phase = "idle";
    }
  }

  // One processed frame (CameraWorker._record): `keypoints` Float32Array(1662) (copied), `frame` any object kept as
  // is (the annotated picture), `stamp` its capture time, `aspect` width / height of the frame.
  // A take: the frames from PREROLL_SECONDS before its countdown ends until its sign is over (takeSpan). Only the
  // sign, with CAPTURE_MARGIN_SECONDS around it, is stored. A take is recorded again when it shows no sign, a sign
  // cut at either end, landmarks cut around the sign ("other") or a camera stall inside the sign: a gap longer than
  // TAKE_GAP_SECONDS and than 3 usual frame intervals (from the frames processed before and during the take). It
  // is judged once the sign is known; during the countdown a stall only drops the frames before it.
  record(keypoints, frame, stamp, aspect) {
    let take = null, failure = null;
    const recent = this.recent;
    let spacing = 0.0;
    if (recent.length > 2) {
      const steps = new Float64Array(recent.length - 1);
      for (let i = 1; i < recent.length; i++) steps[i - 1] = recent[i] - recent[i - 1];
      spacing = median(steps);
    }
    recent.push(stamp);
    if (recent.length > RECENT_FRAMES) recent.shift();
    if (this.phase === "countdown" && this.clock() >= this.deadline) {
      this.phase = "recording";
      this.takeState = "wait";
      this.takeNumber += 1;
      this.takeStart = stamp;
      this.checked = -Infinity;
    }
    if (this.phase === "idle") return;
    const limit = Math.max(TAKE_GAP_SECONDS, 3 * spacing);
    if (this.phase === "countdown" && this.stamps.length && stamp - this.stamps[this.stamps.length - 1] > limit) {
      this._clear();
    }
    this.keypoints.push(Float32Array.from(keypoints));
    this.frames.push(frame);
    this.stamps.push(stamp);
    if (this.phase === "countdown") {
      this._dropBefore(stamp - PREROLL_SECONDS);
      return;
    }
    const elapsed = stamp - this.takeStart;
    if (stamp - this.checked >= TAKE_CHECK_SECONDS || elapsed >= TAKE_MAX_SECONDS) {
      this.checked = stamp;
      const stamps = Float64Array.from(this.stamps);
      const [state, span, activity] = takeSpan(this.keypoints, stamps, aspect, this.takeStart, stamp);
      this.takeState = state;
      if (state === "wait" && !activity.active.some((value) => value)) {
        this._dropBefore(stamp - PREROLL_SECONDS);   // nothing yet: the analysis stays short
      } else if (state !== "signing" && span && gapWithin(stamps, Math.max(span[0] - 1, 0), span[1] + 1, limit)) {
        failure = "stall";   // a gap inside the sign, or the sign began during it
      } else if (state !== "signing" && span && this.alteredWithin(
        stamps[span[0]] - Math.max(TAKE_EDGE_SECONDS, CAPTURE_MARGIN_SECONDS), stamps[span[1] - 1] + CAPTURE_MARGIN_SECONDS)) {
        // Landmarks cut out inside what is stored (or just before the sign: its start is unknown): a hole or a
        // cut sign would be stored.
        failure = "other";
      } else if (state === "done") {
        const start = stamps[span[0]] - CAPTURE_MARGIN_SECONDS, end = stamps[span[1] - 1] + CAPTURE_MARGIN_SECONDS;
        const keep = [];
        for (let i = 0; i < stamps.length; i++) if (stamps[i] >= start && stamps[i] <= end) keep.push(i);
        take = [keep.map((i) => this.keypoints[i]), keep.map((i) => this.frames[i]), keep.map((i) => stamps[i])];
      } else if (Object.hasOwn(TAKE_FAILURES, state)) {
        failure = state;
        if (state === "none" && this.alteredWithin(this.takeStart, stamp)) {
          failure = "other";   // no sign seen because the person's landmarks were not taken
        }
      }
    }
    if (take) {
      this._clear();
      this.restarts = 0;
      this.takesLeft -= 1;
      this._nextTake();
    } else if (failure === null) {
      this.callbacks.progress?.(this.takeNumber, Math.trunc(elapsed * 10), this.takeNumber + this.takesLeft - 1,
        this.takeState);
    }
    if (failure !== null) {
      this._clear();
      this.restarts += 1;
      if (this.restarts >= TAKE_RESTARTS) this.takesLeft = 0;
      else this.takeNumber -= 1;
      this._nextTake();
    }
    const finished = this.phase === "idle";
    if (failure !== null) {
      if (finished) {
        this.callbacks.message?.(`${TAKE_FAILURES[failure]} Đã ${TAKE_RESTARTS} lượt liên tiếp không đạt — đã dừng `
          + "ghi." + (failure === "stall" ? " Hãy đóng bớt ứng dụng khác rồi ghi lại." : ""));
        this.callbacks.finished?.();
      } else {
        this.callbacks.message?.(TAKE_FAILURES[failure]);
      }
      return;
    }
    if (take) {
      this.callbacks.sample?.(...takeSample(...take));
      if (finished) this.callbacks.finished?.();
    }
  }

  // What the countdown / REC indicator shows (CameraWorker._overlay): phase, remaining whole seconds of the
  // countdown (ceil, at least 1; 0 outside it), seconds recorded in this take, and its takeSpan state.
  overlay() {
    const phase = this.phase;
    const recorded = phase === "recording" && this.stamps.length ? this.stamps[this.stamps.length - 1] - this.takeStart : 0.0;
    const remaining = phase === "countdown" ? Math.max(1, Math.ceil(this.deadline - this.clock())) : 0;
    return { phase, remaining, recorded, state: this.takeState };
  }
}

// np.any(np.diff(stamps[low:high]) > limit) (high clipped to the length).
function gapWithin(stamps, low, high, limit) {
  const end = Math.min(high, stamps.length);
  for (let i = low + 1; i < end; i++) if (stamps[i] - stamps[i - 1] > limit) return true;
  return false;
}

export const _internal = { pyFixed, median, searchLeft, maxWhere, gapWithin };
