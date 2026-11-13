// The camera loop (vsl_ui.CameraWorker.run on the phone): newest camera frame -> light correction -> Holistic
// (one tracker per person while a profile is logged in: people.js) -> drawing -> webcam takes (take.js) or the
// recognition worker. Frames are stamped with their capture time; a slow phone skips frames like the PC's
// FrameGrabber (only the newest is processed) and recognition gets bigger batches, never stale ones.
//
// Speed on phones: the picture on the stage is the camera's own <video> (smooth whatever the analysis costs) with the
// landmarks drawn on a transparent canvas over it; frames are analysed at most PROCESS_MAX pixels wide, and, without
// a profile, by a pool of detection workers (detectors.js: light correction + Holistic off the page's thread), each
// taking a frame while the others work, results applied in capture order. The personal lock and enrollment analyse
// on the page's thread (one frame at a time) as before.

import { Camera, CameraAborted, CameraError } from "./camera.js";
import { createHolistic, FACE_TESSELATION, preferredHolisticDelegate } from "./holistic.js";
import { DetectorPool, detectorCount } from "./detectors.js";
import { AdaptiveLightCorrection } from "./light.js";
import { drawDense, drawIdentity, drawMuted, drawSimplified } from "./draw.js";
import { remapKeypoints } from "./layout.js";
import { ALTERED_REASONS, COUNTDOWN_SECONDS, TAKE_HINTS, TakeRecorder } from "./take.js";
import { ENROLL_FRAMES, Enrollment, PersonLock } from "./lock.js";

export { COUNTDOWN_SECONDS, TAKE_HINTS, ENROLL_FRAMES };

export const PROCESS_MAX = 640; // longest side of the frames analysed (the camera's picture may be larger)
// Frames analysed per second at most: enough for live translation and takes (the PC's webcam runs at 15-30), without a
// fast phone heating up and draining its battery on frames that add nothing.
export const MAX_ANALYSIS_FPS = 20;
export const FRAME_FAILURES = 5; // frames in a row a worker cannot analyse before the page's thread takes over
export const PREVIEW_WIDTH = 320; // small copies of the annotated frames: the picture saved with a sample
export const PREVIEW_SECONDS = 0.1; // ... one every this many seconds of capture time
export const PREVIEW_KEEP_SECONDS = 12.0; // ... kept this long (a take or a sign is never longer)
export const REJECTION_REASONS = ["head", "arm", "rest", "hand", "face"];
export const REJECTION_SHOW_SECONDS = 1.5;
export const BLOCKED_SECONDS = 1.0; // nothing of the registered person taken this long while someone else was in the way
// SFace int8 (the phone's face model) against an enrollment: other people reached 0.378 on look-alike signers where the
// PC's fp32 model stays below 0.363; the same person never scored below 0.616 (tools/golden_faces.py faces_identity).
export const PHONE_MATCH_THRESHOLD = 0.40;
export const CAMERA_LOST_SECONDS = 3.0; // no frame for this long after the camera worked: it was disconnected or taken
export const CAMERA_START_SECONDS = 10.0; // ... or for this long after opening it
export const REOPEN_LIMIT = 2; // an interrupted camera is opened again this many times...
export const REOPEN_WINDOW_SECONDS = 30.0; // ... within this long, then reported

const now = () => performance.now() / 1000;

// The size frames are analysed at: the camera's picture scaled down to at most PROCESS_MAX on its longer side.
export function processSize(width, height, limit = PROCESS_MAX) {
  if (!width || !height) return [0, 0];
  const scale = Math.min(1, limit / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

function makeCanvas(width, height) {
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(width, height) : document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

// Small JPEG copies of the annotated frames by capture time; a sample's picture is the one nearest its middle.
class PreviewRing {
  constructor() {
    this.items = [];
    this.canvas = null;
    this.last = -Infinity;
  }

  // A copy of the frame of `stamp` (width x height) when one is due: paint(context, w, h) draws it, small.
  offer(stamp, width, height, paint) {
    if (stamp - this.last < PREVIEW_SECONDS) return;
    this.last = stamp;
    const w = PREVIEW_WIDTH, h = Math.max(1, Math.round((PREVIEW_WIDTH * height) / width));
    if (!this.canvas || this.canvas.width !== w || this.canvas.height !== h) this.canvas = makeCanvas(w, h);
    const context = this.canvas.getContext("2d");
    paint(context, w, h);
    const blob = this.canvas.convertToBlob
      ? this.canvas.convertToBlob({ type: "image/jpeg", quality: 0.82 })
      : new Promise((resolve) => this.canvas.toBlob(resolve, "image/jpeg", 0.82));
    this.items.push({ stamp, blob: blob.catch(() => null) });
    while (this.items.length && this.items[0].stamp < stamp - PREVIEW_KEEP_SECONDS) this.items.shift();
  }

  nearest(stamp) {
    let best = null;
    for (const item of this.items) if (!best || Math.abs(item.stamp - stamp) < Math.abs(best.stamp - stamp)) best = item;
    return best ? best.blob : Promise.resolve(null);
  }

  clear() {
    this.items = [];
    this.last = -Infinity;
  }
}

export class Pipeline {
  // callbacks: status(text|null), started(), failed(message), message(text), frame({fps, frames}),
  // light(exposure, advice), identity(status), overlay({countdown}), takeProgress(take, tenths, total, state),
  // sampleReady(keypoints, frames, details), recordingFinished(), enrollProgress(instruction, bins, captured),
  // enrollFinished(enrollment), signFinished(sign)
  constructor({ video, view, recognition, callbacks }) {
    this.camera = new Camera(video);
    this.view = view;
    this.viewContext = view.getContext("2d");
    this.recognition = recognition;
    this.callbacks = callbacks;
    this.mode = "translation";
    this.identity = null; // the logged-in person's enrolled embeddings (Float32Array, n x 128), or null
    this.settings = { dense: false, autoLight: true, bias: 0 };
    this.session = 0;
    this.starting = false;
    this.holistic = null; // one Holistic on the page's thread (no profile logged in, no detection workers)
    this.pool = null; // detectors.DetectorPool (no profile logged in)
    this.usePool = false; // this session analyses with the pool
    this.workersFailed = false; // the workers could not run here: the page's thread analyses from then on
    this.people = null; // people.People (profile logged in)
    this.engine = null; // faces.FaceEngine, loaded when first needed
    this.enrollment = null;
    this.work = makeCanvas(640, 480);
    this.corrected = null; // the corrected picture of a frame analysed by a worker (for the sample's picture)
    this.light = new AdaptiveLightCorrection();
    this.previews = new PreviewRing();
    this.takes = new TakeRecorder({
      clock: now,
      progress: (...args) => this.callbacks.takeProgress?.(...args),
      sample: (...args) => this.callbacks.sampleReady?.(...args),
      message: (text) => this.callbacks.message?.(text),
      finished: () => this.callbacks.recordingFinished?.(),
    });
    this.fps = 0;
    this.frames = 0;
    this.delegate = null;
    this.perf = {}; // milliseconds per stage of a frame (moving average): what the phone spends its time on
    recognition.onSign = (sign) => this._signFinished(sign);
  }

  get running() {
    return this.camera.running && !this.starting;
  }

  configure(settings) {
    Object.assign(this.settings, settings);
    this.light.autoEnabled = this.settings.autoLight;
    this.light.exposureBiasEv = this.settings.bias;
  }

  setMode(mode) {
    if (mode === this.mode) return;
    // Enrollment and the personal lock's trackers (strict while collecting) are built for one mode.
    const rebuild = mode === "enroll" || this.mode === "enroll"
      || (Boolean(this.identity) && (mode === "collect") !== (this.mode === "collect"));
    this.mode = mode;
    if (mode !== "collect") this.cancelRecording();
    if ((this.running || this.starting) && rebuild) this.restart();
  }

  setIdentity(embeddings) {
    this.identity = embeddings && embeddings.length ? embeddings : null;
    if (this.running || this.starting) this.restart(); // like VslWindow.set_active_profile: the camera restarts
  }

  async restart() {
    this.stop();
    try {
      await this.start();
    } catch (error) {
      this.callbacks.failed?.(error.message || String(error));
    }
  }

  startRecording(takes) {
    this.takes.startRecording(takes);
  }

  cancelRecording() {
    this.takes.cancelRecording();
  }

  async switchFacing() {
    const facing = this.camera.facing === "user" ? "environment" : "user";
    this.stop();
    await this.start(facing);
  }

  // Load what this mode needs, then open the camera (like CameraWorker.run: models first, so leaving early never
  // strands an open camera).
  async start(facing = this.camera.facing) {
    if (this.starting) return;
    this.starting = true;
    const session = ++this.session;
    try {
      this.callbacks.status?.("Đang nạp mô hình nhận dạng…");
      if (this.mode === "enroll" || this.identity) await this._faceEngine();
      if (session !== this.session) return;
      if (this.mode === "enroll") {
        this.enrollment = new Enrollment();
        this.enrollDone = false;
      }
      this.usePool = false;
      let usePool = false; // this session's choice (a superseded start must never change the current one's)
      if (this.identity && this.mode !== "enroll") {
        this._closePool(); // the personal lock runs its own trackers: no memory for idle workers beside them
        await this._createPeople(session);
      } else if (this.mode === "enroll") {
        this._closePool(); // enrollment needs only the face engine: no Holistic is kept busy or loaded for it
      } else {
        if (this.settings.workers !== false && !this.workersFailed && typeof Worker !== "undefined") {
          try {
            if (!this.poolPromise) {
              const created = DetectorPool.create(detectorCount(), preferredHolisticDelegate())
                .catch((error) => { if (this.poolPromise === created) this.poolPromise = null; throw error; });
              this.poolPromise = created;
            }
            const pool = await this.poolPromise;
            if (session !== this.session) return;
            if (pool.broken) throw new Error("the detection workers stopped");
            this.pool = pool;
            this.delegate = pool.delegate;
            usePool = true;
          } catch (error) {
            if (session !== this.session) return;
            console.warn("detection workers unavailable, analysing on the page:", error);
            this.workersFailed = true;
            this._closePool();
          }
        }
        if (!usePool) {
          // The promise is kept, not the result: overlapping starts share one instance instead of leaking one.
          if (!this.holisticPromise) {
            const created = createHolistic().catch((error) => { if (this.holisticPromise === created) this.holisticPromise = null; throw error; });
            this.holisticPromise = created;
          }
          this.holistic = await this.holisticPromise;
          if (session !== this.session) return;
          this.delegate = this.holistic.delegate;
        }
      }
      if (session !== this.session) return;
      this.usePool = usePool;
      this.callbacks.status?.("Đang mở camera…");
      const opened = await this.camera.start(facing);
      if (session !== this.session) {
        if (this.camera.generation === opened) this.camera.stop(); // still ours: no newer start opened its own
        return;
      }
      const track = this.camera.stream?.getVideoTracks()[0];
      const lost = () => {
        if (session !== this.session) return;
        // An interrupted camera (another app took it a moment, the system paused it) is opened again, at most
        // REOPEN_LIMIT times within REOPEN_WINDOW_SECONDS; after that it is reported.
        const recent = (this.reopened ?? []).filter((stamp) => now() - stamp < REOPEN_WINDOW_SECONDS);
        if (recent.length >= REOPEN_LIMIT) {
          this._fail("Camera đã bị ngắt (ứng dụng khác dùng camera hoặc quyền bị thu hồi).");
          return;
        }
        this.reopened = [...recent, now()];
        const recording = this.takes.overlay().phase !== "idle"; // restart() cancels the takes
        this.callbacks.status?.("Camera bị ngắt — đang mở lại…");
        this.restart();
        if (recording) {
          this.callbacks.recordingFinished?.(); // the collect page gets its controls back
          this.callbacks.message?.("Camera bị ngắt giữa lúc ghi — đã dừng ghi, hãy bấm Bắt đầu ghi lại.");
        }
      };
      track?.addEventListener("ended", lost);
      if (track?.readyState === "ended") queueMicrotask(lost); // ended before the listener was added
    } catch (error) {
      if (session === this.session && !(error instanceof CameraAborted)) {
        this.session++;
        this.starting = false; // a failed start must not leave the pipeline "starting" (it would never start again)
        this.camera.stop();
        this.people?.close?.(); // nothing of this session is left for the next one (it may start without stop())
        this.people = null;
        this.enrollment = null;
        this.callbacks.status?.(null);
        throw error instanceof CameraError ? error : new Error(`Lỗi xử lý camera: ${error.message || error}`);
      }
      return; // superseded by a newer start / stop
    } finally {
      if (session === this.session) this.starting = false;
    }
    this.callbacks.status?.(null);
    this.viewContext.clearRect(0, 0, this.view.width, this.view.height);
    this.light.reset();
    if (this.usePool) this.pool.resetLight();
    this.previews.clear();
    this.recognition.reset();
    this.identityState = { last: null, rejection: -Infinity, taken: -Infinity };
    this.callbacks.started?.({ width: this.camera.width, height: this.camera.height, mirrored: this.camera.mirrored });
    if (this.identity && this.mode !== "enroll") this.callbacks.identity?.("searching");
    if (this.usePool) this._loopPool(session, this.pool);
    else this._loop(session);
  }

  stop() {
    this.session++;
    this.starting = false;
    this.camera.stop();
    this.viewContext.clearRect(0, 0, this.view.width, this.view.height); // the overlay of a camera no longer shown
    this.callbacks.stopped?.();
    this.takes.cancelRecording();
    this.people?.close?.();
    this.people = null;
    this.enrollment = null;
    this.callbacks.overlay?.({ countdown: 0 });
  }

  async _faceEngine() {
    if (!this.enginePromise) {
      const created = import("./faces.js").then(({ FaceEngine }) => FaceEngine.create())
        .catch((error) => { if (this.enginePromise === created) this.enginePromise = null; throw error; });
      this.enginePromise = created;
    }
    this.engine = await this.enginePromise;
    return this.engine;
  }

  async _createPeople(session) {
    this.peopleModule ??= await import("./people.js");
    const { People, MAX_OTHERS } = this.peopleModule;
    const instances = [];
    for (let i = 0; i <= MAX_OTHERS; i++) instances.push(await createHolistic());
    if (session !== this.session) {
      instances.forEach((instance) => instance.close());
      return;
    }
    this.delegate = instances[0].delegate;
    const rows = [];
    for (let i = 0; i < this.identity.length / 128; i++) rows.push(this.identity.subarray(i * 128, (i + 1) * 128));
    const lock = new PersonLock(this.engine, rows, PHONE_MATCH_THRESHOLD);
    const tools = {
      // tools.detect(holistic, view): view = lock.prepareRegion(frame, region) (crop, blanked faces, or blank).
      detect: (holistic, view) => holistic.detectView(view),
      extract: (results) => results.keypoints,
      remap: (keypoints, crop, width, height) => (crop ? remapKeypoints(keypoints, crop, width, height) : keypoints),
    };
    this.people = new People(lock, () => instances.shift(), tools, MAX_OTHERS, this.mode === "collect");
  }

  _closePool() {
    this.pool?.close();
    this.pool = null;
    this.poolPromise?.then((pool) => pool.close(), () => {});
    this.poolPromise = null;
  }

  _fail(message) {
    this.stop();
    this.callbacks.failed?.(message);
  }

  async _loop(session) {
    let lastFrame = now();
    let firstFrame = true;
    let rate = 0, previous = null;
    while (session === this.session) {
      const limit = firstFrame ? CAMERA_START_SECONDS : CAMERA_LOST_SECONDS;
      const stamp = await Promise.race([
        this.camera.nextFrame(),
        new Promise((resolve) => setTimeout(() => resolve(null), (limit - (now() - lastFrame)) * 1000 + 50)),
      ]);
      if (session !== this.session) return;
      if (stamp === null) {
        if (now() - lastFrame >= limit) {
          this._fail(firstFrame ? "Camera không gửi hình — hãy đóng ứng dụng khác đang dùng camera rồi thử lại."
            : "Mất tín hiệu camera — camera bị ngắt hoặc ứng dụng khác đang dùng.");
          return;
        }
        continue;
      }
      firstFrame = false;
      try {
        await this._process(stamp, session);
      } catch (error) {
        if (session === this.session) this._fail(`Lỗi xử lý camera: ${error.message || error}`);
        console.error(error);
        return;
      }
      lastFrame = now(); // the camera is lost when no frame comes while we wait, however long a frame took
      if (previous !== null && stamp > previous) rate = rate ? 0.9 * rate + 0.1 / (stamp - previous) : 1 / (stamp - previous);
      previous = stamp;
      this.frames++;
      this.fps = rate;
      if (this.frames % 5 === 0) this.callbacks.frame?.({ fps: rate, frames: this.frames });
    }
  }

  // Without a profile: camera frames go to idle detection workers; results are applied in capture order. A frame
  // that arrives while every worker is busy is skipped (the next one is newer).
  async _loopPool(session, pool) {
    const pending = [];
    let wake = null;
    let failures = 0; // frames in a row a worker could not analyse
    const fallBack = (reason) => {
      if (session !== this.session) return;
      console.warn("detection workers failed, analysing on the page:", reason);
      this.workersFailed = true; // the page's thread analyses from now on
      this._closePool();
      const recording = this.takes.overlay().phase !== "idle"; // restart() cancels the takes
      this.restart();
      if (recording) {
        this.callbacks.recordingFinished?.(); // the collect page gets its controls back
        this.callbacks.message?.("Bộ nhận dạng chạy nền dừng giữa lúc ghi — đã chuyển sang chạy trực tiếp, hãy bấm Bắt đầu ghi lại.");
      }
    };
    const consume = async () => {
      let rate = 0, previous = null;
      while (session === this.session) {
        if (!pending.length) {
          await new Promise((resolve) => { wake = resolve; });
          continue;
        }
        const job = pending[0];
        let result;
        try {
          result = await job.promise;
        } catch (error) {
          pending.shift();
          if (session !== this.session) return;
          if (error.frame && ++failures < FRAME_FAILURES) continue; // that frame failed; the next ones go on
          fallBack(error); // a worker crashed (memory) or keeps failing (its GPU context was lost)
          return;
        }
        failures = 0;
        pending.shift();
        if (session !== this.session) return;
        try {
          this._deliver(result, job);
        } catch (error) {
          if (session === this.session) this._fail(`Lỗi xử lý camera: ${error.message || error}`);
          console.error(error);
          return;
        }
        if (previous !== null && result.stamp > previous) {
          rate = rate ? 0.9 * rate + 0.1 / (result.stamp - previous) : 1 / (result.stamp - previous);
        }
        previous = result.stamp;
        this.frames++;
        this.fps = rate;
        if (this.frames % 5 === 0) this.callbacks.frame?.({ fps: rate, frames: this.frames });
      }
    };
    consume();
    let lastFrame = now();
    let firstFrame = true;
    let submittedStamp = -Infinity;
    while (session === this.session) {
      const limit = firstFrame ? CAMERA_START_SECONDS : CAMERA_LOST_SECONDS;
      const stamp = await Promise.race([
        this.camera.nextFrame(),
        new Promise((resolve) => setTimeout(() => resolve(null), (limit - (now() - lastFrame)) * 1000 + 50)),
      ]);
      if (session !== this.session) break;
      if (stamp === null) {
        if (now() - lastFrame >= limit) {
          this._fail(firstFrame ? "Camera không gửi hình — hãy đóng ứng dụng khác đang dùng camera rồi thử lại."
            : "Mất tín hiệu camera — camera bị ngắt hoặc ứng dụng khác đang dùng.");
          break;
        }
        continue;
      }
      firstFrame = false;
      lastFrame = now();
      if (pool.broken && !pending.length) { // a worker crashed between frames: nothing to wait for
        fallBack(new Error("the detection workers stopped"));
        break;
      }
      if (pool.broken || !pool.idle || stamp - submittedStamp < 1 / MAX_ANALYSIS_FPS - 0.004) {
        this.skipped = (this.skipped ?? 0) + 1;
        continue;
      }
      submittedStamp = stamp;
      const [width, height] = processSize(this.camera.width, this.camera.height);
      if (!width) continue;
      const started = performance.now();
      const image = this.camera.grab(this.work, width, height).getImageData(0, 0, width, height);
      this._time("grab", started);
      const promise = pool.submit({ width, height, buffer: image.data.buffer, stamp,
        light: { autoEnabled: this.settings.autoLight, bias: this.settings.bias } });
      pending.push({ promise, stamp, mirrored: this.camera.mirrored, submitted: performance.now() });
      wake?.();
      wake = null;
    }
    wake?.();
  }

  // One analysed frame from a worker: its landmarks over the live picture, its copy for samples, then the take or
  // the translator (the page's part of _process).
  _deliver(result, job) {
    const begun = performance.now();
    const { width, height, stamp } = result;
    if (this.view.width !== width || this.view.height !== height) {
      this.view.width = width;
      this.view.height = height;
    }
    const display = this.viewContext;
    display.clearRect(0, 0, width, height);
    const dense = this.settings.dense || this.mode === "collect";
    (dense ? drawDense : drawSimplified)(display, result.drawing, width, height, null, FACE_TESSELATION);
    this.previews.offer(stamp, width, height, (target, w, h) => {
      if (!this.corrected || this.corrected.width !== width || this.corrected.height !== height) this.corrected = makeCanvas(width, height);
      this.corrected.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(result.buffer), width, height), 0, 0);
      target.drawImage(this.corrected, 0, 0, w, h); // the corrected frame, as Holistic saw it
      target.drawImage(this.view, 0, 0, w, h); // and its landmarks
    });
    const keypoints = result.keypoints;
    if (this.mode === "collect") {
      const previews = this.previews;
      const info = { stamp, width, height, mirrored: job.mirrored, get preview() { return previews.nearest(this.stamp); } };
      this.takes.record(keypoints, info, stamp, width / height);
    } else if (this.mode === "translation") {
      this.recognition.push(stamp, keypoints, width, height);
    }
    if (this.frames % 15 === 0) this.callbacks.light?.(result.exposure, result.advice);
    this._overlay(display, width, height);
    const ema = (stage, value) => { this.perf[stage] = this.perf[stage] === undefined ? value : 0.9 * this.perf[stage] + 0.1 * value; };
    ema("light", result.ms.light);
    ema("holistic", result.ms.holistic);
    ema("latency", performance.now() - job.submitted);
    ema("deliver", performance.now() - begun);
  }

  _time(stage, started) {
    const now = performance.now();
    const spent = now - started;
    this.perf[stage] = this.perf[stage] === undefined ? spent : 0.9 * this.perf[stage] + 0.1 * spent;
    return now;
  }

  // One frame (CameraWorker.run's loop body).
  async _process(stamp, session) {
    const begun = performance.now();
    let mark = begun;
    const camera = this.camera;
    const [width, height] = processSize(camera.width, camera.height);
    if (!width || !height) return;
    const context = camera.grab(this.work, width, height);
    mark = this._time("grab", mark);
    let image = context.getImageData(0, 0, width, height);
    mark = this._time("read", mark);
    image = this.light.process(image, image);
    mark = this._time("light", mark);
    context.putImageData(image, 0, 0);
    mark = this._time("write", mark);
    const frame = { width, height, data: image.data, canvas: this.work, stamp };

    if (this.view.width !== width || this.view.height !== height) {
      this.view.width = width;
      this.view.height = height;
    }
    const display = this.viewContext;
    display.clearRect(0, 0, width, height); // the camera's own picture is under this canvas: only landmarks are drawn

    let keypoints;
    let region = null, faces = null;
    if (this.enrollment) {
      faces = await this.engine.detect(frame);
      if (session !== this.session) return;
      const instruction = await this.enrollment.offerAsync(this.engine, frame, faces);
      if (session !== this.session) return;
      this.callbacks.enrollProgress?.(instruction, this.enrollment.progressText(), this.enrollment.embeddings.length);
      if (this.enrollment.done && !this.enrollDone) {
        this.enrollDone = true;
        this.callbacks.enrollFinished?.(this.enrollment);
      }
      keypoints = null;
    } else if (this.people) {
      // Only the registered person's own landmarks are kept; nobody else's, ever.
      mark = performance.now();
      const detected = await this.engine.detect(frame); // what lock.locate would ask for (timed apart)
      if (session !== this.session) return;
      mark = this._time("faces", mark);
      const scene = await this.people.process(frame, stamp, detected);
      mark = this._time("people", mark);
      if (session !== this.session) return;
      region = scene.region;
      keypoints = scene.keypoints;
      if (region.skip || ALTERED_REASONS.includes(scene.reason)) {
        this.takes.markAltered(stamp); // before the take or the translator sees this frame (one list, like the PC's)
      }
      const ids = this.identityState;
      if (scene.signer && scene.signer[2].accepted) ids.taken = stamp;
      if (REJECTION_REASONS.includes(scene.reason)) ids.rejection = stamp;
      let status = scene.status;
      if (!region.skip && stamp - ids.rejection <= REJECTION_SHOW_SECONDS) {
        status = stamp - ids.taken > BLOCKED_SECONDS ? "blocked" : "mixed";
      }
      if (status !== ids.last) {
        ids.last = status;
        this.callbacks.identity?.(status);
      }
      this._drawScene(display, scene, width, height);
    } else {
      mark = performance.now();
      const result = this.holistic.run(this.work, width, height, stamp);
      mark = this._time("holistic", mark);
      keypoints = result.keypoints;
      const dense = this.settings.dense || this.mode === "collect";
      (dense ? drawDense : drawSimplified)(display, result.drawing, width, height, null, FACE_TESSELATION);
      mark = this._time("draw", mark);
    }
    drawIdentity(display, region, this.enrollment ? faces : null);
    mark = performance.now();
    this.previews.offer(stamp, width, height, (target, w, h) => {
      target.drawImage(this.work, 0, 0, w, h); // the corrected frame, as Holistic saw it
      target.drawImage(this.view, 0, 0, w, h); // and its landmarks
    });
    mark = this._time("preview", mark);

    if (keypoints) {
      if (this.mode === "collect") {
        const previews = this.previews;
        const info = { stamp, width, height, mirrored: camera.mirrored, get preview() { return previews.nearest(this.stamp); } };
        this.takes.record(keypoints, info, stamp, width / height);
      } else if (this.mode === "translation") {
        this.recognition.push(stamp, keypoints, width, height);
      }
    }
    if (this.frames % 15 === 0) this.callbacks.light?.(this.light.effectiveExposureEv, this.light.advice);
    this._overlay(display, width, height);
    this._time("total", begun);
  }

  // The registered person's landmarks in colour; other people and every rejected part in grey.
  _drawScene(display, scene, width, height) {
    const { keptResults, rejectedResults } = this.peopleModule ?? {};
    for (const [results, crop] of scene.others ?? []) if (results?.drawing) drawMuted(display, results.drawing, width, height, crop);
    if (!scene.signer) return;
    const [results, crop, verdict] = scene.signer;
    if (!results) return;
    const rejected = rejectedResults ? rejectedResults(results, verdict) : [null, null];
    if (rejected[0]) drawMuted(display, rejected[0].drawing ?? rejected[0], width, height, crop, rejected[1]);
    const kept = keptResults ? keptResults(results, verdict) : (verdict.accepted ? results : null);
    if (kept) {
      const dense = this.settings.dense || this.mode === "collect";
      (dense ? drawDense : drawSimplified)(display, kept.drawing ?? kept, width, height, crop, FACE_TESSELATION);
    }
  }

  // Countdown and REC indicator (CameraWorker._overlay), drawn after the preview copy.
  _overlay(display, width, height) {
    const take = this.takes.overlay();
    this.callbacks.overlay?.({ countdown: take.phase === "countdown" ? take.remaining : 0 });
    if (take.phase !== "recording") return;
    const color = take.state === "signing" ? "rgb(60,200,80)" : "rgb(230,40,40)"; // green once the sign is seen
    display.fillStyle = color;
    display.beginPath();
    display.arc(28, 28, 11, 0, Math.PI * 2);
    display.fill();
    display.font = "bold 22px system-ui, sans-serif";
    display.fillText(`REC ${Math.max(take.recorded, 0).toFixed(1)}s`, 48, 36);
  }

  // A finished sign from the recognition worker: its picture and whether a frame of it was altered (decided now,
  // while the marks of its frames are kept).
  _signFinished(sign) {
    const complete = {
      ...sign, mirrored: this.camera.mirrored, preview: this.previews.nearest(sign.middle),
      altered: this.takes.alteredWithin(sign.range[0], sign.range[1]),
    };
    this.callbacks.signFinished?.(complete);
  }
}
