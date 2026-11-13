// MediaPipe Tasks Vision HolisticLandmarker -> KEYPOINT_SIZE frames (the phone's counterpart of
// landmarks.create_holistic / detect / extract_keypoints). Each instance keeps its own tracking state, like a
// Python Holistic object: people.People gives every person their own instance.

import { FilesetResolver, HolisticLandmarker } from "../vendor/tasks-vision/vision_bundle.mjs";
import { FACE, KEYPOINT_SIZE, LEFT_HAND, POSE, RIGHT_HAND } from "./layout.js";

const BASE = new URL("../", import.meta.url);
// Face mesh connections for the dense drawing (mp_holistic.FACEMESH_TESSELATION): [{start, end}, ...].
export const FACE_TESSELATION = HolisticLandmarker.FACE_LANDMARKS_TESSELATION ?? [];
let filesetPromise = null;
let preferredDelegate = "GPU";

function fileset() {
  filesetPromise ??= FilesetResolver.forVisionTasks(new URL("vendor/tasks-vision/wasm", BASE).href);
  return filesetPromise;
}

// Same thresholds as landmarks.create_holistic (min_detection_confidence / min_tracking_confidence 0.5).
const OPTIONS = {
  runningMode: "VIDEO",
  minPoseDetectionConfidence: 0.5,
  minPosePresenceConfidence: 0.5,
  minFaceDetectionConfidence: 0.5,
  minFacePresenceConfidence: 0.5,
  minHandLandmarksConfidence: 0.5,
  outputFaceBlendshapes: false,
  outputPoseSegmentationMasks: false,
};

// "GPU" (default; falls back to CPU by itself) or "CPU" for every later instance (tests: software GPUs are slow).
export function preferredHolisticDelegate() {
  return preferredDelegate;
}

export function preferDelegate(delegate) {
  preferredDelegate = delegate === "CPU" ? "CPU" : "GPU";
}

// A Holistic instance (GPU when the browser allows it, else CPU). `delegate`: "GPU" | "CPU" to force one.
export async function createHolistic(delegate = preferredDelegate) {
  const vision = await fileset();
  const modelAssetPath = new URL("models/holistic_landmarker.task", BASE).href;
  try {
    const landmarker = await HolisticLandmarker.createFromOptions(vision, { ...OPTIONS, baseOptions: { modelAssetPath, delegate } });
    return new Holistic(landmarker, delegate);
  } catch (error) {
    if (delegate !== "GPU") throw error;
    preferredDelegate = "CPU"; // this device has no usable GPU path: every later instance goes straight to CPU
    const landmarker = await HolisticLandmarker.createFromOptions(vision, { ...OPTIONS, baseOptions: { modelAssetPath, delegate: "CPU" } });
    return new Holistic(landmarker, "CPU");
  }
}

// HolisticLandmarkerResult -> Float32Array(1662) laid out like extract_keypoints (the face mesh's 468 points:
// the model's 10 iris points are left out).
export function toKeypoints(result) {
  const values = new Float32Array(KEYPOINT_SIZE);
  const pose = result?.poseLandmarks?.[0];
  if (pose?.length) {
    for (let i = 0; i < 33 && i < pose.length; i++) {
      const at = POSE[0] + i * 4;
      values[at] = pose[i].x; values[at + 1] = pose[i].y; values[at + 2] = pose[i].z;
      values[at + 3] = pose[i].visibility ?? 0;
    }
  }
  const face = result?.faceLandmarks?.[0];
  if (face?.length) {
    for (let i = 0; i < 468 && i < face.length; i++) {
      const at = FACE[0] + i * 3;
      values[at] = face[i].x; values[at + 1] = face[i].y; values[at + 2] = face[i].z;
    }
  }
  for (const [part, hand] of [[LEFT_HAND, result?.leftHandLandmarks?.[0]], [RIGHT_HAND, result?.rightHandLandmarks?.[0]]]) {
    if (!hand?.length) continue;
    for (let i = 0; i < 21; i++) {
      const at = part[0] + i * 3;
      values[at] = hand[i].x; values[at + 1] = hand[i].y; values[at + 2] = hand[i].z;
    }
  }
  return values;
}

// A Holistic for single random frames (collect_window's scrubbing: collectors.create_static_holistic): no tracking
// state between calls, so jumping around a video never mixes two moments. detect(source, width, height, crop) ->
// {keypoints (whole frame), drawing (in the crop), crop}.
export async function createStaticHolistic(delegate = preferredDelegate) {
  const vision = await fileset();
  const modelAssetPath = new URL("models/holistic_landmarker.task", BASE).href;
  const options = { ...OPTIONS, runningMode: "IMAGE" };
  let landmarker;
  try {
    landmarker = await HolisticLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath, delegate } });
  } catch (error) {
    if (delegate !== "GPU") throw error;
    landmarker = await HolisticLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath, delegate: "CPU" } });
  }
  return new StaticHolistic(landmarker);
}

export class StaticHolistic {
  constructor(landmarker) {
    this.landmarker = landmarker;
    this.view = null;
  }

  detect(source, width, height, crop = null) {
    let input = source;
    if (crop) {
      const [x0, y0, x1, y1] = crop;
      if (!this.view || this.view.width !== x1 - x0 || this.view.height !== y1 - y0) this.view = makeCanvas(x1 - x0, y1 - y0);
      this.view.getContext("2d").drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0);
      input = this.view;
    }
    const result = this.landmarker.detect(input);
    const local = toKeypoints(result);
    return { keypoints: crop ? remap(local, crop, width, height) : local, drawing: toDrawing(result), crop };
  }

  close() {
    this.landmarker?.close();
    this.landmarker = null;
  }
}

// The same landmarks as plain objects for drawing ({pose, face, left, right}: arrays of {x, y, z, visibility}),
// normalised to the image Holistic saw.
export function toDrawing(result) {
  return {
    pose: result?.poseLandmarks?.[0] ?? null,
    face: result?.faceLandmarks?.[0] ?? null,
    left: result?.leftHandLandmarks?.[0] ?? null,
    right: result?.rightHandLandmarks?.[0] ?? null,
  };
}

function makeCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  return canvas;
}

export class Holistic {
  constructor(landmarker, delegate) {
    this.landmarker = landmarker;
    this.delegate = delegate;
    this.lastTimestamp = 0;
    this.view = null; // canvas for crops / masks / blank resets
  }

  // MediaPipe needs strictly increasing timestamps per instance (milliseconds).
  _timestamp(stampSeconds) {
    const wanted = Math.round(stampSeconds * 1000);
    this.lastTimestamp = Math.max(wanted, this.lastTimestamp + 1);
    return this.lastTimestamp;
  }

  get tesselation() {
    return FACE_TESSELATION;
  }

  _canvas(width, height) {
    if (!this.view || this.view.width !== width || this.view.height !== height) this.view = makeCanvas(width, height);
    return this.view;
  }

  // Run on `source` (a canvas / video / image of size width x height) or on the crop [x0, y0, x1, y1] of it with
  // `masks` (face boxes [x, y, w, h], padded by a quarter like identity.prepare_region) blanked. Returns
  // { keypoints: whole-frame Float32Array(1662), drawing (in the crop's coordinates), crop }.
  run(source, width, height, stampSeconds, crop = null, masks = []) {
    let input = source;
    if (crop || masks.length) {
      const [x0, y0, x1, y1] = crop ?? [0, 0, width, height];
      const canvas = this._canvas(x1 - x0, y1 - y0);
      const context = canvas.getContext("2d");
      context.drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0);
      context.fillStyle = "#000";
      for (const [x, y, w, h] of masks) {
        const padX = Math.trunc(0.25 * w), padY = Math.trunc(0.25 * h);
        const left = Math.max(0, x - padX - x0), top = Math.max(0, y - padY - y0);
        const right = Math.min(x1, x + w + padX) - x0, bottom = Math.min(y1, y + h + padY) - y0;
        if (right > left && bottom > top) context.fillRect(left, top, right - left, bottom - top);
      }
      input = canvas;
    }
    const result = this.landmarker.detectForVideo(input, this._timestamp(stampSeconds));
    const local = toKeypoints(result);
    const keypoints = crop ? remap(local, crop, width, height) : local;
    return { keypoints, drawing: toDrawing(result), crop };
  }

  // people.js's tools.detect: `view` is lock.prepareRegion(frame, region) — {frame: {canvas, width, height, stamp},
  // crop: [x0, y0, x1, y1] | null, width, height, blanks: [[left, top, right, bottom]] in view pixels, blank} — or a
  // blank image of width x height (blank: true, a tracker reset). Returns {keypoints: Float32Array(1662) in the
  // view's coordinates, drawing}.
  detectView(view) {
    const stamp = view.frame?.stamp ?? performance.now() / 1000;
    const width = Math.max(1, Math.round(view.width)), height = Math.max(1, Math.round(view.height));
    let input;
    if (view.blank || view.crop || view.blanks?.length) {
      const canvas = this._canvas(width, height);
      const context = canvas.getContext("2d");
      context.fillStyle = "#000";
      if (view.blank) {
        context.fillRect(0, 0, width, height);
      } else {
        const [x0, y0] = view.crop ?? [0, 0];
        context.drawImage(view.frame.canvas, x0, y0, width, height, 0, 0, width, height);
        for (const [left, top, right, bottom] of view.blanks ?? []) context.fillRect(left, top, right - left, bottom - top);
      }
      input = canvas;
    } else {
      input = view.frame.canvas;
    }
    const result = this.landmarker.detectForVideo(input, this._timestamp(stamp));
    return { keypoints: toKeypoints(result), drawing: toDrawing(result) };
  }

  // Forget the tracked body (a blank image clears MediaPipe's tracking), like people.Tracker.reset.
  reset(width, height, stampSeconds) {
    const canvas = this._canvas(width, height);
    const context = canvas.getContext("2d");
    context.fillStyle = "#000";
    context.fillRect(0, 0, width, height);
    this.landmarker.detectForVideo(canvas, this._timestamp(stampSeconds));
  }

  close() {
    this.landmarker?.close();
    this.landmarker = null;
  }
}

function remap(values, crop, width, height) {
  // Crop -> whole frame (landmarks.remap_keypoints), kept local to avoid a copy when no crop is used.
  const [x0, y0, x1, y1] = crop;
  const scaleX = (x1 - x0) / width, scaleY = (y1 - y0) / height;
  for (const [part, columns] of [[POSE, 4], [FACE, 3], [LEFT_HAND, 3], [RIGHT_HAND, 3]]) {
    let any = false;
    for (let i = part[0]; i < part[1]; i++) if (Math.abs(values[i]) > 1e-6) { any = true; break; }
    if (!any) continue;
    for (let i = part[0]; i < part[1]; i += columns) {
      values[i] = values[i] * scaleX + x0 / width;
      values[i + 1] = values[i + 1] * scaleY + y0 / height;
      values[i + 2] = values[i + 2] * scaleX;
    }
  }
  return values;
}
