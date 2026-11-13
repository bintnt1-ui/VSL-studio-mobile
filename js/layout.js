// Holistic keypoint layout shared by every capture and recognition path (port of landmarks.py).
//
// A frame is a Float32Array of KEYPOINT_SIZE values: pose 33 x (x, y, z, visibility), face 468 x (x, y, z),
// left hand 21 x (x, y, z) and right hand 21 x (x, y, z). Missing parts are zeros. A sequence is an array of
// frames. Coordinates are normalised to the whole frame (x / width, y / height); z shares x's scale.

export const POSE = [0, 132];
export const FACE = [132, 1536];
export const LEFT_HAND = [1536, 1599];
export const RIGHT_HAND = [1599, 1662];
export const KEYPOINT_SIZE = 1662;
export const PARTS = [[POSE, 4], [FACE, 3], [LEFT_HAND, 3], [RIGHT_HAND, 3]];

// Anatomical left/right pairs used when mirroring a sequence.
export const POSE_MIRROR_PAIRS = [
  [1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
  [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32],
];
// Face mesh pairs among the landmarks that recognition reads (eyes, mouth, brows).
export const FACE_MIRROR_PAIRS = [[33, 263], [61, 291], [105, 334]];
// The only face landmarks recognition reads (translate.gesture_features): the data bundle keeps these.
export const FACE_POINTS_USED = [1, 13, 14, 33, 61, 105, 263, 291, 334];

export const POSE_SKELETON = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28],
];
export const HAND_SKELETON = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
export const FACE_OUTLINE_POINTS = [1, 33, 263, 362, 61, 291, 199];

export function emptyFrame() {
  return new Float32Array(KEYPOINT_SIZE);
}

// Whether a part [start, end) of a frame holds any detected value.
export function present(frame, part) {
  for (let i = part[0]; i < part[1]; i++) if (Math.abs(frame[i]) > 1e-6) return true;
  return false;
}

// Keypoints detected in a crop [x0, y0, x1, y1] (pixels) to whole-frame coordinates (remap_keypoints).
export function remapKeypoints(keypoints, crop, width, height) {
  const values = Float32Array.from(keypoints);
  if (!crop) return values;
  const [x0, y0, x1, y1] = crop;
  const scaleX = (x1 - x0) / width, scaleY = (y1 - y0) / height;
  for (const [part, columns] of PARTS) {
    if (!present(values, part)) continue;
    for (let i = part[0]; i < part[1]; i += columns) {
      values[i] = values[i] * scaleX + x0 / width;
      values[i + 1] = values[i + 1] * scaleY + y0 / height;
      values[i + 2] = values[i + 2] * scaleX;
    }
  }
  return values;
}

function mirrorBlock(frames, part, columns, pairs, sourcePart = part) {
  // Returns per-frame mirrored copies of `sourcePart` laid out for `part` (x -> 1 - x where detected).
  return frames.map((frame) => {
    const count = (sourcePart[1] - sourcePart[0]) / columns;
    const points = frame.slice(sourcePart[0], sourcePart[1]);
    let any = false;
    for (let i = 0; i < points.length; i++) if (Math.abs(points[i]) > 1e-6) { any = true; break; }
    if (any) for (let p = 0; p < count; p++) points[p * columns] = 1 - points[p * columns];
    for (const [left, right] of pairs) {
      for (let c = 0; c < columns; c++) {
        const a = left * columns + c, b = right * columns + c;
        const swap = points[a]; points[a] = points[b]; points[b] = swap;
      }
    }
    return points;
  });
}

// The sequence as it would look in a horizontally flipped image (mirror_keypoints): exact for pose, both
// hands and the face landmarks recognition reads.
export function mirrorKeypoints(frames) {
  const pose = mirrorBlock(frames, POSE, 4, POSE_MIRROR_PAIRS);
  const face = mirrorBlock(frames, FACE, 3, FACE_MIRROR_PAIRS);
  const left = mirrorBlock(frames, LEFT_HAND, 3, [], RIGHT_HAND);
  const right = mirrorBlock(frames, RIGHT_HAND, 3, [], LEFT_HAND);
  return frames.map((_, i) => {
    const out = new Float32Array(KEYPOINT_SIZE);
    out.set(pose[i], POSE[0]);
    out.set(face[i], FACE[0]);
    out.set(left[i], LEFT_HAND[0]);
    out.set(right[i], RIGHT_HAND[0]);
    return out;
  });
}
