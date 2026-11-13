// Landmark overlays on a 2D canvas (port of the drawing in landmarks.py and CameraWorker._draw_identity /
// _draw_scene). Colours are vsl_ui's (given there as BGR). Drawings are {pose, face, left, right} landmark lists
// normalised to the image Holistic saw: `crop` [x0, y0, x1, y1] (pixels) places them in the frame.

import { FACE_OUTLINE_POINTS, HAND_SKELETON, POSE_SKELETON } from "./layout.js";

export const COLORS = {
  pose: "rgb(255,190,60)", left: "rgb(120,220,80)", right: "rgb(70,170,255)", face: "rgb(255,120,210)",
  muted: "rgb(150,150,150)", dense: "rgb(255,0,0)", denseLine: "rgb(224,224,224)",
  locked: "rgb(120,220,80)", held: "rgb(255,190,60)", crop: "rgb(80,200,255)",
};

function placer(crop, width, height) {
  if (!crop) return (p) => [p.x * width, p.y * height];
  const [x0, y0, x1, y1] = crop;
  return (p) => [x0 + p.x * (x1 - x0), y0 + p.y * (y1 - y0)];
}

function skeleton(context, points, connections, color, radius, place, only = null) {
  if (!points?.length) return;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2;
  context.beginPath();
  for (const [a, b] of connections) {
    if (only && (!only.includes(a) || !only.includes(b))) continue;
    if (!points[a] || !points[b]) continue;
    const [ax, ay] = place(points[a]), [bx, by] = place(points[b]);
    context.moveTo(ax, ay);
    context.lineTo(bx, by);
  }
  context.stroke();
  const indices = only ?? points.map((_, i) => i);
  for (const i of indices) {
    if (!points[i]) continue;
    const [x, y] = place(points[i]);
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
  }
}

// Readable sparse overlay used while translating (landmarks.draw_simplified).
export function drawSimplified(context, drawing, width, height, crop = null) {
  const place = placer(crop, width, height);
  skeleton(context, drawing.pose, POSE_SKELETON, COLORS.pose, 4, place);
  skeleton(context, drawing.left, HAND_SKELETON, COLORS.left, 2, place);
  skeleton(context, drawing.right, HAND_SKELETON, COLORS.right, 2, place);
  if (drawing.face?.length) skeleton(context, drawing.face, [], COLORS.face, 3, place, FACE_OUTLINE_POINTS);
}

// Full-density overlay for collecting and the Holistic check (landmarks.draw_dense): `tesselation` is the face
// mesh connection list (holistic.js FACE_TESSELATION).
export function drawDense(context, drawing, width, height, crop = null, tesselation = []) {
  const place = placer(crop, width, height);
  const pose = drawing.pose?.map((p) => ((p.visibility ?? 1) < 0.5 ? null : p)); // like drawing_utils
  skeleton(context, pose, POSE_SKELETON, COLORS.denseLine, 2, place);
  skeleton(context, pose, [], COLORS.dense, 2, place, pose ? pose.map((p, i) => (p ? i : -1)).filter((i) => i >= 0) : null);
  if (drawing.face?.length) {
    context.strokeStyle = COLORS.denseLine;
    context.lineWidth = 0.5;
    context.beginPath();
    for (const { start, end } of tesselation) {
      const a = drawing.face[start], b = drawing.face[end];
      if (!a || !b) continue;
      const [ax, ay] = place(a), [bx, by] = place(b);
      context.moveTo(ax, ay);
      context.lineTo(bx, by);
    }
    context.stroke();
  }
  for (const hand of [drawing.left, drawing.right]) {
    skeleton(context, hand, HAND_SKELETON, COLORS.denseLine, 2, place);
    skeleton(context, hand, [], COLORS.dense, 2, place, hand ? hand.map((_, i) => i) : null);
  }
}

// Grey sparse overlay of landmarks seen but not used (other people, rejected parts); `posePoints`: only these
// pose landmarks (null: the whole pose).
export function drawMuted(context, drawing, width, height, crop = null, posePoints = null) {
  const place = placer(crop, width, height);
  skeleton(context, drawing.pose, POSE_SKELETON, COLORS.muted, 3, place, posePoints);
  skeleton(context, drawing.left, HAND_SKELETON, COLORS.muted, 2, place);
  skeleton(context, drawing.right, HAND_SKELETON, COLORS.muted, 2, place);
  if (drawing.face?.length) skeleton(context, drawing.face, [], COLORS.muted, 2, place, FACE_OUTLINE_POINTS);
}

// Face boxes and the crop (CameraWorker._draw_identity): green = the registered person, orange = where they are
// held, grey crossed = other people.
export function drawIdentity(context, region, enrollmentFaces = null) {
  context.lineWidth = 2;
  if (enrollmentFaces) {
    context.strokeStyle = COLORS.locked;
    for (const face of enrollmentFaces) context.strokeRect(...face.box);
    return;
  }
  if (!region) return;
  context.strokeStyle = COLORS.muted;
  for (const [x, y, w, h] of region.others ?? []) {
    context.strokeRect(x, y, w, h);
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + w, y + h);
    context.stroke();
  }
  if (region.target) {
    context.lineWidth = 3;
    context.strokeStyle = region.status === "locked" ? COLORS.locked : COLORS.held;
    context.strokeRect(...region.target);
  }
  if (region.crop) {
    const [x0, y0, x1, y1] = region.crop;
    context.lineWidth = 1;
    context.strokeStyle = COLORS.crop;
    context.strokeRect(x0, y0, x1 - x0 - 1, y1 - y0 - 1);
  }
}
