// Several people in view: each gets their own Holistic, only the registered signer's landmarks are used
// (port of people.py).
//
// lock.PersonLock says which face is the logged-in person. Holistic follows one body at a time and, with
// someone close by, can join that person's arm or hand to the signer's skeleton, or follow them entirely. So
// while a profile is logged in every visible person gets their own Holistic tracker (a crop around their
// face, the other faces blanked; the people nearest the signer first), and the signer's skeleton is judged
// against the others before anything is recorded or translated (`judge`):
// * its head must be the verified face and its shoulders as wide as a moment ago; while that face is hidden
//   its head must stay at the held face and its shoulders where the accepted body was a moment ago and near
//   where the face was last verified (body continuity), else nothing is taken, and after CHAIN_SECONDS
//   without such a body the face has to be verified again;
// * each arm must be anatomically possible and each hand must sit at its arm's wrist (in shoulder widths of
//   the locked body);
// * a hand another person's skeleton also holds (their hand or pose wrist on it) belongs to whoever held it
//   a moment ago (`HandMemory`); else to the person whose shoulder is nearer (the signer needs
//   OWNERSHIP_MARGIN less). Someone whose tracker gives nothing for a frame still contests with their latest
//   skeleton (a stand-in).
// A wrong head or body takes nothing. An arm reaching into someone else's hand (or an impossible arm) is put
// down at rest, unseen, its hand dropped (`strict`, recorded samples: the whole frame is dropped); both arms
// so: nothing. A hand away from its wrist, a hand someone else holds and a face mesh away from the verified
// face are dropped alone. Nothing is accepted before the face has been verified.
//
// Holistic is injected: People(lock, create, tools, maxOthers, strict) with create() -> a built Holistic instance
// (synchronously; optional close()), tools.detect(holistic, view) -> results (synchronous; view:
// lock.prepareRegion(frame, region) plus `stamp`, or for a reset {frame, crop: null, masks: [], width, height,
// blanks: [], blank: true, stamp}: a black image that clears the tracking), tools.extract(results) ->
// Float32Array(1662) in view coordinates, tools.remap(keypoints, crop, frameWidth, frameHeight) -> whole-frame
// keypoints (layout.remapKeypoints). results = {keypoints, drawing: {pose, face, left, right}} (holistic.js).
// People.process is async only because the face lock awaits SFace (ONNX Runtime Web); one call at a time.
// Frames are objects with width / height (pipeline: {width, height, data, canvas, stamp}). Points are [x, y]
// frame pixels; a body's pose is a Float64Array(66), its hands Float64Array(42) (or null), its face mesh
// Float64Array(936) (or null).

import { FACE, KEYPOINT_SIZE, LEFT_HAND, POSE, RIGHT_HAND } from "./layout.js";
import { Region, _iou, prepareRegion, steadyCrop } from "./lock.js";

// Arm limits in shoulder widths. Webcam takes never exceed upper arm 1.08, forearm 1.21, shoulder-to-wrist
// 1.80; the hand's wrist lies within 0.63 of the pose wrist in 99.99% of the stored frames (max 0.74).
export const MAX_UPPER_ARM = 1.3;
export const MAX_FOREARM = 1.4;
export const MAX_REACH = 2.1;
export const MAX_HAND_OFFSET = 0.8;
export const HAND_PER_SHOULDER = 0.25; // wrist -> middle knuckle in shoulder widths (stored frames: median 0.24)
export const CHECKED_VISIBILITY = 0.5; // an arm is judged when its hand is detected or its pose wrist this visible
export const REFERENCE_SCALE = 0.8; // turned shoulders look narrow: limits use at least this much of the locked width...
export const WIDTH_SHRINK = 0.02; // ... which follows narrower shoulders only this much per frame
export const HEAD_MARGIN = 0.5; // the skeleton's nose may lie this many face-box sizes outside the verified face box
export const HELD_MARGIN = 1.0; // ... and this many outside the held face box while the face is hidden
export const FACE_TO_NOSE = 0.6; // face mesh centre to pose nose, shoulder widths (stored frames: under 0.3)
export const BODY_SHIFT = 0.5; // hidden face: shoulders may move this many shoulder widths between accepted frames...
export const BODY_WIDTH_RANGE = Object.freeze([0.6, 1.6]); // ... and their width change by this factor...
export const BODY_DRIFT = 1.0; // ... and stay this many shoulder widths from where the face was last verified...
export const CHAIN_SECONDS = 0.4; // ... at most this long apart, else the body is no longer known to be the person
export const WIDTH_RESET_SECONDS = 5.0; // the usual shoulder width is measured afresh after this long without the person
// shoulder width / face mesh width: stored frames median 2.08, never above 2.49; a head turned far to the side
// shows a narrower face (3.0 at 70 degrees); a skeleton joined to a neighbour's shoulder ~4
export const SHOULDERS_PER_FACE = 3.2;
export const MIN_SHOULDERS_PER_FACE = 0.7; // stored frames: above 0.97 in 99.9% (a body turned sideways); collapsed: ~0.3
export const SAME_HAND = 1.0; // two skeletons hold the same hand when its centres are within this many hand lengths
export const OWNERSHIP_MARGIN = 0.15; // the signer keeps a shared hand when its shoulder is this much nearer
export const MEMORY_SECONDS = 0.6; // a hand stays with whoever held it without a contest this recently...
export const MEMORY_HAND_LENGTHS = 2.0; // ... within this many hand lengths of where it was
export const MAX_OTHERS = 2; // other people with their own tracker (more are only blanked)
export const OTHER_TIMEOUT_SECONDS = 1.0; // a tracker whose person has not been seen this long is free again
export const RESET_AFTER_SECONDS = 0.5; // a tracker idle this long starts afresh (its tracking state is stale)
export const CROP_JUMP_IOU = 0.7; // a tracker whose input region moved more than this starts afresh
export const REST_UPPER_ARM = 0.8, REST_FOREARM = 0.65, REST_HAND = 0.15; // an arm put down at rest, shoulder widths

export const SIDES = Object.freeze([[11, 13, 15], [12, 14, 16]]); // shoulder, elbow, wrist of the left / right hand's arm
export const ARM_POINTS = Object.freeze([[13, 15, 17, 19, 21], [14, 16, 18, 20, 22]]); // elbow, wrist, pinky, index, thumb
export const HAND_SLICES = Object.freeze([LEFT_HAND, RIGHT_HAND]);
export const REJECTED = Object.freeze(["none", "head", "body", "arm"]); // verdicts that take nothing from the frame

const hypot = Math.hypot;

function _center(box) {
  return [box[0] + box[2] / 2, box[1] + box[3] / 2];
}

export function _inside(point, box, margin = 0.0) {
  const [x, y, w, h] = box;
  return x - margin * w <= point[0] && point[0] <= x + w + margin * w
    && y - margin * h <= point[1] && point[1] <= y + h + margin * h;
}

function sameBox(first, second) {
  return first[0] === second[0] && first[1] === second[1] && first[2] === second[2] && first[3] === second[3];
}

// A reference shoulder width: back up quickly (facing the camera again, closer), down slowly (narrower
// shoulders are mostly a turned body, whose arms are not shorter).
export function _growOrShrink(width, measured) {
  if (width <= 0) return measured;
  if (measured > width) return width + 0.5 * (measured - width);
  return width + WIDTH_SHRINK * (measured - width);
}

// 2 x count pixel block of a part (x, y of each point), or null when the part is all zeros (x, y, z).
function pixelBlock(keypoints, start, count, width, height) {
  let any = false;
  for (let i = start, end = start + 3 * count; i < end; i++) {
    if (Math.abs(keypoints[i]) > 1e-6) { any = true; break; }
  }
  if (!any) return null;
  const out = new Float64Array(2 * count);
  for (let p = 0; p < count; p++) {
    out[2 * p] = keypoints[start + 3 * p] * width;
    out[2 * p + 1] = keypoints[start + 3 * p + 1] * height;
  }
  return out;
}

// One Holistic skeleton in frame pixels (`keypoints`: whole-frame normalised, zeros where missing).
// `reference`: the person's usual shoulder width (their tracker's), 0 when unknown; `owner`: which tracker
// found it (tells other people's hands apart in HandMemory); `standIn`: an earlier skeleton standing in for
// someone their tracker missed now.
export class Body {
  constructor(keypoints, size, reference = 0.0, owner = null) {
    this.keypoints = keypoints instanceof Float32Array ? keypoints : Float32Array.from(keypoints);
    const width = size[0], height = size[1];
    this.size = [width, height];
    this.reference = reference;
    this.owner = owner;
    this.standIn = false;
    const kp = this.keypoints;
    const pose = this.pose = new Float64Array(66);
    const visibility = this.visibility = new Float64Array(33);
    let hasPose = false;
    for (let i = 0; i < 33; i++) {
      const x = kp[4 * i], y = kp[4 * i + 1];
      if (Math.abs(x) > 1e-6 || Math.abs(y) > 1e-6) hasPose = true;
      pose[2 * i] = x * width;
      pose[2 * i + 1] = y * height;
      visibility[i] = kp[4 * i + 3];
    }
    this.hasPose = hasPose;
    this.hands = [pixelBlock(kp, LEFT_HAND[0], 21, width, height), pixelBlock(kp, RIGHT_HAND[0], 21, width, height)];
    this.face = pixelBlock(kp, FACE[0], 468, width, height);
  }

  point(index) {
    return [this.pose[2 * index], this.pose[2 * index + 1]];
  }

  get nose() {
    return this.point(0);
  }

  get shoulderCenter() {
    const p = this.pose;
    return [(p[22] + p[24]) / 2, (p[23] + p[25]) / 2];
  }

  get shoulderWidth() {
    const p = this.pose;
    return hypot(p[22] - p[24], p[23] - p[25]);
  }

  // Shoulder width that a body turned sideways does not shrink (with a known `reference`).
  get scale() {
    return Math.max(this.shoulderWidth, REFERENCE_SCALE * this.reference, 1.0);
  }
}

function distance(first, second) {
  return hypot(first[0] - second[0], first[1] - second[1]);
}

// Two skeletons of one body: shoulders in the same place and as wide.
export function _sameBody(first, second) {
  if (!(first.hasPose && second.hasPose)) return false;
  const width = Math.max(first.shoulderWidth, second.shoulderWidth, 1.0);
  const ratio = first.shoulderWidth / Math.max(second.shoulderWidth, 1.0);
  return distance(first.shoulderCenter, second.shoulderCenter) <= BODY_SHIFT * width
    && BODY_WIDTH_RANGE[0] <= ratio && ratio <= BODY_WIDTH_RANGE[1];
}

// Wrist -> middle knuckle of a hand (Float64Array(42) pixels), at least 4 px.
export function _handLength(hand) {
  return Math.max(hypot(hand[18] - hand[0], hand[19] - hand[1]), 4.0);
}

// Centre of a hand's 21 points (numpy mean: sequential sum / 21).
export function handCenter(hand) {
  let x = 0, y = 0;
  for (let p = 0; p < 42; p += 2) { x += hand[p]; y += hand[p + 1]; }
  return [x / 21, y / 21];
}

function wristOf(hand) {
  return [hand[0], hand[1]];
}

export function _sameHand(first, second) {
  const limit = SAME_HAND * Math.max(_handLength(first), _handLength(second));
  return distance(handCenter(first), handCenter(second)) <= limit;
}

// The registered person's body in the latest accepted frame: what a hidden face is followed by.
export class BodyLock {
  constructor() {
    this.reset();
  }

  reset() {
    this.center = null; // shoulder centre of the latest accepted frame
    this.anchor = null; // ... of the latest accepted frame with the face verified
    this.width = 0.0; // usual shoulder width (scale of the arm limits)
    this.lastWidth = 0.0; // shoulder width of the latest accepted frame
    this.stamp = -Infinity;
  }

  recent(stamp) {
    return this.center !== null && stamp - this.stamp <= CHAIN_SECONDS;
  }

  // Shoulders that can be this person's: never wider than SHOULDERS_PER_FACE face widths of its own face
  // mesh, as wide as a moment ago (BODY_WIDTH_RANGE), and not far wider than within WIDTH_RESET_SECONDS.
  sameBuild(body, stamp) {
    const width = body.shoulderWidth;
    let faceWidth = 0.0;
    if (body.face !== null) {
      let low = Infinity, high = -Infinity;
      const face = body.face;
      for (let p = 0; p < face.length; p += 2) {
        if (face[p] < low) low = face[p];
        if (face[p] > high) high = face[p];
      }
      faceWidth = high - low;
    }
    if (faceWidth > 1.0 && width > SHOULDERS_PER_FACE * faceWidth) return false;
    if (this.lastWidth <= 0 || this.center === null || stamp - this.stamp > WIDTH_RESET_SECONDS) {
      // No reference: only a skeleton with its face mesh, its shoulders neither joined nor collapsed, starts one.
      return faceWidth > 1.0 && width >= MIN_SHOULDERS_PER_FACE * faceWidth;
    }
    if (this.recent(stamp)) {
      const ratio = width / this.lastWidth;
      return BODY_WIDTH_RANGE[0] <= ratio && ratio <= BODY_WIDTH_RANGE[1];
    }
    // Later: never far wider than the usual width, which a turned body hardly lowers.
    return width <= BODY_WIDTH_RANGE[1] * Math.max(this.width, this.lastWidth);
  }

  follows(body, stamp) {
    if (!this.recent(stamp) || this.anchor === null || this.width <= 0) return false;
    const center = body.shoulderCenter;
    return distance(center, this.center) <= BODY_SHIFT * this.width
      && distance(center, this.anchor) <= BODY_DRIFT * this.width
      && this.sameBuild(body, stamp);
  }

  update(body, stamp, verified) {
    const width = body.shoulderWidth;
    if (this.center === null || stamp - this.stamp > WIDTH_RESET_SECONDS) {
      this.width = width; // after a long gap the person may sit elsewhere: no old scale
    } else {
      this.width = _growOrShrink(this.width, width);
    }
    this.center = body.shoulderCenter;
    this.lastWidth = width;
    this.stamp = stamp;
    if (verified) this.anchor = this.center;
  }

  scale(body) {
    const scale = Math.max(body.shoulderWidth, REFERENCE_SCALE * this.width, 1.0);
    // Never wider than the usual build allows: a joined skeleton must not loosen every limit.
    return this.width > 0 ? Math.min(scale, BODY_WIDTH_RANGE[1] * this.width) : scale;
  }
}

// Whose hand was where a moment ago: a hand stays with its person while two skeletons claim it. The signer's
// hands are learned only from frames where nobody else could claim them, never from a decision of this
// memory or of the nearer shoulder; another person's hand is also learned while this memory keeps it theirs.
// own[side]: {place: [x, y], length, stamp} | null; foreign: [{place, length, stamp, owner, side}].
export class HandMemory {
  constructor() {
    this.reset();
  }

  reset() {
    this.own = [null, null]; // per side: the signer's latest uncontested hand
    this.foreign = []; // other people's hands
  }

  // [was the signer's, was someone else's] for a hand centre (or pose wrist) seen now. `current`: the other
  // people's other hands in this frame, as [owner, side, centre]: a remembered hand that the same hand of the
  // same person is nearer to now than this hand is is still visible there, not this hand.
  recall(point, side, stamp, current = []) {
    const near = (entry) => stamp - entry.stamp <= MEMORY_SECONDS
      && distance(point, entry.place) <= MEMORY_HAND_LENGTHS * entry.length;
    const explained = (entry) => current.some(([who, which, centre]) => entry.owner != null && who === entry.owner
      && which === entry.side && distance(entry.place, centre) < distance(entry.place, point));
    const own = this.own[side];
    const theirs = this.foreign.some((entry) => near(entry) && !explained(entry));
    return [own != null && near(own), theirs];
  }

  // Learn from this frame what nobody contested: the signer's kept hands (only when every other face in view
  // had a skeleton to contest them: `complete`), other people's hands the signer's tracker did not also hold
  // (only from an accepted signer skeleton).
  remember(verdict, signer, others, stamp, complete) {
    this.foreign = this.foreign.filter((entry) => stamp - entry.stamp <= MEMORY_SECONDS); // also when rejected
    if (!verdict.accepted) return;
    const sure = verdict.sure;
    signer.hands.forEach((hand, side) => {
      if (hand !== null && verdict.hands[side] && sure[side] && complete) {
        this.own[side] = { place: handCenter(hand), length: _handLength(hand), stamp };
      }
    });
    // Their hands contested by the signer's tracker are not learned, unless this memory kept them theirs.
    const mine = signer.hands.filter((hand, side) => hand !== null
      && !(verdict.basis[side] === "memory" && !verdict.hands[side]));
    for (const other of others) {
      other.hands.forEach((hand, side) => {
        if (hand !== null && !mine.some((own) => _sameHand(hand, own))) {
          this.foreign.push({ place: handCenter(hand), length: _handLength(hand), stamp, owner: other.owner, side });
        }
      });
    }
  }
}

// What may be used of the signer tracker's skeleton. reason: "" all kept | "rest" | "hand" | "face" (parts
// left out) | one of REJECTED (nothing kept). hands / rested: per side [left, right]; face: face mesh kept;
// basis: per hand what deciding whose it is rested on: "free" (nobody claimed it), "memory", "shoulder".
export class Verdict {
  constructor(keypoints, reason, hands = [false, false], face = false, rested = [false, false], basis = ["", ""]) {
    this.keypoints = keypoints;
    this.reason = reason;
    this.hands = hands;
    this.face = face;
    this.rested = rested;
    this.basis = basis;
  }

  get accepted() {
    return !REJECTED.includes(this.reason);
  }

  // Per hand: uncontested (nobody else claimed it, nothing remembered against it).
  get sure() {
    return this.basis.map((basis) => basis === "free");
  }
}

function _rejected(reason) {
  return new Verdict(new Float32Array(KEYPOINT_SIZE), reason);
}

// [body, side] of another person's skeleton holding the same hand, or its pose wrist on it, or null.
export function _holder(hand, others) {
  for (const other of others) {
    for (let side = 0; side < 2; side++) {
      const theirs = other.hands[side];
      if (theirs !== null && _sameHand(hand, theirs)) return [other, side];
    }
  }
  const wrist = wristOf(hand);
  const reach = _handLength(hand);
  for (const other of others) {
    for (let side = 0; side < 2; side++) {
      const index = SIDES[side][2];
      if (other.visibility[index] >= CHECKED_VISIBILITY && distance(other.point(index), wrist) <= reach) {
        return [other, side];
      }
    }
  }
  return null;
}

// [body, side] of another skeleton with a hand, or a visible pose wrist, on this pose wrist (an arm whose own
// hand was not found).
export function _wristHolder(point, others, scale) {
  for (const other of others) {
    for (let side = 0; side < 2; side++) {
      const theirs = other.hands[side];
      if (theirs !== null && distance(point, wristOf(theirs)) <= SAME_HAND * _handLength(theirs)) return [other, side];
    }
  }
  const reach = SAME_HAND * HAND_PER_SHOULDER * scale;
  for (const other of others) {
    for (let side = 0; side < 2; side++) {
      const index = SIDES[side][2];
      if (other.visibility[index] >= CHECKED_VISIBILITY && distance(other.point(index), point) <= reach) {
        return [other, side];
      }
    }
  }
  return null;
}

export function _armPossible(body, side, scale) {
  const [shoulder, elbow, wrist] = SIDES[side];
  return distance(body.point(elbow), body.point(shoulder)) <= MAX_UPPER_ARM * scale
    && distance(body.point(wrist), body.point(elbow)) <= MAX_FOREARM * scale
    && distance(body.point(wrist), body.point(shoulder)) <= MAX_REACH * scale;
}

// [whether the hand at `point` (its centre; `wrist`: its wrist) on the signer's `side` arm belongs to someone
// else, what that rests on: "free" (nobody claims it), "memory" or "shoulder"]. For an arm whose hand was not
// found both are its pose wrist; `holder`: [body, side] of another skeleton claiming it now; `current`: the
// other people's other hands.
export function _theirs(point, wrist, side, signer, holder, scale, memory, stamp, current) {
  const [mine, theirs] = memory != null ? memory.recall(point, side, stamp, current) : [false, false];
  if (holder === null) {
    // Nobody claims it now; someone else had a hand there a moment ago (their tracker may just have lost it):
    // theirs, even if the signer's hand was also near.
    return theirs ? [true, "memory"] : [false, "free"];
  }
  if (mine !== theirs) return [theirs, "memory"]; // it stays with whoever had it before the contest
  const [other, theirSide] = holder;
  const nearer = distance(wrist, signer.point(SIDES[side][0])) / scale;
  const theirDistance = distance(wrist, other.point(SIDES[theirSide][0])) / other.scale;
  return [nearer + OWNERSHIP_MARGIN > theirDistance, "shoulder"];
}

// Put an arm down at rest below its shoulder, unseen (visibility 0), as MediaPipe reports an arm out of view.
export function _restArm(keypoints, body, side, scale) {
  const shoulderIndex = SIDES[side][0];
  const [width, height] = body.size;
  const x = body.pose[2 * shoulderIndex];
  const elbowY = body.pose[2 * shoulderIndex + 1] + REST_UPPER_ARM * scale;
  const wristY = elbowY + REST_FOREARM * scale;
  const handY = wristY + REST_HAND * scale;
  const places = [elbowY, wristY, handY, handY, handY];
  const depth = keypoints[4 * shoulderIndex + 2];
  ARM_POINTS[side].forEach((index, i) => {
    keypoints[4 * index] = x / width;
    keypoints[4 * index + 1] = places[i] / height;
    keypoints[4 * index + 2] = depth;
    keypoints[4 * index + 3] = 0.0;
  });
  keypoints.fill(0, HAND_SLICES[side][0], HAND_SLICES[side][1]);
}

// What of the signer tracker's skeleton belongs to the registered person (see the module comment).
// `target`: the verified face box this frame, or null while it is hidden (`held`: where it was);
// `otherFaces`: everybody else's face boxes; `others`: their skeletons (only those whose head is at their own
// face, plus stand-ins); `memory`: who held which hand a moment ago (the caller teaches it); `strict`: an arm
// of someone else drops the whole frame.
export function judge(signer, target, otherFaces, others, lock, stamp, memory = null, strict = false, held = null) {
  if (!signer.hasPose) return _rejected("none");
  const nose = signer.nose;
  if (otherFaces.some((box) => _inside(nose, box))) return _rejected("head");
  if (target != null) {
    if (!_inside(nose, target, HEAD_MARGIN)) return _rejected("head");
    if (!lock.sameBuild(signer, stamp)) return _rejected("body"); // shoulders joined to someone else's (or collapsed)
  } else if (!lock.follows(signer, stamp) || (held != null && !_inside(nose, held, HELD_MARGIN))) {
    return _rejected("body");
  }
  const scale = lock.scale(signer);
  // Hands seen now (a stand-in's stale copy is no evidence that a remembered hand is still somewhere else):
  // [owner, side, hand].
  const seenHands = [];
  for (const other of others) {
    if (other.standIn) continue;
    other.hands.forEach((hand, which) => { if (hand !== null) seenHands.push([other.owner, which, hand]); });
  }
  const outcome = []; // per side: "keep", "drop" (hand only), "rest" (hand and arm), "" (nothing there)
  const basis = ["", ""]; // per side: what deciding whose hand it is rested on (_theirs)
  for (let side = 0; side < 2; side++) {
    const [shoulder, elbow, wrist] = SIDES[side];
    const hand = signer.hands[side];
    if (hand === null && signer.visibility[wrist] < CHECKED_VISIBILITY) {
      // An arm out of view: nothing of it is used for signing, but a visible elbow far from its shoulder is
      // someone else's arm (translate reads elbows).
      const elbowSeen = signer.visibility[elbow] >= CHECKED_VISIBILITY;
      const tooFar = distance(signer.point(elbow), signer.point(shoulder)) > MAX_UPPER_ARM * scale;
      outcome.push(elbowSeen && tooFar ? "rest" : "");
      continue;
    }
    if (!_armPossible(signer, side, scale)) {
      outcome.push("rest");
      continue;
    }
    const wristPoint = signer.point(wrist);
    if (hand === null) { // a visible arm without its hand: is its wrist on someone else's hand?
      const holder = _wristHolder(wristPoint, others, scale);
      const current = seenHands.filter(([, , other]) => distance(wristOf(other), wristPoint) > SAME_HAND * _handLength(other))
        .map(([owner, which, other]) => [owner, which, handCenter(other)]);
      const [onTheirs] = _theirs(wristPoint, wristPoint, side, signer, holder, scale, memory, stamp, current);
      outcome.push(onTheirs ? "rest" : "");
      continue;
    }
    const handWrist = wristOf(hand);
    const onWrist = distance(handWrist, wristPoint) <= MAX_HAND_OFFSET * scale;
    const current = seenHands.filter(([, , other]) => !_sameHand(hand, other))
      .map(([owner, which, other]) => [owner, which, handCenter(other)]);
    const [theirs, why] = _theirs(handCenter(hand), handWrist, side, signer, _holder(hand, others), scale,
      memory, stamp, current);
    basis[side] = why;
    if (theirs) outcome.push(onWrist ? "rest" : "drop"); // an arm on their hand is not the signer's arm
    else outcome.push(onWrist ? "keep" : "drop");
  }
  const rests = outcome.filter((result) => result === "rest").length;
  if (rests === 2 || (strict && rests > 0)) return _rejected("arm");
  const keypoints = Float32Array.from(signer.keypoints);
  const reasons = new Set();
  outcome.forEach((result, side) => {
    if (result === "rest") {
      _restArm(keypoints, signer, side, scale);
      reasons.add("rest");
    } else if (result === "drop") {
      keypoints.fill(0, HAND_SLICES[side][0], HAND_SLICES[side][1]);
      reasons.add("hand");
    }
  });
  let faceKept = signer.face !== null;
  if (faceKept) {
    const center = handCenterOf(signer.face, 468);
    if (target != null) faceKept = _inside(center, target, HEAD_MARGIN);
    else faceKept = distance(center, nose) <= FACE_TO_NOSE * scale;
    faceKept = faceKept && !otherFaces.some((box) => _inside(center, box));
    if (!faceKept) {
      keypoints.fill(0, FACE[0], FACE[1]);
      reasons.add("face");
    }
  }
  const reason = ["rest", "hand", "face"].find((name) => reasons.has(name)) ?? "";
  return new Verdict(keypoints, reason, outcome.map((result) => result === "keep"), faceKept,
    outcome.map((result) => result === "rest"), basis);
}

function handCenterOf(points, count) {
  let x = 0, y = 0;
  for (let p = 0; p < 2 * count; p += 2) { x += points[p]; y += points[p + 1]; }
  return [x / count, y / count];
}

// ------------------------------------------------------------------ trackers

// The input region changed enough for the tracked body to belong to another image.
export function _jumped(previous, crop) {
  if (previous == null || crop == null) return !(previous == null && crop == null);
  const asBox = (region) => [region[0], region[1], region[2] - region[0], region[3] - region[1]];
  return _iou(asBox(previous), asBox(crop)) < CROP_JUMP_IOU;
}

// One Holistic instance following one person: its tracking state belongs to that person.
export class Tracker {
  constructor(holistic) {
    this.holistic = holistic;
    this.box = null; // face box of the person it follows (other people's trackers)
    this.crop = null; // input region of the last run
    this.seen = -Infinity; // when its person was last seen
    this.ran = -Infinity; // when it last ran
    this.stale = true; // its tracking state does not belong to the next input
    this.width = 0.0; // its person's usual shoulder width (other people's trackers)
    this.body = null; // its person's latest skeleton whose head was at their face...
    this.bodySeen = -Infinity; // ... and when
    this.results = null;
  }

  // Forget the tracked body: a blank (black) image of the view's size clears MediaPipe's tracking.
  reset(frame, width, height, tools, stamp = frame?.stamp ?? 0) {
    tools.detect(this.holistic, { frame, crop: null, masks: [], width, height, blanks: [], blank: true, stamp });
    this.results = null;
    this.stale = false;
  }

  run(frame, crop, masks, stamp, tools) {
    const view = prepareRegion(frame, new Region("locked", crop, masks, null, [], 0.0));
    view.stamp = stamp;
    if (this.stale || _jumped(this.crop, crop) || stamp - this.ran > RESET_AFTER_SECONDS) {
      this.reset(frame, view.width, view.height, tools, stamp); // what it tracked belongs to another input or moment
    }
    this.crop = crop;
    this.ran = stamp;
    this.results = tools.detect(this.holistic, view);
    const keypoints = tools.remap(tools.extract(this.results), crop, frame.width, frame.height);
    return new Body(keypoints, [frame.width, frame.height], this.width);
  }
}

// keypoints: the registered person's landmarks to record / translate (zeros: none this frame); status:
// identity status (locked / hidden / searching / lost); reason: what was left out of the signer tracker's
// skeleton ("" nothing); region: lock.Region; signer: [results, crop, verdict] of the signer tracker when it ran,
// else null; others: [[results, crop], ...] of other people's trackers, drawn grey.
export class Scene {
  constructor(keypoints, status, reason, region, signer = null, others = []) {
    this.keypoints = keypoints;
    this.status = status;
    this.reason = reason;
    this.region = region;
    this.signer = signer;
    this.others = others;
  }
}

// Close every Holistic instance, even when one of them fails to close (the first error is thrown after).
export function _close(trackers) {
  let error = null;
  for (const tracker of trackers) {
    const close = tracker.holistic?.close;
    if (typeof close !== "function") continue;
    try {
      close.call(tracker.holistic);
    } catch (failure) {
      error = error ?? failure;
    }
  }
  if (error !== null) throw error;
}

// Face lock + one Holistic tracker for the signer and one for each person around (camera loop only).
// `strict`: drop a whole frame where an arm of someone else is joined to the signer (recorded samples).
export class People {
  constructor(lock, create, tools, maxOthers = MAX_OTHERS, strict = false) {
    this.lock = lock;
    this.tools = tools;
    this.strict = strict;
    const trackers = [];
    try {
      for (let i = 0; i < 1 + maxOthers; i++) trackers.push(new Tracker(create()));
    } catch (error) {
      _close(trackers);
      throw error;
    }
    this.signer = trackers[0];
    this.slots = trackers.slice(1);
    this.body = new BodyLock();
    this.memory = new HandMemory();
    this.cover = [null, -Infinity]; // the signer's hand points in the latest accepted frame (_coverPoints), stamp
    this.freed = -Infinity; // when a tracker that followed the person's own face was freed
    this.closed = false;
  }

  close() {
    this.closed = true;
    _close([this.signer, ...this.slots]);
  }

  // [tracker, face box] for the other faces nearest the signer (`boxes` nearest first): a tracker keeps
  // following the face nearest to the one it followed; a new face takes a free tracker, else the tracker of a
  // farther face; faces beyond the trackers are only blanked.
  _assign(boxes, stamp) {
    const chosen = boxes.slice(0, this.slots.length);
    const assigned = [], used = new Set(), fresh = [];
    for (const box of chosen) {
      const center = _center(box);
      let best = null; // [distance, index]: nearest, first index of ties (min of tuples)
      this.slots.forEach((slot, index) => {
        if (slot.box == null || used.has(index)) return;
        const away = distance(center, _center(slot.box));
        if (away <= 1.5 * box[2] && (best === null || away < best[0])) best = [away, index];
      });
      if (best !== null) {
        used.add(best[1]);
        assigned.push([this.slots[best[1]], box]);
      } else {
        fresh.push(box);
      }
    }
    for (const box of fresh) {
      // A free tracker first, else one whose person is farther away or gone.
      const spare = [];
      for (let index = 0; index < this.slots.length; index++) if (!used.has(index) && this.slots[index].box == null) spare.push(index);
      for (let index = 0; index < this.slots.length; index++) if (!used.has(index) && this.slots[index].box != null) spare.push(index);
      used.add(spare[0]);
      const slot = this.slots[spare[0]];
      slot.stale = true; // a new person: start afresh
      slot.crop = null;
      slot.width = 0.0;
      slot.body = null;
      assigned.push([slot, box]);
    }
    for (const [slot, box] of assigned) {
      slot.box = box;
      slot.seen = stamp;
    }
    return assigned;
  }

  // Run the other people's trackers. Returns [bodies whose head is at their own face, stand-ins, drawings]:
  // a stand-in is the latest skeleton (within MEMORY_SECONDS) of someone seen a moment ago whose tracker gave
  // nothing now; their hands still count in the contests, and nothing is learned from the frame. Never the
  // signer's own body (`signer`).
  _others(frame, region, stamp, mine, signer = null) {
    const bodies = [], drawings = [];
    let boxes;
    if (mine != null) {
      const center = _center(mine);
      boxes = region.others.map((box) => [box, distance(_center(box), center)])
        .sort((a, b) => a[1] - b[1]).map(([box]) => box);
    } else {
      boxes = region.others.slice().sort((a, b) => b[2] - a[2]); // nobody to protect: the largest faces
    }
    const answered = new Set();
    for (const [slot, box] of this._assign(boxes, stamp)) {
      const crop = steadyCrop(slot.crop, box, frame.width, frame.height);
      const blanked = region.others.filter((other) => !sameBox(other, box));
      if (mine != null) blanked.push(mine);
      const body = slot.run(frame, crop, blanked, stamp, this.tools);
      if (!body.hasPose) continue;
      if (_inside(body.nose, box, HEAD_MARGIN) && !(mine != null && _inside(body.nose, mine))) {
        slot.width = _growOrShrink(slot.width, body.shoulderWidth);
        body.reference = slot.width;
        body.owner = this.slots.indexOf(slot);
        slot.body = body;
        slot.bodySeen = stamp;
        answered.add(slot);
        bodies.push(body);
        drawings.push([slot.results, crop]);
      } else {
        slot.stale = true; // it follows somebody else's body: start afresh next frame
      }
    }
    const standIns = this.slots.filter((slot) => slot.box != null && !answered.has(slot) && slot.body !== null
      && stamp - slot.bodySeen <= MEMORY_SECONDS && !(signer !== null && _sameBody(slot.body, signer)))
      .map((slot) => slot.body);
    for (const body of standIns) body.standIn = true;
    return [bodies, standIns, drawings];
  }

  // No face in view: whoever Holistic finds is drawn grey (seen, not accepted).
  _unverified(frame, stamp) {
    const slot = this.slots.length ? this.slots[0] : null;
    if (slot === null || slot.box != null) return [];
    const body = slot.run(frame, null, [], stamp, this.tools);
    return body.hasPose ? [[slot.results, null]] : [];
  }

  // Where the signer's hands are in an accepted frame: the points of the hands kept, and the visible pose
  // wrist of an arm (not put down) whose hand was not found. Float64Array [x0, y0, x1, y1, ...] pixels.
  static _coverPoints(signer, verdict) {
    const parts = [];
    signer.hands.forEach((hand, side) => { if (verdict.hands[side]) parts.push(hand); });
    SIDES.forEach(([, , wrist], side) => {
      if (signer.hands[side] === null && !verdict.rested[side] && signer.visibility[wrist] >= CHECKED_VISIBILITY) {
        parts.push(signer.point(wrist));
      }
    });
    const out = new Float64Array(parts.reduce((total, part) => total + part.length, 0));
    let offset = 0;
    for (const part of parts) { out.set(part, offset); offset += part.length; }
    return out;
  }

  // One camera frame (a Promise of its Scene). stamp: capture time, seconds; faces: its Face detections (null:
  // lock.engine.detect).
  async process(frame, stamp, faces = null) {
    for (const slot of this.slots) if (stamp - slot.seen > OTHER_TIMEOUT_SECONDS) slot.box = null;
    const [points, seen] = this.cover;
    this.lock.coverPoints = stamp - seen <= CHAIN_SECONDS ? points : null;
    const region = await this.lock.locate(frame, faces);
    // Closed while SFace ran (the camera stopped): its Holistic instances must not run any more; nothing is taken.
    if (this.closed) return new Scene(new Float32Array(KEYPOINT_SIZE), region.status, "", region, null, []);
    return this._scene(frame, stamp, region);
  }

  _scene(frame, stamp, region) {
    if (region.skip) {
      this.signer.stale = true;
      if (this.lock.trackBox == null || stamp - this.body.stamp > WIDTH_RESET_SECONDS) {
        // Nobody is held (never verified, released) or the person was away long: nothing to continue from.
        this.body.reset();
        this.memory.reset();
      }
      // Otherwise the body and hand memories expire on their own: a frame skipped for the person's own covered
      // face does not end the body continuity.
      const drawings = region.others.length ? this._others(frame, region, stamp, null)[2] : this._unverified(frame, stamp);
      return new Scene(new Float32Array(KEYPOINT_SIZE), region.status, "", region, null, drawings);
    }
    const mine = region.target; // the verified face, or where it was while hidden
    const locked = region.status === "locked";
    const signer = this.signer.run(frame, region.crop, region.masks, stamp, this.tools);
    for (const slot of this.slots) {
      if (slot.box != null && _iou(slot.box, mine) >= 0.3 && (slot.body === null || _sameBody(slot.body, signer))) {
        // It followed the person's own face while it was not recognised: that skeleton is theirs, never
        // 'someone else' to contest their hands.
        slot.box = null;
        slot.body = null;
        this.freed = stamp;
      }
    }
    const [others, standIns, drawings] = this._others(frame, region, stamp, mine, signer);
    const verdict = judge(signer, locked ? mine : null, region.others, others.concat(standIns), this.body, stamp,
      this.memory, this.strict, locked ? null : mine);
    // The signer's hands are learned only when everyone around had a skeleton to contest them (a tracker freed a
    // moment ago may have been the only one to know someone was there).
    let tracked = 0;
    for (const slot of this.slots) if (slot.box != null) tracked += 1;
    const complete = !standIns.length && others.length === region.others.length && region.others.length === tracked
      && stamp - this.freed > MEMORY_SECONDS;
    this.memory.remember(verdict, signer, others, stamp, complete);
    if (verdict.accepted) {
      // The anchor of the hidden-face continuity moves only with a face recognised by SFace this frame.
      this.body.update(signer, stamp, region.verified);
      this.cover = [People._coverPoints(signer, verdict), stamp];
    } else if (verdict.reason === "head" || (verdict.reason === "body" && locked)) {
      this.signer.stale = true; // it follows someone else (or a joined body): start afresh
    } else if (verdict.reason === "body" && stamp - this.body.stamp > CHAIN_SECONDS) {
      // Hidden face, no body following the person's for CHAIN_SECONDS: no longer known to be the person.
      this.body.reset();
      this.memory.reset();
      this.lock.release();
    }
    return new Scene(verdict.keypoints, region.status, verdict.reason, region,
      [this.signer.results, region.crop, verdict], drawings);
  }
}

// ------------------------------------------------------------------ drawing
// `results` is what tools.detect returned: {keypoints, drawing: {pose, face, left, right}} with arrays of
// {x, y, z, visibility} in the tracker's view (null = not detected), the counterpart of MediaPipe's results.

function drawingOf(results) {
  const drawing = results.drawing ?? {};
  return { pose: drawing.pose ?? null, face: drawing.face ?? null, left: drawing.left ?? null, right: drawing.right ?? null };
}

// The signer's results without the rejected parts, for drawing (kept_results): {drawing} with the parts not
// kept null and a rested arm's points (not the signer's arm) moved to its shoulder with visibility 0; or null.
// The input is not changed.
export function keptResults(results, verdict) {
  if (results == null || !verdict.accepted) return null;
  const drawing = drawingOf(results);
  let pose = drawing.pose;
  if (pose != null && verdict.rested.some(Boolean)) {
    pose = pose.map((point) => ({ ...point }));
    verdict.rested.forEach((rested, side) => {
      if (!rested) return; // not the signer's arm: drawn grey by rejectedResults, not here
      const shoulder = pose[SIDES[side][0]];
      for (const index of ARM_POINTS[side]) {
        Object.assign(pose[index], { x: shoulder.x, y: shoulder.y, z: shoulder.z, visibility: 0.0 });
      }
    });
  }
  return {
    drawing: {
      pose,
      face: verdict.face ? drawing.face : null,
      left: verdict.hands[0] ? drawing.left : null,
      right: verdict.hands[1] ? drawing.right : null,
    },
  };
}

// [the rejected parts of the signer's results ({drawing}), the pose points to draw: null = all] for grey drawing
// (draw.drawMuted), rejected_results: [null, []] without results or when nothing was rejected; [results, null]
// when the whole skeleton was rejected; else [{drawing: pose + the parts not kept}, the shoulder, elbow and wrist
// of each rested arm].
export function rejectedResults(results, verdict) {
  if (results == null) return [null, []];
  if (!verdict.accepted) return [results, null];
  const drawing = drawingOf(results);
  const parts = {
    pose: drawing.pose,
    face: verdict.face ? null : drawing.face,
    left: verdict.hands[0] ? null : drawing.left,
    right: verdict.hands[1] ? null : drawing.right,
  };
  const arms = [];
  verdict.rested.forEach((rested, side) => {
    if (rested) arms.push(SIDES[side][0], ...ARM_POINTS[side].slice(0, 2));
  });
  if (!arms.length && parts.face == null && parts.left == null && parts.right == null) return [null, []];
  return [{ drawing: parts }, arms];
}
