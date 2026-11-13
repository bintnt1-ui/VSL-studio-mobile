// Display parts of the phone's multi-source collector (ports of collector_window.py's TimelineWidget, FrameCanvas
// regions / trajectory and FilmStrip), for touch: tap or drag to seek, drag the A / B handles, pinch or the +/−
// buttons to zoom, drag with two fingers to pan.

const COLORS = {
  background: "#0f172a", lane: "#1e293b", grid: "#334155", text: "#94a3b8", caption: "#a78bfa", activity: "rgba(56,189,248,0.6)",
  deleted: "#991b1b", saved: "#16a34a", proposal: "#f59e0b", proposalOff: "#92400e", proposalSaved: "#15803d",
  proposalDeleted: "#7f1d1d", skipped: "#475569", ab: "rgba(53,97,237,0.28)", marker: "#3b82f6", handle: "#2563eb",
  seek: "#475569", playhead: "#ef4444", signer: "rgb(255,170,60)", textRegion: "rgb(80,200,255)",
};
const LEFT = 14, RIGHT = 14, HEIGHT = 122;

export function secondsText(frame, fps) {
  const seconds = frame / (fps || 30);
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}s`;
}

export class Timeline {
  // callbacks: seek(frame), marker(which "a" | "b", frame), proposal(index)
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.callbacks = callbacks;
    this.reset(0, 30);
    this.pointers = new Map();
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", (event) => this._down(event));
    canvas.addEventListener("pointermove", (event) => this._move(event));
    canvas.addEventListener("pointerup", (event) => this._up(event));
    canvas.addEventListener("pointercancel", (event) => this._up(event));
    canvas.addEventListener("dblclick", (event) => this._double(event));
    canvas.addEventListener("wheel", (event) => this._wheel(event), { passive: false });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  reset(frameCount, fps) {
    this.frameCount = Math.max(0, Math.trunc(frameCount));
    this.fps = fps || 30;
    this.position = 0;
    this.a = this.b = null;
    this.proposals = [];
    this.saved = [];
    this.deleted = [];
    this.captions = [];
    this.activity = null;
    this.view = [0, Math.max(0, this.frameCount - 1)];
    this.draw();
  }

  set(values) {
    Object.assign(this, values);
    this.draw();
  }

  _width() {
    return this.canvas.clientWidth || 320;
  }

  xOf(frame) {
    const [start, end] = this.view;
    return LEFT + ((frame - start) / Math.max(end - start, 1)) * (this._width() - LEFT - RIGHT);
  }

  frameAt(x) {
    const [start, end] = this.view;
    const frame = start + ((x - LEFT) / Math.max(this._width() - LEFT - RIGHT, 1)) * (end - start);
    return Math.round(Math.min(Math.max(frame, 0), Math.max(this.frameCount - 1, 0)));
  }

  ensureVisible(frame) {
    const [start, end] = this.view;
    if (start <= frame && frame <= end) return;
    const width = end - start;
    const first = Math.min(Math.max(0, frame - Math.trunc(width / 5)), Math.max(0, this.frameCount - 1 - width));
    this.view = [first, first + width];
  }

  setPosition(frame) {
    this.position = frame;
    this.ensureVisible(frame);
    this.draw();
  }

  // factor < 1 zooms in around `anchor` (a frame), like the PC's wheel.
  zoom(factor, anchor = this.position) {
    if (this.frameCount <= 1) return;
    const [start, end] = this.view;
    const minimum = Math.min(this.frameCount - 1, Math.max(30, Math.trunc(this.fps * 2)));
    const width = Math.trunc(Math.min(this.frameCount - 1, Math.max(minimum, (end - start) * factor)));
    const share = (anchor - start) / Math.max(end - start, 1);
    let first = Math.round(anchor - share * width);
    first = Math.min(Math.max(0, first), Math.max(0, this.frameCount - 1 - width));
    this.view = [first, first + width];
    this.draw();
  }

  pan(frames) {
    const [start, end] = this.view;
    const first = Math.min(Math.max(0, start + frames), Math.max(0, this.frameCount - 1 - (end - start)));
    this.view = [first, first + (end - start)];
    this.draw();
  }

  draw() {
    const canvas = this.canvas, ratio = window.devicePixelRatio || 1;
    const width = this._width();
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(HEIGHT * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(HEIGHT * ratio);
    }
    const g = canvas.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    g.fillStyle = COLORS.background;
    g.fillRect(0, 0, width, HEIGHT);
    if (this.frameCount <= 0) {
      g.fillStyle = "#64748b";
      g.font = "13px system-ui, sans-serif";
      g.textAlign = "center";
      g.fillText("Chưa có video", width / 2, HEIGHT / 2);
      g.textAlign = "left";
      return;
    }
    const left = LEFT, right = width - RIGHT;
    this._ruler(g, left, right);
    for (const [start, end, text] of this.captions) this._bar(g, start, end, 24, 14, COLORS.caption, text);
    g.fillStyle = COLORS.lane;
    g.fillRect(left, 42, right - left, 16);
    if (this.activity?.length) this._activity(g, 42, 16, left, right);
    g.fillStyle = COLORS.lane;
    g.fillRect(left, 62, right - left, 28);
    for (const [start, end] of this.deleted) this._bar(g, start, end, 64, 24, COLORS.deleted, "⊘ đã xóa");
    for (const [start, end, label] of this.saved) this._bar(g, start, end, 64, 24, COLORS.saved, `✓ ${label}`);
    for (const proposal of this.proposals) {
      const color = { saved: COLORS.proposalSaved, deleted: COLORS.proposalDeleted, skipped: COLORS.skipped }[proposal.state]
        ?? (proposal.checked !== false ? COLORS.proposal : COLORS.proposalOff);
      this._bar(g, proposal.start, proposal.end, 68, 16, color, proposal.label ?? "");
    }
    if (this.a !== null) {
      const end = this.b !== null ? this.b : this.position;
      const [first, last] = [Math.min(this.a, end), Math.max(this.a, end)];
      const x0 = this.xOf(first), x1 = this.xOf(last + 1);
      g.fillStyle = COLORS.ab;
      g.fillRect(x0, 20, Math.max(2, x1 - x0), 74);
      for (const [marker, name] of [[this.a, "A"], [this.b, "B"]]) {
        if (marker === null) continue;
        const x = this.xOf(name === "A" ? marker : marker + 1);
        g.strokeStyle = COLORS.marker;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(x, 20);
        g.lineTo(x, 96);
        g.stroke();
        g.fillStyle = this.drag === name.toLowerCase() ? "#60a5fa" : COLORS.handle;
        roundRect(g, x - 11, 1, 22, 18, 4);
        g.fill();
        g.fillStyle = "#fff";
        g.font = "bold 12px system-ui, sans-serif";
        g.textAlign = "center";
        g.fillText(name, x, 14);
        g.textAlign = "left";
      }
    }
    g.fillStyle = COLORS.seek;
    g.fillRect(left, 104, right - left, 8);
    const x = this.xOf(this.position);
    g.fillStyle = COLORS.playhead;
    g.fillRect(left, 104, Math.max(0, x - left), 8);
    g.strokeStyle = COLORS.playhead;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(x, 20);
    g.lineTo(x, 104);
    g.stroke();
    g.beginPath();
    g.arc(x, 108, 8, 0, Math.PI * 2);
    g.fill();
  }

  _ruler(g, left, right) {
    const [start, end] = this.view;
    const seconds = (end - start) / this.fps;
    const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const step = steps.find((value) => seconds / value <= Math.max(4, (right - left) / 90)) ?? steps[steps.length - 1];
    g.font = "10px system-ui, sans-serif";
    const first = Math.floor(start / this.fps / step), last = Math.ceil(end / this.fps / step);
    for (let tick = first; tick <= last; tick++) {
      const frame = tick * step * this.fps;
      const x = this.xOf(frame);
      if (x < left - 1 || x > right + 1) continue;
      g.strokeStyle = COLORS.grid;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x, 14);
      g.lineTo(x, 96);
      g.stroke();
      g.fillStyle = COLORS.text;
      const label = secondsText(Math.trunc(frame), this.fps);
      g.fillText(step < 1 ? label.slice(0, -1) : label.slice(0, 5), x + 3, 11);
    }
  }

  _bar(g, start, end, top, height, color, text) {
    const left = LEFT, right = this._width() - RIGHT;
    let x0 = this.xOf(start), x1 = this.xOf(end + 1);
    if (x1 < left || x0 > right) return;
    x0 = Math.max(x0, left);
    x1 = Math.min(x1, right);
    const width = Math.max(2, x1 - x0);
    g.fillStyle = color;
    roundRect(g, x0, top, width, height, 3);
    g.fill();
    if (text && width > 24) {
      g.save();
      g.beginPath();
      g.rect(x0 + 3, top, width - 5, height);
      g.clip();
      g.fillStyle = "#fff";
      g.font = "10px system-ui, sans-serif";
      g.fillText(text, x0 + 4, top + height / 2 + 4);
      g.restore();
    }
  }

  _activity(g, top, height, left, right) {
    const [start, end] = this.view;
    const columns = Math.max(1, Math.trunc(right - left));
    g.fillStyle = COLORS.activity;
    g.beginPath();
    g.moveTo(left, top + height);
    for (let column = 0; column < columns; column++) {
      const frame = Math.trunc(start + ((end - start) * column) / Math.max(columns - 1, 1));
      const value = this.activity[Math.min(Math.max(frame, 0), this.activity.length - 1)] || 0;
      g.lineTo(left + column, top + height - value * height);
    }
    g.lineTo(left + columns, top + height);
    g.closePath();
    g.fill();
  }

  _local(event) {
    const box = this.canvas.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  _handleAt(x, y) {
    if (y > 26) return null;
    for (const [marker, name] of [[this.a, "a"], [this.b, "b"]]) {
      if (marker !== null && Math.abs(this.xOf(name === "a" ? marker : marker + 1) - x) <= 16) return name;
    }
    return null;
  }

  _down(event) {
    if (this.frameCount <= 0) return;
    this.canvas.setPointerCapture?.(event.pointerId);
    const [x, y] = this._local(event);
    this.pointers.set(event.pointerId, x);
    if (this.pointers.size === 2) { // pinch / two-finger pan
      const [p, q] = [...this.pointers.values()];
      this.pinch = { distance: Math.abs(p - q) || 1, center: (p + q) / 2, view: [...this.view] };
      this.drag = null;
      return;
    }
    this.drag = this._handleAt(x, y) ?? "seek";
    if (this.drag === "seek") this.callbacks.seek?.(this.frameAt(x));
    this.draw();
  }

  _move(event) {
    if (!this.pointers.has(event.pointerId)) return;
    const [x] = this._local(event);
    this.pointers.set(event.pointerId, x);
    if (this.pinch && this.pointers.size >= 2) {
      const [p, q] = [...this.pointers.values()];
      const distance = Math.abs(p - q) || 1;
      this.view = [...this.pinch.view];
      this.zoom(this.pinch.distance / distance, this.frameAt(this.pinch.center));
      const shift = this.frameAt(this.pinch.center) - this.frameAt((p + q) / 2);
      this.pan(shift);
      return;
    }
    if (this.drag === "seek") this.callbacks.seek?.(this.frameAt(x));
    else if (this.drag === "a" || this.drag === "b") {
      let frame = this.frameAt(x);
      if (this.drag === "b") frame = Math.max(0, frame - 1);
      this.callbacks.marker?.(this.drag, frame);
    }
  }

  _up(event) {
    this.pointers.delete(event.pointerId);
    if (this.pointers.size < 2) this.pinch = null;
    if (!this.pointers.size) {
      this.drag = null;
      this.draw();
    }
  }

  _double(event) {
    const frame = this.frameAt(this._local(event)[0]);
    const index = this.proposals.findIndex((p) => p.start <= frame && frame <= p.end);
    if (index >= 0) this.callbacks.proposal?.(index);
  }

  _wheel(event) {
    if (this.frameCount <= 1) return;
    event.preventDefault();
    const delta = event.deltaY || event.deltaX;
    if (event.shiftKey) {
      const [start, end] = this.view;
      this.pan(Math.trunc((end - start) * 0.15) * (delta < 0 ? -1 : 1));
    } else {
      this.zoom(delta < 0 ? 0.8 : 1.25, this.frameAt(this._local(event)[0]));
    }
  }
}

function roundRect(g, x, y, width, height, radius) {
  g.beginPath();
  if (g.roundRect) {
    g.roundRect(x, y, width, height, radius);
    return;
  }
  g.rect(x, y, width, height);
}

// Signer / text regions ([x, y, w, h] in video pixels) on a canvas showing the video at full size (FrameCanvas).
export function drawRegions(context, regions) {
  for (const [kind, region] of Object.entries(regions)) {
    if (!region) continue;
    const [x, y, w, h] = region;
    const color = kind === "signer" ? COLORS.signer : COLORS.textRegion;
    context.lineWidth = Math.max(2, context.canvas.width / 320);
    context.strokeStyle = color;
    context.setLineDash(kind === "text" ? [10, 6] : []);
    context.strokeRect(x, y, w, h);
    context.setLineDash([]);
    const label = kind === "signer" ? "Người ký" : "Chữ";
    const size = Math.max(14, Math.round(context.canvas.width / 40));
    context.font = `bold ${size}px system-ui, sans-serif`;
    const width = context.measureText(label).width + 10;
    context.fillStyle = color;
    context.fillRect(x, Math.max(0, y - size - 6), width, size + 6);
    context.fillStyle = "#0b111c";
    context.fillText(label, x + 5, Math.max(size, y - 4));
  }
}

// Paths of both wrists and index fingertips over the 30 saved frames (SegmentSample.trajectory), up to `upTo`.
export function trajectory(keypoints) {
  const paths = [];
  for (const start of [1536, 1599]) {
    const valid = keypoints.map((frame) => frame.subarray(start, start + 63).some((v) => Math.abs(v) > 1e-6));
    if (!valid.some(Boolean)) continue;
    for (const landmark of [0, 8]) {
      paths.push({ points: keypoints.map((frame) => [frame[start + landmark * 3], frame[start + landmark * 3 + 1]]), valid,
        color: landmark === 0 ? (start === 1536 ? "rgb(120,220,80)" : "rgb(70,170,255)") : "rgb(255,120,210)" });
    }
  }
  return paths;
}

export function drawTrajectory(context, paths, upTo = Infinity) {
  const { width, height } = context.canvas;
  context.lineWidth = Math.max(2, width / 300);
  for (const path of paths) {
    context.strokeStyle = path.color;
    context.beginPath();
    let open = false;
    path.points.forEach(([x, y], index) => {
      if (index > upTo || !path.valid[index]) {
        open = false;
        return;
      }
      if (!open) context.moveTo(x * width, y * height);
      else context.lineTo(x * width, y * height);
      open = true;
    });
    context.stroke();
  }
}

// Drawing a region on the stage: pointer drag on `canvas` (CSS-scaled, intrinsic size = video size) -> region [x, y, w, h].
export class RegionTool {
  constructor(canvas, done) {
    this.canvas = canvas;
    this.done = done;
    this.kind = null;
    this.start = null;
    canvas.addEventListener("pointerdown", (event) => {
      if (!this.kind) return;
      event.preventDefault();
      canvas.setPointerCapture?.(event.pointerId);
      this.start = this._point(event);
      this.current = this.start;
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.kind || !this.start) return;
      this.current = this._point(event);
      this.done(this.kind, this._region(), false);
    });
    const finish = (event) => {
      if (!this.kind || !this.start) return;
      this.current = this._point(event);
      const region = this._region();
      const kind = this.kind;
      this.start = null;
      this.done(kind, region[2] >= 16 && region[3] >= 16 ? region : null, true);
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", () => { this.start = null; });
  }

  activate(kind) {
    this.kind = kind;
    this.canvas.style.touchAction = kind ? "none" : "";
    this.canvas.style.cursor = kind ? "crosshair" : "";
  }

  _point(event) {
    const box = this.canvas.getBoundingClientRect();
    // object-fit: contain — the picture may not fill the element.
    const scale = Math.min(box.width / this.canvas.width, box.height / this.canvas.height);
    const offsetX = (box.width - this.canvas.width * scale) / 2, offsetY = (box.height - this.canvas.height * scale) / 2;
    const x = (event.clientX - box.left - offsetX) / scale, y = (event.clientY - box.top - offsetY) / scale;
    return [Math.min(Math.max(x, 0), this.canvas.width), Math.min(Math.max(y, 0), this.canvas.height)];
  }

  _region() {
    const [x0, y0] = this.start, [x1, y1] = this.current;
    return [Math.round(Math.min(x0, x1)), Math.round(Math.min(y0, y1)), Math.round(Math.abs(x1 - x0)), Math.round(Math.abs(y1 - y0))];
  }
}

// The 30 frames exactly as stored (FilmStrip): thumbnails (canvases or ImageBitmaps) in a scrolling row; tapping one
// calls pick(index).
export function renderFilmstrip(container, thumbnails, pick) {
  container.replaceChildren(...thumbnails.map((thumbnail, index) => {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "film-frame";
    cell.title = `Frame ${index + 1}/30`;
    const canvas = document.createElement("canvas");
    canvas.width = thumbnail.width;
    canvas.height = thumbnail.height;
    canvas.getContext("2d").drawImage(thumbnail, 0, 0);
    cell.append(canvas, Object.assign(document.createElement("span"), { textContent: String(index + 1) }));
    cell.addEventListener("click", () => pick?.(index));
    return cell;
  }));
}
