// The phone's multi-source collector ("Thu thập đa nguồn": collector_window.py's workflow on one screen). Sources:
// videos picked from the phone (several at once, named by a CSV / text file, by their folder, or — QIPEDC files — by
// their dictionary code), the QIPEDC dictionary shipped with the app (search, topics; its videos are saved from the
// QIPEDC site and picked as files, since QIPEDC lets no other site read them), links to videos whose host allows it,
// and videos shared to the installed app (Android). Each video: timeline with A–B, signer and text regions, live
// landmarks, proposals from hand activity and captions (OCR), the 30 frames exactly as stored with the trajectory, save;
// videos whose sign name is known are saved in a batch (collectors.import_labelled).

import { VideoClip, computeSegment, scanVideo, showFrame, linspaceIndices } from "./videocollect.js";
import { createStaticHolistic, FACE_TESSELATION } from "./holistic.js";
import { drawDense } from "./draw.js";
import { MIN_SEGMENT_FRAMES, overlaps } from "./segmentation.js";
import { RegionTool, Timeline, drawRegions, drawTrajectory, renderFilmstrip, secondsText, trajectory } from "./collectorui.js";

const MIN_TEMPLATE_HAND_COVERAGE = 0.40;
const FRAME_COUNT = 30;
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

export class Collector {
  // env: {store, recognition, bundle, view (stage canvas), video (<video> for clips), status(text), confirm(title, text,
  // ok) -> Promise<bool>, vocabulary() -> [names], saved() (after a save: refresh lists)}
  constructor(env) {
    this.env = env;
    this.items = [];
    this.current = null; // {entry, clip, state}
    this.position = 0;
    this.playing = false;
    this.busy = null; // "open" | "preview" | "scan" | "save" | "batch" | "ocr"
    this.shown = true; // the stage is the collector's (suspend() / resume())
    this.stop = false;
    this.preview = null; // computed segment for the current A–B and region
    this.previewKey = null;
    this.previewTimer = 0;
    this.liveTimer = 0;
    this.live = null; // StaticHolistic
    this.liveDrawing = null;
    this.catalog = null; // QIPEDC
    this.ocrPromise = null; // one text reader, made the first time it is needed
    this.timeline = new Timeline($("timeline"), {
      seek: (frame) => this.seek(frame),
      marker: (which, frame) => this.markerMoved(which, frame),
      proposal: (index) => this.activateProposal(index),
    });
    this.regionTool = new RegionTool(env.view, (kind, region, finished) => this.regionDrawn(kind, region, finished));
    this.wire();
  }

  // ------------------------------------------------------------------ sources

  wire() {
    $("pickVideo").addEventListener("click", () => $("videoFile").click());
    $("videoFile").addEventListener("change", () => {
      const files = [...$("videoFile").files];
      $("videoFile").value = "";
      this.addFiles(files);
    });
    $("openUrl").addEventListener("click", () => this.addUrl($("sourceUrl").value.trim()));
    $("sourceUrl").addEventListener("keydown", (event) => { if (event.key === "Enter") this.addUrl($("sourceUrl").value.trim()); });
    $("openQipedc").addEventListener("click", () => this.toggleQipedc());
    $("qipedcSearch").addEventListener("input", () => this.renderQipedc());
    $("qipedcTopic").addEventListener("change", () => this.renderQipedc());
    $("qipedcAdd").addEventListener("click", () => this.addChosenWords());
    $("saveLabelled").addEventListener("click", () => this.saveLabelled());
    $("stepBack").addEventListener("click", () => this.step(-1));
    $("stepForward").addEventListener("click", () => this.step(1));
    $("playPause").addEventListener("click", () => this.togglePlay());
    $("zoomIn").addEventListener("click", () => this.timeline.zoom(0.6));
    $("zoomOut").addEventListener("click", () => this.timeline.zoom(1.6));
    $("setA").addEventListener("click", () => this.setMarker("a"));
    $("setB").addEventListener("click", () => this.setMarker("b"));
    $("toolSigner").addEventListener("click", () => this.chooseTool("signer"));
    $("toolText").addEventListener("click", () => this.chooseTool("text"));
    $("clearRegions").addEventListener("click", () => this.clearRegions());
    $("liveLandmarks").addEventListener("change", () => {
      this.liveFailed = false;
      this.redraw(true);
    });
    $("showTrajectory").addEventListener("change", () => this.redraw());
    $("scanItem").addEventListener("click", () => this.scan());
    $("stopScan").addEventListener("click", () => { this.stop = true; });
    $("stopBatch").addEventListener("click", () => { this.stop = true; });
    $("saveSegment").addEventListener("click", () => this.saveCurrent());
    $("saveChecked").addEventListener("click", () => this.saveChecked());
    $("closeEditor").addEventListener("click", async () => {
      if (!(await this.claim("close"))) return;
      this.busy = null;
      this.leave();
    });
    $("segmentName").addEventListener("input", () => { if (this.current) this.current.state.labelFrom = "typed"; });
  }

  status(text) {
    this.env.status(text);
  }

  async sources() {
    this.sourcesModule ??= await import("./sources.js");
    return this.sourcesModule;
  }

  async qipedcCatalog() {
    if (!this.catalog) {
      const src = await this.sources();
      try {
        this.catalog = await src.loadQipedc();
      } catch {
        throw new Error("Chưa có danh mục QIPEDC trong bản ứng dụng này.");
      }
    }
    return this.catalog;
  }

  catalogEntries() {
    return this.catalog?.catalog ?? [];
  }

  // Deletions known on the phone: the PC's (bundle) and the phone's own (phone-collected samples deleted here).
  async deletionIndex() {
    const local = await this.env.store.deletedProvenance();
    return { segments: [...(this.env.bundle.segments ?? []), ...local.filter((key) => key.startsWith("seg:"))],
      sources: [...(this.env.bundle.sources ?? []), ...local.filter((key) => key.startsWith("src:"))] };
  }

  newEntry(item, file) {
    return { key: crypto.randomUUID?.() ?? `${Date.now()}-${this.items.length}`, file, item,
      status: file ? "chưa mở" : "chờ tệp video",
      state: { regions: {}, proposals: [], scan: null, a: null, b: null,
        labelFrom: item.label ? (String(item.mediaId).startsWith("qipedc:") ? "dictionary" : "folder") : "typed" } };
  }

  // Videos (and label files) picked together: names from a CSV / text file, the folder, or the QIPEDC code.
  async addFiles(files) {
    if (!files.length) return;
    const src = await this.sources();
    let catalog = null;
    try {
      catalog = (await this.qipedcCatalog()).catalog;
    } catch { /* no dictionary in this build: files keep their own names */ }
    // A label table picked with the videos: a .csv is found by labelFiles itself; a .txt / .tsv is passed explicitly.
    const table = files.find((file) => /\.(tsv|txt)$/i.test(file.name));
    let items;
    try {
      items = await src.labelFiles(files, table ?? null, { csvName: table?.name ?? "labels.csv", catalog });
    } catch (error) {
      this.status(error.message || String(error));
      return;
    }
    let added = 0, filled = 0;
    for (const item of items) {
      const file = item.file;
      const waiting = this.items.find((existing) => !existing.file && existing.item.mediaId === item.mediaId);
      if (waiting) { // a QIPEDC word chosen in the dictionary, waiting for its video
        waiting.file = file;
        waiting.status = "chưa mở";
        filled++;
        continue;
      }
      if (this.items.some((existing) => existing.item.mediaId === item.mediaId)) continue;
      this.items.push(this.newEntry(item, file));
      added++;
    }
    this.renderList();
    this.status(added || filled
      ? `Đã thêm ${added} video${filled ? `, ghép ${filled} video QIPEDC vào từ đã chọn` : ""}. Bấm một video để xem và tìm đoạn ký; video đã có tên có thể “Lưu tất cả video có tên”.`
      : "Không có video mới (các video này đã có trong danh sách).");
    if (added && !this.current) this.open(this.items[this.items.length - added]);
  }

  async addUrl(text) {
    if (!text) return;
    const src = await this.sources();
    let kind;
    try {
      kind = src.classify(text);
    } catch (error) {
      this.status(error.message || String(error));
      return;
    }
    const link = kind === "url" ? src.linkKind(text) : kind;
    if (link === "page") {
      this.status(src.pageLinkMessage(text));
      return;
    }
    if (kind === "qipedc" || link === "qipedc") {
      await this.toggleQipedc(true);
      $("qipedcSearch").value = text.replace(/^qipedc:/i, "");
      this.renderQipedc();
      this.status("QIPEDC không cho trang khác đọc video của họ: hãy lưu video từ trang QIPEDC về máy rồi chọn ở “Chọn video…” (giữ tên mã, ví dụ D0001B.mp4).");
      return;
    }
    if (kind !== "url") {
      this.status("Chỉ mở được link video (.mp4…) từ trang cho phép tải, hoặc mã QIPEDC.");
      return;
    }
    this.status("Đang tải video từ link…");
    try {
      const item = await src.urlItem(text);
      if (this.items.some((existing) => existing.item.mediaId === item.mediaId)) {
        this.status("Link này đã có trong danh sách.");
        return;
      }
      const file = await src.fetchVideo(text, { progress: (fraction) => this.status(`Đang tải video từ link… ${Math.round((fraction ?? 0) * 100)}%`) });
      const entry = this.newEntry(item, file);
      this.items.push(entry);
      $("sourceUrl").value = "";
      this.renderList();
      this.open(entry);
    } catch (error) {
      this.status(error.message || String(error));
    }
  }

  // Videos shared to the installed app (Android share sheet).
  async takeShared() {
    const src = await this.sources();
    const files = await src.takeShared();
    if (files.length) await this.addFiles(files);
    return files.length;
  }

  async toggleQipedc(open = null) {
    const panel = $("qipedcPanel");
    const show = open ?? panel.hidden;
    panel.hidden = !show;
    if (!show) return;
    try {
      await this.qipedcCatalog();
    } catch (error) {
      $("qipedcResults").replaceChildren(el("li", { className: "muted small", textContent: error.message }));
      return;
    }
    const topics = this.catalog?.topics ?? [];
    $("qipedcTopic").replaceChildren(el("option", { value: "", textContent: "Tất cả (tìm theo từ)" }),
      ...topics.map((topic) => el("option", { value: topic.key, textContent: topic.title })));
    this.renderQipedc();
  }

  // Words chosen in the dictionary become items waiting for their video (QipedcDialog.selected_entries).
  async addChosenWords() {
    const src = await this.sources();
    const ids = [...$("qipedcResults").querySelectorAll("input[type=checkbox]:checked")].map((box) => box.value);
    if (!ids.length) {
      this.status("Hãy đánh dấu các từ cần thu trong danh sách.");
      return;
    }
    let added = 0;
    for (const item of src.qipedcChoose(ids, this.catalogEntries()).map((entry) => src.qipedcItem(entry))) {
      if (this.items.some((existing) => existing.item.mediaId === item.mediaId)) continue;
      this.items.push(this.newEntry(item, null));
      added++;
    }
    this.renderList();
    this.status(`Đã thêm ${added} từ QIPEDC. Với mỗi từ: bấm “Mở video ↗”, lưu video về máy (giữ tên mã), rồi “Chọn video…” chọn các tệp đó — ứng dụng tự ghép vào đúng từ.`);
  }

  // Search results / a topic's words: whether the data store has the sign, and a link to save the video from QIPEDC.
  async renderQipedc() {
    const src = await this.sources();
    const entries = this.catalogEntries();
    const query = $("qipedcSearch").value.trim();
    const topicKey = $("qipedcTopic").value;
    let shown = [];
    if (topicKey) {
      const topic = (this.catalog?.topics ?? []).find((t) => t.key === topicKey);
      const ids = new Set((topic?.words ?? []).flatMap((word) => word.ids ?? []));
      shown = entries.filter((entry) => ids.has(entry.id));
    } else if (query) {
      shown = src.qipedcSearch(query, entries).slice(0, 60);
    }
    const labels = new Set(this.env.store.signs().map((sign) => sign.label.toLowerCase()));
    $("qipedcResults").replaceChildren(...(shown.length ? shown.map((entry) => {
      const id = entry.id ?? entry._id;
      const item = src.qipedcItem(entry);
      const have = item.label && labels.has(String(item.label).toLowerCase());
      const link = el("a", { href: item.url, target: "_blank", rel: "noopener", textContent: "Mở video ↗", download: `${id}.mp4` });
      return el("li", {}, el("input", { type: "checkbox", value: id, checked: !have }),
        el("span", { textContent: `${entry.word}${item.variant ? ` (${item.variant})` : ""}` }),
        el("span", { className: `badge${have ? " phone" : ""}`, textContent: have ? "đã có trong kho" : "chưa có" }),
        el("span", { className: "muted small", textContent: id }), link);
    }) : [el("li", { className: "muted small", textContent: query || topicKey ? "Không có từ nào khớp." : "Gõ một từ để tìm trong từ điển QIPEDC, hoặc chọn chủ đề." })]));
  }

  renderList() {
    $("sourceList").replaceChildren(...this.items.map((entry) => {
      const item = entry.item;
      const li = el("li", { className: entry === this.current?.entry ? "active" : "" },
        el("strong", { textContent: item.title || entry.file?.name || item.mediaId }),
        el("span", { className: "muted small", textContent: `${item.platform ?? ""}${item.label ? ` · tên: ${item.label}` : ""}` }),
        el("span", { className: "badge", textContent: entry.status }));
      if (!entry.file && item.url) li.append(el("a", { href: item.url, target: "_blank", rel: "noopener", textContent: "Mở video ↗" }));
      li.addEventListener("click", (event) => {
        if (event.target.tagName === "A") return;
        if (!entry.file) {
          this.status(`“${item.title}” chưa có video: bấm “Mở video ↗”, lưu về máy (giữ tên ${String(item.mediaId).replace("qipedc:", "")}.mp4) rồi “Chọn video…”.`);
          return;
        }
        this.open(entry);
      });
      return li;
    }));
    // Never disabled for a moment's work (a tap then waits or says why: idle()); only when there is nothing to save.
    $("saveLabelled").disabled = !this.items.some((entry) => entry.item.label && entry.file);
  }

  // ------------------------------------------------------------------ one video

  async open(entry) {
    if (this.current?.entry === entry) return;
    if (!(await this.claim("open"))) return;
    if (this.current?.entry === entry) {
      this.busy = null;
      return;
    }
    this.leave(false);
    this.status(`Đang mở “${entry.item.title || entry.file.name}”…`);
    let clip;
    try {
      clip = await new VideoClip(entry.file, this.env.video).open();
    } catch (error) {
      entry.status = "không mở được";
      this.status(error.message || String(error));
      if (this.busy === "open") this.busy = null;
      this.renderList();
      return;
    }
    clip.mediaId = entry.item.mediaId; // QIPEDC / link items keep their own media id
    if (this.busy === "open") this.busy = null;
    this.current = { entry, clip, state: entry.state };
    entry.status = entry.status === "chưa mở" ? "đang xem" : entry.status;
    $("itemEditor").hidden = false;
    $("editorTitle").textContent = entry.item.title || entry.file.name;
    $("editorInfo").textContent = `${clip.width}x${clip.height} · ${clip.fps} fps · ${(clip.frameCount / clip.fps).toFixed(1)} giây`
      + (entry.item.variant ? ` · biến thể miền ${entry.item.variant}` : "");
    $("segmentName").value = entry.item.label ?? "";
    this.timeline.reset(clip.frameCount, clip.fps);
    await this.refreshSaved();
    if (this.current?.entry !== entry) return;
    if (entry.state.scan) this.applyScan(entry.state.scan, true);
    this.timeline.set({ a: entry.state.a, b: entry.state.b });
    this.preview = null;
    this.previewKey = null;
    renderFilmstrip($("filmstrip"), []);
    this.renderList();
    this.renderProposals();
    await this.seek(entry.state.a ?? 0);
    this.updateSegment();
    this.schedulePreview(); // A–B kept from an earlier visit
    this.status(entry.item.label ? `Video có tên “${entry.item.label}”: bấm “Tìm đoạn ký” (tự chọn đoạn từ lúc tay bắt đầu đến lúc tay hạ) rồi Lưu.`
      : "Bấm “Tìm đoạn ký” để ứng dụng đề xuất các đoạn, hoặc tự đặt A–B. Khoanh “Vùng chữ” để đọc tên từ phụ đề; “Vùng người ký” khi video có nhiều người.");
  }

  // The collector is no longer shown (another tab, the camera source): no playback, no tool on the stage.
  suspend() {
    this.shown = false;
    this.stopPlaying();
    clearTimeout(this.liveTimer);
    clearTimeout(this.previewTimer);
    this.regionTool.activate(null);
    $("toolSigner").classList.remove("active");
    $("toolText").classList.remove("active");
    $("playPause").textContent = "▶︎";
  }

  // Shown again: the stage shows the frame (or stays dark while a scan / save uses the video, until it is done).
  resume() {
    this.shown = true;
    if (this.current && this.previewKey !== this.key()) this.schedulePreview(); // cleared by suspend()
    if (this.current && (!this.busy || this.busy === "open")) {
      this.seek(this.position);
      return true;
    }
    return false;
  }

  leave(hide = true) {
    this.playing = false;
    this.env.video.pause();
    this.regionTool.activate(null);
    this.current?.clip.close();
    this.current = null;
    if (hide) $("itemEditor").hidden = true;
    this.renderList();
  }

  // Saved segments of this video (the phone's and the bundle's) and segments the user deleted on the PC.
  async refreshSaved() {
    const current = this.current;
    if (!current) return;
    const mediaId = current.clip.mediaId;
    const saved = [];
    for (const sample of this.env.store.samples()) {
      if (sample.origin === "phone" && sample.metadata?.media_id === mediaId) {
        saved.push([Number(sample.metadata.start_frame), Number(sample.metadata.end_frame), sample.display]);
      } else if (sample.media && sample.media[0] === mediaId) {
        saved.push([sample.media[1], sample.media[2], sample.display]);
      }
    }
    let deleted = [];
    try {
      const src = await this.sources();
      const index = await this.deletionIndex();
      deleted = (await src.deletedSegments(mediaId, index)).map(([start, end]) => [Math.round(start * current.clip.fps), Math.round(end * current.clip.fps)]);
      current.isDeleted = await src.deletedChecker(mediaId, index);
    } catch {
      deleted = [];
      current.isDeleted = null;
    }
    current.saved = saved;
    current.deleted = deleted;
    this.timeline.set({ saved, deleted });
  }

  async seek(frame) {
    const current = this.current;
    if (!current) return;
    this.position = Math.min(Math.max(0, Math.round(frame)), current.clip.frameCount - 1);
    this.timeline.setPosition(this.position);
    $("timeText").textContent = `${secondsText(this.position, current.clip.fps)} / ${secondsText(current.clip.frameCount - 1, current.clip.fps)}`;
    if (this.playing || !this.shown) return;
    if (this.busy && this.busy !== "open") return; // the video belongs to that work: the frame is shown when it is done
    const ticket = (this.seekTicket = (this.seekTicket ?? 0) + 1);
    await showFrame(current.clip, this.env.view, this.position);
    if (ticket !== this.seekTicket) return;
    this.liveDrawing = null;
    this.redraw();
    this.scheduleLive();
  }

  step(amount) {
    this.seek(this.position + amount);
  }

  // Landmarks of the frame shown (a static Holistic, inside the signer region), after scrubbing settles.
  scheduleLive() {
    clearTimeout(this.liveTimer);
    if (!$("liveLandmarks").checked || this.playing || !this.shown) return;
    this.liveTimer = setTimeout(async () => {
      try {
        const current = this.current;
        if (!current || this.busy || !this.shown) return;
        this.livePromise ??= createStaticHolistic().catch((error) => { this.livePromise = null; throw error; });
        this.live = await this.livePromise;
        if (current !== this.current || this.busy || !this.shown) return;
        let canvas;
        try {
          canvas = await current.clip.frame(this.position);
        } catch {
          return; // closed meanwhile
        }
        const crop = current.clip.crop(current.state.regions.signer);
        const result = this.live.detect(canvas, canvas.width, canvas.height, crop);
        if (current !== this.current || !this.shown) return;
        const scale = current.clip.width / canvas.width;
        this.liveDrawing = { drawing: result.drawing, crop: crop ? crop.map((value) => value * scale) : null };
        await showFrame(current.clip, this.env.view, this.position);
        this.redraw();
      } catch (error) {
        if (!this.liveFailed) this.status(`Không chạy được điểm mốc trực tiếp: ${error.message || error}`);
        this.liveFailed = true;
        $("liveLandmarks").checked = false;
      }
    }, 180);
  }

  // The frame on the stage with the regions, the live landmarks and the trajectory of the segment.
  redraw(live = false) {
    const current = this.current;
    if (!current) return;
    const view = this.env.view, context = view.getContext("2d");
    if (this.liveDrawing && $("liveLandmarks").checked) {
      drawDense(context, this.liveDrawing.drawing, view.width, view.height, this.liveDrawing.crop, FACE_TESSELATION);
    }
    drawRegions(context, current.state.regions);
    if (this.preview && this.previewKey === this.key() && $("showTrajectory").checked) {
      const indices = this.preview.indices;
      let upTo = indices.findIndex((index) => index > this.position);
      upTo = upTo < 0 ? indices.length : upTo;
      drawTrajectory(context, trajectory(this.preview.keypoints), upTo - 1);
    }
    if (live) this.scheduleLive();
  }

  togglePlay() {
    const current = this.current;
    if (!current) return;
    if (this.playing) {
      this.playing = false;
      current.clip.video.pause();
      $("playPause").textContent = "▶︎";
      this.seek(this.position);
      return;
    }
    if (!this.shown) return;
    if (this.busy) {
      this.idle().then((idle) => { if (idle && this.shown && this.current === current && !this.playing) this.togglePlay(); });
      return;
    }
    this.playing = true;
    $("playPause").textContent = "⏸";
    const video = current.clip.video, view = this.env.view;
    const end = this.timeline.b !== null && this.position < this.timeline.b ? (this.timeline.b + 1) / current.clip.fps : Infinity;
    current.clip.frame(this.position).then(() => {
      if (!this.playing || !this.shown || this.current !== current) return false; // stopped while the first frame was sought
      return video.play().then(() => true);
    }).then((started) => {
      if (!started) return;
      if (!this.playing || !this.shown || this.current !== current) { video.pause(); return; }
      const draw = () => {
        if (!this.playing || !this.shown || this.current !== current) {
          if (this.current === current) video.pause();
          return;
        }
        view.getContext("2d").drawImage(video, 0, 0, view.width, view.height);
        drawRegions(view.getContext("2d"), current.state.regions);
        this.position = Math.min(current.clip.frameCount - 1, Math.floor(video.currentTime * current.clip.fps));
        this.timeline.setPosition(this.position);
        $("timeText").textContent = `${secondsText(this.position, current.clip.fps)} / ${secondsText(current.clip.frameCount - 1, current.clip.fps)}`;
        if (video.ended || video.paused || video.currentTime >= end) {
          this.togglePlay();
          return;
        }
        if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(draw);
        else requestAnimationFrame(draw);
      };
      draw();
    }).catch(() => { this.playing = false; $("playPause").textContent = "▶︎"; });
  }

  // Work on the clip (30 frames, a scan) seeks the video itself: playback stops first.
  stopPlaying() {
    if (!this.playing) return;
    this.playing = false;
    this.current?.clip.video.pause();
    $("playPause").textContent = "▶︎";
  }

  chooseTool(kind) {
    const active = this.regionTool.kind === kind ? null : kind;
    this.regionTool.activate(active);
    $("toolSigner").classList.toggle("active", active === "signer");
    $("toolText").classList.toggle("active", active === "text");
    if (active) this.status(active === "signer" ? "Kéo trên hình để khoanh người ký (khi video có nhiều người)."
      : "Kéo trên hình để khoanh vùng phụ đề (tên ký hiệu) — ứng dụng sẽ đọc chữ trong vùng này khi quét.");
  }

  async regionDrawn(kind, region, finished) {
    const current = this.current;
    if (!current) return;
    if (!finished) { // live rectangle while dragging
      if (this.busy) return; // the stage follows the video's owner; the region is drawn when the drag ends
      await showFrame(current.clip, this.env.view, this.position);
      drawRegions(this.env.view.getContext("2d"), { ...current.state.regions, [kind]: region });
      return;
    }
    this.chooseTool(kind); // one region per drag
    if (region) current.state.regions[kind] = region;
    this.preview = null;
    this.previewKey = null;
    await this.seek(this.position);
    if (kind === "text" && region) this.readCaptionNow();
    this.schedulePreview();
  }

  clearRegions() {
    if (!this.current) return;
    this.current.state.regions = {};
    this.preview = null;
    this.previewKey = null;
    this.seek(this.position);
    this.schedulePreview();
  }

  // OCR of the text region on the frame shown: the candidates become name suggestions.
  async readCaptionNow() {
    const current = this.current;
    const region = current?.state.regions.text;
    if (!region || !(await this.claim("ocr"))) return;
    if (current !== this.current) {
      this.busy = null;
      return;
    }
    this.status("Đang đọc chữ trong vùng chữ… (lần đầu tải bộ đọc chữ khoảng 15 MB)");
    try {
      const { readCaption } = await import("./videocollect.js");
      const ocr = await this.ocr();
      await current.clip.frame(this.position);
      const image = current.clip.regionImage(region);
      const [candidates, raw] = await readCaption(ocr, image, this.env.vocabulary(), true);
      if (current !== this.current) return;
      this.showSuggestions(candidates, raw);
    } catch (error) {
      this.status(`Không đọc được chữ: ${error.message || error}`);
    } finally {
      if (this.busy === "ocr") this.busy = null;
      if (current === this.current) this.seek(this.position);
    }
  }

  ocr() {
    this.ocrPromise ??= import("./ocr.js").then((module) => module.createOcr())
      .catch((error) => { this.ocrPromise = null; throw error; });
    return this.ocrPromise;
  }

  showSuggestions(candidates, raw) {
    const current = this.current;
    $("ocrSuggestions").replaceChildren(...candidates.slice(0, 5).map(([text, confidence]) => {
      const chip = el("button", { type: "button", textContent: `${text} (${Math.round(confidence * 100)}%)` });
      chip.addEventListener("click", () => {
        $("segmentName").value = text;
        current.state.labelFrom = "ocr";
        current.state.ocrText = raw;
        current.state.ocrConfidence = confidence;
      });
      return chip;
    }));
    this.status(candidates.length ? `Đọc được: “${raw}”. Bấm một gợi ý để đặt tên.` : `Không đọc được tên trong vùng chữ${raw ? ` (thấy “${raw}”)` : ""}.`);
  }

  // ------------------------------------------------------------------ A–B and the segment

  setMarker(which) {
    const state = this.current?.state;
    if (!state) return;
    if (which === "a") {
      state.a = this.position;
      if (state.b !== null && state.b < state.a) state.b = null;
    } else {
      state.b = this.position;
      if (state.a === null || state.a > state.b) state.a = Math.max(0, state.b - Math.round(this.current.clip.fps));
    }
    state.activeProposal = null;
    this.timeline.set({ a: state.a, b: state.b });
    this.updateSegment();
    this.schedulePreview();
  }

  markerMoved(which, frame) {
    const state = this.current?.state;
    if (!state) return;
    if (which === "a") state.a = Math.min(frame, state.b ?? frame);
    else state.b = Math.max(frame, state.a ?? frame);
    this.timeline.set({ a: state.a, b: state.b });
    this.seek(which === "a" ? state.a : state.b);
    this.updateSegment();
    this.schedulePreview();
  }

  updateSegment() {
    const current = this.current;
    const state = current?.state;
    const ready = state && state.a !== null && state.b !== null;
    $("segmentBox").hidden = !current;
    if (!ready) {
      $("segmentInfo").textContent = "Đặt A (đầu) và B (cuối) của một lần ký — hoặc bấm một đề xuất.";
      return;
    }
    const frames = state.b - state.a + 1;
    let text = `Đoạn ${secondsText(state.a, current.clip.fps)}–${secondsText(state.b + 1, current.clip.fps)} · ${(frames / current.clip.fps).toFixed(2)} giây`;
    if (frames < MIN_SEGMENT_FRAMES) text += ` · quá ngắn (cần ít nhất ${MIN_SEGMENT_FRAMES} frame)`;
    if (this.preview && this.previewKey === this.key()) text += ` · tay hiện ${Math.round(this.preview.coverage * 100)}% frame`;
    $("segmentInfo").textContent = text;
  }

  key() {
    const state = this.current?.state;
    return state ? JSON.stringify([state.a, state.b, state.regions.signer ?? null]) : null;
  }

  // The 30 frames exactly as stored, computed shortly after A–B or the signer region change.
  schedulePreview() {
    clearTimeout(this.previewTimer);
    const state = this.current?.state;
    if (!state || state.a === null || state.b === null || state.b - state.a + 1 < MIN_SEGMENT_FRAMES) return;
    this.previewTimer = setTimeout(() => this.computePreview(), 500);
  }

  async computePreview(now = false) {
    const current = this.current;
    if (!current) return null;
    if (this.busy) {
      if (this.busy !== "preview" && !now) this.schedulePreview(); // after that work (the running preview redoes itself)
      return null;
    }
    if (this.playing && !now) { // the user is watching: the 30 frames are taken once playback stops
      this.schedulePreview();
      return null;
    }
    const key = this.key();
    if (this.preview && this.previewKey === key) return this.preview;
    const { a, b, regions } = current.state;
    this.stopPlaying();
    this.busy = "preview";
    let finish;
    this.previewDone = new Promise((resolve) => { finish = resolve; });
    $("filmstrip").replaceChildren(el("p", { className: "muted small", textContent: "Đang lấy 30 frame của đoạn…" }));
    try {
      const segment = await this.segmentOf(current.clip, a, b, regions.signer, true);
      if (current !== this.current || this.key() !== key) return null;
      this.preview = segment;
      this.previewKey = key;
      renderFilmstrip($("filmstrip"), segment.thumbnails, (index) => this.seek(segment.indices[index]));
      this.updateSegment();
      return segment;
    } catch (error) {
      if (current === this.current) $("filmstrip").replaceChildren(el("p", { className: "muted small", textContent: error.message || String(error) }));
      return null;
    } finally {
      if (this.busy === "preview") this.busy = null;
      finish();
      if (current === this.current && !this.busy) {
        this.seek(this.position); // the frame the user moved to meanwhile
        if (this.key() !== key) this.schedulePreview(); // A–B or the signer region changed meanwhile
      }
    }
  }

  // Wait for the 30 frames being computed; a tap meanwhile must never be lost. Returns false (with a message) when
  // something else is running. Work that will use the video takes it with claim() instead.
  async idle() {
    while (this.busy === "preview") await this.previewDone;
    if (!this.busy) return true;
    this.status({ scan: "Đang quét video — đợi quét xong (hoặc bấm Dừng).", save: "Đang lưu…", batch: "Đang lưu các video có tên…",
      open: "Đang mở video…", ocr: "Đang đọc chữ trong vùng chữ…" }[this.busy] ?? "Đang bận, thử lại sau giây lát.");
    return false;
  }

  // idle(), then `kind` takes the collector in the same step that found it free (two taps waiting on one preview
  // never both start). Returns false when something else holds it.
  async claim(kind) {
    while (this.busy === "preview") await this.previewDone;
    if (this.busy) return this.idle();
    this.busy = kind;
    return true;
  }

  // The 30 frames of [a, b] as stored, with the share of frames showing a hand.
  async segmentOf(clip, a, b, region, thumbnails = false) {
    const segment = await computeSegment(clip, a, b, { region, thumbnails });
    const { gestureFeatures, handCoverage } = await import("./translate.js");
    segment.coverage = handCoverage(gestureFeatures(segment.keypoints));
    return segment;
  }

  // ------------------------------------------------------------------ proposals

  async scan() {
    const current = this.current;
    if (!current || !(await this.claim("scan"))) return;
    if (current !== this.current) {
      this.busy = null;
      return;
    }
    this.stopPlaying();
    this.stop = false;
    $("scanItem").disabled = true;
    $("stopScan").hidden = false;
    $("scanProgress").hidden = false;
    const { regions } = current.state;
    const label = current.entry.item.label || null;
    const textRegion = label ? null : regions.text ?? null;
    let ocr = null;
    try {
      if (textRegion) {
        this.status("Đang tải bộ đọc chữ… (một lần, khoảng 15 MB)");
        ocr = await this.ocr();
      } else if (!label) {
        this.status("Đề xuất chỉ dựa vào chuyển động tay (chưa khoanh vùng chữ nên sẽ không có tên). Khoanh vùng chữ để có tên tự động.");
      }
      const result = await scanVideo(current.clip, {
        region: regions.signer, textRegion, ocr, vocabulary: this.env.vocabulary(), label,
        progress: (fraction, text) => { $("scanProgress").value = fraction; this.status(text); },
        cancelled: () => this.stop,
      });
      if (current !== this.current) return;
      current.state.scan = result;
      this.applyScan(result);
      const fresh = current.state.proposals.filter((p) => p.state === "proposed").length;
      current.entry.status = `${current.state.proposals.length} đề xuất`;
      this.renderList();
      if (!current.state.proposals.length) {
        this.status(`Không tìm thấy đoạn ký hiệu nào (không thấy tay chuyển động${textRegion ? " hoặc không đọc được chữ trong vùng chữ" : ""}).`);
      } else {
        this.status(`Đề xuất ${current.state.proposals.length} đoạn (${fresh} mới). Bấm từng đề xuất để xem, sửa tên rồi Lưu; hoặc “Lưu các đề xuất đã chọn”.`);
        const first = current.state.proposals.findIndex((p) => p.state === "proposed");
        if (first >= 0) this.activateProposal(first);
      }
    } catch (error) {
      this.status(this.stop ? "Đã dừng quét." : `Quét video không thành công: ${error.message || error}`);
    } finally {
      if (this.busy === "scan") this.busy = null;
      $("scanItem").disabled = false;
      $("stopScan").hidden = true;
      $("scanProgress").hidden = true;
      if (current === this.current) this.seek(this.position);
    }
  }

  // keep: a video opened again keeps its proposals (checks, states); only saved / deleted are marked again.
  applyScan(result, keep = false) {
    const current = this.current;
    const saved = current.saved ?? [];
    const deleted = current.deleted ?? [];
    const before = keep ? current.state.proposals ?? [] : [];
    current.state.proposals = result.proposals.map((proposal, index) => {
      if (before[index] && before[index].state !== "proposed") return before[index];
      const entry = { start: proposal.start, end: proposal.end, label: proposal.label, text: proposal.text,
        confidence: proposal.confidence, reason: proposal.reason, flags: proposal.flags ?? [], state: "proposed",
        checked: before[index] ? before[index].checked : Boolean(proposal.label) && !(proposal.flags ?? []).length };
      if (saved.some(([first, last]) => overlaps(proposal.start, proposal.end, first, last))) Object.assign(entry, { state: "saved", checked: false });
      else if (current.isDeleted ? current.isDeleted(proposal.start, proposal.end, current.clip.fps)
        : deleted.some(([first, last]) => overlaps(proposal.start, proposal.end, first, last))) Object.assign(entry, { state: "deleted", checked: false });
      return entry;
    });
    this.timeline.set({ activity: result.activityCurve ?? null, captions: (result.captions ?? []).map(([start, end, name]) => [start, end, name]),
      proposals: current.state.proposals });
    this.renderProposals();
  }

  renderProposals() {
    const current = this.current;
    const proposals = current?.state.proposals ?? [];
    $("proposalList").replaceChildren(...proposals.map((proposal, index) => {
      const check = el("input", { type: "checkbox", checked: proposal.checked, disabled: proposal.state !== "proposed" });
      check.addEventListener("click", (event) => {
        event.stopPropagation();
        proposal.checked = check.checked;
        this.timeline.draw();
        $("saveChecked").disabled = !proposals.some((p) => p.checked && p.state === "proposed");
      });
      const states = { saved: "đã lưu", deleted: "đã xóa trước đây", skipped: "bỏ qua" };
      const li = el("li", { className: `${proposal.state}${current.state.activeProposal === index ? " active" : ""}` }, check,
        el("strong", { textContent: `${secondsText(proposal.start, current.clip.fps)}–${secondsText(proposal.end + 1, current.clip.fps)}` }),
        el("span", { textContent: proposal.label || "(chưa có tên)" }),
        ...proposal.flags.map((flag) => el("span", { className: "flag", textContent: flag })),
        states[proposal.state] ? el("span", { className: "badge", textContent: states[proposal.state] }) : null);
      li.addEventListener("click", () => this.activateProposal(index));
      return li;
    }));
    $("saveChecked").disabled = !proposals.some((p) => p.checked && p.state === "proposed");
  }

  activateProposal(index) {
    const current = this.current;
    const proposal = current?.state.proposals[index];
    if (!proposal) return;
    current.state.activeProposal = index;
    current.state.a = proposal.start;
    current.state.b = proposal.end;
    if (proposal.label) {
      $("segmentName").value = proposal.label;
      current.state.labelFrom = current.entry.item.label ? current.state.labelFrom : "ocr";
      current.state.ocrText = proposal.text;
      current.state.ocrConfidence = proposal.confidence;
    }
    this.timeline.set({ a: proposal.start, b: proposal.end });
    this.timeline.ensureVisible(proposal.start);
    this.renderProposals();
    this.updateSegment();
    this.seek(proposal.start);
    this.schedulePreview();
  }

  // ------------------------------------------------------------------ saving

  async metadataFor(entry, clip, segment, labelFrom, state) {
    const { sampleMetadataFor } = await import("./collectbatch.js");
    return sampleMetadataFor(entry.item, segment, clip.fps, [clip.width, clip.height], labelFrom, state.regions.text ?? null,
      labelFrom === "ocr" ? state.ocrText ?? "" : "", labelFrom === "ocr" ? state.ocrConfidence ?? null : null,
      segment.region ?? null);
  }

  // What a save stores besides the frames, fixed when the save starts (taps during the save change the next one).
  snapshot(state, extra = {}) {
    return { regions: { ...state.regions }, labelFrom: state.labelFrom, ocrText: state.ocrText, ocrConfidence: state.ocrConfidence,
      ...extra };
  }

  // Store one segment as a sample (collectors.save_segment through the phone's data store).
  async store(entry, clip, segment, display, labelFrom, state, revive = false) {
    if (segment.coverage < MIN_TEMPLATE_HAND_COVERAGE) {
      throw new Error(`Chỉ thấy tay ở ${Math.round(segment.coverage * 100)}% số frame (cần ≥ ${MIN_TEMPLATE_HAND_COVERAGE * 100}%); mẫu này không dùng được để nhận dạng nên không lưu.`);
    }
    const metadata = await this.metadataFor(entry, clip, segment, labelFrom, state);
    const { sampleGeometry } = await import("./translate.js");
    const [aspect, seconds] = sampleGeometry(metadata);
    const saved = await this.env.store.saveSample(display, segment.keypoints, { display, metadata, preview: segment.preview, aspect, seconds, revive });
    this.env.recognition.addSample({ id: saved.id, label: saved.label, display: this.env.store.displayOf(saved.label), keypoints: segment.keypoints,
      aspect, seconds, source: metadata.source });
    return saved;
  }

  async saveCurrent() {
    const current = this.current;
    if (!current || !(await this.idle())) return;
    const state = current.state;
    if (state.a === null || state.b === null) {
      this.status("Hãy chọn đoạn: điểm đầu A và điểm cuối B.");
      return;
    }
    const display = $("segmentName").value.trim();
    if (!display) {
      this.status("Hãy nhập tên ký hiệu hợp lệ (ô Tên ký hiệu).");
      $("segmentName").focus();
      return;
    }
    clearTimeout(this.previewTimer); // computed below if needed, not behind the confirmation
    if (current.saved?.some(([first, last]) => overlaps(state.a, state.b, first, last, 0.8))
      && !(await this.env.confirm("Đoạn đã lưu", "Đoạn này trùng với một mẫu đã lưu từ video này. Vẫn lưu thêm?", "Lưu thêm"))) {
      this.schedulePreview();
      return;
    }
    let revive = false;
    if (current.deleted?.some(([first, last]) => overlaps(state.a, state.b, first, last, 0.5))) {
      if (!(await this.env.confirm("Đoạn đã bị xóa", "Bạn đã từng xóa một mẫu của đoạn này. Vẫn lưu lại?", "Lưu lại"))) {
        this.schedulePreview();
        return;
      }
      revive = true;
    }
    if (!(await this.claim("save"))) return;
    // What is saved is fixed now: a proposal tapped or A–B moved while saving changes the next save, not this one.
    const { a, b } = state;
    const proposalIndex = state.activeProposal;
    const fixed = this.snapshot(state);
    const region = fixed.regions.signer ?? null;
    try {
      if (current !== this.current || current.clip.closed) return;
      if (a === null || b === null) {
        this.status("Hãy chọn đoạn: điểm đầu A và điểm cuối B.");
        return;
      }
      this.stopPlaying();
      let segment = this.preview && this.previewKey === JSON.stringify([a, b, region]) ? this.preview : null;
      if (!segment) {
        this.status(`Đang lấy 30 frame của “${display}”…`);
        segment = await this.segmentOf(current.clip, a, b, region);
      }
      this.status(`Đang lưu “${display}”…`);
      const saved = await this.store(current.entry, current.clip, segment, display, fixed.labelFrom ?? "typed", fixed, revive);
      const proposal = proposalIndex === null || proposalIndex === undefined ? null : state.proposals[proposalIndex];
      if (proposal) Object.assign(proposal, { state: "saved", checked: false });
      this.status(`Đã lưu “${display}” ${secondsText(a, current.clip.fps)}–${secondsText(b + 1, current.clip.fps)} (tay ${Math.round(segment.coverage * 100)}%; ký hiệu này có ${saved.count} mẫu).`);
      current.entry.status = "đã lưu mẫu";
      this.env.saved?.();
      await this.refreshSaved();
      if (current !== this.current) return;
      this.timeline.set({ proposals: state.proposals });
      this.renderProposals();
      this.renderList();
      if (state.a !== a || state.b !== b || state.activeProposal !== proposalIndex || $("segmentName").value.trim() !== display
        || state.labelFrom !== fixed.labelFrom || state.ocrText !== fixed.ocrText) return; // the user went on meanwhile
      state.a = state.b = null;
      state.activeProposal = null;
      this.timeline.set({ a: null, b: null });
      if (state.labelFrom === "ocr" || state.labelFrom === "typed") {
        if (!current.entry.item.label) $("segmentName").value = "";
        state.labelFrom = current.entry.item.label ? state.labelFrom : "typed";
      }
      const next = state.proposals.findIndex((p) => p.state === "proposed");
      this.busy = null;
      if (next >= 0) this.activateProposal(next);
      else this.updateSegment();
    } catch (error) {
      this.status(`Chưa lưu “${display}”: ${error.message || error}`);
    } finally {
      if (this.busy === "save") this.busy = null;
      if (current === this.current) {
        this.seek(this.position);
        if (this.previewKey !== this.key()) this.schedulePreview(); // the 30 frames of the A–B shown now
      }
    }
  }

  // "Lưu các đề xuất đã chọn": every checked proposal with its name, one after another.
  async saveChecked() {
    const current = this.current;
    if (!current || !(await this.claim("save"))) return;
    const chosen = current.state.proposals.map((p, i) => [p, i]).filter(([p]) => p.checked && p.state === "proposed");
    if (!chosen.length || current !== this.current) {
      this.busy = null;
      return;
    }
    this.stopPlaying();
    let done = 0, failed = 0;
    for (const [proposal] of chosen) {
      if (current !== this.current || current.clip.closed) break; // closed: nothing more is taken from it
      const display = proposal.label || $("segmentName").value.trim();
      if (!display) { failed++; continue; }
      this.status(`Đang lưu ${done + failed + 1}/${chosen.length}: “${display}”…`);
      try {
        const fixed = this.snapshot(current.state, { ocrText: proposal.text, ocrConfidence: proposal.confidence });
        const labelFrom = proposal.label && !current.entry.item.label ? "ocr" : fixed.labelFrom ?? "typed";
        const segment = await this.segmentOf(current.clip, proposal.start, proposal.end, fixed.regions.signer ?? null);
        if (current !== this.current || current.clip.closed) break;
        await this.store(current.entry, current.clip, segment, display, labelFrom, fixed);
        Object.assign(proposal, { state: "saved", checked: false });
        done++;
      } catch (error) {
        failed++;
        this.status(`“${display}”: ${error.message || error}`);
      }
    }
    if (this.busy === "save") this.busy = null;
    if (done) {
      this.env.saved?.();
      current.entry.status = "đã lưu mẫu";
    }
    if (current === this.current) {
      this.seek(this.position);
      await this.refreshSaved();
      this.timeline.set({ proposals: current.state.proposals });
      this.renderProposals();
    }
    this.renderList();
    this.status(`Đã lưu ${done}/${chosen.length} đề xuất${failed ? ` (${failed} chưa lưu — xem lý do ở trên)` : ""}`
      + `${current === this.current ? "" : " (video đã đóng: các đề xuất còn lại chưa lưu)"}.`);
  }

  // "Lưu tất cả video có tên": collectors.import_labelled for every video whose sign name is known (QIPEDC, CSV,
  // folder): the span from the first to the last hand activity, saved unless already there or deleted before.
  async saveLabelled() {
    if (!(await this.claim("batch"))) return;
    try {
      await this.runBatch();
    } catch (error) {
      this.status(`Không lưu được các video có tên: ${error.message || error}`);
    } finally {
      if (this.busy === "batch") this.busy = null;
      $("stopBatch").hidden = true;
      this.renderList();
    }
  }

  async runBatch() {
    const todo = this.items.filter((entry) => entry.item.label && entry.file && !["đã lưu", "đã có"].includes(entry.status));
    if (!todo.length) return;
    this.leave();
    this.stop = false;
    $("stopBatch").hidden = false;
    this.renderList();
    const { importLabelled, savedSegmentsIn } = await import("./collectbatch.js");
    const { Cancelled } = await import("./videocollect.js");
    const src = await this.sources();
    const index = await this.deletionIndex();
    let saved = 0;
    for (const [number, entry] of todo.entries()) {
      if (this.stop) break;
      entry.status = "đang xử lý…";
      this.renderList();
      this.status(`Video ${number + 1}/${todo.length}: “${entry.item.title}”…`);
      let clip = null;
      try {
        const [outcome] = await importLabelled(entry.item, {
          open: async () => {
            clip = await new VideoClip(entry.file, this.env.video).open();
            clip.mediaId = entry.item.mediaId;
            return clip;
          },
          scan: (opened, label) => scanVideo(opened, { label, cancelled: () => this.stop,
            progress: (fraction) => this.status(`Video ${number + 1}/${todo.length}: tìm đoạn ký ${Math.round(fraction * 100)}%`) }),
          compute: (opened, first, last) => computeSegment(opened, first, last, { cancelled: () => this.stop }),
          nearDuplicate: (label, keypoints, aspect, frameSeconds) => this.env.recognition.nearDuplicate(label, keypoints, aspect, frameSeconds),
          savedSegments: (mediaId) => savedSegmentsIn(this.env.store.samples(), mediaId),
          sourceDeleted: (mediaId) => src.sourceDeleted(mediaId, index),
          deletedBefore: (mediaId, first, last, fps) => src.deletedBefore(mediaId, first, last, fps, index),
          save: async (label, keypoints, options) => {
            const result = await this.env.store.saveSample(label, keypoints, options);
            this.env.recognition.addSample({ id: result.id, label: result.label, display: this.env.store.displayOf(result.label), keypoints,
              aspect: options.aspect, seconds: options.seconds, source: options.metadata?.source ?? "video" });
            return result;
          },
          close: (opened) => opened.close(),
        }, { Cancelled });
        entry.status = outcome;
        if (outcome === "đã lưu") saved++;
      } catch (error) {
        entry.status = error instanceof Cancelled ? "đã dừng" : `lỗi: ${error.message || error}`;
      } finally {
        clip = null; // closed by importLabelled's close effect
      }
      this.renderList();
    }
    this.busy = null;
    this.env.saved?.();
    this.renderList();
    this.status(`${this.stop ? "Đã dừng. " : ""}Đã lưu ${saved}/${todo.length} video có tên. Trạng thái từng video ở danh sách trên.`);
  }
}

export { linspaceIndices };
