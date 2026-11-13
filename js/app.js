// VSL Studio Mobile: the single-window app. Port of vsl_ui.py's camera pages (translation, collection, face
// enrollment), data store dialog and account dialog; the camera loop is pipeline.js (CameraWorker), recognition
// runs in recognition.worker.js (RecognitionLoop) through recognition.js.

import { loadBundle } from "./bundle.js";
import { Store, DatasetError, DuplicateSampleError, IdentityError, FRAME_COUNT } from "./store.js";
import { canonicalLabel, accentSuggestion, plainText, sameLabel } from "./naming.js";
import * as sentence from "./sentence.js";
import { Recognition } from "./recognition.js";
import { Pipeline, COUNTDOWN_SECONDS, TAKE_HINTS, ENROLL_FRAMES } from "./pipeline.js";
import { POSE_SKELETON, HAND_SKELETON, POSE, LEFT_HAND, RIGHT_HAND } from "./layout.js";
import { appAddress, detectPlatform, installGuide } from "./install.js";

const MIN_TEMPLATE_HAND_COVERAGE = 0.40; // translate.MIN_TEMPLATE_HAND_COVERAGE (checked against translate.js at boot)
const MIN_TEMPLATES_PER_LABEL = 1;

const $ = (id) => document.getElementById(id);
const app = $("app");
const params = new URLSearchParams(location.search);

const HELP = {
  translate: "Nhận mọi ký hiệu trong kho, kể cả từ chỉ có một mẫu QIPEDC; tự chuẩn hóa vị trí, vóc người, tốc độ ký, rung tay, khung hình và tốc độ khung hình của camera, tay nào cũng được. Làm từng ký hiệu rồi hạ tay xuống: kết quả chốt ngay khi tay hạ; khi đánh vần, giữ yên mỗi chữ một chút. Các từ được ghép thành câu theo ngữ pháp tiếng Việt; dừng tay lâu thì sang câu mới. Sau mỗi ký hiệu, bấm “Đúng là …” với từ bạn vừa ký (hoặc “Từ khác…”): lần ký đó được lưu làm mẫu của bạn và câu được sửa.",
  collect: `Nhập tên ký hiệu, chọn số lượt rồi bấm Bắt đầu ghi. Mỗi lượt: đếm ngược ${COUNTDOWN_SECONDS} giây, làm ký hiệu một lần rồi hạ tay xuống — lượt tự kết thúc khi tay đã hạ. Chỉ đoạn đang ký được lưu, thành ${FRAME_COUNT} frame cách đều theo thời gian, dù camera nhanh hay chậm.`,
  enroll: `Chỉ một người trong khung hình. Hãy quay mặt từ trái sang phải: ứng dụng chụp ${ENROLL_FRAMES} frame chia đều ba hướng để nhận ra bạn khi có người khác đi vào.`,
};
const IDENTITY_TEXT = {
  locked: "Đang theo dõi đúng người đã đăng ký",
  hidden: "Mặt đang bị che hoặc quay đi — vẫn theo đúng cơ thể người đã đăng ký",
  mixed: "Có người khác sát bên — chỉ lấy landmark của người đã đăng ký, phần của người khác vẽ màu xám",
  blocked: "Người khác đang chen vào sát người đã đăng ký — chưa lấy được landmark nào; hãy ngồi tách ra một chút",
  searching: "Chưa nhận ra khuôn mặt người đã đăng ký — hãy nhìn thẳng vào camera (chưa lấy landmark của ai)",
  lost: "Không thấy người đã đăng ký — không lấy landmark của người khác trong khung hình",
};
const SOURCE_NAMES = {
  webcam: "webcam", "youtube-subtitle": "YouTube (phụ đề)", YouTube: "YouTube", video: "video",
  ocr: "video (nhận chữ)", OCR: "video (nhận chữ)", manual: "video (theo mốc)", qipedc: "Từ điển QIPEDC (Bộ GD&ĐT)",
};
const CAMERA_TABS = new Set(["translate", "collect"]);

const state = {
  bundle: null, store: null, recognition: null, pipeline: null, version: {},
  tab: "translate", profile: null, cameraWanted: false,
  // translation page (CameraPage)
  words: [], breaks: [], lastSignEnd: null, lastCandidates: [],
  correction: { candidates: [], index: null, rested: false, number: 0 },
  lastSign: null, // {number, sample: {frames, duration, coverage} | null, aspect, size, mirrored, preview, altered}
  // collection
  recording: null, // {text}
  collectSource: "camera", // "camera" | "video" (a video from the phone)
  collector: null, collectorLoading: null,
  // accounts
  pendingAccount: null, enrollment: null,
  // data store
  selectedSign: null, selectedSample: null, previews: new Map(),
  ready: false, installPrompt: null,
};

// ------------------------------------------------------------------ small helpers

function setStatus(element, text) {
  $(element).textContent = text ? `●  ${text}` : "";
}

const pct = (value) => `${Math.round(value * 100)}%`;

function show(element, visible) {
  $(element).hidden = !visible;
}

// The app's one dialog: {title, text, input: {value, type, placeholder} | null, choices: [text], image: Blob | null,
// ok, validate(value) -> error text | null}. Resolves with the input's value (or true without input), null if cancelled.
function ask({ title, text = "", input = null, choices = [], image = null, ok = "Đồng ý", danger = false, validate = null }) {
  const dialog = $("dialog"), field = $("dialogInput"), okButton = $("dialogOk"), picture = $("dialogImage");
  $("dialogTitle").textContent = title;
  $("dialogText").textContent = text;
  $("dialogError").textContent = "";
  field.hidden = !input;
  field.type = input?.type ?? "text";
  field.value = input?.value ?? "";
  field.placeholder = input?.placeholder ?? "";
  field.autocomplete = input?.type === "password" ? "current-password" : "off";
  if (input?.type === "password") field.removeAttribute("list"); else field.setAttribute("list", "signNames");
  okButton.textContent = ok;
  okButton.classList.toggle("danger", danger);
  okButton.classList.toggle("primary", !danger);
  const row = $("dialogChoices");
  row.replaceChildren(...choices.map((choice) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = choice;
    button.addEventListener("click", () => { field.value = choice; field.focus(); });
    return button;
  }));
  row.hidden = !choices.length;
  if (picture.dataset.url) URL.revokeObjectURL(picture.dataset.url);
  picture.hidden = !image;
  if (image) {
    picture.dataset.url = URL.createObjectURL(image);
    picture.src = picture.dataset.url;
  }
  return new Promise((resolve) => {
    const finish = (value) => {
      okButton.removeEventListener("click", accept);
      dialog.removeEventListener("close", cancel);
      resolve(value);
    };
    const accept = (event) => {
      event.preventDefault();
      const value = input ? field.value.trim() : true;
      const error = validate ? validate(value) : null;
      if (error) {
        $("dialogError").textContent = error;
        return;
      }
      dialog.removeEventListener("close", cancel);
      dialog.close("ok");
      finish(value);
    };
    const cancel = () => finish(null);
    okButton.addEventListener("click", accept);
    dialog.addEventListener("close", cancel);
    dialog.showModal();
    if (input) {
      field.focus();
      if (input.type !== "password") field.select();
    }
  });
}

const confirmAction = (title, text, ok = "Đồng ý", danger = false) => ask({ title, text, ok, danger });

function nameError(value) {
  try {
    canonicalLabel(value);
    return null;
  } catch (error) {
    return error.message || "Nhập tên ký hiệu trước khi lưu.";
  }
}

// ------------------------------------------------------------------ tabs and camera

function setTab(tab) {
  if (state.pendingAccount && tab !== "account") return; // finish or cancel the running enrollment first
  if (tab !== "collect") state.collector?.suspend();
  state.tab = tab;
  app.dataset.mode = tab;
  for (const button of document.querySelectorAll(".tabbar button")) button.classList.toggle("active", button.dataset.tab === tab);
  for (const pane of document.querySelectorAll(".pane")) pane.hidden = pane.dataset.pane !== tab;
  if (tab === "data") renderData();
  if (tab === "account") renderAccounts();
  if (tab === "more") renderAbout();
  if (tab === "collect") refreshVocabulary();
  syncCamera();
}

// Which camera mode the visible page needs; the camera runs only while a camera page is shown and the app is visible.
function cameraMode() {
  if (state.pendingAccount) return "enroll";
  if (state.tab === "collect" && state.collectSource === "video") return null;
  return CAMERA_TABS.has(state.tab) ? state.tab : null;
}

async function syncCamera() {
  const pipeline = state.pipeline;
  if (!pipeline) return;
  const mode = cameraMode();
  const visible = document.visibilityState === "visible";
  app.classList.toggle("video-source", state.tab === "collect" && state.collectSource === "video" && !state.pendingAccount);
  if (!mode || !visible || (!state.cameraWanted && mode !== "enroll")) {
    if (pipeline.running || pipeline.starting) pipeline.stop();
    if (state.recording) recordingFinished();
    showCameraOff(mode && !state.cameraWanted ? "Bấm để bật camera" : null);
    if (state.tab === "collect" && state.collectSource === "video") showVideoStage();
    return;
  }
  if (state.recording && mode !== "collect") recordingFinished(); // setMode cancels the takes
  pipeline.setMode(mode === "translate" ? "translation" : mode);
  if (pipeline.running || pipeline.starting) return;
  showCameraOff(null, "Đang mở camera…");
  try {
    await pipeline.start();
  } catch (error) {
    showCameraOff(error.message || String(error));
    state.cameraWanted = false;
    if (state.pendingAccount) cancelEnrollment(error.message);
    return;
  }
  if (cameraMode() !== mode || document.visibilityState !== "visible") syncCamera(); // the page changed meanwhile
}

// The camera's own picture on the stage (or not: width 0), in its own shape, mirrored for the front camera; its shape
// follows the picture when the phone is turned.
$("video").addEventListener("resize", () => {
  const video = $("video");
  if ($("stage").classList.contains("camera-live") && video.videoWidth && video.videoHeight) {
    showLiveCamera(video.videoWidth, video.videoHeight, $("stage").classList.contains("mirrored"));
  }
});

function showLiveCamera(width, height, mirrored) {
  const stage = $("stage");
  const live = Boolean(width && height);
  stage.classList.toggle("camera-live", live);
  stage.classList.toggle("mirrored", live && mirrored);
  stage.classList.toggle("portrait", live && height > width);
  if (live) stage.style.setProperty("--camera-aspect", `${width} / ${height}`);
  else $("video").style.filter = "";
}

function showCameraOff(text, banner = null) {
  const off = $("cameraOff");
  off.hidden = !text;
  if (text) $("cameraOffText").textContent = text;
  showBanner(banner);
}

let bannerTimer = 0;
function showBanner(text, seconds = 0) {
  clearTimeout(bannerTimer);
  $("banner").hidden = !text;
  $("banner").textContent = text ?? "";
  if (text && seconds) bannerTimer = setTimeout(() => showBanner(null), seconds * 1000);
}

// ------------------------------------------------------------------ pipeline callbacks (CameraWorker signals)

function pipelineCallbacks() {
  return {
    status: (text) => showBanner(text),
    failed: (message) => {
      showCameraOff(message);
      state.cameraWanted = false;
      if (state.recording) recordingFinished();
      if (state.pendingAccount) cancelEnrollment(message);
      setStatus(state.tab === "collect" ? "collectStatus" : "translateStatus", message);
    },
    started: ({ width, height, mirrored } = {}) => {
      showLiveCamera(width, height, mirrored);
      showCameraOff(null);
      if (state.tab === "collect" && !state.recording) setStatus("collectStatus", "Camera sẵn sàng");
    },
    message: (text) => setStatus(state.tab === "collect" ? "collectStatus" : "translateStatus", text),
    frame: ({ fps, frames }) => {
      $("fps").textContent = fps ? `${fps.toFixed(0)} fps` : "";
      state.fps = fps;
      state.frames = frames;
    },
    light: (exposure, advice) => {
      $("lightAdvice").textContent = `${exposure >= 0 ? "+" : ""}${exposure.toFixed(1)} EV · ${advice}`;
      // The live picture is the camera's own: brightened like the analysed frames (approximately), so what is seen is
      // what Holistic sees.
      $("video").style.filter = Math.abs(exposure) < 0.05 ? "" : `brightness(${Math.min(1.8, Math.max(0.6, 2 ** exposure)).toFixed(2)})`;
    },
    stopped: () => showLiveCamera(0, 0, false),
    identity: (status) => showIdentity(status),
    overlay: ({ countdown }) => {
      $("bigCount").hidden = !countdown;
      $("bigCount").textContent = countdown ? String(countdown) : "";
    },
    takeProgress: showTakeProgress,
    sampleReady: saveTakeSample,
    recordingFinished,
    enrollProgress: showEnrollProgress,
    enrollFinished: finishEnrollment,
    signFinished: (sign) => { state.lastSign = sign; },
  };
}

function showIdentity(status) {
  const pill = $("identity");
  if (!state.profile) {
    pill.hidden = true;
    return;
  }
  pill.hidden = false;
  pill.textContent = `👤 ${state.profile.name}: ${IDENTITY_TEXT[status] ?? status ?? "đang tìm người đã đăng ký..."}`;
  pill.classList.toggle("ok", ["locked", "hidden", "mixed"].includes(status));
  pill.classList.toggle("warn", !["locked", "hidden", "mixed"].includes(status));
}

// ------------------------------------------------------------------ recognizer (show_recognizer, show_prediction...)

function showRecognizer({ names, missing }) {
  let text;
  if (names.length) {
    const shown = names.slice(0, 12).join(", ") + (names.length > 12 ? `… và ${names.length - 12} ký hiệu khác` : "");
    text = `Sẵn sàng ${names.length} ký hiệu (${shown}). Làm từng ký hiệu rồi hạ tay xuống.`;
  } else {
    text = `Chưa có ký hiệu nào đủ ${MIN_TEMPLATES_PER_LABEL} mẫu rõ tay để thông dịch.`;
  }
  const lacking = Object.entries(missing ?? {});
  if (lacking.length) text += `  ·  Cần thêm mẫu: ${lacking.map(([name, count]) => `${name} (${count}/${MIN_TEMPLATES_PER_LABEL})`).join(", ")}`;
  setStatus("translateStatus", text);
}

function showPrediction(event) {
  const prediction = event.prediction;
  if (prediction.candidates.length) state.lastCandidates = prediction.candidates.map(([, display]) => display);
  const names = prediction.candidates.map(([, display]) => `“${display}”`).join(" · ");
  $("liveScore").textContent = prediction.confidence > 0 ? `${Math.round(prediction.confidence * 100)}%` : "";
  let rested = false;
  if (event.kind === "final") { // rest measured from the latest finished sign, even one that was not recognised
    rested = state.lastSignEnd !== null && event.start - state.lastSignEnd > sentence.SENTENCE_PAUSE_SECONDS;
    state.lastSignEnd = event.end;
  }
  if (event.append) {
    state.words.push(prediction.display);
    state.breaks.push(rested);
    updateSentence();
  }
  if (event.kind === "final") offerCorrection(prediction, event.append ? state.words.length - 1 : null, event.sign, rested);
  if (event.kind === "signing" || (event.kind === "hold" && prediction.label === null)) {
    $("liveWord").textContent = "Đang ký…" + (prediction.label !== null ? ` “${prediction.display}”?` : "");
    $("candidates").textContent = names ? `Gần giống: ${names}` : "";
    return;
  }
  if (prediction.label !== null) {
    $("liveWord").textContent = prediction.display + (event.kind === "hold" ? " (giữ tay)" : "");
    $("candidates").textContent = prediction.candidates.length > 1 ? `Gần giống: ${names}` : "";
    return;
  }
  $("liveWord").textContent = prediction.bestLabel ? "Chưa chắc — làm lại rõ hơn" : "Đang chờ ký hiệu...";
  $("candidates").textContent = names
    ? `Gần giống nhất: ${names}. Nếu là một trong các từ này, bấm “Đúng là …” để dạy ứng dụng.` : "";
}

// After a finished sign: one button per nearest sign ("Đúng là: …"); `index` = the sentence position of its word.
function offerCorrection(prediction, index, number, rested = false) {
  const correction = state.correction;
  correction.candidates = prediction.candidates.slice(0, 3).map(([label, display]) => [label, display]);
  correction.index = index;
  correction.rested = rested;
  correction.number = correction.candidates.length ? number : 0;
  $("correctionButtons").replaceChildren(...correction.candidates.map(([, display], position) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `“${display}”`;
    button.title = "Lần ký vừa rồi là từ này: lưu làm mẫu của bạn (lần sau nhận đúng) và sửa câu.";
    button.addEventListener("click", () => confirmSign(position));
    return button;
  }));
  show("corrections", Boolean(correction.number));
}

const offered = (captured) => Boolean(state.correction.number) && captured != null && captured.number === state.correction.number;

function applyCorrection(display) {
  const correction = state.correction;
  if (correction.index !== null && correction.index < state.words.length) {
    state.words[correction.index] = display;
  } else {
    state.words.push(display);
    state.breaks.push(correction.rested);
  }
  updateSentence();
  show("corrections", false); // one sample per sign
  correction.number = 0;
}

async function confirmSign(position) {
  const correction = state.correction;
  if (position >= correction.candidates.length) return;
  const latest = state.lastSign;
  if (!offered(latest)) {
    setStatus("translateStatus", "Đã có ký hiệu mới sau lần ký này — hãy chọn trong hàng “Đúng là” mới.");
    return;
  }
  const [label, display] = correction.candidates[position];
  const captured = capturedSample(latest);
  if (!captured) {
    if (latest?.altered) {
      applyCorrection(display); // someone else was too close to keep this sign as a sample; the sentence is corrected
      setStatus("translateStatus", `Đã sửa câu thành “${display}”. Lần ký này không được lưu làm mẫu: giữa lúc ký, ứng dụng chưa nhận ra người đã đăng ký hoặc có người khác ở quá sát — hãy nhìn vào camera, ngồi tách khỏi người khác khi muốn dạy từ này.`);
    }
    return;
  }
  const stored = await storeSign(label, display, latest);
  if (stored !== null && offered(latest)) applyCorrection(stored);
}

function updateSentence() {
  const words = state.words;
  let text;
  if ($("grammar").checked) {
    const groups = sentence.sentences(words, null, sentence.SENTENCE_PAUSE_SECONDS, state.breaks);
    text = groups.slice(-3).map((group) => sentence.compose(group)).join(" ");
  } else {
    text = words.slice(-12).join(" ");
  }
  $("sentence").textContent = text || "…";
  $("gloss").textContent = words.length ? "Ký hiệu: " + words.slice(-12).join(" · ") : "";
}

function removeLastWord() {
  if (state.words.length) {
    state.words.pop();
    state.breaks.length = state.words.length;
  }
  if (state.correction.index !== null && state.correction.index >= state.words.length) state.correction.index = null;
  updateSentence();
}

function clearSentence() {
  state.words.length = 0;
  state.breaks.length = 0;
  state.correction.index = null;
  updateSentence();
}

// The sample of `captured` (a finished sign) or null (the status says why).
function capturedSample(captured) {
  if (!captured) {
    setStatus("translateStatus", "Chưa có ký hiệu nào kết thúc: làm ký hiệu, hạ tay xuống rồi bấm lưu.");
    return null;
  }
  if (!captured.sample) {
    setStatus("translateStatus", "Lần ký vừa rồi quá ngắn hoặc không rõ tay — làm lại rồi lưu.");
    return null;
  }
  if (captured.altered) {
    setStatus("translateStatus", "Giữa lúc ký, ứng dụng chưa nhận ra người đã đăng ký hoặc có người khác ở quá sát (một phần landmark đã bị loại) — không lưu làm mẫu. Nhìn vào camera, ngồi tách ra rồi ký lại.");
    return null;
  }
  const coverage = captured.sample.coverage;
  if (coverage < MIN_TEMPLATE_HAND_COVERAGE) {
    setStatus("translateStatus", `Lần ký vừa rồi chỉ thấy tay ở ${pct(coverage)} số frame (cần từ ${pct(MIN_TEMPLATE_HAND_COVERAGE)}) — không lưu. Giữ tay trong khung hình rồi làm lại.`);
    return null;
  }
  return captured.sample;
}

// "Lưu ký hiệu vừa làm…" / "Từ khác…": name the latest finished sign and keep it as the user's own sample.
async function saveSign() {
  const latest = state.lastSign;
  if (!capturedSample(latest)) return;
  const isOffered = offered(latest);
  const suggestions = isOffered ? state.correction.candidates.map(([, display]) => display) : state.lastCandidates;
  const text = await ask({
    title: "Lưu ký hiệu vừa làm",
    text: "Đây là lần ký gần nhất camera thấy. Nhập đúng tên từ bạn vừa ký: mẫu được lưu vào kho dữ liệu (nguồn webcam) và dùng ngay để nhận dạng. Hai, ba mẫu của chính bạn cho mỗi từ hay dùng giúp nhận đúng hơn nhiều so với chỉ có mẫu của người khác.",
    input: { value: suggestions[0] ?? "", placeholder: "Tên ký hiệu, ví dụ: xin chào" },
    choices: suggestions, image: await latest.preview, ok: "Lưu mẫu", validate: nameError,
  });
  if (!text) return;
  const stored = await storeSign(text, text, latest);
  if (stored !== null && isOffered && offered(latest)) applyCorrection(stored); // still that sign's row after the dialog
}

// Save the captured sign as the user's own sample of `text`, recognised from the next sign on. Returns the word's
// display name, or null when nothing was saved (the status says why).
async function storeSign(text, display, captured) {
  const { frames, duration, coverage } = captured.sample;
  const [width, height] = captured.size;
  const metadata = {
    source: "webcam", orientation: captured.mirrored ? "mirrored" : "original", hand_coverage: coverage.toFixed(2),
    frame_size: `${width}x${height}`, duration: duration.toFixed(3), captured_in: "translation",
  };
  if (state.profile) metadata.signer = state.profile.id; // random account id only, never the name
  const seconds = duration / (FRAME_COUNT - 1);
  let saved;
  try {
    saved = await state.store.saveSample(text, frames, { display, metadata, preview: await captured.preview, aspect: captured.aspect, seconds });
  } catch (error) {
    if (error instanceof DuplicateSampleError) {
      setStatus("translateStatus", `Lần ký này đã được lưu cho “${display}”.`);
      return display;
    }
    setStatus("translateStatus", `Không lưu được mẫu: ${error.message || error}`);
    return null;
  }
  const shown = state.recognition.displays.get(saved.label) ?? state.store.displayOf(saved.label);
  state.recognition.addSample({ id: saved.id, label: saved.label, display: shown, keypoints: frames, aspect: captured.aspect, seconds, source: "webcam" });
  setStatus("translateStatus", `Đã lưu mẫu của bạn cho “${shown}” (ký hiệu này có ${saved.count} mẫu). Ứng dụng dùng mẫu này ngay từ lần ký tiếp theo.`);
  return shown;
}

function speak() {
  const text = $("sentence").textContent.replace(/^…$/, "");
  if (!text) return;
  if (!("speechSynthesis" in window)) {
    setStatus("translateStatus", "Trình duyệt này không đọc được thành tiếng.");
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "vi-VN";
  const voice = speechSynthesis.getVoices().find((v) => v.lang?.toLowerCase().startsWith("vi"));
  if (voice) utterance.voice = voice;
  else setStatus("translateStatus", "Máy chưa có giọng đọc tiếng Việt — có thể cài thêm trong Cài đặt → Chuyển văn bản thành giọng nói.");
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}

// ------------------------------------------------------------------ collection (start_recording, save_sample...)

function refreshVocabulary() {
  // dataset.known_vocabulary: the target words plus every sign in the data store, one entry per label.
  const vocabulary = new Map();
  for (const word of state.bundle.vocabulary) {
    try { vocabulary.set(canonicalLabel(word).toLowerCase(), word); } catch { /* not a sign name */ }
  }
  for (const sign of state.store.signs()) if (!vocabulary.has(sign.label.toLowerCase())) vocabulary.set(sign.label.toLowerCase(), sign.display);
  const words = [...vocabulary.values()].sort((a, b) => a.localeCompare(b, "vi", { sensitivity: "base" }));
  $("signNames").replaceChildren(...words.map((word) => Object.assign(document.createElement("option"), { value: word })));
}

function signHint() {
  const text = $("signName").value.trim();
  const labels = state.store.signs().map((sign) => sign.display);
  const suggestion = text ? accentSuggestion(text, labels, [...state.bundle.vocabulary, ...sentence.lexiconWords()]) : null;
  $("signHint").hidden = !suggestion;
  if (suggestion) $("signHint").textContent = `Có phải “${suggestion}”? Gõ đúng dấu để mẫu được gộp vào ký hiệu đó.`;
}

function startRecording() {
  if (!state.pipeline?.running) {
    setStatus("collectStatus", "Camera chưa bật — bấm “Bật camera” trước khi ghi.");
    return;
  }
  const text = $("signName").value.trim();
  try {
    canonicalLabel(text);
  } catch {
    setStatus("collectStatus", "Nhập tên ký hiệu trước khi ghi.");
    return;
  }
  const takes = Math.max(1, Math.min(10, Number.parseInt($("takes").value, 10) || 1)); // the PC's 1..10
  state.recording = { text };
  for (const id of ["signName", "takes", "record"]) $(id).disabled = true;
  $("cancelRecord").hidden = false;
  $("takeProgress").hidden = false;
  $("takeProgress").value = 0;
  state.pipeline.startRecording(takes);
  setStatus("collectStatus", `Chuẩn bị “${text}”: đếm ngược ${COUNTDOWN_SECONDS} giây, rồi làm ký hiệu một lần và hạ tay xuống.`);
}

function showTakeProgress(take, tenths, total, takeState = "wait") {
  const hint = TAKE_HINTS[takeState] ?? "";
  $("takeProgress").value = tenths / 10;
  setStatus("collectStatus", `Đang ghi “${state.recording?.text ?? ""}” — lượt ${take}/${total}: ${(tenths / 10).toFixed(1)} giây${hint ? ` · ${hint}` : ""}`);
}

async function saveTakeSample(keypoints, frames, details) {
  const recording = state.recording;
  if (!recording) return;
  const { gestureFeatures, handCoverage } = await import("./translate.js");
  const coverage = handCoverage(gestureFeatures(keypoints));
  if (coverage < MIN_TEMPLATE_HAND_COVERAGE) {
    setStatus("collectStatus", `Lượt vừa rồi chỉ thấy tay ở ${pct(coverage)} số frame — không lưu. Giữ tay trong khung hình, đủ sáng rồi thử lại.`);
    return;
  }
  const middle = frames[Math.floor(frames.length / 2)];
  const metadata = { source: "webcam", orientation: middle.mirrored ? "mirrored" : "original", hand_coverage: coverage.toFixed(2), ...details };
  metadata.frame_size = `${middle.width}x${middle.height}`; // recognition corrects the frame's aspect ratio
  if (state.profile) metadata.signer = state.profile.id; // random account id only: the name stays on the phone
  const duration = Number.parseFloat(details.duration);
  try {
    const saved = await state.store.saveSample(recording.text, keypoints, {
      display: recording.text, metadata, preview: await middle.preview, aspect: middle.width / middle.height,
      seconds: duration / (FRAME_COUNT - 1),
    });
    state.recognition.addSample({ id: saved.id, label: saved.label, display: state.store.displayOf(saved.label), keypoints,
      aspect: middle.width / middle.height, seconds: duration / (FRAME_COUNT - 1), source: "webcam" });
    setStatus("collectStatus", `Đã lưu “${recording.text}” (ký hiệu này có ${saved.count} mẫu).`);
  } catch (error) {
    setStatus("collectStatus", `Không lưu được mẫu: ${error.message || error}`);
  }
}

function recordingFinished() {
  state.recording = null;
  for (const id of ["signName", "takes", "record"]) $(id).disabled = false;
  $("cancelRecord").hidden = true;
  $("takeProgress").hidden = true;
  $("bigCount").hidden = true;
  refreshVocabulary();
}

function cancelRecording() {
  state.pipeline?.cancelRecording();
  recordingFinished();
  setStatus("collectStatus", "Đã dừng ghi.");
}

// ------------------------------------------------------------------ multi-source collection (collector.js)

function setCollectSource(source) {
  if (state.recording) return; // a running take ends first (the collector's work goes on in the background)
  if (source !== "video") state.collector?.suspend();
  state.collectSource = source;
  $("sourceCamera").classList.toggle("active", source === "camera");
  $("sourceVideo").classList.toggle("active", source === "video");
  show("cameraCollect", source === "camera");
  show("videoCollect", source === "video");
  if (source === "video") ensureCollector();
  syncCamera();
}

// The collector is loaded the first time it is opened (the camera pages never need it).
async function ensureCollector() {
  if (!state.collector) {
    state.collectorLoading ??= import("./collector.js").then(({ Collector }) => {
      state.collector = new Collector({
        store: state.store, recognition: state.recognition, bundle: state.bundle, view: $("view"), video: $("clipVideo"),
        status: (text) => setStatus("videoStatus", text),
        confirm: (title, text, ok) => confirmAction(title, text, ok),
        vocabulary: () => [...$("signNames").options].map((option) => option.value),
        saved: () => { refreshVocabulary(); state.previews.clear(); },
      });
      // Left the video source while the collector was loading: it must not draw on the stage.
      if (state.tab !== "collect" || state.collectSource !== "video") state.collector.suspend();
      return state.collector;
    });
    await state.collectorLoading;
  }
  return state.collector;
}

// The stage while the collector is shown: the video's frame (or an empty stage before a video is opened).
async function showVideoStage() {
  showCameraOff(null);
  const collector = state.collector;
  if (collector?.resume()) return; // its video's frame (a scan or a save draws it once done)
  const view = $("view");
  const context = view.getContext("2d");
  context.fillStyle = "#0b111c";
  context.fillRect(0, 0, view.width, view.height);
}

// ------------------------------------------------------------------ data store (DataDialog)

function sourceText(sample) {
  if (sample.origin === "phone") {
    const title = sample.metadata?.title;
    return sample.source === "video" ? `video${title ? ` “${title.slice(0, 60)}”` : ""} (thu trên điện thoại)` : "webcam của điện thoại";
  }
  return SOURCE_NAMES[sample.source] ?? (sample.source || "webcam (cũ)");
}

function renderData() {
  const signs = state.store.signs();
  const query = plainText($("search").value.trim()).toLowerCase();
  const shown = query ? signs.filter((sign) => plainText(sign.display).toLowerCase().includes(query) || sign.label.toLowerCase().includes(query)) : signs;
  const samples = signs.reduce((sum, sign) => sum + sign.samples.length, 0);
  const phone = signs.reduce((sum, sign) => sum + sign.samples.filter((s) => s.origin === "phone").length, 0);
  $("dataSummary").textContent = `${signs.length} ký hiệu · ${samples} mẫu` + (phone ? ` (${phone} mẫu thu trên điện thoại)` : "")
    + (query ? ` · ${shown.length} khớp “${$("search").value.trim()}”` : "");
  $("signList").replaceChildren(...shown.map((sign) => {
    const item = document.createElement("li");
    item.tabIndex = 0;
    const name = document.createElement("span");
    name.textContent = sign.display;
    item.append(name);
    if (sign.samples.some((s) => s.origin === "phone")) item.append(Object.assign(document.createElement("span"), { className: "badge phone", textContent: "điện thoại" }));
    item.append(Object.assign(document.createElement("span"), { className: "count", textContent: `${sign.samples.length} mẫu` }));
    item.addEventListener("click", () => openSign(sign.label));
    item.addEventListener("keydown", (event) => { if (event.key === "Enter") openSign(sign.label); });
    return item;
  }));
  if (state.selectedSign) {
    const sign = signs.find((s) => sameLabel(s.label, state.selectedSign));
    if (sign) renderSign(sign); else closeSign();
  }
}

function openSign(label) {
  state.selectedSign = label;
  state.selectedSample = null;
  $("detailStatus").textContent = "";
  const sign = state.store.signs().find((s) => s.label === label);
  if (sign) renderSign(sign, true);
}

function closeSign() {
  state.selectedSign = null;
  state.selectedSample = null;
  show("signDetail", false);
  show("signList", true);
}

function renderSign(sign, fresh = false) {
  show("signList", false);
  show("signDetail", true);
  $("detailTitle").textContent = sign.display;
  $("detailInfo").textContent = `thư mục ${sign.label} · ${sign.samples.length} mẫu`;
  if (fresh) {
    $("renameInput").value = sign.display;
    $("renameHint").hidden = true;
    const others = state.store.signs().filter((s) => s.label !== sign.label).map((s) => s.display);
    const suggestion = accentSuggestion(sign.display, others, [...state.bundle.vocabulary, ...sentence.lexiconWords()]);
    if (suggestion) {
      $("renameInput").value = suggestion; // "Đổi tên / gộp" applies it (and asks before merging)
      const merge = others.includes(suggestion);
      $("renameHint").hidden = false;
      $("renameHint").textContent = `Gợi ý: “${sign.display}” có thể là “${suggestion}” (viết thiếu dấu).`
        + (merge ? " Đã có ký hiệu này: bấm “Đổi tên / gộp” để gộp các mẫu vào đó — hai tên cho cùng một ký hiệu làm ứng dụng phân vân giữa chúng."
          : " Bấm “Đổi tên / gộp” để sửa tên.");
    }
  }
  $("sampleList").replaceChildren(...sign.samples.map((sample, index) => {
    const item = document.createElement("li");
    item.classList.toggle("selected", sample.id === state.selectedSample);
    const image = document.createElement("img");
    image.alt = "";
    samplePicture(sample).then((url) => { if (url) image.src = url; });
    item.append(image, Object.assign(document.createElement("div"), { textContent: `#${index + 1}` }));
    item.title = sourceText(sample);
    item.addEventListener("click", () => {
      state.selectedSample = sample.id;
      for (const other of $("sampleList").children) other.classList.toggle("selected", other === item);
      $("deleteSample").disabled = false;
      $("detailStatus").textContent = `Mẫu #${index + 1}: nguồn ${sourceText(sample)}`
        + (sample.seconds ? `, ${(sample.seconds * (FRAME_COUNT - 1)).toFixed(1)} giây` : "");
    });
    return item;
  }));
  $("deleteSample").disabled = !sign.samples.some((s) => s.id === state.selectedSample);
}

// A sample's picture: its preview photo (phone samples) or its middle frame drawn as a skeleton (built-in samples
// carry no images in the bundle).
async function samplePicture(sample) {
  if (state.previews.has(sample.id)) return state.previews.get(sample.id);
  let url = null;
  if (sample.preview) {
    url = URL.createObjectURL(sample.preview);
  } else {
    const frame = await state.recognition.middleFrame(sample.id).catch(() => null);
    if (frame) url = skeletonPicture(frame, sample.aspect ?? 4 / 3);
  }
  state.previews.set(sample.id, url);
  return url;
}

function skeletonPicture(frame, aspect) {
  const width = 160, height = Math.round(width / aspect);
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0b111c";
  context.fillRect(0, 0, width, height);
  const point = (base, i, stride) => [frame[base + i * stride] * width, frame[base + i * stride + 1] * height];
  const draw = (base, stride, connections, color, count) => {
    let any = false;
    for (let i = base; i < base + count * stride; i++) if (Math.abs(frame[i]) > 1e-6) { any = true; break; }
    if (!any) return;
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    for (const [a, b] of connections) {
      const [ax, ay] = point(base, a, stride), [bx, by] = point(base, b, stride);
      context.moveTo(ax, ay);
      context.lineTo(bx, by);
    }
    context.stroke();
  };
  draw(POSE[0], 4, POSE_SKELETON, "rgb(255,190,60)", 33);
  draw(LEFT_HAND[0], 3, HAND_SKELETON, "rgb(120,220,80)", 21);
  draw(RIGHT_HAND[0], 3, HAND_SKELETON, "rgb(70,170,255)", 21);
  return canvas.toDataURL("image/png");
}

async function renameSign() {
  const label = state.selectedSign;
  const newName = $("renameInput").value.trim();
  if (!label || !newName) return;
  let target;
  try {
    target = canonicalLabel(newName);
  } catch (error) {
    $("detailStatus").textContent = `Tên không hợp lệ: ${error.message}`;
    return;
  }
  const existing = state.store.signs().find((s) => sameLabel(s.label, target));
  if (existing && !sameLabel(target, label)) {
    const yes = await confirmAction("Gộp ký hiệu", `Đã có ký hiệu “${existing.display}”. Gộp toàn bộ mẫu của “${state.store.displayOf(label)}” vào đó?`, "Gộp");
    if (!yes) return;
  }
  try {
    target = await state.store.renameLabel(label, newName);
  } catch (error) {
    $("detailStatus").textContent = `Không đổi tên được: ${error.message || error}`;
    return;
  }
  dataChanged();
  openSign(state.store.signs().find((s) => sameLabel(s.label, target))?.label ?? target);
  $("detailStatus").textContent = `Đã đổi tên thành “${state.store.displayOf(target)}”.`;
}

async function deleteSample() {
  const sign = state.store.signs().find((s) => s.label === state.selectedSign);
  const index = sign?.samples.findIndex((s) => s.id === state.selectedSample) ?? -1;
  if (!sign || index < 0) return;
  if (!(await confirmAction("Xóa mẫu", `Xóa mẫu #${index + 1} của “${sign.display}”?`, "Xóa", true))) return;
  await state.store.removeSamples([state.selectedSample]);
  state.selectedSample = null;
  dataChanged();
  if (sign.samples.length > 1) openSign(sign.label); else closeSign();
}

async function deleteSign() {
  const sign = state.store.signs().find((s) => s.label === state.selectedSign);
  if (!sign) return;
  if (!(await confirmAction("Xóa ký hiệu", `Xóa toàn bộ ${sign.samples.length} mẫu của “${sign.display}”?`, "Xóa", true))) return;
  await state.store.removeLabel(sign.label);
  dataChanged();
  closeSign();
}

// Renames and deletions change the recognizer's signs: it is rebuilt from the data store in the worker.
function dataChanged() {
  state.recognition.setSamples(catalog());
  state.correction.number = 0; // its candidates may name a sign that was just renamed, merged or deleted
  state.correction.candidates = [];
  show("corrections", false);
  renderData();
}

// What the worker needs to build the recognizer: built-in samples by id (their keypoints are in its own bundle
// copy) with their current label and name, plus the phone's samples with their keypoints.
function catalog() {
  const builtin = [], phone = [];
  for (const sample of state.store.samples()) {
    if (sample.origin === "builtin") builtin.push({ id: sample.id, label: sample.label, display: sample.display });
    else phone.push({ id: sample.id, label: sample.label, display: sample.display, keypoints: sample.keypoints,
      aspect: sample.aspect, seconds: sample.seconds, source: sample.source });
  }
  return { builtin, phone };
}

async function exportSync() {
  let exported = await state.store.exportSync(state.version.build ?? "");
  if (!exported.count) {
    const pending = state.store.pendingChanges();
    if (!pending) {
      setStatus("dataStatus", "Chưa có thay đổi nào trên điện thoại để gửi về máy tính.");
      return;
    }
    const again = await confirmAction("Xuất lại", `Không có thay đổi mới. ${pending} thay đổi đã xuất trước đây nhưng máy tính chưa nhận (có thể tệp chưa được nhập, hoặc máy tính chưa cập nhật lại ứng dụng điện thoại). Xuất lại tất cả những gì máy tính chưa có?`, "Xuất lại");
    if (!again) return;
    exported = await state.store.exportSync(state.version.build ?? "", { all: true });
  }
  const { count, file } = exported;
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:]/g, "").replace("T", "-");
  const name = `vsl-dien-thoai-${stamp}.vslsync`;
  const shared = new File([file], name, { type: "application/json" });
  try {
    if (navigator.canShare?.({ files: [shared] })) {
      await navigator.share({ files: [shared], title: "VSL Studio — tệp đồng bộ" });
    } else {
      const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(file), download: name });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
    }
  } catch (error) {
    if (error?.name === "AbortError") return; // the share sheet was closed
    setStatus("dataStatus", `Không xuất được tệp: ${error.message || error}`);
    return;
  }
  await state.store.markExported(exported);
  setStatus("dataStatus", `Đã xuất ${count} thay đổi (${name}). Trên máy tính: Kho dữ liệu → “Nhập từ điện thoại…” rồi chọn tệp này. Các mẫu vẫn ở đây cho đến khi máy tính cập nhật lại ứng dụng điện thoại.`);
}

// ------------------------------------------------------------------ accounts (AccountDialog, enrollment)

async function renderAccounts() {
  const profiles = await state.store.profiles();
  show("accountActive", Boolean(state.profile));
  if (state.profile) $("activeName").textContent = state.profile.name;
  $("profileList").replaceChildren(...profiles.map((profile) => {
    const item = document.createElement("li");
    const name = Object.assign(document.createElement("span"), { textContent: profile.name });
    item.append(name);
    if (state.profile?.id === profile.id) item.append(Object.assign(document.createElement("span"), { className: "badge phone", textContent: "đang dùng" }));
    const actions = Object.assign(document.createElement("span"), { className: "count" });
    const login = Object.assign(document.createElement("button"), { type: "button", textContent: "Đăng nhập" });
    login.disabled = state.profile?.id === profile.id || Boolean(state.pendingAccount);
    login.addEventListener("click", (event) => { event.stopPropagation(); loginProfile(profile); });
    const remove = Object.assign(document.createElement("button"), { type: "button", textContent: "Xóa", className: "danger" });
    remove.disabled = Boolean(state.pendingAccount);
    remove.addEventListener("click", (event) => { event.stopPropagation(); deleteProfile(profile); });
    actions.append(login, remove);
    item.append(actions);
    return item;
  }));
  if (!profiles.length) $("profileList").replaceChildren(Object.assign(document.createElement("p"), { className: "muted small", textContent: "Chưa có tài khoản nào. Tài khoản giúp camera chỉ lấy landmark của bạn khi có người khác trong khung hình." }));
}

async function loginProfile(profile) {
  const password = await ask({ title: `Đăng nhập “${profile.name}”`, input: { type: "password", placeholder: "Mật khẩu" }, ok: "Đăng nhập" });
  if (password === null) return;
  if (!(await state.store.verifyPassword(profile, password))) {
    setStatus("accountStatusLine", "Sai mật khẩu.");
    return;
  }
  setProfile(profile);
  setStatus("accountStatusLine", `Đã đăng nhập “${profile.name}”. Camera chỉ theo dõi landmark của người này.`);
}

async function deleteProfile(profile) {
  const password = await ask({ title: `Xóa tài khoản “${profile.name}”`, text: "Khuôn mặt và mật khẩu của tài khoản này bị xóa khỏi điện thoại. Các mẫu đã thu vẫn giữ nguyên.", input: { type: "password", placeholder: "Mật khẩu" }, ok: "Xóa tài khoản", danger: true });
  if (password === null) return;
  try {
    await state.store.deleteProfile(profile, password);
  } catch (error) {
    setStatus("accountStatusLine", error.message || String(error));
    return;
  }
  if (state.profile?.id === profile.id) setProfile(null);
  setStatus("accountStatusLine", `Đã xóa tài khoản “${profile.name}”.`);
  renderAccounts();
}

function setProfile(profile) {
  state.profile = profile;
  $("accountChip").textContent = profile ? `👤 ${profile.name}` : "Khách";
  showIdentity(profile ? null : undefined);
  state.pipeline?.setIdentity(profile ? profile.embeddings : null); // the camera restarts with the new identity
  renderAccounts();
}

function openAccountForm() {
  $("accountForm").reset();
  $("accountStatus").textContent = "";
  show("accountForm", true);
  $("accountName").focus();
}

async function submitAccount(event) {
  event.preventDefault();
  const name = $("accountName").value, password = $("accountPassword").value, confirm = $("accountConfirm").value;
  let clean;
  try {
    clean = await state.store.validateNewAccount(name, password, confirm);
  } catch (error) {
    $("accountStatus").textContent = error.message;
    return;
  }
  show("accountForm", false);
  state.pendingAccount = { name: clean, password };
  app.classList.add("enrolling");
  show("enrollCard", true);
  $("enrollInstruction").textContent = "Đang tải mô hình khuôn mặt (một lần, khoảng 23 MB)…";
  $("enrollBins").textContent = HELP.enroll;
  $("enrollProgress").value = 0;
  syncCamera();
}

function showEnrollProgress(instruction, bins, captured) {
  $("enrollInstruction").textContent = instruction;
  $("enrollBins").textContent = bins;
  $("enrollProgress").value = captured;
}

function endEnrollment() {
  state.pendingAccount = null;
  app.classList.remove("enrolling");
  show("enrollCard", false);
  syncCamera();
}

function cancelEnrollment(reason = null) {
  endEnrollment();
  setStatus("accountStatusLine", reason ? `Chưa tạo tài khoản: ${reason}` : "Đã hủy tạo tài khoản.");
  renderAccounts();
}

async function finishEnrollment(enrollment) {
  const pending = state.pendingAccount;
  if (!pending) return;
  endEnrollment();
  let profile;
  try {
    profile = await state.store.createProfile(pending.name, pending.password, enrollment);
  } catch (error) {
    setStatus("accountStatusLine", `Không tạo được tài khoản: ${error.message || error}`);
    renderAccounts();
    return;
  }
  setProfile(profile);
  setStatus("accountStatusLine", `Đã ghi nhận khuôn mặt của “${profile.name}” (${enrollment.embeddings.length} frame) và đăng nhập. Từ giờ camera chỉ theo dõi landmark của bạn khi có người khác trong khung hình.`);
}

// ------------------------------------------------------------------ settings, install, update

async function loadSettings() {
  const store = state.store;
  $("grammar").checked = await store.setting("grammar", true);
  $("denseView").checked = await store.setting("denseView", false);
  $("autoLight").checked = await store.setting("autoLight", true);
  $("bias").value = await store.setting("bias", 0);
  $("biasValue").textContent = Number($("bias").value).toFixed(2).replace(/\.?0+$/, "") || "0";
  state.cameraWanted = await store.setting("cameraWanted", false);
  applySettings();
}

function applySettings() {
  state.pipeline?.configure({ dense: $("denseView").checked, autoLight: $("autoLight").checked, bias: Number($("bias").value) });
}

function saveSetting(key, value) {
  state.store.setSetting(key, value).catch(() => {});
}

// ------------------------------------------------------------------ installing on the home screen

function platform() {
  return detectPlatform();
}

// The top-bar button and the suggestion banner while the app is not installed; the "Thêm" page's install card.
function renderInstall() {
  const where = platform();
  $("installChip").hidden = where.standalone;
  $("install").textContent = where.standalone ? "Đã cài — cách cài lên máy khác" : "⬇ Cài ứng dụng vào màn hình chính";
  $("installHelp").textContent = where.standalone ? "Đã cài trên màn hình chính."
    : `Đang mở bằng ${where.label}. ${state.installPrompt ? "Có thể cài ngay bằng một lần bấm." : "Bấm nút trên để xem từng bước cài."}`;
}

async function maybeSuggestInstall() {
  const where = platform();
  if (where.standalone || params.has("e2e")) return;
  if (await state.store.setting("installHintDismissed", false)) return;
  $("installBanner").hidden = false;
}

function openInstallGuide() {
  const where = platform();
  const guide = installGuide(where, Boolean(state.installPrompt));
  $("installTitle").textContent = guide.title;
  $("installDevice").textContent = `Thiết bị: ${where.label}`;
  $("installSteps").replaceChildren(...guide.steps.map((step) => Object.assign(document.createElement("li"), { textContent: step })));
  $("installNote").textContent = guide.note;
  $("installNow").hidden = guide.action !== "prompt";
  $("copyLink").classList.toggle("primary", guide.action === "copy");
  $("installStatus").textContent = "";
  $("appAddress").textContent = appAddress();
  $("installDialog").showModal();
}

async function installNow() {
  const prompt = state.installPrompt;
  if (!prompt) {
    openInstallGuide();
    return;
  }
  state.installPrompt = null;
  await prompt.prompt();
  const choice = await prompt.userChoice.catch(() => null);
  $("installStatus").textContent = choice?.outcome === "accepted" ? "●  Đang cài… biểu tượng VSL Studio sẽ có trên màn hình chính."
    : "●  Chưa cài. Có thể bấm “⬇ Cài app” bất cứ lúc nào.";
  if (choice?.outcome === "accepted") $("installBanner").hidden = true;
  renderInstall();
  if ($("installDialog").open) openInstallGuide();
}

async function copyLink() {
  const address = appAddress();
  try {
    await navigator.clipboard.writeText(address);
    $("installStatus").textContent = "●  Đã sao chép liên kết — mở Safari / Chrome rồi dán vào thanh địa chỉ.";
  } catch {
    // Clipboard blocked (some in-app browsers): show the address to copy by hand.
    $("installStatus").textContent = `●  Hãy sao chép liên kết này: ${address}`;
  }
}

async function shareApp() {
  const address = appAddress();
  const data = { title: "VSL Studio", text: "VSL Studio — phiên dịch và thu mẫu ngôn ngữ ký hiệu Việt Nam trên điện thoại.", url: address };
  if (navigator.share) {
    try {
      await navigator.share(data);
      return;
    } catch (error) {
      if (error?.name === "AbortError") return;
    }
  }
  if (!$("installDialog").open) openInstallGuide();
  copyLink();
}

function renderAbout() {
  const signs = state.store.signs();
  const samples = signs.reduce((sum, sign) => sum + sign.samples.length, 0);
  const built = state.version.built ? new Date(state.version.built).toLocaleString("vi-VN") : "bản đang phát triển";
  $("about").textContent = `VSL Studio Mobile · dữ liệu ${built}${state.version.commit ? ` (${state.version.commit})` : ""} · ${signs.length} ký hiệu, ${samples} mẫu · Holistic chạy trên ${state.pipeline?.delegate ?? "GPU/CPU"} · hoạt động không cần mạng sau lần mở đầu.`;
  renderInstall();
}

function registerServiceWorker() {
  const local = ["localhost", "127.0.0.1"].includes(location.hostname);
  if (!("serviceWorker" in navigator) || params.has("nosw") || (local && !params.has("sw"))) return;
  let hadController = Boolean(navigator.serviceWorker.controller);
  navigator.serviceWorker.register("sw.js").then((registration) => {
    const offer = (worker) => {
      if (!worker || !navigator.serviceWorker.controller) return; // first install: nothing to replace
      $("update").hidden = false;
      $("update").onclick = () => worker.postMessage("update");
      showBanner("Có bản VSL Studio mới — vào Thêm → “Cập nhật”.", 6);
    };
    if (registration.waiting) offer(registration.waiting);
    registration.addEventListener("updatefound", () => {
      const worker = registration.installing;
      worker?.addEventListener("statechange", () => { if (worker.state === "installed") offer(worker); });
    });
    setInterval(() => registration.update().catch(() => {}), 60 * 60 * 1000);
  }).catch(() => {});
  let reloading = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (!hadController) { // the first install took control: the page already runs this version
      hadController = true;
      return;
    }
    if (reloading) return;
    reloading = true;
    location.reload();
  });
}

// ------------------------------------------------------------------ wiring and boot

function wire() {
  for (const button of document.querySelectorAll(".tabbar button")) button.addEventListener("click", () => setTab(button.dataset.tab));
  $("accountChip").addEventListener("click", () => setTab("account"));
  $("startCamera").addEventListener("click", () => {
    state.cameraWanted = true;
    saveSetting("cameraWanted", true);
    syncCamera();
  });
  $("stopCamera").addEventListener("click", () => {
    state.cameraWanted = false;
    saveSetting("cameraWanted", false);
    if (state.recording) cancelRecording();
    syncCamera();
  });
  $("switchCamera").addEventListener("click", () => {
    if (state.recording) cancelRecording(); // the takes cannot continue on another camera
    // A camera that fails to switch is reported like any camera failure (the stage offers "Bật camera" again).
    state.pipeline?.switchFacing().catch((error) => pipelineCallbacks().failed(error.message || String(error)));
  });
  $("dialogCancel").addEventListener("click", () => $("dialog").close("cancel"));
  document.addEventListener("visibilitychange", () => syncCamera());

  $("grammar").addEventListener("change", () => { updateSentence(); saveSetting("grammar", $("grammar").checked); });
  $("undoWord").addEventListener("click", removeLastWord);
  $("clearSentence").addEventListener("click", clearSentence);
  $("saveSign").addEventListener("click", saveSign);
  $("otherWord").addEventListener("click", saveSign);
  $("speak").addEventListener("click", speak);

  $("signName").addEventListener("input", signHint);
  $("signName").addEventListener("keydown", (event) => { if (event.key === "Enter") startRecording(); });
  $("record").addEventListener("click", startRecording);
  $("cancelRecord").addEventListener("click", cancelRecording);
  $("sourceCamera").addEventListener("click", () => setCollectSource("camera"));
  $("sourceVideo").addEventListener("click", () => setCollectSource("video"));


  $("search").addEventListener("input", () => { closeSign(); renderData(); });
  $("backToList").addEventListener("click", () => { closeSign(); renderData(); });
  $("renameSign").addEventListener("click", renameSign);
  $("deleteSample").addEventListener("click", deleteSample);
  $("deleteSign").addEventListener("click", deleteSign);
  $("exportSync").addEventListener("click", exportSync);

  $("newAccount").addEventListener("click", openAccountForm);
  $("accountForm").addEventListener("submit", submitAccount);
  $("accountCancel").addEventListener("click", () => show("accountForm", false));
  $("enrollCancel").addEventListener("click", () => cancelEnrollment());
  $("logout").addEventListener("click", () => { setProfile(null); setStatus("accountStatusLine", "Đã đăng xuất."); });

  $("denseView").addEventListener("change", () => { applySettings(); saveSetting("denseView", $("denseView").checked); });
  $("autoLight").addEventListener("change", () => { applySettings(); saveSetting("autoLight", $("autoLight").checked); });
  $("bias").addEventListener("input", () => {
    $("biasValue").textContent = Number($("bias").value).toString();
    applySettings();
    saveSetting("bias", Number($("bias").value));
  });
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault(); // offered from our own button / guide instead of the browser's mini-bar
    state.installPrompt = event;
    renderInstall();
  });
  window.addEventListener("appinstalled", () => {
    state.installPrompt = null;
    $("installBanner").hidden = true;
    renderInstall();
    showBanner("Đã cài VSL Studio lên màn hình chính.", 5);
  });
  matchMedia("(display-mode: standalone)").addEventListener?.("change", renderInstall);
  $("installChip").addEventListener("click", openInstallGuide);
  $("install").addEventListener("click", openInstallGuide);
  $("installBannerOpen").addEventListener("click", () => { $("installBanner").hidden = true; openInstallGuide(); });
  $("installBannerLater").addEventListener("click", () => { $("installBanner").hidden = true; saveSetting("installHintDismissed", true); });
  $("installNow").addEventListener("click", installNow);
  $("copyLink").addEventListener("click", copyLink);
  $("shareLink").addEventListener("click", shareApp);
  $("shareApp").addEventListener("click", shareApp);
  $("installClose").addEventListener("click", () => $("installDialog").close());
  $("appQr").addEventListener("error", () => { $("appQr").hidden = true; });
  $("collectHelp").textContent = HELP.collect;
  $("translateHelp").textContent = HELP.translate;
}

async function boot() {
  const splash = (text) => { $("splashText").textContent = text; };
  registerServiceWorker();
  wire();
  try {
    splash("Đang tải dữ liệu mẫu…");
    state.version = await fetch("version.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
    state.bundle = await loadBundle("data/", { keypoints: false });
    state.store = await Store.open(state.bundle);
    await loadSettings();
    state.recognition = new Recognition({
      ready: showRecognizer, // a rebuild; a sample just saved only extends the recognizer silently
      events: (events) => events.forEach(showPrediction),
      error: (message) => setStatus("translateStatus", `Lỗi nhận dạng: ${message}`),
    });
    state.recognition.setSamples(catalog());
    state.pipeline = new Pipeline({ video: $("video"), view: $("view"), recognition: state.recognition, callbacks: pipelineCallbacks() });
    applySettings();
  } catch (error) {
    splash(`Không mở được VSL Studio: ${error.message || error}`);
    $("splashProgress").hidden = true;
    console.error(error);
    return;
  }
  updateSentence();
  setStatus("translateStatus", "Đang nạp dữ liệu mẫu để thông dịch...");
  $("splash").hidden = true;
  state.ready = true;
  setTab("translate");
  renderInstall();
  maybeSuggestInstall();
  if (params.has("install")) openInstallGuide(); // a shared link or the QR code can open the guide directly
  if (params.has("shared")) { // videos shared to the app from the gallery / Zalo / files (Android)
    setTab("collect");
    setCollectSource("video");
    ensureCollector().then((collector) => collector.takeShared()).catch((error) => setStatus("videoStatus", error.message || String(error)));
  }
}

// Hooks for tools/test_mobile_e2e.py (fake camera) — read-only state and the camera button.
window.__vsl = {
  get ready() { return state.ready && Boolean(state.recognition?.ready); },
  startCamera() { $("startCamera").click(); return true; },
  state() {
    return { words: [...state.words], sentence: $("sentence").textContent, fps: state.fps ?? 0, frames: state.frames ?? 0,
      live: $("liveWord").textContent, status: $("translateStatus").textContent };
  },
};

if (params.has("e2e")) window.__vsl.app = state; // tools/test_mobile_app.py reads the app's state
if (params.get("delegate") === "CPU") import("./holistic.js").then(({ preferDelegate }) => preferDelegate("CPU"));

boot();
