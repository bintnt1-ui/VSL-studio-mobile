// The recognition thread (vsl_ui.RecognitionLoop + CameraWorker.translate_frames): builds the recognizer from the
// built-in bundle and the phone's samples, feeds every stamped frame, in order, to live_translation's
// LiveTranslator and posts what it decided. Frames are added as they arrive; update() runs whenever the thread is
// free, so a slow phone simply hands over bigger batches (the translator decides every burst in a late batch).

import { decodeBundle, frameColumns, halfToFloat } from "./bundle.js";
import { KEYPOINT_SIZE } from "./layout.js";
import * as translate from "./translate.js";
import { LiveTranslator } from "./live.js";

const FRAME_COUNT = 30;
const BASE = new URL("../data/", import.meta.url);

let bundle = null; // {index, buffer}: the float16 bundle, decoded per rebuild (the decoded copy is ~110 MB)
let catalog = null; // the latest data store catalog
let recognizer = null;
let translator = null;
let building = null;
let rebuildAgain = false;
let included = new Set(); // ids of the samples the recognizer knows
const pendingAdds = [];
let lastNumber = 0;
let base = 0; // sign numbers stay unique across camera sessions (a new translator counts from 1 again)
let size = [640, 480];
let scheduled = false;

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

async function loadBundleData() {
  bundle ??= (async () => { // a failed download is tried again next time (not cached)
    const [index, buffer] = await Promise.all([
      fetch(new URL("index.json", BASE)).then((r) => { if (!r.ok) throw new Error(`index.json: ${r.status}`); return r.json(); }),
      fetch(new URL("keypoints.bin", BASE)).then((r) => { if (!r.ok) throw new Error(`keypoints.bin: ${r.status}`); return r.arrayBuffer(); }),
    ]);
    return { index, buffer };
  })().catch((error) => {
    bundle = null;
    throw error;
  });
  return bundle;
}

function unflatten(values) {
  const frames = [];
  for (let i = 0; i < values.length / KEYPOINT_SIZE; i++) frames.push(values.subarray(i * KEYPOINT_SIZE, (i + 1) * KEYPOINT_SIZE));
  return frames;
}

function displayOf(model, label) {
  const displays = model.displays;
  return displays instanceof Map ? displays.get(label) : displays[label];
}

// The samples as they are in the data store now: built-in ones (renamed / deleted on the phone) and the phone's.
async function currentSamples(wanted) {
  const { index, buffer } = await loadBundleData();
  const decoded = decodeBundle(index, buffer).samples;
  const byId = new Map(decoded.map((sample) => [sample.id, sample]));
  const samples = [];
  for (const entry of wanted.builtin) {
    const sample = byId.get(entry.id);
    if (sample) samples.push({ ...sample, label: entry.label, display: entry.display });
  }
  for (const entry of wanted.phone) samples.push({ ...entry, keypoints: unflatten(entry.keypoints) });
  return samples;
}

async function rebuild() {
  if (building) {
    rebuildAgain = true;
    return;
  }
  building = (async () => {
    do {
      rebuildAgain = false;
      const wanted = catalog;
      const samples = await currentSamples(wanted);
      const started = performance.now();
      const loaded = translate.loadRecognizer(samples, { keepKeypoints: false }); // ~100 MB less on the phone
      recognizer = loaded.recognizer;
      included = new Set(samples.map((sample) => sample.id));
      const adds = pendingAdds.splice(0).filter((sample) => !included.has(sample.id));
      if (adds.length) extend(adds);
      if (translator) translator.recognizer = recognizer;
      else translator = new LiveTranslator(recognizer);
      const labels = [...recognizer.labels];
      post({
        type: "ready", names: labels.map((label) => displayOf(recognizer, label)),
        missing: loaded.missing instanceof Map ? Object.fromEntries(loaded.missing) : (loaded.missing ?? {}),
        displays: labels.map((label) => [label, displayOf(recognizer, label)]),
        templates: recognizer.templates.length, seconds: (performance.now() - started) / 1000,
      });
    } while (rebuildAgain);
  })().catch((error) => post({ type: "error", message: `Không nạp được dữ liệu mẫu: ${error.message || error}` }))
    .finally(() => { building = null; });
}

function extend(samples) {
  const templates = translate.usableTemplates(samples);
  if (!templates.length) return;
  recognizer = recognizer.extended(templates);
  for (const sample of samples) included.add(sample.id);
  if (translator) translator.recognizer = recognizer;
  post({ type: "extended", displays: recognizer.labels.map((label) => [label, displayOf(recognizer, label)]),
    templates: recognizer.templates.length });
}

function addSample(entry) {
  const sample = { ...entry, keypoints: unflatten(entry.keypoints) };
  if (!recognizer || building) pendingAdds.push(sample);
  else if (!included.has(sample.id)) extend([sample]);
}

// translate_frames: everything since the last call, then what the translator decided.
function run() {
  scheduled = false;
  if (!translator) return;
  let events;
  try {
    events = translator.update();
  } catch (error) {
    post({ type: "error", message: error.message || String(error) });
    return;
  }
  const sign = translator.latestSign();
  if (sign && sign.number !== lastNumber) { // a sign just ended: its sample, ready to be saved
    lastNumber = sign.number;
    post({ type: "sign", sign: describeSign(sign) });
  }
  if (events.length) post({ type: "events", events: events.map(describeEvent) });
}

function describeSign(sign) {
  const [first, last] = sign.span;
  const times = sign.times;
  const aspect = translator.aspect;
  let sample = null;
  const captured = translate.captureSample(sign.keypoints, aspect, FRAME_COUNT, times, sign.span);
  if (captured) {
    const [frames, duration] = captured;
    const list = Array.isArray(frames) ? frames : unflatten(frames);
    sample = { frames: list.map((frame) => new Float32Array(frame)), duration,
      coverage: translate.handCoverage(translate.gestureFeatures(list)) };
  }
  return {
    number: base + sign.number, aspect, size: [...size],
    middle: (times[first] + times[last - 1]) / 2,
    range: [times[first] - translate.CAPTURE_MARGIN_SECONDS, times[last - 1] + translate.CAPTURE_MARGIN_SECONDS],
    sample,
  };
}

function describeEvent(event) {
  const p = event.prediction;
  return {
    kind: event.kind, start: event.start, end: event.end, append: event.append, sign: event.sign ? base + event.sign : 0,
    prediction: {
      label: p.label, display: p.display, confidence: p.confidence, distance: p.distance, margin: p.margin,
      bestLabel: p.bestLabel ?? null, candidates: (p.candidates ?? []).map(([label, display, distance]) => [label, display, distance]),
    },
  };
}

async function middleFrame(id) {
  const { index, buffer } = await loadBundleData();
  const sample = index.samples.find((s) => s.id === id);
  if (!sample) return null;
  const columns = frameColumns(index.facePoints);
  const width = index.frameValues;
  const halves = new Uint16Array(buffer, (sample.offset + Math.floor(sample.frames / 2)) * width * 2, width);
  const frame = new Float32Array(KEYPOINT_SIZE);
  for (let c = 0; c < width; c++) frame[columns[c]] = halfToFloat(halves[c]);
  return frame;
}

self.onmessage = async (event) => {
  const message = event.data;
  switch (message.type) {
    case "frame":
      if (!translator) return; // the recognizer is still being built: nothing to translate with yet
      size = [message.width, message.height];
      translator.aspect = message.width / message.height;
      translator.add(message.stamp, message.keypoints);
      if (!scheduled) {
        scheduled = true;
        setTimeout(run, 0);
      }
      break;
    case "samples":
      catalog = { builtin: message.builtin, phone: message.phone };
      rebuild();
      break;
    case "add":
      addSample(message.sample);
      break;
    case "reset": // a camera start: a recognizer that failed to load (network) is tried again
      if (!recognizer && !building && catalog) rebuild(); // (a build already running is not started twice)
      if (recognizer) translator = new LiveTranslator(recognizer);
      base += lastNumber + 1;
      lastNumber = 0;
      break;
    case "middleFrame":
      try {
        post({ type: "reply", id: message.id, value: await middleFrame(message.sample) });
      } catch (error) {
        post({ type: "reply", id: message.id, error: error.message || String(error) });
      }
      break;
    case "nearDuplicate": // collectors.near_duplicate (collectbatch.js): a stored sample that is the same recording
      try {
        const { nearDuplicateIn, samplesOutside } = await import("./collectbatch.js");
        while (building) await building; // a rebuild under way: what was added meanwhile is in once it is done
        if (!recognizer) throw new Error("Chưa nạp được dữ liệu mẫu.");
        const data = catalog?.builtin?.length ? await loadBundleData() : null;
        const outside = samplesOutside(recognizer, message.label, catalog, data); // too few hand frames to be templates
        post({ type: "reply", id: message.id, value: nearDuplicateIn(recognizer, message.label, unflatten(message.keypoints),
          { aspect: message.aspect, frameSeconds: message.frameSeconds, outside }) });
      } catch (error) {
        post({ type: "reply", id: message.id, error: error.message || String(error) });
      }
      break;
    default:
      break;
  }
};
