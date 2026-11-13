// Recognition off the camera loop (vsl_ui.RecognitionLoop): the camera hands every stamped frame to
// recognition.worker.js, which runs the recognizer and live_translation's LiveTranslator and answers with events.

export class Recognition {
  // callbacks: ready({names, missing}), events([LiveEvent]), sign(SignInfo), error(message)
  constructor(callbacks) {
    this.callbacks = callbacks;
    this.ready = false;
    this.displays = new Map(); // label -> the recognizer's name for it
    this.requests = new Map();
    this.nextRequest = 1;
    this.onSign = null; // set by the pipeline: completes a finished sign with its preview and altered flag
    this.worker = new Worker(new URL("./recognition.worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = (event) => this._message(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault?.();
      callbacks.error?.(event.message || "lỗi không rõ trong luồng nhận dạng");
    };
  }

  _message(message) {
    switch (message.type) {
      case "ready":
        this.ready = true;
        this.displays = new Map(message.displays);
        this.callbacks.ready?.(message);
        break;
      case "extended": // a saved sample was added: nothing to announce (the save already said so)
        this.ready = true;
        this.displays = new Map(message.displays);
        break;
      case "sign":
        this.onSign?.(message.sign);
        break;
      case "events":
        this.callbacks.events?.(message.events);
        break;
      case "reply": {
        const request = this.requests.get(message.id);
        this.requests.delete(message.id);
        if (message.error) request?.reject(new Error(message.error));
        else request?.resolve(message.value);
        break;
      }
      case "error":
        this.callbacks.error?.(message.message);
        break;
      default:
        break;
    }
  }

  _ask(type, payload) {
    const id = this.nextRequest++;
    return new Promise((resolve, reject) => {
      this.requests.set(id, { resolve, reject });
      this.worker.postMessage({ type, id, ...payload });
    });
  }

  // The data store's samples ({builtin: [{id, label, display}], phone: [{..., keypoints: Float32Array[]}]}):
  // the recognizer is rebuilt from them.
  setSamples(catalog) {
    this.worker.postMessage({ type: "samples", builtin: catalog.builtin,
      phone: catalog.phone.map((sample) => ({ ...sample, keypoints: flatten(sample.keypoints) })) });
  }

  // A sample saved just now: recognised from the next sign on (Recognizer.extended, nothing else re-prepared).
  addSample(sample) {
    this.worker.postMessage({ type: "add", sample: { ...sample, keypoints: flatten(sample.keypoints) } });
  }

  // One processed frame: capture time (seconds), Float32Array(1662) keypoints (transferred), frame size.
  push(stamp, keypoints, width, height) {
    const copy = new Float32Array(keypoints);
    this.worker.postMessage({ type: "frame", stamp, keypoints: copy, width, height }, [copy.buffer]);
  }

  // A new camera session: the live translator starts afresh (no burst spans two sessions).
  reset() {
    this.worker.postMessage({ type: "reset" });
  }

  // The middle frame of a built-in sample (data store pictures).
  middleFrame(id) {
    return this._ask("middleFrame", { sample: id });
  }

  // collectors.near_duplicate: resolves to the id of a stored sample of sign `label` that is the same recording as
  // `keypoints` (FRAME_COUNT frames, or one flat Float32Array), else null. aspect: the new recording's frame width /
  // height (null: 4:3); frameSeconds: time between its frames (null: 1/15 s). Answered after a rebuild under way and
  // after every sample added before the request (addSample), so a sample saved just now is compared too.
  nearDuplicate(label, keypoints, aspect = null, frameSeconds = null) {
    return this._ask("nearDuplicate", { label, keypoints: flatten(keypoints), aspect, frameSeconds });
  }
}

export function flatten(frames) {
  if (frames instanceof Float32Array) return frames;
  const size = frames[0]?.length ?? 0;
  const out = new Float32Array(frames.length * size);
  frames.forEach((frame, i) => out.set(frame, i * size));
  return out;
}
