// A pool of detection workers (detector.worker.js: light correction + Holistic off the page's thread). The camera
// pipeline gives each new frame to an idle worker, so on a phone where one frame takes 300 ms, two or three frames are
// analysed at the same time: several times the frames per second of one Holistic on the page's thread, and the page
// itself (the live camera picture, buttons, the sentence) never waits for a frame.

export const INIT_SECONDS = 90; // a worker that is not ready by then (model download on a slow network) is given up

// How many workers this device runs at once: cores left after the page and the recognition worker, at most 3 (each
// holds its own Holistic, about 100-200 MB): 1 on a phone with 2 GB or less, 2 with 4 GB or less.
export function detectorCount(cores = navigator.hardwareConcurrency || 4, memory = navigator.deviceMemory || 4) {
  const byMemory = memory <= 2 ? 1 : memory <= 4 ? 2 : 3;
  return Math.max(1, Math.min(byMemory, cores - 2));
}

function ready(worker, delegate) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Bộ nhận dạng chạy nền không khởi động được (quá lâu).")), INIT_SECONDS * 1000);
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === "ready") {
        clearTimeout(timer);
        resolve(message.delegate);
      } else if (message.type === "error") {
        clearTimeout(timer);
        reject(new Error(message.message));
      }
    };
    worker.onerror = (event) => {
      clearTimeout(timer);
      reject(new Error(event.message || "worker error"));
    };
    worker.postMessage({ type: "init", delegate });
  });
}

export class DetectorPool {
  constructor(workers, delegate) {
    this.delegate = delegate;
    this.broken = false;
    this.slots = workers.map((worker) => ({ worker, job: null, resetLight: false }));
    for (const slot of this.slots) {
      slot.worker.onmessage = (event) => this._message(slot, event.data);
      slot.worker.onerror = (event) => this._crash(slot, event.message || "worker error");
      slot.worker.onmessageerror = () => this._crash(slot, "message error");
    }
  }

  // `count` workers, each with its own Holistic, ready to take frames; rejects (and closes them) if any cannot start.
  static async create(count, delegate) {
    const workers = [];
    try {
      for (let i = 0; i < count; i++) workers.push(new Worker(new URL("./detector.worker.js", import.meta.url), { type: "module" }));
      // One after another: the first fills the browser's cache with the model, the others then start from it. The
      // first uses the GPU (when asked: it falls back to the CPU by itself), the others the CPU's other cores: several
      // Holistic instances on one phone GPU only queue behind each other, while CPU cores work side by side.
      const delegates = [];
      for (const [index, worker] of workers.entries()) delegates.push(await ready(worker, index === 0 ? delegate : "CPU"));
      return new DetectorPool(workers, [...new Set(delegates)].join(" + "));
    } catch (error) {
      for (const worker of workers) worker.terminate();
      throw error;
    }
  }

  // Every worker starts its exposure over with its next frame (a new camera session, like light.reset()).
  resetLight() {
    for (const slot of this.slots) slot.resetLight = true;
  }

  get size() {
    return this.slots.length;
  }

  get idle() {
    return this.broken ? 0 : this.slots.filter((slot) => !slot.job).length;
  }

  // Analyse one frame: {width, height, buffer (RGBA, transferred), stamp, light, resetLight}. Resolves with the
  // worker's result ({keypoints, drawing, buffer (the corrected picture), exposure, advice, ms}).
  submit(frame) {
    const slot = this.slots.find((candidate) => !candidate.job);
    if (!slot || this.broken) return Promise.reject(new Error("no idle detector"));
    const resetLight = slot.resetLight;
    slot.resetLight = false;
    return new Promise((resolve, reject) => {
      slot.job = { resolve, reject };
      slot.worker.postMessage({ type: "frame", ...frame, resetLight }, [frame.buffer]);
    });
  }

  _message(slot, message) {
    const job = slot.job;
    slot.job = null;
    if (!job) return;
    if (message.type === "result") job.resolve(message);
    else job.reject(Object.assign(new Error(message.message || "frame failed"), { frame: true }));
  }

  _crash(slot, reason) {
    this.broken = true; // a crashed worker (out of memory): the pipeline goes back to analysing on the page
    const job = slot.job;
    slot.job = null;
    job?.reject(new Error(`Bộ nhận dạng chạy nền dừng: ${reason}`));
  }

  close() {
    this.broken = true;
    for (const slot of this.slots) {
      slot.job?.reject(new Error("closed"));
      slot.job = null;
      slot.worker.terminate();
    }
  }
}
