// A detection worker: light correction (light.js, exactly the PC's camera_enhancement) and Holistic (holistic.js) on
// camera frames, off the page's thread. The camera pipeline keeps a few of these busy at once: each takes every
// n-th frame, so a phone whose Holistic needs 300 ms still analyses several frames per second, and the page (the live
// camera picture, buttons) never waits for it.
//
// in:  {type: "init", delegate}                         -> {type: "ready", delegate} | {type: "error", message}
//      {type: "frame", id, width, height, buffer, stamp, light: {autoEnabled, bias}, keepImage}
//                                                       -> {type: "result", id, stamp, keypoints, drawing, buffer,
//                                                           exposure, advice, ms: {light, holistic}}
// The frame's RGBA buffer is transferred in and, corrected, back out (the page reuses it for the next frame).

// MediaPipe loads its wasm glue with importScripts, which module workers do not have; it then asks self.import, which
// here runs the glue as a classic script (its `var ModuleFactory` becomes a global, as importScripts would make it).
self.import = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  (0, eval)(await response.text()); // eslint-disable-line no-eval
};

let holistic = null;
let light = null;

async function init(delegate) {
  const [{ createHolistic, preferDelegate }, { AdaptiveLightCorrection }] = await Promise.all([
    import("./holistic.js"), import("./light.js")]);
  if (delegate) preferDelegate(delegate);
  holistic = await createHolistic();
  light = new AdaptiveLightCorrection();
  return holistic.delegate;
}

function frame(message) {
  const { id, width, height, buffer, stamp } = message;
  const started = performance.now();
  const pixels = new Uint8ClampedArray(buffer);
  light.autoEnabled = message.light?.autoEnabled ?? true;
  light.exposureBiasEv = message.light?.bias ?? 0;
  if (message.resetLight) light.reset();
  const image = { data: pixels, width, height };
  light.process(image, image); // in place: the corrected picture is what Holistic sees, as on the PC
  const corrected = performance.now();
  const result = holistic.run(new ImageData(pixels, width, height), width, height, stamp);
  const done = performance.now();
  const keypoints = result.keypoints instanceof Float32Array ? result.keypoints : Float32Array.from(result.keypoints);
  self.postMessage({
    type: "result", id, stamp, keypoints, drawing: result.drawing, buffer: pixels.buffer, width, height,
    exposure: light.effectiveExposureEv, advice: light.advice, ms: { light: corrected - started, holistic: done - corrected },
  }, [pixels.buffer, keypoints.buffer]);
}

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "init") {
    try {
      self.postMessage({ type: "ready", delegate: await init(message.delegate) });
    } catch (error) {
      self.postMessage({ type: "error", message: error?.message || String(error) });
    }
  } else if (message.type === "frame") {
    try {
      frame(message);
    } catch (error) {
      self.postMessage({ type: "failed", id: message.id, message: error?.message || String(error) });
    }
  } else if (message.type === "close") {
    holistic?.close();
    holistic = null;
    self.close();
  }
};
