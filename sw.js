// Service worker: VSL Studio Mobile works offline after the first visit. tools/publish_mobile.py stamps BUILD,
// PRECACHE (every file the app needs at once), LAZY (files fetched when first used: ONNX Runtime and the face
// models, the no-SIMD wasm, Tesseract) and LAZY_FILES (each of those files' content hash).
// - Each build has its own cache, filled straight from the network (not the HTTP cache: a new build never gets the
//   previous build's files) and answered cache-first, so the app always runs one consistent build.
// - Lazy files live in one cache, each under its own content hash: an update keeps every file that did not change
//   (offline login and enrollment keep working) and drops only the changed or removed ones.
const BUILD = "ddb00b2-20261113050515";
const PRECACHE = ["./", "app.css", "data/index.json", "data/keypoints.bin", "data/qipedc.json", "icons/icon-192.png", "icons/icon-512-maskable.png", "icons/icon-512.png", "icons/qr.png", "index.html", "js/app.js", "js/bundle.js", "js/camera.js", "js/collectbatch.js", "js/collector.js", "js/collectorui.js", "js/detector.worker.js", "js/detectors.js", "js/draw.js", "js/faces.js", "js/holistic.js", "js/install.js", "js/layout.js", "js/light.js", "js/live.js", "js/lock.js", "js/naming.js", "js/ocr.js", "js/people.js", "js/pipeline.js", "js/recognition.js", "js/recognition.worker.js", "js/segmentation.js", "js/sentence.js", "js/sources.js", "js/store.js", "js/take.js", "js/translate.js", "js/videocollect.js", "manifest.webmanifest", "models/holistic_landmarker.task", "vendor/tasks-vision/vision_bundle.mjs", "vendor/tasks-vision/wasm/vision_wasm_internal.js", "vendor/tasks-vision/wasm/vision_wasm_internal.wasm", "version.json"];
const LAZY = ["vendor/ort/", "models/face_", "vendor/tasks-vision/wasm/vision_wasm_nosimd", "vendor/tesseract/"];
const LAZY_FILES = {"models/face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4", "models/face_recognition_sface_2021dec_int8.onnx": "2b0e941e6f16cc04", "vendor/ort/ort-wasm-simd-threaded.mjs": "ad9ee5889344e212", "vendor/ort/ort-wasm-simd-threaded.wasm": "3cbf90f74a919ff8", "vendor/ort/ort.wasm.min.mjs": "71cb6a78b3989d8e", "vendor/tasks-vision/wasm/vision_wasm_nosimd_internal.js": "942654dbdcebce08", "vendor/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm": "c9b1b69bb87baf9e", "vendor/tesseract/tesseract-core-lstm.wasm.js": "775a35df6f2ae100", "vendor/tesseract/tesseract-core-simd-lstm.wasm.js": "9d7c43fb206dc9f4", "vendor/tesseract/tesseract.esm.min.js": "da6267cfe5036ae7", "vendor/tesseract/vie.traineddata.gz": "2284f610f262a1b1", "vendor/tesseract/worker.min.js": "38645599043239c0"};
const CACHE = `vsl-studio-${BUILD}`;
const ASSET_CACHE = "vsl-assets";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(PRECACHE.map((url) => new Request(url, { cache: "reload" })))));
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const assets = await caches.open(ASSET_CACHE);
    for (const request of await assets.keys()) {
      if (assetKey(request.url) !== request.url) await assets.delete(request); // changed or no longer published
    }
    for (const name of await caches.keys()) {
      if (name.startsWith("vsl-studio-") && name !== CACHE) await caches.delete(name);
      if (name.startsWith("vsl-assets-")) { // the earlier one-hash-for-all cache: its unchanged files move over
        try {
          const old = await caches.open(name);
          for (const request of await old.keys()) {
            const key = assetKey(request.url);
            if (!key || await assets.match(key)) continue;
            const response = await old.match(request);
            if (await contentHash(response) === LAZY_FILES[relativePath(request.url)]) await assets.put(key, response);
          }
        } catch (error) {
          console.warn("asset cache:", error); // fetched again when needed
        }
        await caches.delete(name);
      }
    }
    await self.clients.claim();
  })());
});

self.addEventListener("message", (event) => {
  if (event.data === "update") self.skipWaiting(); // the app asked to switch to this new version now
});

function relativePath(url) {
  const path = new URL(url).pathname;
  const base = new URL(self.registration.scope).pathname;
  return path.startsWith(base) ? path.slice(base.length) : path;
}

function lazy(url) {
  return LAZY.some((prefix) => relativePath(url).startsWith(prefix));
}

// The first 16 hex digits of the SHA-256 of a response's body (publish_mobile.py hashes the files the same way).
async function contentHash(response) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", await response.clone().arrayBuffer()));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

// The cache key of a lazy file: its address with its content hash (null: not a published file, never cached).
function assetKey(url) {
  const relative = relativePath(url);
  const hash = LAZY_FILES[relative];
  return hash ? new URL(`${relative}?v=${hash}`, self.registration.scope).href : null;
}

// Share target (manifest.webmanifest "share_target"): Android's share sheet (Gallery, Zalo, Files…) sends videos to the
// installed app as a POST of ./share-target. They are kept in the cache "vsl-shared" (an unread earlier share is
// replaced) and the app opens at ./?shared=1, where js/sources.js takeShared() hands them to the video collector.
const SHARED_CACHE = "vsl-shared";

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "POST" || url.origin !== self.location.origin
    || url.pathname !== new URL("share-target", self.registration.scope).pathname) return;
  event.respondWith((async () => {
    let count = 0;
    try {
      const form = await request.formData();
      const cache = await caches.open(SHARED_CACHE);
      for (const old of await cache.keys()) await cache.delete(old);
      for (const file of form.getAll("video")) {
        if (typeof file === "string" || !file.size) continue;
        const headers = {
          "Content-Type": file.type || "application/octet-stream",
          "X-VSL-Name": encodeURIComponent(file.name || "video.mp4"),
          "X-VSL-Modified": String(file.lastModified || Date.now()),
        };
        const key = new URL(`shared/${String(count).padStart(4, "0")}`, self.registration.scope);
        await cache.put(new Request(key), new Response(file, { headers }));
        count++;
      }
    } catch (error) {
      console.warn("share target:", error); // the app still opens, without the videos
    }
    return Response.redirect(new URL(count ? "./?shared=1" : "./", self.registration.scope).href, 303);
  })());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const asset = lazy(request.url);
    const key = asset ? assetKey(request.url) : request;
    const cache = await caches.open(asset ? ASSET_CACHE : CACHE);
    const cached = key ? await cache.match(key, { ignoreSearch: request.mode === "navigate" }) : null;
    if (cached) return cached;
    try {
      // A lazy file comes from the network itself (never the HTTP cache's copy of an earlier build) and is kept only
      // when it is this build's file (its content hash): a stale or partial answer is used once, not kept.
      const response = await fetch(asset ? new Request(request, { cache: "reload" }) : request);
      if (key && response.ok && response.type === "basic" && response.status === 200) {
        if (!asset) cache.put(key, response.clone());
        else if (await contentHash(response) === LAZY_FILES[relativePath(request.url)]) await cache.put(key, response.clone());
      }
      return response;
    } catch (error) {
      if (request.mode === "navigate") return (await caches.match("index.html")) ?? Response.error();
      throw error;
    }
  })());
});
