// The phone camera (the FrameGrabber + capture part of vsl_ui.CameraWorker): a <video> fed by getUserMedia,
// frames drawn into a work canvas — mirrored for the front camera, like the PC's webcam frames, so samples
// recorded here look like the PC's webcam samples — each with the moment it was captured (seconds).

export class CameraError extends Error {}

export const CAMERA_START_SECONDS = 10.0; // vsl_ui: a camera that opens but sends no picture this long is reported

// An older start() superseded by a newer start() or stop(): its stream is closed and it is not an error to show.
export class CameraAborted extends Error {}
export const START_ATTEMPTS = 4; // a camera held a moment by another app is opened again (0.5, 1, 2 s later)
// A camera that worked here and then went missing (its track ended: released by the system or another app, the
// browser's capture service restarting) is looked for this many times (0.5, 1, 2, 4, 8 s later) before it is reported.
export const REAPPEAR_ATTEMPTS = 6;

// What is asked of the camera: its whole picture. A fixed 640x480 made phones crop their picture to that shape (in
// portrait the middle third of the height: the person's hands and body left the frame, as if zoomed in); "none" asks
// for a picture the sensor makes without cropping, a size near 1280 x 960 keeps the full field of view (4:3 sensors),
// and the frames are scaled down for analysis anyway (pipeline.PROCESS_MAX).
export function constraints(facing) {
  return {
    audio: false,
    video: {
      facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 960 }, frameRate: { ideal: 30 },
      resizeMode: { ideal: "none" },
    },
  };
}

// Resolves when the video presents its first picture.
function firstPicture(video) {
  return new Promise((resolve) => {
    if (video.requestVideoFrameCallback) {
      video.requestVideoFrameCallback(() => resolve());
      return;
    }
    const check = () => (video.readyState >= 2 && video.videoWidth ? resolve() : requestAnimationFrame(check));
    check();
  });
}

function within(promise, seconds) {
  let timer = 0;
  return Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("timeout")), seconds * 1000); })])
    .finally(() => clearTimeout(timer));
}

// The widest view the camera has: phones that can zoom (Android) sometimes open zoomed in; the smallest zoom shows
// the most of the person. Never an error (not every camera can zoom).
async function widest(track, wanted) {
  try {
    const zoom = track?.getCapabilities?.().zoom;
    if (zoom && Number.isFinite(zoom.min) && track.getSettings?.().zoom !== zoom.min) {
      // The same picture as asked when opening (applyConstraints replaces every constraint), at the smallest zoom.
      await track.applyConstraints({ ...wanted, advanced: [{ zoom: zoom.min }] });
    }
  } catch {
    // keeps the camera's own zoom
  }
}

export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
    this.facing = "user";
    this.lastFrame = -1;
    this.generation = 0; // every start() / stop(): an older start's stream is closed instead of kept
  }

  get running() {
    return Boolean(this.stream);
  }

  get width() {
    return this.video.videoWidth;
  }

  get height() {
    return this.video.videoHeight;
  }

  get mirrored() {
    return this.facing === "user";
  }

  // Open the camera and wait for its first picture. A track that ends at once (the camera is still held by another
  // app or tab for a moment) is opened again a few times before it is reported.
  async start(facing = this.facing) {
    this.stop();
    const generation = this.generation;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new CameraError("Trình duyệt này không cho dùng camera (cần mở trang qua https).");
    }
    for (let attempt = 1; ; attempt++) {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints(facing));
      } catch (error) {
        if (generation !== this.generation) throw new CameraAborted("superseded");
        // Held by another app or tab for a moment: opened again a few times before it is reported. Missing
        // ("NotFoundError") only counts as a moment when this camera worked here before (else there is no camera).
        const busy = ["NotReadableError", "AbortError", "TrackStartError"].includes(error?.name);
        const missing = ["NotFoundError", "DevicesNotFoundError"].includes(error?.name) && this.everOpened;
        if ((busy && attempt < START_ATTEMPTS) || (missing && attempt < REAPPEAR_ATTEMPTS)) {
          await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** (attempt - 1)));
          if (generation !== this.generation) throw new CameraAborted("superseded");
          continue;
        }
        const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
        throw new CameraError(denied
          ? "Chưa được phép dùng camera — hãy cho phép camera trong cài đặt trình duyệt rồi thử lại."
          : "Không mở được camera. Hãy đóng ứng dụng khác đang dùng camera rồi thử lại.");
      }
      this.everOpened = true; // the camera exists: if it goes missing later, that is for a moment
      const close = () => { for (const track of stream.getTracks()) track.stop(); };
      if (generation !== this.generation) { // stopped or started again meanwhile: never leave this one running
        close();
        throw new CameraAborted("superseded");
      }
      const track = stream.getVideoTracks()[0];
      const ended = new Promise((_, reject) => {
        if (track?.readyState === "ended") reject(new Error("ended"));
        track?.addEventListener("ended", () => reject(new Error("ended")), { once: true });
      });
      ended.catch(() => {});
      this.stream = stream;
      this.video.srcObject = stream;
      try {
        // Playing and a first picture, or the track ending, or nothing for CAMERA_START_SECONDS.
        await within(Promise.race([ended, (async () => {
          await this.video.play();
          await firstPicture(this.video);
        })()]), CAMERA_START_SECONDS);
        if (track?.readyState === "ended") throw new Error("ended");
      } catch (error) {
        if (generation !== this.generation) throw new CameraAborted("superseded");
        close();
        this.stream = null;
        this.video.srcObject = null;
        if (error.message === "ended" && attempt < START_ATTEMPTS) {
          await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** (attempt - 1)));
          if (generation !== this.generation) throw new CameraAborted("superseded");
          continue;
        }
        throw new CameraError(error.message === "ended"
          ? "Camera đang bị ứng dụng khác dùng hoặc vừa bị ngắt — hãy đóng ứng dụng đó rồi bật lại camera."
          : "Camera không gửi hình — hãy đóng ứng dụng khác đang dùng camera rồi thử lại.");
      }
      if (generation !== this.generation) throw new CameraAborted("superseded");
      this.facing = facing; // only a camera that works becomes the one the next start uses
      await widest(track, constraints(facing).video);
      if (generation !== this.generation) throw new CameraAborted("superseded"); // stopped while it zoomed out
      return generation; // which opening this is: a later start or stop moves this.generation past it

    }
  }

  stop() {
    this.generation++;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.video.srcObject = null;
  }

  async switchFacing() {
    await this.start(this.facing === "user" ? "environment" : "user");
  }

  // Resolves with the capture time (seconds) of the next new video frame.
  nextFrame() {
    const video = this.video;
    if (video.requestVideoFrameCallback) {
      return new Promise((resolve) => video.requestVideoFrameCallback((now, meta) => {
        resolve((meta.captureTime ?? meta.expectedDisplayTime ?? now) / 1000);
      }));
    }
    return new Promise((resolve) => {
      const wait = () => {
        if (video.currentTime !== this.lastFrame) {
          this.lastFrame = video.currentTime;
          resolve(performance.now() / 1000);
        } else requestAnimationFrame(wait);
      };
      requestAnimationFrame(wait);
    });
  }

  // Draw the current frame into `canvas` (resized to width x height: by default the video's size), mirrored for the
  // front camera.
  grab(canvas, width = this.video.videoWidth, height = this.video.videoHeight) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.save();
    if (this.mirrored) {
      context.translate(width, 0);
      context.scale(-1, 1);
    }
    context.drawImage(this.video, 0, 0, width, height);
    context.restore();
    return context;
  }
}
