// Installing VSL Studio on the home screen: which phone / browser this is and the steps that work there. Chrome, Edge
// and Samsung Internet on Android (and Chrome / Edge on computers) install in one tap (beforeinstallprompt); iPhone
// and iPad install only from Safari's Share menu; the browsers inside Zalo, Facebook, Messenger, Instagram, TikTok,
// Line cannot install at all: the page must be opened in the phone's browser first.

const IN_APP = [
  ["Zalo", /Zalo/i],
  ["Messenger", /Messenger|FB_IAB\/MESSENGER|\bOrca-Android/i],
  ["Facebook", /FBAN|FBAV|FB_IAB|FBIOS|FB4A/i],
  ["Instagram", /Instagram/i],
  ["TikTok", /musical_ly|BytedanceWebview|TikTok|trill_/i],
  ["Line", /\bLine\//],
  ["Telegram", /Telegram/i],
];

// {os: "ios" | "android" | "desktop", browser, inApp (app name or null), standalone, label}
export function detectPlatform(ua = navigator.userAgent, nav = navigator, standaloneQuery = null) {
  const ios = /iPhone|iPad|iPod/i.test(ua) || (nav.platform === "MacIntel" && nav.maxTouchPoints > 1);
  const android = /Android/i.test(ua);
  const os = ios ? "ios" : android ? "android" : "desktop";
  const inApp = (IN_APP.find(([, pattern]) => pattern.test(ua)) ?? [null])[0]
    ?? (android && /; wv\)/.test(ua) ? "ứng dụng này" : null); // an Android WebView inside some app
  let browser = "other";
  if (ios) browser = /CriOS/.test(ua) ? "chrome" : /FxiOS/.test(ua) ? "firefox" : /EdgiOS/.test(ua) ? "edge" : inApp ? "other" : "safari";
  else if (/SamsungBrowser/.test(ua)) browser = "samsung";
  else if (/Firefox\//.test(ua)) browser = "firefox";
  else if (/EdgA?\//.test(ua)) browser = "edge";
  else if (/OPR\/|Opera/.test(ua)) browser = "opera";
  else if (/Chrome\//.test(ua)) browser = "chrome";
  else if (/Safari\//.test(ua)) browser = "safari";
  const standalone = Boolean(standaloneQuery ?? (typeof matchMedia === "function"
    && (matchMedia("(display-mode: standalone)").matches || matchMedia("(display-mode: fullscreen)").matches))) || nav.standalone === true;
  const names = { safari: "Safari", chrome: "Chrome", firefox: "Firefox", edge: "Edge", samsung: "Samsung Internet", opera: "Opera", other: "trình duyệt này" };
  const device = os === "ios" ? (/iPad/i.test(ua) || nav.maxTouchPoints > 1 && !/iPhone/i.test(ua) ? "iPad" : "iPhone") : os === "android" ? "Android" : "máy tính";
  return { os, browser, inApp, standalone, label: inApp ? `${inApp} trên ${device}` : `${names[browser]} trên ${device}` };
}

// What to show: {title, steps: [text], note, action: "prompt" | "copy" | null}. `canPrompt`: the browser offered its
// one-tap install (beforeinstallprompt was received).
export function installGuide(platform, canPrompt) {
  const { os, browser, inApp, standalone } = platform;
  const offline = "Lần mở đầu cần mạng (khoảng 50 MB, nên dùng Wi-Fi); sau đó VSL Studio chạy không cần mạng.";
  if (standalone) {
    return { title: "Đã cài trên màn hình chính", action: null, note: "",
      steps: ["Bạn đang dùng VSL Studio như một ứng dụng: mở từ biểu tượng trên màn hình chính, không cần mạng.",
        "Để cài lên máy khác: bấm “Chia sẻ ứng dụng” hoặc cho máy kia quét mã QR bên dưới."] };
  }
  if (inApp) {
    return { title: "Mở bằng trình duyệt để cài", action: "copy", note: offline,
      steps: [`Bạn đang mở VSL Studio bên trong ${inApp}: trình duyệt của ${inApp} không cài được ứng dụng lên màn hình chính.`,
        os === "ios" ? "Bấm nút ⋯ (hoặc biểu tượng la bàn) → “Mở bằng Safari” / “Mở trong trình duyệt”."
          : "Bấm nút ⋮ hoặc ⋯ ở góc trên → “Mở bằng trình duyệt” / “Mở trong Chrome”.",
        `Hoặc bấm “Sao chép liên kết” bên dưới, mở ${os === "ios" ? "Safari" : "Chrome"} rồi dán vào thanh địa chỉ.`,
        "Rồi bấm “⬇ Cài app” ở đầu trang để xem bước cài."] };
  }
  const safariSteps = ["Bấm nút Chia sẻ (ô vuông có mũi tên lên ⬆︎) ở thanh dưới cùng — trên iPad ở góc trên bên phải.",
    "Kéo danh sách xuống, chọn “Thêm vào MH chính” (Add to Home Screen).",
    "Bấm “Thêm” ở góc trên bên phải.",
    "Mở VSL Studio từ biểu tượng mới trên màn hình chính."];
  const iosStorage = "Trên iPhone / iPad, ứng dụng trên màn hình chính có kho dữ liệu riêng: mẫu và tài khoản tạo trong Safari không chuyển sang. Nếu đã thu mẫu trong Safari, hãy “⇪ Đồng bộ” (Kho từ) và nhập vào máy tính trước, rồi dùng ứng dụng từ biểu tượng.";
  if (os === "ios") {
    if (browser === "safari") return { title: "Cài lên iPhone / iPad", action: null, note: `${offline} ${iosStorage}`, steps: safariSteps };
    return { title: "Mở bằng Safari để cài", action: "copy", note: offline,
      steps: ["Trên iPhone / iPad chỉ Safari cài được ứng dụng web lên màn hình chính.",
        "Bấm “Sao chép liên kết” bên dưới, mở Safari rồi dán vào thanh địa chỉ.", ...safariSteps] };
  }
  if (canPrompt) {
    return { title: "Cài lên " + (os === "android" ? "điện thoại" : "máy tính"), action: "prompt", note: offline,
      steps: ["Bấm “Cài đặt ngay” bên dưới, rồi bấm “Cài đặt” trong hộp thoại hiện ra.",
        os === "android" ? "Biểu tượng VSL Studio xuất hiện trên màn hình chính (hoặc trong ngăn ứng dụng)."
          : "VSL Studio mở trong cửa sổ riêng và có trong menu Start / Dock."] };
  }
  if (os === "android") {
    const steps = {
      chrome: ["Bấm nút ⋮ ở góc trên bên phải của Chrome.", "Chọn “Cài đặt ứng dụng” (hoặc “Thêm vào màn hình chính” → “Cài đặt”)."],
      samsung: ["Bấm nút ≡ (menu) ở thanh dưới, hoặc biểu tượng tải xuống ⤓ trên thanh địa chỉ.", "Chọn “Thêm trang vào” → “Màn hình chờ” (hoặc “Cài đặt”)."],
      firefox: ["Bấm nút ⋮.", "Chọn “Cài đặt” (hoặc “Thêm vào màn hình chính”)."],
      edge: ["Bấm nút ⋯ ở thanh dưới.", "Chọn “Thêm vào điện thoại” / “Cài đặt ứng dụng”."],
      opera: ["Bấm nút ⋮.", "Chọn “Thêm vào màn hình chính”."],
    }[browser];
    if (steps) return { title: "Cài lên điện thoại", action: null, note: offline, steps: [...steps, "Mở VSL Studio từ biểu tượng mới trên màn hình chính."] };
    return { title: "Mở bằng Chrome để cài", action: "copy", note: offline,
      steps: ["Trình duyệt này có thể không cài được ứng dụng web.", "Bấm “Sao chép liên kết”, mở Chrome rồi dán vào thanh địa chỉ.",
        "Trong Chrome: nút ⋮ → “Cài đặt ứng dụng”."] };
  }
  if (browser === "chrome" || browser === "edge") {
    return { title: "Cài lên máy tính", action: null, note: offline,
      steps: ["Bấm biểu tượng cài đặt ⊕ ở cuối thanh địa chỉ (hoặc menu ⋮ / ⋯ → “Cài đặt VSL Studio”).", "Bấm “Cài đặt”."] };
  }
  if (browser === "safari") {
    return { title: "Cài lên máy Mac", action: null, note: offline,
      steps: ["Trên thanh menu: Tệp (File) → “Thêm vào Dock” (macOS 14 trở lên)."] };
  }
  return { title: "Mở bằng Chrome hoặc Edge để cài", action: "copy", note: offline,
    steps: ["Trình duyệt này không cài được ứng dụng web.", "Bấm “Sao chép liên kết”, mở Chrome hoặc Edge rồi dán vào thanh địa chỉ."] };
}

// The app's public address (without query or hash).
export function appAddress(location = window.location) {
  return location.origin + location.pathname.replace(/index\.html$/, "");
}
