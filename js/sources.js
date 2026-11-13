// Where collection videos come from on the phone: port of video_sources.py where a phone can do it, the deleted-segment
// checks of dataset.py / collectors.py, and the Android share target.
//
// QIPEDC, YouTube, TikTok… send no CORS headers: a web app can neither read their data nor the pixels of their videos.
// So the phone works from files the user picks or shares to the app (labelled by sub-folder or a CSV, like
// folder_items), links to video files on hosts that allow it (CORS), and data the PC ships in the bundle: the QIPEDC
// dictionary (data/qipedc.json, tools/build_mobile.py qipedc) and the deleted segments / sources (data/index.json
// "segments" / "sources"). A video downloaded from QIPEDC keeps its id as file name (D0001B.mp4): it becomes the
// dictionary item again (qipedcIdFromName / itemForFile), with the PC's media id and sign name.
//
// Items are plain objects shaped like video_sources.MediaItem: {mediaId, title, origin, platform, url, label, variant,
// details} (+ file: the File to open, for picked, shared or downloaded videos). Never a local path.

import { overlaps } from "./segmentation.js";

export const VIDEO_EXTENSIONS = [".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp",
  ".ts", ".mts", ".m2ts", ".gif"];
export const QIPEDC_BASE = "https://qipedc.moet.gov.vn";
export const QIPEDC_VARIANTS = Object.freeze({ B: "Bắc", T: "Trung", N: "Nam" });
// re.match(r"^[WD]\d{4,6}[BNT]?$"): Python's \d is any decimal digit and its $ also matches before a final newline.
export const QIPEDC_ID = /^[WD]\p{Nd}{4,6}[BNT]?\n?$/u;
export const LABEL_COLUMNS = ["label", "word", "gloss", "text", "meaning", "vietnamese", "tu", "nhan", "tu_vung", "ky_hieu"];
export const FILE_COLUMNS = ["file", "filename", "file_name", "video", "video_name", "video_id", "path", "name", "id"];
export const YOUTUBE_HOSTS = ["youtube.com", "youtu.be", "youtube-nocookie.com"];
export const FILE_PLATFORM = "Tệp trên máy";
export const CSV_DELIMITERS = ",;\t|";
export const FIELD_LIMIT = 131072; // csv.field_size_limit()
export const SHARED_CACHE = "vsl-shared"; // sw.js keeps the videos shared to the app here

// A source cannot be used; the message is shown to the user (video_sources.SourceError).
export class SourceError extends Error {
  constructor(message) {
    super(message);
    this.name = "SourceError";
  }
}

// The user stopped a download (video_sources.Cancelled).
export class Cancelled extends Error {
  constructor(message = "Đã hủy tải video.") {
    super(message);
    this.name = "Cancelled";
  }
}

// csv.Error (a broken CSV the PC would not read either).
export class CsvError extends Error {
  constructor(message) {
    super(message);
    this.name = "Error";
  }
}

// ------------------------------------------------------------------ Python string helpers

// str.isspace() / re's \s for str patterns (not JavaScript's \s: no U+FEFF, but \x1c-\x1f and \x85).
const SPACE_CLASS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const SPACE = new RegExp(`[${SPACE_CLASS}]`, "u");
const SPACE_RUNS = new RegExp(`[${SPACE_CLASS}]+`, "gu");
const WORD = "\\p{L}\\p{N}_"; // re's \w for str patterns (str.isalnum() or "_")
const NOT_WORD_RUNS = new RegExp(`[^${WORD}]+`, "gu");

// str.strip() (whitespace) or str.strip(chars).
export function pyStrip(text, chars = null) {
  const strip = chars === null ? (ch) => SPACE.test(ch) : (ch) => chars.includes(ch);
  const points = Array.from(text);
  let start = 0, end = points.length;
  while (start < end && strip(points[start])) start++;
  while (end > start && strip(points[end - 1])) end--;
  return start === 0 && end === points.length ? text : points.slice(start, end).join("");
}

const FOLDED = new Map();
// str.casefold(): full case folding (lower(upper(lower(c))) per character gives it, but for the dotless ı, which
// does not fold, and Cherokee, which folds to upper case).
export function casefold(text) {
  if (/^[\x00-\x7f]*$/.test(text)) return text.toLowerCase();
  let out = "";
  for (const ch of text) {
    let folded = FOLDED.get(ch);
    if (folded === undefined) {
      folded = ch === "\u0131" ? ch : ch.toLowerCase().toUpperCase().toLowerCase();
      FOLDED.set(ch, folded);
    }
    out += folded;
  }
  return out;
}

const nfc = (text) => text.normalize("NFC");

// Code-point comparison of Python strings (JavaScript's < compares UTF-16 units).
function comparePy(a, b) {
  const x = Array.from(a), y = Array.from(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const p = x[i].codePointAt(0), q = y[i].codePointAt(0);
    if (p !== q) return p - q;
  }
  return x.length - y.length;
}

// round(x) for floats: to the nearest integer, ties to even (exact on the double).
export function pyRound(value) {
  const floor = Math.floor(value), diff = value - floor;
  if (diff < 0.5) return floor + 0;
  if (diff > 0.5) return floor + 1;
  return (floor % 2 === 0 ? floor : floor + 1) + 0;
}

// int(text) for the decimal texts tombstone keys hold (ASCII digits, sign, underscores, surrounding whitespace).
function pyInt(text) {
  const value = pyStrip(text);
  return /^[+-]?[0-9]+(?:_[0-9]+)*$/.test(value) ? Number(value.replaceAll("_", "")) + 0 : null;
}

// float(text) for metadata values (finite decimals; inf / nan count as unreadable).
function pyFloat(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const text = pyStrip(String(value ?? ""));
  return /^[+-]?(?:[0-9]+(?:_[0-9]+)*(?:\.(?:[0-9]+(?:_[0-9]+)*)?)?|\.[0-9]+(?:_[0-9]+)*)(?:[eE][+-]?[0-9]+(?:_[0-9]+)*)?$/.test(text)
    ? Number(text.replaceAll("_", "")) : null;
}

// ------------------------------------------------------------------ pathlib (Windows) and urllib.parse

// pathlib._WindowsFlavour.splitroot (Python 3.10) -> [drive, root, rest], with "\\" as separator.
function splitRoot(part) {
  const sep = "\\";
  let first = part.slice(0, 1), second = part.slice(1, 2), prefix = "";
  if (second === sep && first === sep) {
    if (part.startsWith("\\\\?\\")) { // an extended path
      prefix = part.slice(0, 4);
      part = part.slice(4);
      if (part.startsWith("UNC\\")) {
        prefix += part.slice(0, 3);
        part = sep + part.slice(3);
      }
    }
    first = part.slice(0, 1);
    second = part.slice(1, 2);
  }
  const third = part.slice(2, 3);
  if (second === sep && first === sep && third !== sep) { // \\machine\mountpoint\...
    const index = part.indexOf(sep, 2);
    if (index !== -1) {
      let index2 = part.indexOf(sep, index + 1);
      if (index2 !== index + 1) {
        if (index2 === -1) index2 = part.length;
        return prefix ? [prefix + part.slice(1, index2), sep, part.slice(index2 + 1)]
          : [part.slice(0, index2), sep, part.slice(index2 + 1)];
      }
    }
  }
  let drive = "", root = "";
  if (second === ":" && /^[A-Za-z]$/.test(first)) {
    drive = part.slice(0, 2);
    part = part.slice(2);
    first = third;
  }
  if (first === sep) {
    root = first;
    part = part.replace(/^\\+/, "");
  }
  return [prefix + drive, root, part];
}

// Path(text).name on the PC (Windows: "/" and "\" separate, a drive or UNC share is no name).
export function windowsName(text) {
  if (!text) return "";
  const [, , rest] = splitRoot(String(text).replaceAll("/", "\\"));
  const parts = rest.split("\\").filter((part) => part && part !== ".");
  return parts.length ? parts[parts.length - 1] : "";
}

function nameStem(name) {
  const i = name.lastIndexOf(".");
  return i > 0 && i < name.length - 1 ? name.slice(0, i) : name;
}

function nameSuffix(name) {
  const i = name.lastIndexOf(".");
  return i > 0 && i < name.length - 1 ? name.slice(i) : "";
}

// Path(text).stem / .suffix on the PC.
export const pathStem = (text) => nameStem(windowsName(text));
export const pathSuffix = (text) => nameSuffix(windowsName(text));

const USES_PARAMS = new Set(["", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtspu", "sip", "sips",
  "mms", "sftp", "tel"]);

// urllib.parse.urlparse (Python 3.10): {scheme, netloc, path, params, query, fragment}; % escapes are kept.
// Like Python it throws on unbalanced IPv6 brackets and netlocs that NFKC turns into separators (bracketed hosts
// are not validated further).
export function urlparse(url) {
  url = String(url).replace(/^[\x00-\x20]+/, "").replace(/[\t\r\n]/g, "");
  let scheme = "", netloc = "", query = "", fragment = "", params = "";
  const colon = url.indexOf(":");
  if (colon > 0 && /^[A-Za-z0-9+\-.]+$/.test(url.slice(0, colon))) {
    scheme = url.slice(0, colon).toLowerCase();
    url = url.slice(colon + 1);
  }
  if (url.slice(0, 2) === "//") {
    let end = url.length;
    for (const ch of "/?#") {
      const found = url.indexOf(ch, 2);
      if (found >= 0) end = Math.min(end, found);
    }
    netloc = url.slice(2, end);
    url = url.slice(end);
    if (netloc.includes("[") !== netloc.includes("]")) throw new Error("Invalid IPv6 URL");
  }
  const hash = url.indexOf("#");
  if (hash >= 0) {
    fragment = url.slice(hash + 1);
    url = url.slice(0, hash);
  }
  const question = url.indexOf("?");
  if (question >= 0) {
    query = url.slice(question + 1);
    url = url.slice(0, question);
  }
  if (netloc && !/^[\x00-\x7f]*$/.test(netloc)) { // _checknetloc
    const bare = netloc.replace(/[@:#?]/g, ""), normal = bare.normalize("NFKC");
    if (bare !== normal && /[/?#@:]/.test(normal)) {
      throw new Error(`netloc '${netloc}' contains invalid characters under NFKC normalization`);
    }
  }
  if (USES_PARAMS.has(scheme) && url.includes(";")) {
    let i = url.indexOf(";", url.includes("/") ? url.lastIndexOf("/") : 0);
    if (i >= 0) {
      params = url.slice(i + 1);
      url = url.slice(0, i);
    }
  }
  return { scheme, netloc, path: url, params, query, fragment };
}

const UTF8 = new TextDecoder("utf-8", { ignoreBOM: true }); // invalid bytes -> U+FFFD, like errors="replace"

// urllib.parse.unquote: runs of ASCII with %XX escapes decoded as UTF-8; other characters kept.
export function unquote(text) {
  if (!text.includes("%")) return text;
  return text.replace(/[\x00-\x7f]+/g, (run) => {
    if (!run.includes("%")) return run;
    const bytes = [];
    for (let i = 0; i < run.length; i++) {
      if (run[i] === "%" && /^[0-9A-Fa-f]{2}$/.test(run.slice(i + 1, i + 3))) {
        bytes.push(parseInt(run.slice(i + 1, i + 3), 16));
        i += 2;
      } else {
        bytes.push(run.charCodeAt(i));
      }
    }
    return UTF8.decode(new Uint8Array(bytes));
  });
}

async function digestHex(algorithm, text) {
  const digest = await crypto.subtle.digest(algorithm, new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

// ------------------------------------------------------------------ sources

export function isUrl(text) {
  const lowered = pyStrip(String(text)).toLowerCase();
  return lowered.startsWith("http://") || lowered.startsWith("https://");
}

// video_sources.classify for what a phone can be given as text: "url", "qipedc" or "invalid" (the PC also takes a
// file or folder path here; a phone has none: files are picked or shared).
export function classify(text) {
  text = pyStrip(String(text ?? ""));
  if (!text) return "invalid";
  if (text.toLowerCase().startsWith("qipedc:")) return "qipedc";
  if (isUrl(text)) return "url";
  return "invalid";
}

// The 11-character id of a YouTube link (watch, youtu.be, shorts, embed, live); null for other sites.
export function youtubeId(url) {
  const host = urlparse(pyStrip(url)).netloc.toLowerCase().split("@").at(-1).split(":")[0];
  if (!YOUTUBE_HOSTS.some((name) => host === name || host.endsWith(`.${name}`))) return null;
  const match = /(?:v=|youtu\.be\/|shorts\/|embed\/|live\/)([A-Za-z0-9_-]{11})/.exec(url);
  return match ? match[1] : null;
}

// video_sources._direct_media_url: a link to a video file (by its path's extension; .ts pages are not).
export function directMediaUrl(url) {
  const path = urlparse(url).path.toLowerCase();
  return VIDEO_EXTENSIONS.some((extension) => extension !== ".ts" && path.endsWith(extension));
}

export function qipedcIdFromUrl(url) {
  const parsed = urlparse(url);
  if (!parsed.netloc.toLowerCase().endsWith("qipedc.moet.gov.vn")) return null;
  const match = /\/videos\/([WD]\p{Nd}{4,6}[BNT]?)\.mp4$/u.exec(parsed.path);
  return match ? match[1] : null;
}

const PLATFORMS = [ // a web page's platform by its host (the PC asks yt-dlp's extractor)
  [["youtube.com", "youtu.be", "youtube-nocookie.com"], "YouTube"], [["facebook.com", "fb.watch", "fb.com"], "Facebook"],
  [["tiktok.com"], "TikTok"], [["vimeo.com"], "Vimeo"], [["dailymotion.com", "dai.ly"], "Dailymotion"],
  [["instagram.com"], "Instagram"], [["twitter.com", "x.com"], "X/Twitter"],
];

export function webPlatform(url) {
  let host = "";
  try {
    host = urlparse(pyStrip(url)).netloc.toLowerCase().split("@").at(-1).split(":")[0];
  } catch {
    return "Web";
  }
  for (const [hosts, name] of PLATFORMS) if (hosts.some((h) => host === h || host.endsWith(`.${h}`))) return name;
  return "Web";
}

// What a phone can do with a link: "qipedc" (a dictionary video: by its id), "direct" (a video file), "page"
// (YouTube, TikTok, any other page: only the PC's collector reads those, with yt-dlp).
export function linkKind(url) {
  if (qipedcIdFromUrl(url)) return "qipedc";
  if (directMediaUrl(url)) return "direct";
  return "page";
}

// Why a page link cannot be collected on the phone.
export function pageLinkMessage(url) {
  const platform = webPlatform(url);
  const where = platform === "Web" ? "Link này là một trang web, không phải file video" : `Video ${platform}`;
  return `${where}: chỉ VSL Studio trên máy tính (“Thu thập đa nguồn”) mới thu được, vì trang này không cho ứng dụng `
    + "web đọc video. Trên điện thoại: tải video về máy rồi chọn tệp, hoặc dán link trực tiếp tới file video (.mp4, .webm…).";
}

function mediaItem(fields) {
  return { url: null, label: null, variant: null, details: {}, ...fields };
}

// A video file picked or shared on the phone (video_sources.file_item): media id "file:" + the PC's file fingerprint,
// so both know it is the same video. origin: the file name (or its path inside a picked folder), never a local path.
export async function fileItem(file, label = null, origin = null) {
  const { fileFingerprint } = await import("./videocollect.js"); // loaded only here: it brings MediaPipe
  return mediaItem({ mediaId: `file:${await fileFingerprint(file)}`, title: nameStem(file.name), origin: origin ?? file.name,
    platform: FILE_PLATFORM, label, file });
}

// A link to a video file (video_sources.url_items' direct case): media id "url:" + sha1(url)[:16].
export async function urlItem(url) {
  const name = windowsName(unquote(urlparse(url).path));
  return mediaItem({ mediaId: `url:${(await digestHex("SHA-1", url)).slice(0, 16)}`, title: pathStem(name) || url,
    origin: url, platform: "Web", url });
}

// video_sources.url_items for the links a phone can use: a QIPEDC video URL (by its id: the dictionary item) or a
// link to a video file; any other page throws a SourceError explaining that only the PC collects it.
export async function urlItems(url, catalog = null) {
  url = pyStrip(url);
  const videoId = qipedcIdFromUrl(url);
  if (videoId) return [qipedcItem(qipedcLookup(videoId, catalog ?? await loadCatalog()))];
  if (directMediaUrl(url)) return [await urlItem(url)];
  throw new SourceError(pageLinkMessage(url));
}

// video_sources.items_for for typed text: a link, or "qipedc:ID,ID".
export async function itemsFor(text, catalog = null) {
  const kind = classify(text);
  if (kind === "url") return urlItems(text, catalog);
  if (kind === "qipedc") {
    const ids = text.slice(text.indexOf(":") + 1).split(",").map((part) => pyStrip(part)).filter(Boolean);
    if (!ids.length) return [];
    catalog ??= await loadCatalog();
    return ids.map((videoId) => qipedcItem(qipedcLookup(videoId, catalog)));
  }
  throw new SourceError("Không nhận ra nguồn: hãy dán link video, hoặc chọn file/thư mục video.");
}

function refusedMessage(url) {
  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    return "Không có mạng: không tải được video. Hãy kết nối mạng rồi thử lại.";
  }
  const videoId = qipedcIdFromUrl(url);
  if (videoId) {
    return `Từ điển QIPEDC không cho ứng dụng web tải video. Hãy mở ${qipedcVideoUrl(videoId)} trong trình duyệt, tải video `
      + `về máy (giữ tên tệp ${videoId}.mp4) rồi chọn tệp đó: ứng dụng nhận ra mã từ điển.`;
  }
  let host = "";
  try {
    host = urlparse(url).netloc.split("@").at(-1);
  } catch { /* the message without the host */ }
  return `Trang ${host || "này"} không cho ứng dụng web tải video (không có CORS). Hãy tải video về máy rồi chọn nó như một tệp.`;
}

// Download a video file from a host that allows web apps (CORS) -> File (named after the link). progress(fraction or
// -1, text); cancelled() -> true, or an AbortSignal, stops it (Cancelled). A host that refuses web apps (no CORS: a
// TypeError, an opaque answer) -> SourceError telling the user to download the video and pick it as a file.
export async function fetchVideo(url, { progress = null, cancelled = null, signal = null } = {}) {
  url = pyStrip(url);
  let response;
  try {
    response = await fetch(url, { mode: "cors", credentials: "omit", signal });
  } catch (error) {
    if (error?.name === "AbortError") throw new Cancelled();
    throw new SourceError(refusedMessage(url));
  }
  if (response.type === "opaque" || response.type === "opaqueredirect") throw new SourceError(refusedMessage(url));
  if (!response.ok) {
    throw new SourceError(`Không tải được video: HTTP Error ${response.status}: ${response.statusText || "lỗi máy chủ"}`);
  }
  const total = Number(response.headers.get("Content-Length")) || 0;
  const type = (response.headers.get("Content-Type") || "").split(";")[0].trim();
  const name = windowsName(unquote(urlparse(url).path)) || "video.mp4";
  const chunks = [];
  let done = 0;
  const reader = response.body?.getReader();
  if (!reader) return new File([await response.blob()], name, { type });
  for (;;) {
    if (cancelled?.()) {
      reader.cancel().catch(() => {});
      throw new Cancelled();
    }
    let part;
    try {
      part = await reader.read();
    } catch (error) {
      if (error?.name === "AbortError") throw new Cancelled();
      throw new SourceError(`Không tải được video: ${error?.message || error}`);
    }
    if (part.done) break;
    chunks.push(part.value);
    done += part.value.length;
    progress?.(total ? done / total : -1, `${(done / 1e6).toFixed(1)} MB`);
  }
  // A body shorter than Content-Length is cut off; a longer one was compressed on the way (fetch decodes it).
  if (total && done < total) throw new SourceError(`Tải chưa đủ dữ liệu (${done}/${total} byte); hãy thử lại.`);
  return new File(chunks, name, { type });
}

// ------------------------------------------------------------------ label CSV files (video_sources.read_label_csv)

// re patterns of csv.Sniffer._guess_quote_and_delimiter; Python's MULTILINE ^ / $ only see "\n" (JavaScript's m flag
// also "\r"), so they are spelled out.
const LINE_START = "(?:^|(?<=\\n))", LINE_END = "(?:$|(?=\\n))";
const NOT_WORD_QUOTE = `[^${WORD}\\n"']`;
const QUOTE_PATTERNS = [
  `(?<delim>${NOT_WORD_QUOTE})(?<space> ?)(?<quote>["']).*?\\k<quote>\\k<delim>`,
  `(?:${LINE_START}|\\n)(?<quote>["']).*?\\k<quote>(?<delim>${NOT_WORD_QUOTE})(?<space> ?)`,
  `(?<delim>${NOT_WORD_QUOTE})(?<space> ?)(?<quote>["']).*?\\k<quote>(?:${LINE_END}|\\n)`,
  `(?:${LINE_START}|\\n)(?<quote>["']).*?\\k<quote>(?:${LINE_END}|\\n)`,
].map((source) => new RegExp(source, "gsu"));

const escapeRegExp = (text) => text.replace(/[\\^$.*+?()[\]{}|/]/g, "\\$&");

function firstMax(map) { // max(d, key=d.get): the first key with the largest value
  let best = null, value = -Infinity;
  for (const [key, count] of map) if (count > value) [best, value] = [key, count];
  return best;
}

function countOf(text, part) {
  return text.split(part).length - 1;
}

function guessQuoteAndDelimiter(data, delimiters) {
  let matches = [];
  for (const regexp of QUOTE_PATTERNS) {
    matches = [...data.matchAll(regexp)];
    if (matches.length) break;
  }
  if (!matches.length) return ["", false, null, 0];
  const quotes = new Map(), delims = new Map();
  let spaces = 0;
  for (const match of matches) {
    const { quote, delim, space } = match.groups;
    if (quote) quotes.set(quote, (quotes.get(quote) ?? 0) + 1);
    if (!("delim" in match.groups)) continue;
    if (delim && (delimiters === null || delimiters.includes(delim))) delims.set(delim, (delims.get(delim) ?? 0) + 1);
    if (space) spaces++;
  }
  const quotechar = firstMax(quotes);
  let delim, skipinitialspace;
  if (delims.size) {
    delim = firstMax(delims);
    skipinitialspace = delims.get(delim) === spaces;
    if (delim === "\n") delim = "";
  } else {
    delim = "";
    skipinitialspace = 0;
  }
  // an extra quote between delimiters: a doubled-quote format
  const d = escapeRegExp(delim), q = quotechar, notWord = `[^${WORD}]`;
  const doubled = new RegExp(`((${d})|${LINE_START})${notWord}*${q}[^${d}\\n]*${q}[^${d}\\n]*${q}${notWord}*((${d})|${LINE_END})`, "u");
  return [quotechar, doubled.test(data), delim, skipinitialspace];
}

function guessDelimiter(sample, delimiters) {
  const data = sample.split("\n").filter(Boolean);
  const chunkLength = Math.min(10, data.length);
  let iteration = 0, start = 0, end = chunkLength;
  const charFrequency = new Map(), modes = new Map(), delims = new Map(); // keyed by char code (7-bit ASCII)
  const skip = (delim) => countOf(data[0], delim) === countOf(data[0], `${delim} `);
  while (start < data.length) {
    iteration++;
    for (const line of data.slice(start, end)) {
      const counts = new Array(127).fill(0);
      for (let i = 0; i < line.length; i++) {
        const code = line.charCodeAt(i);
        if (code < 127) counts[code]++;
      }
      for (let code = 0; code < 127; code++) {
        let meta = charFrequency.get(code);
        if (!meta) charFrequency.set(code, (meta = new Map()));
        meta.set(counts[code], (meta.get(counts[code]) ?? 0) + 1);
      }
    }
    for (const [code, meta] of charFrequency) {
      const items = [...meta];
      if (items.length === 1 && items[0][0] === 0) continue;
      if (items.length > 1) { // the mode of the frequencies, less the other frequencies' counts
        let best = 0;
        for (let i = 1; i < items.length; i++) if (items[i][1] > items[best][1]) best = i;
        const others = items.reduce((sum, item, i) => (i === best ? sum : sum + item[1]), 0);
        modes.set(code, [items[best][0], items[best][1] - others]);
      } else {
        modes.set(code, items[0]);
      }
    }
    const total = Math.min(chunkLength * iteration, data.length);
    let consistency = 1.0;
    const threshold = 0.9;
    while (delims.size === 0 && consistency >= threshold) {
      for (const [code, mode] of modes) {
        if (mode[0] > 0 && mode[1] > 0 && mode[1] / total >= consistency
          && (delimiters === null || delimiters.includes(String.fromCharCode(code)))) delims.set(code, mode);
      }
      consistency -= 0.01;
    }
    if (delims.size === 1) {
      const delim = String.fromCharCode([...delims.keys()][0]);
      return [delim, skip(delim)];
    }
    start = end;
    end += chunkLength;
  }
  if (!delims.size) return ["", 0];
  if (delims.size > 1) {
    for (const delim of [",", "\t", ";", " ", ":"]) if (delims.has(delim.charCodeAt(0))) return [delim, skip(delim)];
  }
  const items = [...delims].map(([code, mode]) => [mode, String.fromCharCode(code)]);
  items.sort((a, b) => a[0][0] - b[0][0] || a[0][1] - b[0][1] || comparePy(a[1], b[1]));
  const delim = items[items.length - 1][1];
  return [delim, skip(delim)];
}

export const EXCEL = Object.freeze({ delimiter: ",", quotechar: '"', doublequote: true, skipinitialspace: false });

// csv.Sniffer().sniff(sample, delimiters) -> {delimiter, quotechar, doublequote, skipinitialspace}; throws CsvError
// ("Could not determine delimiter") like Python.
export function sniffCsv(sample, delimiters = null) {
  let [quotechar, doublequote, delimiter, skipinitialspace] = guessQuoteAndDelimiter(sample, delimiters);
  if (!delimiter) [delimiter, skipinitialspace] = guessDelimiter(sample, delimiters);
  if (!delimiter) throw new CsvError("Could not determine delimiter");
  return { delimiter, quotechar: quotechar || '"', doublequote: Boolean(doublequote), skipinitialspace: Boolean(skipinitialspace) };
}

// Lines of a text file opened with newline="": ends at "\n", "\r" or "\r\n", kept.
function* fileLines(text) {
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 10 || code === 13) {
      if (code === 13 && text.charCodeAt(i + 1) === 10) i++;
      yield text.slice(start, i + 1);
      start = i + 1;
    }
  }
  if (start < text.length) yield text.slice(start);
}

const START_RECORD = 0, START_FIELD = 1, IN_FIELD = 2, IN_QUOTED_FIELD = 3, QUOTE_IN_QUOTED_FIELD = 4, EAT_CRNL = 5;

// csv.reader(text file, dialect) (the C state machine of Python 3.10, QUOTE_MINIMAL, no escapechar, not strict):
// records as arrays of strings; a blank line is []. Throws CsvError on NUL or a field over FIELD_LIMIT.
export function* csvRows(text, dialect = EXCEL) {
  const { delimiter, quotechar, doublequote, skipinitialspace } = dialect;
  const lines = fileLines(text);
  for (;;) {
    const fields = [];
    let field = "", length = 0, state = START_RECORD;
    const save = () => {
      fields.push(field);
      field = "";
      length = 0;
    };
    const add = (ch) => {
      if (length >= FIELD_LIMIT) throw new CsvError(`field larger than field limit (${FIELD_LIMIT})`);
      field += ch;
      length++;
    };
    const process = (ch) => { // ch === null: the end of a line
      const newline = ch === "\n" || ch === "\r" || ch === null;
      switch (state) {
        case START_RECORD:
          if (ch === null) break; // an empty line: []
          if (ch === "\n" || ch === "\r") {
            state = EAT_CRNL;
            break;
          }
          state = START_FIELD;
        // falls through
        case START_FIELD:
          if (newline) {
            save();
            state = ch === null ? START_RECORD : EAT_CRNL;
          } else if (ch === quotechar) {
            state = IN_QUOTED_FIELD;
          } else if (ch === " " && skipinitialspace) {
            // spaces at the start of a field are skipped
          } else if (ch === delimiter) {
            save();
          } else {
            add(ch);
            state = IN_FIELD;
          }
          break;
        case IN_FIELD:
          if (newline) {
            save();
            state = ch === null ? START_RECORD : EAT_CRNL;
          } else if (ch === delimiter) {
            save();
            state = START_FIELD;
          } else {
            add(ch);
          }
          break;
        case IN_QUOTED_FIELD:
          if (ch === null) break;
          if (ch === quotechar) state = doublequote ? QUOTE_IN_QUOTED_FIELD : IN_FIELD;
          else add(ch);
          break;
        case QUOTE_IN_QUOTED_FIELD:
          if (ch === quotechar) {
            add(ch);
            state = IN_QUOTED_FIELD;
          } else if (ch === delimiter) {
            save();
            state = START_FIELD;
          } else if (newline) {
            save();
            state = ch === null ? START_RECORD : EAT_CRNL;
          } else {
            add(ch);
            state = IN_FIELD;
          }
          break;
        case EAT_CRNL:
          if (ch === null) state = START_RECORD;
          else if (ch !== "\n" && ch !== "\r") {
            throw new CsvError("new-line character seen in unquoted field - do you need to open the file in universal-newline mode?");
          }
          break;
      }
    };
    let ended = false;
    do {
      const line = lines.next();
      if (line.done) {
        if (length !== 0 || state === IN_QUOTED_FIELD) {
          save();
          ended = true;
          break;
        }
        return;
      }
      for (const ch of line.value) {
        if (ch === "\0") throw new CsvError("line contains NUL");
        process(ch);
      }
      process(null);
    } while (state !== START_RECORD);
    yield fields;
    if (ended) return;
  }
}

function pickColumn(columns, candidates) {
  const lowered = new Map();
  for (const column of columns) {
    if (column) lowered.set(pyStrip(pyStrip(column).toLowerCase().replace(NOT_WORD_RUNS, "_"), "_"), column);
  }
  for (const candidate of candidates) if (lowered.has(candidate)) return lowered.get(candidate);
  return null;
}

function firstCodePoints(text, count) {
  let units = 0;
  for (let points = 0; points < count && units < text.length; points++) {
    const code = text.charCodeAt(units);
    units += code >= 0xd800 && code <= 0xdbff && (text.charCodeAt(units + 1) & 0xfc00) === 0xdc00 ? 2 : 1;
  }
  return text.slice(0, units);
}

// video_sources.read_label_csv: Map {file stem (casefold) -> sign name} from a CSV with a file column and a label
// column (delimiter sniffed among , ; tab |). text: the file decoded as UTF-8 (a leading BOM is dropped: utf-8-sig);
// name: the file's name for the error message.
export function readLabelCsv(text, name = "labels.csv") {
  if (text.startsWith("\ufeff")) text = text.slice(1);
  let dialect;
  try {
    dialect = sniffCsv(firstCodePoints(text, 8192), CSV_DELIMITERS);
  } catch (error) {
    if (!(error instanceof CsvError)) throw error;
    dialect = EXCEL;
  }
  const rows = csvRows(text, dialect);
  const header = rows.next();
  const columns = header.done ? [] : header.value;
  const fileColumn = pickColumn(columns, FILE_COLUMNS), labelColumn = pickColumn(columns, LABEL_COLUMNS);
  if (!fileColumn || !labelColumn || fileColumn === labelColumn) {
    throw new SourceError(`${name}: không xác định được cột tên file và cột nhãn (các cột: ${columns.join(", ")}).`);
  }
  const mapping = new Map();
  for (const row of rows) {
    if (!row.length) continue; // csv.DictReader skips blank lines
    const record = new Map(); // dict(zip(fieldnames, row)), missing fields None
    for (let i = 0; i < Math.min(columns.length, row.length); i++) record.set(columns[i], row[i]);
    for (const column of columns.slice(row.length)) record.set(column, null);
    const file = pyStrip(record.get(fileColumn) || "");
    const label = nfc(pyStrip(record.get(labelColumn) || ""));
    if (file && label) mapping.set(casefold(pathStem(file.replaceAll("\\", "/"))), label);
  }
  return mapping;
}

// A CSV file picked on the phone -> readLabelCsv. The PC refuses a file that is not UTF-8 (UnicodeDecodeError): so
// does the phone, with a Vietnamese message.
export async function readLabelCsvFile(file) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer());
  } catch {
    throw new SourceError(`${file.name}: không đọc được tệp (hãy lưu bảng dưới dạng “CSV UTF-8”).`);
  }
  return readLabelCsv(text, file.name);
}

// ------------------------------------------------------------------ files picked together (video_sources.folder_items)

// Path parts of a picked file below the picked folder: webkitRelativePath "Folder/sub/a.mp4" -> ["sub", "a.mp4"]
// (the folder itself is the root); a file picked alone -> [its name].
function relativeParts(file) {
  const parts = String(file.webkitRelativePath || "").split("/").filter(Boolean);
  return parts.length > 1 ? parts.slice(1) : [file.name];
}

function rootName(files) {
  for (const file of files) {
    const parts = String(file.webkitRelativePath || "").split("/").filter(Boolean);
    if (parts.length > 1) return parts[0];
  }
  return null;
}

export function isVideoName(name) {
  return VIDEO_EXTENSIONS.includes(nameSuffix(name).toLowerCase()) && !name.startsWith(".");
}

// video_sources.folder_items for files picked together (a folder picked with its sub-folders, or several files):
// the videos (no hidden files or folders), sorted like the PC, each labelled by the CSV (`csv`: a File or the text;
// null = the only .csv file picked at the top, like the PC's folder), else by its sub-folder's name, else unknown.
// catalog (QIPEDC entries): files named by a dictionary id (D0001B.mp4) become the dictionary's items (an addition of
// the phone, where QIPEDC videos can only arrive as downloaded files). -> Promise<item[]>; SourceError when there is
// no video or the CSV cannot be read.
export async function labelFiles(files, csv = null, { csvName = "labels.csv", catalog = null } = {}) {
  files = Array.from(files);
  const root = rootName(files);
  if (csv === null) {
    const tables = files.filter((file) => relativeParts(file).length === 1 && file.name.toLowerCase().endsWith(".csv"));
    if (tables.length === 1) csv = tables[0];
  }
  let mapping = new Map();
  if (typeof csv === "string") mapping = readLabelCsv(csv, csvName);
  else if (csv) mapping = await readLabelCsvFile(csv);
  const videos = files
    .map((file) => ({ file, parts: relativeParts(file) }))
    .filter(({ parts }) => isVideoName(parts[parts.length - 1]) && !parts.some((part) => part.startsWith(".")))
    .map((entry) => ({ ...entry, key: casefold(entry.parts.join("\\")) }))
    .sort((a, b) => comparePy(a.key, b.key));
  if (!videos.length) {
    throw new SourceError(root ? `Thư mục ${root} không có video (${VIDEO_EXTENSIONS.join(", ")}).`
      : `Không có video trong các tệp đã chọn (${VIDEO_EXTENSIONS.join(", ")}).`);
  }
  const items = [];
  for (const { file, parts } of videos) {
    const name = parts[parts.length - 1];
    if (catalog) {
      const dictionary = qipedcFileItem(file, catalog);
      if (dictionary) {
        items.push(dictionary);
        continue;
      }
    }
    let label = mapping.get(casefold(nameStem(name))) ?? null;
    if (label === null && parts.length > 1) label = nfc(parts[parts.length - 2]);
    items.push(await fileItem(file, label, parts.join("/")));
  }
  return items;
}

// One picked or shared video: the QIPEDC item when its name is a dictionary id in the catalogue, else a file item.
export async function itemForFile(file, catalog = null) {
  if (qipedcIdFromName(file.name)) {
    const dictionary = qipedcFileItem(file, catalog ?? await loadCatalog().catch(() => []));
    if (dictionary) return dictionary;
  }
  return fileItem(file);
}

function qipedcFileItem(file, catalog) {
  const videoId = qipedcIdFromName(file.name);
  const entry = videoId && catalog.find((candidate) => (candidate.id ?? candidate._id) === videoId);
  return entry ? { ...qipedcItem(entry), file } : null;
}

// ------------------------------------------------------------------ QIPEDC dictionary

export function qipedcVideoUrl(videoId) {
  return `${QIPEDC_BASE}/videos/${videoId}.mp4`;
}

export function qipedcVariant(videoId) {
  if (!videoId) return null;
  const last = videoId[videoId.length - 1];
  return Object.hasOwn(QIPEDC_VARIANTS, last) ? QIPEDC_VARIANTS[last] : null;
}

// Sign name from a dictionary word: explanations in brackets are dropped, spacing normalised.
export function qipedcWordLabel(word) {
  const text = nfc(word.replace(/\([^)]*\)|\[[^\]]*\]/g, " "));
  return pyStrip(text.replace(SPACE_RUNS, " "), " .,;:") || pyStrip(nfc(word));
}

// The id a video downloaded from QIPEDC keeps as its file name: "D0001B.mp4", "w01497b.MP4", with the suffix a
// browser adds to a second download ("D0001B (1).mp4", "D0001B(1).mp4", "D0001B-1.mp4"); null for other names.
export function qipedcIdFromName(fileName) {
  const name = windowsName(String(fileName ?? ""));
  const stem = VIDEO_EXTENSIONS.includes(nameSuffix(name).toLowerCase()) ? nameStem(name) : name;
  const match = /^([WD][0-9]{4,6}[BNT]?)(?:\s*\(\d+\)|[ _-]\d{1,3})?$/i.exec(pyStrip(stem));
  return match ? match[1].toUpperCase() : null;
}

// entry: a catalogue entry {id, word, ...} (data/qipedc.json; the PC's {_id, word} works too).
export function qipedcItem(entry) {
  const videoId = entry.id ?? entry._id, word = entry.word, variant = qipedcVariant(videoId), url = qipedcVideoUrl(videoId);
  return mediaItem({ mediaId: `qipedc:${videoId}`, title: variant ? `${word} (${variant})` : word, origin: url,
    platform: "QIPEDC", url, label: qipedcWordLabel(word), variant, details: { catalog_word: word, catalog_id: videoId } });
}

export function qipedcLookup(videoId, catalog) {
  videoId = pyStrip(String(videoId));
  if (!QIPEDC_ID.test(videoId)) throw new SourceError(`Mã QIPEDC không hợp lệ: ${videoId}`);
  for (const entry of catalog) if ((entry.id ?? entry._id) === videoId) return entry;
  throw new SourceError(`Không có mã ${videoId} trong từ điển QIPEDC.`);
}

const FOLDED_CATALOGS = new WeakMap();

// Entries whose word contains the query (diacritics matter; case does not): exact words or sign names first, then
// words starting with it, then the others containing it.
export function qipedcSearch(query, catalog) {
  const needle = casefold(pyStrip(nfc(query)));
  if (!needle) return [];
  let folded = FOLDED_CATALOGS.get(catalog);
  if (!folded || folded.length !== catalog.length) {
    folded = catalog.map((entry) => [casefold(entry.word), casefold(entry.label ?? qipedcWordLabel(entry.word))]);
    FOLDED_CATALOGS.set(catalog, folded);
  }
  const exact = [], starts = [], contains = [];
  catalog.forEach((entry, i) => {
    const [word, label] = folded[i];
    if (word === needle || label === needle) exact.push(entry);
    else if (word.startsWith(needle)) starts.push(entry);
    else if (word.includes(needle)) contains.push(entry);
  });
  return [...exact, ...starts, ...contains];
}

// The dictionary dialog's selection (collector_window.QipedcDialog.selected_entries): ids in order without repeats,
// only the regional variants allowed ("" = no suffix, "B", "T", "N") and only ids in the catalogue -> entries.
export function qipedcChoose(ids, catalog, allowed = ["", "B", "T", "N"]) {
  const byId = new Map(catalog.map((entry) => [entry.id ?? entry._id, entry]));
  const chosen = [];
  for (const videoId of new Set(ids)) {
    const suffix = Object.hasOwn(QIPEDC_VARIANTS, videoId.at(-1)) ? videoId.at(-1) : "";
    if (allowed.includes(suffix) && byId.has(videoId)) chosen.push(byId.get(videoId));
  }
  return chosen;
}

const LOADED = new Map();

// data/qipedc.json (tools/build_mobile.py qipedc): {catalog: [{id, word, label, variant}], topics: [{key, title, words:
// [{word, ids}]}]} — the curated everyday topics of video_sources.qipedc_topics, checked against the catalogue.
export function loadQipedc(base = new URL("../data/", import.meta.url).href) {
  if (!LOADED.has(base)) {
    const loading = fetch(`${base}qipedc.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .catch((error) => {
        LOADED.delete(base);
        throw new SourceError(`Không đọc được từ điển QIPEDC của ứng dụng (${error?.message || error}).`);
      });
    LOADED.set(base, loading);
  }
  return LOADED.get(base);
}

async function loadCatalog() {
  return (await loadQipedc()).catalog;
}

// ------------------------------------------------------------------ deleted segments and sources

function keyList(keys) {
  if (!keys) return [];
  return keys instanceof Set ? [...keys] : Array.from(keys);
}

// dataset._source_hash: sha256(media id)[:20], lower-case hex (tombstone keys never reveal the source).
export async function sourceHash(mediaId) {
  return (await digestHex("SHA-256", String(mediaId))).slice(0, 20);
}

// index: anything with the bundle's tombstone keys: the raw data/index.json, decodeBundle's result or loadProvenance()
// ({segments: ["seg:…"], sources: ["src:…"]}; arrays or Sets: add the keys of samples deleted on the phone with
// provenanceKeys). -> [[start, end]] in seconds of deleted samples collected from this video, sorted.
export async function deletedSegments(mediaId, index) {
  const prefix = `seg:${await sourceHash(mediaId)}@`;
  const segments = [];
  for (const key of keyList(index?.segments)) {
    if (!key.startsWith(prefix)) continue;
    const rest = key.slice(prefix.length), dash = rest.indexOf("-");
    if (dash < 0) continue;
    const start = pyInt(rest.slice(0, dash)), end = pyInt(rest.slice(dash + 1));
    if (start === null || end === null) continue;
    segments.push([start / 1000, end / 1000]);
  }
  return segments.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

// collectors.deleted_before: true when the user deleted a sample of (most of) frames [first, last] of this video.
export async function deletedBefore(mediaId, first, last, fps, index) {
  return (await deletedSegments(mediaId, index))
    .some(([start, end]) => overlaps(first, last, pyRound(start * fps), pyRound(end * fps)));
}

// deletedBefore for many proposals of one video: the video's deleted segments looked up once -> (first, last, fps) =>
// boolean, synchronous.
export async function deletedChecker(mediaId, index) {
  const segments = await deletedSegments(mediaId, index);
  return (first, last, fps) => segments.some(([start, end]) => overlaps(first, last, pyRound(start * fps), pyRound(end * fps)));
}

// dataset.source_deleted: true when a whole-video import of this source (dictionary entry, labelled file) was deleted.
export async function sourceDeleted(mediaId, index) {
  return keyList(index?.sources).includes(`src:${await sourceHash(mediaId)}`);
}

// dataset.provenance_keys(metadata): the tombstone keys a deleted sample leaves ("mobile:<id>", "seg:…", "src:…").
export async function provenanceKeys(metadata) {
  const keys = [];
  const mobileId = pyStrip(String(metadata.mobile_id || "")).toLowerCase();
  if (mobileId && /^[0-9a-f]+$/.test(mobileId)) keys.push(`mobile:${mobileId}`);
  const mediaId = pyStrip(String(metadata.media_id || ""));
  if (!mediaId) return keys;
  const source = await sourceHash(mediaId);
  const start = pyFloat(metadata.start), end = pyFloat(metadata.end);
  if (start !== null && end !== null) keys.push(`seg:${source}@${pyRound(start * 1000)}-${pyRound(end * 1000)}`);
  if (metadata.label_from === "dictionary" || metadata.label_from === "folder") keys.push(`src:${source}`);
  return keys;
}

const PROVENANCE = new Map();

// {segments, sources} of the bundle (data/index.json), fetched once.
export function loadProvenance(base = new URL("../data/", import.meta.url).href) {
  if (!PROVENANCE.has(base)) {
    PROVENANCE.set(base, fetch(`${base}index.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`index.json: HTTP ${response.status}`);
        return response.json();
      })
      .then((index) => ({ segments: index.segments ?? [], sources: index.sources ?? [] }))
      .catch((error) => {
        PROVENANCE.delete(base);
        throw error;
      }));
  }
  return PROVENANCE.get(base);
}

// ------------------------------------------------------------------ videos shared to the app (Android share target)

// The videos sw.js received from the share sheet (the app was opened at ./?shared=1) as File objects, in the order
// shared; the cache is emptied unless keep. [] without the Cache API or when nothing was shared.
export async function takeShared({ keep = false } = {}) {
  if (typeof caches === "undefined") return [];
  const cache = await caches.open(SHARED_CACHE);
  const requests = (await cache.keys()).sort((a, b) => (a.url < b.url ? -1 : a.url > b.url ? 1 : 0));
  const files = [];
  for (const request of requests) {
    const response = await cache.match(request);
    if (!response) continue;
    const blob = await response.blob();
    let name = "video.mp4";
    try {
      name = decodeURIComponent(response.headers.get("X-VSL-Name") || "") || name;
    } catch { /* a broken name: the default */ }
    const type = blob.type || response.headers.get("Content-Type") || "";
    const lastModified = Number(response.headers.get("X-VSL-Modified")) || Date.now();
    files.push(new File([blob], name, { type, lastModified }));
  }
  if (!keep) for (const request of requests) await cache.delete(request);
  return files;
}
