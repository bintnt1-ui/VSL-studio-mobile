// Port of segmentation.py (pure logic): hand activity of a video scan -> spans of signing; captions read by OCR in the
// text region -> cleaned sign names, text lines, ranked candidates, caption intervals; both combined into proposals the
// user reviews. Also the caption helpers of collectors.py (match_ocr_label, vote_captions, region_signature, _differs,
// CONFIDENT_READING) and difflib.SequenceMatcher(None, a, b).ratio() / difflib.get_close_matches.
//
// Python details kept: str.split()/strip() and re's \s use Python's whitespace set (not JS \s); \d is any Unicode
// decimal digit; [^\W\d_] is a letter or a non-decimal number; re.IGNORECASE equivalents are spelled out (i also
// matches İ and ı, s matches ſ); isupper / isdigit / casefold follow Python's Unicode tables; lengths, str[0] and
// string order count code points; dicts are Maps (insertion order, first of equal maxima); sorts are stable; np.mean
// sums pairwise like numpy; cv2.cvtColor(BGR2GRAY) and cv2.resize(INTER_AREA) are reproduced bit for bit.

import { KEYPOINT_SIZE, LEFT_HAND, POSE, RIGHT_HAND } from "./layout.js";
import { canonicalLabel, words } from "./naming.js";

export const SCAN_FPS = 10.0; // Holistic samples per second of video during a scan
export const OCR_FPS = 2.0; // caption checks per second (OCR only runs when the region changed)
export const REGION_CHANGE = 6.0; // mean grey-level difference that counts as a new caption
export const REGION_STILL = 4.0; // ... and below this between two checks the region has settled
export const MAX_OCR_GAP_SECONDS = 1.5; // a region that never settles (hands, printed clothes) is still read this often
export const RAISE_MIN = 0.35; // wrist height between hip (0) and shoulder (1) that counts as signing
export const SPEED_MIN = 0.04; // hand speed (shoulder widths per scan step) that counts as signing
export const GAP_FILL_SECONDS = 0.35; // rests shorter than this do not split a sign
export const MIN_SIGN_SECONDS = 0.40; // shorter bursts of activity are noise
export const MAX_SIGN_SECONDS = 6.0; // longer proposals are flagged (continuous signing)
export const CAPTION_MARGIN_SECONDS = 1.5; // a sign may start before / end after its caption (never into a neighbour's)
export const CAPTION_SIMILARITY = 0.84; // OCR variants of the same caption ("Xin lỗi" / "Xin lỗ") are merged
export const NEIGHBOUR_SIMILARITY = 0.6; // consecutive readings of one caption differ more (stylised text)
export const WEAK_CAPTION_CONFIDENCE = 0.3; // proposals below this OCR confidence are not pre-selected
export const SHORT_CAPTION_SECONDS = 0.8; // captions shown more briefly are likely misreads
export const MIN_SEGMENT_FRAMES = 10; // same floor as the manual timeline
export const CONFIDENT_READING = 0.7; // collectors.py: an OCR reading this sure needs no second preprocessing pass

// One proposal: frames [start, end] (inclusive), label, text, confidence, reason, flags.
export class Proposal {
  constructor(start, end, label = "", text = "", confidence = 0.0, reason = "", flags = []) {
    Object.assign(this, { start, end, label, text, confidence, reason, flags });
  }
}

// ------------------------------------------------------------------ Python string semantics

// Python's whitespace (str.isspace(), re \s on str patterns): JS \s lacks \x1c-\x1f and \x85, and adds \ufeff.
const SPACE_CLASS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const S = `[${SPACE_CLASS}]`;
const SPACE_RUN = new RegExp(`${S}+`, "gu");
const SPACE_EDGES = new RegExp(`^${S}+|${S}+$`, "gu");

// str.split() without arguments: runs of Python whitespace, no empty parts.
export function pySplit(text) {
  return text.split(SPACE_RUN).filter((part) => part !== "");
}

// str.strip() without arguments.
export function pyStrip(text) {
  return text.replace(SPACE_EDGES, "");
}

// str.strip(chars): the characters (code points) of `chars` removed from both ends.
function stripChars(text, chars) {
  const points = Array.from(text);
  let start = 0, end = points.length;
  while (start < end && chars.has(points[start])) start++;
  while (end > start && chars.has(points[end - 1])) end--;
  return start === 0 && end === points.length ? text : points.slice(start, end).join("");
}

function codePointLength(text) {
  let count = 0;
  for (const _ of text) count++; // eslint-disable-line no-unused-vars
  return count;
}

// Python string order: by code point (JS < compares UTF-16 units, which differs above U+FFFF).
export function comparePy(a, b) {
  if (a === b) return 0;
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    const x = a.codePointAt(i), y = b.codePointAt(j);
    if (x !== y) return x < y ? -1 : 1;
    i += x > 0xffff ? 2 : 1;
    j += y > 0xffff ? 2 : 1;
  }
  return i < a.length ? 1 : j < b.length ? -1 : 0;
}

// Python tuple order for tuples of numbers and strings: the first unequal element decides.
function compareTuples(p, q) {
  const n = Math.min(p.length, q.length);
  for (let i = 0; i < n; i++) {
    const x = p[i], y = q[i];
    if (typeof x === "string" && typeof y === "string") {
      const order = comparePy(x, y);
      if (order) return order;
    } else if (x !== y) {
      return x < y ? -1 : 1;
    }
  }
  return p.length - q.length;
}

function compareNumbers(x, y) {
  return x < y ? -1 : x > y ? 1 : 0;
}

// str(value) for the values OCR results hold (Python's spelling of None / True / False).
function pyStr(value) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  return String(value);
}

const FLOAT_TEXT = /^[+-]?(?:(?:\d(?:_?\d)*)?\.\d(?:_?\d)*|\d(?:_?\d)*\.?)(?:[eE][+-]?\d(?:_?\d)*)?$/;
const FLOAT_SPECIAL = /^([+-]?)(inf|infinity|nan)$/i;

// float(value): a number, or null where Python raises TypeError / ValueError (ASCII digits only in text).
function pyFloat(value) {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1.0 : 0.0;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "string") {
    const text = pyStrip(value);
    if (FLOAT_TEXT.test(text)) return Number(text.replace(/_/g, ""));
    const special = FLOAT_SPECIAL.exec(text);
    if (special) return special[2].toLowerCase() === "nan" ? NaN : special[1] === "-" ? -Infinity : Infinity;
  }
  return null;
}

// str.isupper() of one character: Unicode's Uppercase property (Lu + Other_Uppercase).
const UPPER = /^\p{Uppercase}$/u;
export function isUpperChar(char) {
  return UPPER.test(char);
}

// re's [^\W\d_] on str patterns (alphanumeric, not a decimal digit, not "_"): letters and non-decimal numbers.
const WORD_NOT_DIGIT = "\\p{L}\\p{Nl}\\p{No}";

// str.isdigit(): decimal digits (\p{Nd}) and the other digits of Python's Unicode table (superscripts, circled ...).
const DIGIT_EXTRA = [[0xb2, 0xb3], [0xb9, 0xb9], [0x1369, 0x1371], [0x19da, 0x19da], [0x2070, 0x2070], [0x2074, 0x2079],
  [0x2080, 0x2089], [0x2460, 0x2468], [0x2474, 0x247c], [0x2488, 0x2490], [0x24ea, 0x24ea], [0x24f5, 0x24fd],
  [0x24ff, 0x24ff], [0x2776, 0x277e], [0x2780, 0x2788], [0x278a, 0x2792], [0x10a40, 0x10a43], [0x10e60, 0x10e68],
  [0x11052, 0x1105a], [0x1f100, 0x1f10a]];
const DECIMAL = /^\p{Nd}$/u;
export function isDigitChar(char) {
  if (DECIMAL.test(char)) return true;
  const code = char.codePointAt(0);
  return DIGIT_EXTRA.some(([first, last]) => code >= first && code <= last);
}

function isDigits(text) {
  if (!text) return false;
  for (const char of text) if (!isDigitChar(char)) return false;
  return true;
}

// str.casefold(): per-character lower() except for these (generated from Python's unicodedata 13.0: every assigned
// character whose casefold() differs from its lower()).
const CASEFOLD = new Map();
for (const [first, last, delta] of [[0xb5, 0xb5, 775], [0x17f, 0x17f, -268], [0x345, 0x345, 116], [0x3c2, 0x3c2, 1],
  [0x3d0, 0x3d0, -30], [0x3d1, 0x3d1, -25], [0x3d5, 0x3d5, -15], [0x3d6, 0x3d6, -22], [0x3f0, 0x3f0, -54],
  [0x3f1, 0x3f1, -48], [0x3f5, 0x3f5, -64], [0x13a0, 0x13f5, 0], [0x13f8, 0x13fd, -8], [0x1c80, 0x1c80, -6222],
  [0x1c81, 0x1c81, -6221], [0x1c82, 0x1c82, -6212], [0x1c83, 0x1c84, -6210], [0x1c85, 0x1c85, -6211],
  [0x1c86, 0x1c86, -6204], [0x1c87, 0x1c87, -6180], [0x1c88, 0x1c88, 35267], [0x1e9b, 0x1e9b, -58],
  [0x1fbe, 0x1fbe, -7173], [0xab70, 0xabbf, -38864]]) {
  for (let code = first; code <= last; code++) CASEFOLD.set(code, String.fromCodePoint(code + delta));
}
for (const [code, folded] of [
  [0xdf, "ss"], [0x149, "ʼn"], [0x1f0, "j\u030c"], [0x390, "ι\u0308\u0301"],
  [0x3b0, "υ\u0308\u0301"], [0x587, "եւ"], [0x1e96, "h\u0331"], [0x1e97, "t\u0308"],
  [0x1e98, "w\u030a"], [0x1e99, "y\u030a"], [0x1e9a, "aʾ"], [0x1e9e, "ss"], [0x1f50, "υ\u0313"],
  [0x1f52, "υ\u0313\u0300"], [0x1f54, "υ\u0313\u0301"], [0x1f56, "υ\u0313\u0342"],
  [0x1f80, "ἀι"], [0x1f81, "ἁι"], [0x1f82, "ἂι"], [0x1f83, "ἃι"],
  [0x1f84, "ἄι"], [0x1f85, "ἅι"], [0x1f86, "ἆι"], [0x1f87, "ἇι"],
  [0x1f88, "ἀι"], [0x1f89, "ἁι"], [0x1f8a, "ἂι"], [0x1f8b, "ἃι"],
  [0x1f8c, "ἄι"], [0x1f8d, "ἅι"], [0x1f8e, "ἆι"], [0x1f8f, "ἇι"],
  [0x1f90, "ἠι"], [0x1f91, "ἡι"], [0x1f92, "ἢι"], [0x1f93, "ἣι"],
  [0x1f94, "ἤι"], [0x1f95, "ἥι"], [0x1f96, "ἦι"], [0x1f97, "ἧι"],
  [0x1f98, "ἠι"], [0x1f99, "ἡι"], [0x1f9a, "ἢι"], [0x1f9b, "ἣι"],
  [0x1f9c, "ἤι"], [0x1f9d, "ἥι"], [0x1f9e, "ἦι"], [0x1f9f, "ἧι"],
  [0x1fa0, "ὠι"], [0x1fa1, "ὡι"], [0x1fa2, "ὢι"], [0x1fa3, "ὣι"],
  [0x1fa4, "ὤι"], [0x1fa5, "ὥι"], [0x1fa6, "ὦι"], [0x1fa7, "ὧι"],
  [0x1fa8, "ὠι"], [0x1fa9, "ὡι"], [0x1faa, "ὢι"], [0x1fab, "ὣι"],
  [0x1fac, "ὤι"], [0x1fad, "ὥι"], [0x1fae, "ὦι"], [0x1faf, "ὧι"],
  [0x1fb2, "ὰι"], [0x1fb3, "αι"], [0x1fb4, "άι"], [0x1fb6, "α\u0342"],
  [0x1fb7, "α\u0342ι"], [0x1fbc, "αι"], [0x1fc2, "ὴι"], [0x1fc3, "ηι"],
  [0x1fc4, "ήι"], [0x1fc6, "η\u0342"], [0x1fc7, "η\u0342ι"], [0x1fcc, "ηι"],
  [0x1fd2, "ι\u0308\u0300"], [0x1fd3, "ι\u0308\u0301"], [0x1fd6, "ι\u0342"],
  [0x1fd7, "ι\u0308\u0342"], [0x1fe2, "υ\u0308\u0300"], [0x1fe3, "υ\u0308\u0301"],
  [0x1fe4, "ρ\u0313"], [0x1fe6, "υ\u0342"], [0x1fe7, "υ\u0308\u0342"], [0x1ff2, "ὼι"],
  [0x1ff3, "ωι"], [0x1ff4, "ώι"], [0x1ff6, "ω\u0342"], [0x1ff7, "ω\u0342ι"],
  [0x1ffc, "ωι"], [0xfb00, "ff"], [0xfb01, "fi"], [0xfb02, "fl"], [0xfb03, "ffi"], [0xfb04, "ffl"],
  [0xfb05, "st"], [0xfb06, "st"], [0xfb13, "մն"], [0xfb14, "մե"], [0xfb15, "մի"],
  [0xfb16, "վն"], [0xfb17, "մխ"]]) {
  CASEFOLD.set(code, folded);
}

const ASCII = /^[\x00-\x7f]*$/;

export function casefold(text) {
  if (ASCII.test(text)) return text.toLowerCase();
  let out = "";
  for (const char of text) out += CASEFOLD.get(char.codePointAt(0)) ?? char.toLowerCase();
  return out;
}

function stripMarkup(text) {
  return pyStr(text).replace(/<[^>]+>|\[[^\]]*\]|\([^)]*\)/gu, " ");
}

// unicodedata.category(char) == "Mn" in Python's Unicode 13: U+1734 and U+1171E became Mc in Unicode 14.
const MARK_CLASS = "\\p{Mn}\\u1734\\u{1171e}";
const MARKS = new RegExp(`[${MARK_CLASS}]`, "gu");

// dataset.slugify: ASCII, lowercase, underscore-separated form; only for tolerant text matching.
const UNDERSCORE = new Set(["_"]);
export function slugify(text) {
  let value = stripMarkup(text).replaceAll("đ", "d").replaceAll("Đ", "D");
  value = value.normalize("NFD").replace(MARKS, "");
  return stripChars(value.replace(/[^A-Za-z0-9]+/gu, "_"), UNDERSCORE).toLowerCase().slice(0, 80);
}

// numpy's pairwise summation (np.add.reduce on a contiguous float64 array).
function pairwiseSum(values, start, n) {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += values[start + i];
    return res;
  }
  if (n <= 128) {
    const r = [];
    for (let j = 0; j < 8; j++) r.push(values[start + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += values[start + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += values[start + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(values, start, n2) + pairwiseSum(values, start + n2, n - n2);
}

// float(np.mean(values)) of a list of Python floats.
function mean(values) {
  return pairwiseSum(values, 0, values.length) / values.length;
}

// The same in float32 (np.mean of a float32 array: pairwise per 8192-element buffer, float32 accumulator).
const f = Math.fround;
function pairwiseSum32(values, start, n) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f(res + values[start + i]);
    return res;
  }
  if (n <= 128) {
    const r = [];
    for (let j = 0; j < 8; j++) r.push(values[start + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = f(r[j] + values[start + i + j]);
    let res = f(f(f(r[0] + r[1]) + f(r[2] + r[3])) + f(f(r[4] + r[5]) + f(r[6] + r[7])));
    for (; i < n; i++) res = f(res + values[start + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f(pairwiseSum32(values, start, n2) + pairwiseSum32(values, start + n2, n - n2));
}

function mean32(values) {
  let total = 0;
  for (let start = 0; start < values.length; start += 8192) {
    total = f(total + pairwiseSum32(values, start, Math.min(8192, values.length - start)));
  }
  return f(total / values.length);
}

// ------------------------------------------------------------------ difflib

function sequence(value) {
  return typeof value === "string" ? Array.from(value, (char) => char.codePointAt(0)) : Array.from(value);
}

// difflib.SequenceMatcher(None, a, b, autojunk): strings compare by code point, arrays by item (===). Only what
// ratio(), quick_ratio(), real_quick_ratio() and get_close_matches need.
export class SequenceMatcher {
  constructor(a = "", b = "", autojunk = true) {
    this.autojunk = autojunk;
    this.setSeq2(b);
    this.setSeq1(a);
  }

  setSeq1(a) {
    this.a = sequence(a);
    this.matches = null;
  }

  setSeq2(b) {
    this.b = sequence(b);
    this.matches = null;
    this.fullbcount = null;
    // __chain_b: indices of every element of b, without the popular ones (autojunk, len(b) >= 200)
    const b2j = new Map();
    this.b.forEach((element, index) => {
      const indices = b2j.get(element);
      if (indices) indices.push(index);
      else b2j.set(element, [index]);
    });
    const n = this.b.length;
    if (this.autojunk && n >= 200) {
      const ntest = Math.floor(n / 100) + 1;
      const popular = [...b2j].filter(([, indices]) => indices.length > ntest).map(([element]) => element);
      for (const element of popular) b2j.delete(element);
    }
    this.b2j = b2j;
  }

  findLongestMatch(alo, ahi, blo, bhi) {
    const { a, b, b2j } = this;
    let besti = alo, bestj = blo, bestsize = 0;
    let j2len = new Map();
    for (let i = alo; i < ahi; i++) {
      const newj2len = new Map();
      const indices = b2j.get(a[i]);
      if (indices) {
        for (const j of indices) {
          if (j < blo) continue;
          if (j >= bhi) break;
          const k = (j2len.get(j - 1) ?? 0) + 1;
          newj2len.set(j, k);
          if (k > bestsize) {
            besti = i - k + 1;
            bestj = j - k + 1;
            bestsize = k;
          }
        }
      }
      j2len = newj2len;
    }
    // extend by equal (popular) elements on both ends; there is no junk without isjunk
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) {
      besti--;
      bestj--;
      bestsize++;
    }
    while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) bestsize++;
    return [besti, bestj, bestsize];
  }

  // sum of the sizes of get_matching_blocks()
  matchCount() {
    if (this.matches !== null) return this.matches;
    const queue = [[0, this.a.length, 0, this.b.length]];
    let matches = 0;
    while (queue.length) {
      const [alo, ahi, blo, bhi] = queue.pop();
      const [i, j, k] = this.findLongestMatch(alo, ahi, blo, bhi);
      if (k) {
        matches += k;
        if (alo < i && blo < j) queue.push([alo, i, blo, j]);
        if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
      }
    }
    this.matches = matches;
    return matches;
  }

  ratio() {
    return calculateRatio(this.matchCount(), this.a.length + this.b.length);
  }

  quickRatio() {
    if (this.fullbcount === null) {
      this.fullbcount = new Map();
      for (const element of this.b) this.fullbcount.set(element, (this.fullbcount.get(element) ?? 0) + 1);
    }
    const avail = new Map();
    let matches = 0;
    for (const element of this.a) {
      const numb = avail.has(element) ? avail.get(element) : this.fullbcount.get(element) ?? 0;
      avail.set(element, numb - 1);
      if (numb > 0) matches++;
    }
    return calculateRatio(matches, this.a.length + this.b.length);
  }

  realQuickRatio() {
    const la = this.a.length, lb = this.b.length;
    return calculateRatio(Math.min(la, lb), la + lb);
  }
}

function calculateRatio(matches, length) {
  return length ? 2.0 * matches / length : 1.0;
}

// difflib.SequenceMatcher(None, a, b).ratio()
export function sequenceRatio(a, b, autojunk = true) {
  return new SequenceMatcher(a, b, autojunk).ratio();
}

// difflib.get_close_matches: the best `n` possibilities scoring at least `cutoff`, best first (ties: the larger string).
export function getCloseMatches(word, possibilities, n = 3, cutoff = 0.6) {
  if (!(n > 0)) throw new RangeError(`n must be > 0: ${n}`);
  if (!(cutoff >= 0.0 && cutoff <= 1.0)) throw new RangeError(`cutoff must be in [0.0, 1.0]: ${cutoff}`);
  const result = [];
  const matcher = new SequenceMatcher("", word);
  for (const x of possibilities) {
    matcher.setSeq1(x);
    if (matcher.realQuickRatio() >= cutoff && matcher.quickRatio() >= cutoff && matcher.ratio() >= cutoff) {
      result.push([matcher.ratio(), x]);
    }
  }
  // heapq.nlargest(n, result): largest (score, x) tuples first
  result.sort((p, q) => compareTuples(q, p));
  return result.slice(0, n).map(([, x]) => x);
}

// ------------------------------------------------------------------ caption text

// re.compile(r"^\s*(?:(?:bài|từ|số|no\.?)\s*)?(?:\d{1,3}|[ivxlc]{1,4})\s*[.)\]:/\-–—]\s*", re.IGNORECASE)
const NUMBERING = new RegExp(`^${S}*(?:(?:[bB][\\u00e0\\u00c0][iI\\u0130\\u0131]|[tT][\\u1eeb\\u1eea]|[sS\\u017f][\\u1ed1\\u1ed0]`
  + `|[nN][oO]\\.?)${S}*)?(?:\\p{Nd}{1,3}|[iIvVxXlLcC\\u0130\\u0131]{1,4})${S}*[.)\\]:/\\-\\u2013\\u2014]${S}*`, "u");
// re.compile(r"^\d{1,2}\s*(?=[^\W\d_]{2})"): "5 Hạnh phúc", "5Manh": numbering whose dot OCR lost
const BARE_NUMBER = new RegExp(`^\\p{Nd}{1,2}${S}*(?=[${WORD_NOT_DIGIT}]{2})`, "u");
// re.split(r"\s+/\s+|\s*\|\s*", line): bilingual captions "Xin chào / Hello"
const PART_SPLIT = new RegExp(`${S}+/${S}+|${S}*\\|${S}*`, "u");
const EDGE_PUNCTUATION = new Set(Array.from(" \t\"'“”‘’«».,;:!¡¿·•-–—_|/\\*#~^=+()[]{}<>"));
const VIETNAMESE = /(?![\x00-\x7f])\p{L}/u;
const LETTER = /\p{L}/u;

// The single-character classes the patterns above are built from (checked against Python's tables by the tests).
export const CHAR_CLASSES = {
  space: new RegExp(`^${S}$`, "u"), // \s, str.isspace()
  letter: /^\p{L}$/u, // str.isalpha()
  decimal: DECIMAL, // \d, str.isdecimal()
  wordNotDigit: new RegExp(`^[${WORD_NOT_DIGIT}]$`, "u"), // [^\W\d_]
  mark: new RegExp(`^[${MARK_CLASS}]$`, "u"), // unicodedata.category(char) == "Mn" (slugify)
};

// Caption text as a sign name: numbering ("1. Xin lỗi") and edge punctuation removed, NFC.
export function cleanCaption(text) {
  text = pyStr(text).normalize("NFC").replaceAll("\u00a0", " ");
  text = pyStrip(text.replace(SPACE_RUN, " "));
  let previous = null;
  while (previous !== text) {
    previous = text;
    text = stripChars(text.replace(NUMBERING, ""), EDGE_PUNCTUATION);
  }
  return text;
}

// True when the text contains letters that only Vietnamese uses (diacritics, đ).
export function hasVietnamese(text) {
  return VIETNAMESE.test(text);
}

// Captions as sign names in lower case ('Xin chào', 'XIN CHÀO', OCR's 'cỦa tôi' -> lower case), except single letters
// ('A') and names whose every word is capitalised ('Đà Lạt', 'Hà Nội').
export function sentenceCase(text) {
  let letters = 0;
  for (const char of text) if (LETTER.test(char)) letters++;
  if (letters <= 1) return text;
  const parts = pySplit(text).filter((word) => LETTER.test(word));
  const titled = parts.length >= 2 && parts.every((word) => {
    const chars = Array.from(word);
    return isUpperChar(chars[0]) && !chars.slice(1).some(isUpperChar);
  });
  return titled ? text : text.toLowerCase();
}

// Shape of a rectangular nest of arrays (numpy's view of a nested list), or null when ragged.
function nestedShape(value) {
  if (!(Array.isArray(value) || ArrayBuffer.isView(value))) return [];
  if (!value.length) return [0];
  let inner = null;
  for (const item of value) {
    const shape = nestedShape(item);
    if (shape === null) return null;
    if (inner === null) inner = shape;
    else if (inner.length !== shape.length || inner.some((size, i) => size !== shape[i])) return null;
  }
  return [value.length, ...inner];
}

function flatten(value, out) {
  if (Array.isArray(value) || ArrayBuffer.isView(value)) for (const item of value) flatten(item, out);
  else out.push(value);
  return out;
}

// np.asarray(box, dtype=np.float32).reshape(-1, 2) -> (top, bottom, left), or null where Python falls back.
function boxEdges(box) {
  if (nestedShape(box) === null) return null;
  const values = flatten(box, []);
  if (!values.length || values.length % 2) return null;
  let top = Infinity, bottom = -Infinity, left = Infinity;
  for (let i = 0; i < values.length; i += 2) {
    const xv = values[i] === null || values[i] === undefined ? NaN : pyFloat(values[i]);
    const yv = values[i + 1] === null || values[i + 1] === undefined ? NaN : pyFloat(values[i + 1]);
    if (xv === null || yv === null) return null;
    const x = f(xv), y = f(yv);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (y < top) top = y;
    if (y > bottom) bottom = y;
    if (x < left) left = x;
  }
  return [top, bottom, left];
}

function isSized(item) {
  return item !== null && item !== undefined && typeof item.length === "number";
}

// EasyOCR detections [[box (4 [x, y] points), text, confidence], ...] grouped into text lines, top to bottom:
// [[text, confidence], ...]. Detections without a usable box keep their order as separate lines.
export function captionLines(detections) {
  const boxes = [];
  let order = -1;
  for (const item of detections) {
    order++;
    if (!isSized(item) || item.length < 3) continue;
    const confidence = pyFloat(item[2]);
    if (confidence === null) continue;
    const text = pyStrip(pyStr(item[1]));
    if (!text) continue;
    const [top, bottom, left] = boxEdges(item[0]) ?? [1e6 + order * 10.0, 1e6 + order * 10.0 + 1.0, 0.0];
    boxes.push([top, bottom, left, text, confidence]);
  }
  boxes.sort((p, q) => compareNumbers((p[0] + p[1]) / 2, (q[0] + q[1]) / 2));
  const lines = [];
  for (const box of boxes) {
    const centre = (box[0] + box[1]) / 2;
    const last = lines[lines.length - 1];
    if (last) {
      let top = last[0][0], bottom = last[0][1];
      for (const item of last) {
        if (item[0] < top) top = item[0];
        if (item[1] > bottom) bottom = item[1];
      }
      if (top <= centre && centre <= bottom) {
        last.push(box);
        continue;
      }
    }
    lines.push([box]);
  }
  return lines.map((line) => {
    line.sort((p, q) => compareNumbers(p[2], q[2]));
    return [line.map((box) => box[3]).join(" "), mean(line.map((box) => box[4]))];
  });
}

// Possible sign names in an OCR result, best first: [[name, confidence], ...]. Lines matching the vocabulary come
// first, then Vietnamese lines (diacritics), then the rest, so a bilingual caption "Xin chào / Hello" yields "xin chào".
export function captionCandidates(detections, vocabulary = []) {
  const known = new Map();
  for (const entry of vocabulary) {
    let key;
    try {
      key = casefold(canonicalLabel(entry));
    } catch {
      continue;
    }
    if (!known.has(key)) known.set(key, entry);
  }
  // Multi-word names tolerate OCR slips ("hanh phúl" -> "hạnh phúc"), as matchOcrLabel does.
  const phrases = new Map();
  for (const entry of vocabulary) if (words(entry).length > 1) phrases.set(slugify(entry), entry);
  const phraseKeys = [...phrases.keys()];
  const ranked = [];
  const seen = new Set();
  captionLines(detections).forEach(([line, confidence], order) => {
    let parts = line.split(PART_SPLIT).filter((part) => pyStrip(part) !== "");
    if (!parts.length) parts = [line];
    for (const part of parts) {
      const cleaned = cleanCaption(part);
      const bare = pyStrip(cleaned.replace(BARE_NUMBER, ""));
      [...new Set([bare, cleaned].filter((item) => item))].forEach((name, variant) => {
        const nameWords = words(name);
        if (!nameWords.length) return;
        let key;
        try {
          key = casefold(canonicalLabel(name));
        } catch {
          return;
        }
        if (seen.has(key)) return;
        seen.add(key);
        const close = phraseKeys.length && nameWords.length > 1 ? getCloseMatches(slugify(name), phraseKeys, 1, 0.82) : [];
        let rank, display;
        if (known.has(key)) [rank, display] = [0, known.get(key)];
        else if (close.length) [rank, display] = [0, phrases.get(close[0])];
        else [rank, display] = [hasVietnamese(name) ? 1 : 2, sentenceCase(name)];
        ranked.push([rank + variant * 3, order, -confidence, display, confidence]);
      });
    }
  });
  ranked.sort(compareTuples);
  return ranked.map(([, , , display, confidence]) => [display, confidence]);
}

export function sameCaption(first, second, threshold = CAPTION_SIMILARITY) {
  if (!first || !second) return first === second;
  const a = slugify(first), b = slugify(second);
  if (a === b) return true;
  return sequenceRatio(a, b) >= threshold;
}

// ------------------------------------------------------------------ intervals

// [start, end] index pairs (inclusive) of consecutive true values.
export function runs(mask) {
  const out = [];
  let start = -1;
  for (let i = 0; i <= mask.length; i++) {
    const on = i < mask.length && Boolean(mask[i]);
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      out.push([start, i - 1]);
      start = -1;
    }
  }
  return out;
}

export function closeGaps(intervals, maxGap) {
  const merged = [];
  for (const [start, end] of [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    const last = merged[merged.length - 1];
    if (last && start - last[1] - 1 <= maxGap) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  return merged;
}

// ------------------------------------------------------------------ hand activity

// Per-row activity signals from Holistic keypoints (rows: frames of KEYPOINT_SIZE values, scan order).
// Computed in float32 like numpy on float32 rows.
export function activity(rows) {
  const count = rows.length;
  const hands = new Uint8Array(count), raised = new Uint8Array(count), moving = new Uint8Array(count);
  const active = new Uint8Array(count), height = new Float64Array(count), speed = new Float64Array(count);
  const value = (row, index) => {
    const v = row[index];
    return Number.isFinite(v) ? f(v) : (Number.isNaN(v) ? 0 : f(v > 0 ? 3.4028234663852886e38 : -3.4028234663852886e38));
  };
  const handValid = [], centres = [], widths = [];
  for (let r = 0; r < count; r++) {
    const row = rows[r];
    const valid = [false, false];
    const centre = [[0, 0], [0, 0]];
    [LEFT_HAND, RIGHT_HAND].forEach(([start], side) => {
      let any = false;
      for (let i = start; i < start + 63; i++) if (Math.abs(value(row, i)) > 1e-6) { any = true; break; }
      valid[side] = any;
      for (let axis = 0; axis < 2; axis++) {
        let sum = 0;
        for (let p = 0; p < 21; p++) sum += value(row, start + p * 3 + axis);
        centre[side][axis] = f(sum / 21);
      }
    });
    handValid.push(valid);
    centres.push(centre);
    const pose = (point, axis) => value(row, POSE[0] + point * 4 + axis);
    let poseValid = false;
    for (const point of [11, 12, 23, 24]) {
      if (Math.abs(pose(point, 0)) > 1e-6 || Math.abs(pose(point, 1)) > 1e-6) poseValid = true;
    }
    const shoulderY = f(f(pose(11, 1) + pose(12, 1)) / 2);
    const hipY = f(f(pose(23, 1) + pose(24, 1)) / 2);
    const torso = Math.max(f(hipY - shoulderY), f(1e-3));
    widths.push(Math.max(f(Math.abs(f(pose(11, 0) - pose(12, 0)))), f(0.05)));
    let best = -Infinity;
    [LEFT_HAND, RIGHT_HAND].forEach(([start], side) => {
      if (!valid[side] || !poseValid) return;
      const wrist = value(row, start + 1);
      const h = f(f(hipY - wrist) / torso);
      if (h > best) best = h;
    });
    const bestHeight = Number.isFinite(best) ? best : NaN;
    hands[r] = valid[0] || valid[1] ? 1 : 0;
    raised[r] = poseValid ? ((Number.isNaN(bestHeight) ? -1.0 : bestHeight) >= RAISE_MIN ? 1 : 0) : hands[r];
    height[r] = Number.isNaN(bestHeight) ? 0.0 : bestHeight;
  }
  for (let r = 1; r < count; r++) {
    let step = 0;
    for (let side = 0; side < 2; side++) {
      if (!(handValid[r][side] && handValid[r - 1][side])) continue;
      const dx = f(centres[r][side][0] - centres[r - 1][side][0]);
      const dy = f(centres[r][side][1] - centres[r - 1][side][1]);
      const norm = f(Math.sqrt(f(f(dx * dx) + f(dy * dy))));
      if (norm > step) step = norm;
    }
    speed[r] = f(step / widths[r]);
  }
  for (let r = 0; r < count; r++) {
    moving[r] = speed[r] >= SPEED_MIN ? 1 : 0;
    active[r] = hands[r] && (raised[r] || moving[r]) ? 1 : 0;
  }
  return { hands, raised, moving, active, height, speed };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// Python round(): half to even.
function roundHalfEven(value) {
  const floor = Math.floor(value), diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

// Activity samples (taken at the given frame indices) -> frame intervals of signing.
export function activeSpans(frames, active, fps, frameCount) {
  if (!frames.length) return [];
  const diffs = [];
  for (let i = 1; i < frames.length; i++) diffs.push(frames[i] - frames[i - 1]);
  const step = frames.length > 1 ? Math.max(1, Math.trunc(median(diffs))) : 1;
  let spans = runs(active).map(([first, last]) => [frames[first], Math.min(frameCount - 1, frames[last] + step - 1)]);
  spans = closeGaps(spans, roundHalfEven(GAP_FILL_SECONDS * fps));
  const minimum = Math.max(MIN_SEGMENT_FRAMES, roundHalfEven(MIN_SIGN_SECONDS * fps));
  return spans.filter(([start, end]) => end - start + 1 >= minimum);
}

// ------------------------------------------------------------------ captions

// OCR samples [[frame, name, confidence], ...] -> [[start, end, name, confidence], ...]. Equal neighbouring readings
// are merged, single failed readings ("" between two equal names) are bridged, and a caption lasts until the next
// reading.
export function captionIntervals(samples, frameCount) {
  const bridged = [...samples].sort((p, q) => compareNumbers(p[0], q[0])).map(([frame, name, confidence]) => [frame, name, confidence]);
  if (!bridged.length) return [];
  for (let index = 1; index < bridged.length - 1; index++) {
    const [frame, name] = bridged[index];
    const before = bridged[index - 1][1], after = bridged[index + 1][1];
    if (before && sameCaption(before, after) && !sameCaption(name, before)) bridged[index] = [frame, before, bridged[index - 1][2]];
  }
  const intervals = [];
  bridged.forEach(([frame, name, confidence], position) => {
    const end = position + 1 < bridged.length ? bridged[position + 1][0] - 1 : frameCount - 1;
    const current = intervals[intervals.length - 1];
    if (current && sameCaption(current[2], name, NEIGHBOUR_SIMILARITY)) {
      current[1] = Math.max(current[1], end);
      current[4].push([name, confidence]);
    } else {
      intervals.push([frame, Math.max(frame, end), name, confidence, [[name, confidence]]]);
    }
  });
  const result = [];
  for (const [start, end, , , readings] of intervals) {
    const votes = new Map();
    for (const [name, confidence] of readings) {
      if (!name) continue;
      if (votes.has(name)) votes.get(name).push(confidence);
      else votes.set(name, [confidence]);
    }
    if (!votes.size) continue;
    // max(votes, key=(count, mean)): the first of equal keys
    let best = null, bestCount = 0, bestMean = 0;
    for (const [name, confidences] of votes) {
      const count = confidences.length, average = mean(confidences);
      if (best === null || count > bestCount || (count === bestCount && average > bestMean)) {
        [best, bestCount, bestMean] = [name, count, average];
      }
    }
    result.push([Math.trunc(start), Math.trunc(end), best, bestMean]);
  }
  return result;
}

// ------------------------------------------------------------------ proposals

// Combine activity spans and caption intervals ([[start, end, name, confidence], ...] or null) into reviewable
// proposals: one per caption and per separate repetition of its sign; without captions one unnamed proposal per burst.
export function propose(frameCount, fps, spans, captions = null) {
  fps = fps && fps > 0 ? fps : 30.0;
  const limit = roundHalfEven(MAX_SIGN_SECONDS * fps);
  const margin = roundHalfEven(CAPTION_MARGIN_SECONDS * fps);
  const proposals = [];
  if (captions && captions.length) {
    const ordered = [...captions].sort(compareTuples);
    ordered.forEach(([start, end, name, confidence], index) => {
      // A sign may begin a little before its caption appears (and end after it disappears), but never inside the
      // neighbouring caption's interval.
      const low = Math.max(0, start - margin, index > 0 ? ordered[index - 1][1] + 1 : 0);
      const high = Math.min(frameCount - 1, end + margin,
        index + 1 < ordered.length ? ordered[index + 1][0] - 1 : frameCount - 1);
      // (the window stops at the neighbouring captions, so each burst is cut between them)
      const inside = spans.filter(([first, last]) => last >= low && first <= high)
        .map(([first, last]) => [Math.max(low, first), Math.min(high, last)])
        // A burst that only touches the caption's edge belongs to the neighbouring sign.
        .filter(([first, last]) => last - first + 1 >= MIN_SEGMENT_FRAMES
          && overlap(first, last, start, end) >= Math.max(MIN_SEGMENT_FRAMES, 0.3 * (last - first + 1)));
      const weak = [];
      if (confidence < WEAK_CAPTION_CONFIDENCE) weak.push("chữ mờ");
      const compact = name.replaceAll(" ", "");
      if (codePointLength(compact) <= 1) weak.push("một ký tự");
      else if (!hasVietnamese(name) && !isDigits(compact)) weak.push("không dấu"); // printed clothes, logos, English
      if (end - start + 1 < SHORT_CAPTION_SECONDS * fps) weak.push("chữ hiện ngắn");
      for (const [first, last] of inside) {
        const flags = [...weak, ...(last - first + 1 > limit ? ["dài"] : [])];
        proposals.push(new Proposal(first, last, name, name, confidence, "chữ + chuyển động tay", flags));
      }
    });
  } else {
    for (const [first, last] of spans) {
      proposals.push(new Proposal(first, last, "", "", 0.0, "chuyển động tay", last - first + 1 > limit ? ["dài"] : []));
    }
  }
  return proposals;
}

// For one-sign clips (dictionary, labelled files): the span from the first to the last activity.
export function singleSignSpan(frames, active, fps, frameCount) {
  const spans = activeSpans(frames, active, fps, frameCount);
  return spans.length ? [spans[0][0], spans[spans.length - 1][1]] : null;
}

export function overlap(first, last, start, end) {
  return Math.max(0, Math.min(last, end) - Math.max(first, start) + 1);
}

// True when two frame intervals share at least `threshold` of the shorter one.
export function overlaps(first, last, start, end, threshold = 0.5) {
  const shorter = Math.max(1, Math.min(last - first, end - start) + 1);
  return overlap(first, last, start, end) / shorter >= threshold;
}

// ------------------------------------------------------------------ OCR helpers of collectors.py

// (exact words, ASCII words, display) per vocabulary entry, longest first.
function phrasesOf(vocabulary) {
  const phrases = [];
  for (const entry of vocabulary) {
    const exact = words(entry);
    if (exact.length) phrases.push([exact, exact.map((word) => slugify(word)), entry]);
  }
  return phrases.sort((p, q) => q[0].length - p[0].length);
}

function sameItems(tokens, index, items) {
  for (let i = 0; i < items.length; i++) if (tokens[index + i] !== items[i]) return false;
  return true;
}

// Diacritics must match, except that multi-word phrases tolerate spelling variants ("cám ơn" ~ "cảm ơn"); single words
// like có/cô/cỏ never collapse.
function contains([exactTokens, asciiTokens], exact, asciiWords) {
  for (let index = 0; index < exactTokens.length - exact.length + 1; index++) {
    if (sameItems(exactTokens, index, exact) || (exact.length > 1 && sameItems(asciiTokens, index, asciiWords))) return true;
  }
  return false;
}

function tokens(text) {
  const exact = words(text);
  return [exact, exact.map((word) => slugify(word).replaceAll("_", ""))];
}

// collectors.match_ocr_label: the best vocabulary word for EasyOCR detections -> [word or null, raw text, confidence].
export function matchOcrLabel(detections, vocabulary, minConfidence = 0.4) {
  const candidates = [];
  for (const item of detections) {
    if (!isSized(item) || item.length < 3) continue;
    const confidence = pyFloat(item[2]);
    if (confidence === null) continue;
    const text = pyStrip(pyStr(item[1]));
    if (text) candidates.push([text, confidence]);
  }
  const rawText = candidates.map(([text]) => text).join(" ");
  const surest = [...candidates].sort((p, q) => compareNumbers(-p[1], -q[1])).map(([text, confidence]) => [tokens(text), text, confidence]);
  const phrases = phrasesOf(vocabulary);
  const rawTokens = tokens(rawText);
  for (const [exact, asciiWords, display] of phrases) {
    for (const [textTokens, , confidence] of surest) {
      if (confidence >= minConfidence && contains(textTokens, exact, asciiWords)) return [display, rawText, confidence];
    }
    if (candidates.length && contains(rawTokens, exact, asciiWords)) { // text split across boxes
      let total = 0; // Python 3.10 sum(): left to right
      for (const [, score] of candidates) total += score;
      const confidence = total / candidates.length;
      if (confidence >= minConfidence) return [display, rawText, confidence];
    }
  }
  // OCR often garbles diacritics or a letter ("Xin lỗ8"); allow a close ASCII match for multi-word phrases only, the
  // user confirms the result anyway.
  const multiWord = new Map();
  for (const [exact, asciiWords, display] of phrases) if (exact.length > 1) multiWord.set(asciiWords.join("_"), display);
  const keys = [...multiWord.keys()];
  for (const [, text, confidence] of surest) {
    if (confidence < minConfidence) continue;
    const match = getCloseMatches(slugify(text), keys, 1, 0.82);
    if (match.length) return [multiWord.get(match[0]), rawText, confidence];
  }
  let best = 0.0;
  candidates.forEach(([, score], index) => {
    if (index === 0 || score > best) best = score;
  });
  return [null, rawText, best];
}

// collectors.vote_captions: combine readings [[candidates, raw], ...] of several frames: each frame's best name counts
// fully, the others a little -> [[[name, best confidence], ...], raw texts joined by " | "].
export function voteCaptions(readings) {
  const scores = new Map(), best = new Map(), raws = [];
  for (const [candidates, raw] of readings) {
    if (raw) raws.push(raw);
    candidates.forEach(([name, confidence], rank) => {
      const value = pyFloat(confidence);
      if (value === null) throw new TypeError(`float() argument: ${confidence}`);
      scores.set(name, (scores.has(name) ? scores.get(name) : 0.0) + value * (rank === 0 ? 1.0 : 0.25));
      const before = best.has(name) ? best.get(name) : 0.0;
      best.set(name, value > before ? value : before);
    });
  }
  const ranked = [...scores.keys()].sort((p, q) => compareNumbers(-scores.get(p), -scores.get(q)));
  return [ranked.map((name) => [name, best.get(name)]), [...new Set(raws)].join(" | ")];
}

// cv2.cvtColor(BGR2GRAY) of 8-bit pixels (OpenCV 4.x, 15-bit fixed point), read from RGBA ImageData-like pixels.
export function greyPixels(image, x0 = 0, y0 = 0, width = image.width, height = image.height) {
  const out = new Uint8Array(width * height);
  const data = image.data, stride = image.width * 4;
  for (let y = 0; y < height; y++) {
    let i = (y0 + y) * stride + x0 * 4, o = y * width;
    for (let x = 0; x < width; x++, i += 4) out[o + x] = (data[i + 2] * 3735 + data[i + 1] * 19235 + data[i] * 9798 + 16384) >> 15;
  }
  return out;
}

function saturateU8(value) { // saturate_cast<uchar>(float): cvRound (half to even), clamped
  const rounded = roundHalfEven(value);
  return rounded < 0 ? 0 : rounded > 255 ? 255 : rounded;
}

function saturateShort(value) {
  const rounded = roundHalfEven(value);
  return rounded < -32768 ? -32768 : rounded > 32767 ? 32767 : rounded;
}

// computeResizeAreaTab: [destination index, source index, float weight] per contribution.
function areaTable(ssize, dsize, scale) {
  const table = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) table.push([dx, sx1 - 1, f((sx1 - fsx1) / cellWidth)]);
    for (let sx = sx1; sx < sx2; sx++) table.push([dx, sx, f(1.0 / cellWidth)]);
    if (fsx2 - sx2 > 1e-3) table.push([dx, sx2, f(Math.min(Math.min(fsx2 - sx2, 1.0), cellWidth) / cellWidth)]);
  }
  return table;
}

// cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA) of one 8-bit channel (Uint8Array sw x sh), bit for bit:
// integer factors average blocks (2x2 rounds half up like OpenCV's SIMD path), other shrinks use float area weights,
// enlargements OpenCV's fixed-point bilinear "area" emulation.
export function resizeArea(src, sw, sh, dw, dh) {
  const out = new Uint8Array(dw * dh);
  if (sw === dw && sh === dh) {
    out.set(src.subarray(0, sw * sh));
    return out;
  }
  const invX = dw / sw, invY = dh / sh;
  const scaleX = 1.0 / invX, scaleY = 1.0 / invY;
  const ix = roundHalfEven(scaleX), iy = roundHalfEven(scaleY);
  const fast = Math.abs(scaleX - ix) < Number.EPSILON && Math.abs(scaleY - iy) < Number.EPSILON;
  if (scaleX >= 1 && scaleY >= 1) {
    if (fast) {
      const scale = f(1 / (ix * iy));
      const dwidth1 = Math.floor(sw / ix);
      for (let dy = 0; dy < dh; dy++) {
        const sy0 = dy * iy;
        if (sy0 >= sh) continue;
        const w = sy0 + iy <= sh ? dwidth1 : 0;
        let dx = 0;
        for (; dx < w; dx++) {
          let sum = 0;
          for (let sy = 0; sy < iy; sy++) for (let sx = 0; sx < ix; sx++) sum += src[(sy0 + sy) * sw + dx * ix + sx];
          out[dy * dw + dx] = ix === 2 && iy === 2 ? Math.min(255, (sum + 2) >> 2) : saturateU8(f(sum * scale));
        }
        for (; dx < dw; dx++) {
          const sx0 = dx * ix;
          let sum = 0, count = 0;
          for (let sy = 0; sy < iy && sy0 + sy < sh; sy++) {
            for (let sx = 0; sx < ix && sx0 + sx < sw; sx++) {
              sum += src[(sy0 + sy) * sw + sx0 + sx];
              count++;
            }
          }
          out[dy * dw + dx] = count ? saturateU8(f(f(sum) / count)) : 0;
        }
      }
      return out;
    }
    const xtab = areaTable(sw, dw, scaleX), ytab = areaTable(sh, dh, scaleY);
    const buf = new Float32Array(dw), sum = new Float32Array(dw);
    let previous = ytab[0][0];
    for (const [dy, sy, beta] of ytab) {
      buf.fill(0);
      const row = sy * sw;
      for (const [dx, sx, alpha] of xtab) buf[dx] = f(buf[dx] + f(src[row + sx] * alpha));
      if (dy !== previous) {
        for (let dx = 0; dx < dw; dx++) {
          out[previous * dw + dx] = saturateU8(sum[dx]);
          sum[dx] = f(beta * buf[dx]);
        }
        previous = dy;
      } else {
        for (let dx = 0; dx < dw; dx++) sum[dx] = f(sum[dx] + f(beta * buf[dx]));
      }
    }
    for (let dx = 0; dx < dw; dx++) out[previous * dw + dx] = saturateU8(sum[dx]);
    return out;
  }
  // enlarging on an axis: bilinear with "area" coefficients, 11-bit fixed point (resizeGeneric_ for 8U)
  const xofs = new Int32Array(dw), alpha0 = new Int32Array(dw), alpha1 = new Int32Array(dw);
  let xmax = dw;
  for (let dx = 0; dx < dw; dx++) {
    let sx = Math.floor(dx * scaleX);
    let fx = f((dx + 1) - (sx + 1) * invX);
    fx = fx <= 0 ? 0 : f(fx - Math.floor(fx));
    if (sx < 0) [fx, sx] = [0, 0];
    if (sx + 1 >= sw) {
      xmax = Math.min(xmax, dx);
      if (sx >= sw - 1) [fx, sx] = [0, sw - 1];
    }
    xofs[dx] = sx;
    alpha0[dx] = saturateShort(f(f(1 - fx) * 2048));
    alpha1[dx] = saturateShort(f(fx * 2048));
  }
  const horizontal = (sy) => {
    const row = sy * sw, values = new Int32Array(dw);
    for (let dx = 0; dx < dw; dx++) {
      const sx = xofs[dx];
      values[dx] = dx < xmax ? src[row + sx] * alpha0[dx] + src[row + sx + 1] * alpha1[dx] : src[row + sx] * 2048;
    }
    return values;
  };
  const clip = (value) => (value < 0 ? 0 : value < sh ? value : sh - 1);
  for (let dy = 0; dy < dh; dy++) {
    const sy = Math.floor(dy * scaleY);
    let fy = f((dy + 1) - (sy + 1) * invY);
    fy = fy <= 0 ? 0 : f(fy - Math.floor(fy));
    const b0 = saturateShort(f(f(1 - fy) * 2048)), b1 = saturateShort(f(fy * 2048));
    const r0 = horizontal(clip(sy)), r1 = horizontal(clip(sy + 1));
    for (let x = 0; x < dw; x++) {
      const value = (((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2) >> 2;
      out[dy * dw + x] = value < 0 ? 0 : value > 255 ? 255 : value;
    }
  }
  return out;
}

// numpy slice bounds of a[start:stop] for an axis of `length` (start >= 0 here).
function sliceBounds(start, stop, length) {
  start = Math.min(start, length);
  if (stop < 0) stop = Math.max(0, stop + length);
  stop = Math.min(stop, length);
  return [start, Math.max(start, stop)];
}

// collectors.region_signature: small grey thumbnail of a region ([x, y, w, h] in the image's pixels) to notice when the
// caption changes. image: RGBA ImageData-like {data, width, height}. -> {width, height, data: Float32Array}.
// The region is cropped like numpy slices frame[max(0, y):y + h, max(0, x):x + w] (integers only, as numpy).
export function regionSignature(image, region) {
  const values = Array.from(region);
  if (values.length !== 4) throw new TypeError(`region needs 4 values (x, y, w, h), got ${values.length}`);
  if (!values.every(Number.isInteger)) throw new TypeError("slice indices must be integers");
  const [x, y, w, h] = values;
  const [y0, y1] = sliceBounds(Math.max(0, y), y + h, image.height);
  const [x0, x1] = sliceBounds(Math.max(0, x), x + w, image.width);
  const cropWidth = x1 - x0, cropHeight = y1 - y0;
  if (!cropWidth || !cropHeight) return { width: 8, height: 8, data: new Float32Array(64) };
  const pixels = greyPixels(image, x0, y0, cropWidth, cropHeight);
  const width = 96;
  const height = Math.min(Math.max(8, roundHalfEven(cropHeight * width / Math.max(cropWidth, 1))), 96);
  return { width, height, data: Float32Array.from(resizeArea(pixels, cropWidth, cropHeight, width, height)) };
}

// collectors._differs: true when there is no earlier signature, the size changed, or the mean absolute grey-level
// difference is above `threshold`.
export function differs(signature, other, threshold) {
  if (!other || signature.width !== other.width || signature.height !== other.height) return true;
  const a = signature.data, b = other.data;
  const diff = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) diff[i] = Math.abs(f(a[i] - b[i]));
  return mean32(diff) > threshold;
}

export const ROW_SIZE = KEYPOINT_SIZE;
