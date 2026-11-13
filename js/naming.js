// Sign names (port of the label functions of dataset.py): canonical labels, display names, accent suggestions.

export const WINDOWS_RESERVED = new Set(["CON", "PRN", "AUX", "NUL",
  ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`), ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`)]);

function stripMarkup(text) {
  return String(text).replace(/<[^>]+>|\[[^\]]*\]|\([^)]*\)/g, " ");
}

// Lower-case Vietnamese words with diacritics kept (NFC).
export function words(text) {
  return stripMarkup(text).normalize("NFC").toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
}

// Folder name of a sign: lower-case words joined by "_", diacritics kept; single letters upper-case.
export function canonicalLabel(text) {
  let label = words(text).join("_").slice(0, 80).replace(/^_+|_+$/g, "");
  if (!label) throw new Error(`Tên ký hiệu không hợp lệ: ${JSON.stringify(String(text))}`);
  if ([...label].length === 1 && /\p{L}/u.test(label)) label = label.toUpperCase();
  if (WINDOWS_RESERVED.has(label.toUpperCase())) label += "_";
  return label;
}

// Lower-case text without Vietnamese diacritics ("Bạn" -> "ban", "đi" -> "di").
export function plainText(text) {
  return String(text).replace(/đ/g, "d").replace(/Đ/g, "D").normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}

function diacritics(text) {
  const marks = String(text).normalize("NFD").match(/\p{M}/gu)?.length ?? 0;
  return marks + (String(text).match(/[đĐ]/g)?.length ?? 0);
}

// `word` is `typed` with diacritics added only to syllables typed without any.
export function addsDiacritics(typed, word) {
  const typedSyllables = typed.toLowerCase().split(/\s+/).filter(Boolean);
  const wordSyllables = word.normalize("NFC").toLowerCase().split(/\s+/).filter(Boolean);
  if (typedSyllables.length !== wordSyllables.length) return false;
  let added = false;
  for (let i = 0; i < typedSyllables.length; i++) {
    const mine = typedSyllables[i], theirs = wordSyllables[i];
    if (plainText(mine) !== plainText(theirs)) return false;
    if (diacritics(mine)) {
      if (mine.normalize("NFC") !== theirs) return false;
    } else if (diacritics(theirs)) {
      added = true;
    }
  }
  return added;
}

// The word a name typed without (some of) its diacritics stands for: the one sign name in `labels` that is the
// name with diacritics added to its bare syllables, else the one such word in `dictionary`; null otherwise.
export function accentSuggestion(display, labels = [], dictionary = []) {
  const typed = String(display).normalize("NFC").trim();
  if (plainText(typed).replace(/ /g, "").length < 2) return null;
  if (dictionary.some((word) => word.normalize("NFC").toLowerCase() === typed.toLowerCase())) return null;
  for (const spellings of [labels, dictionary]) {
    const found = new Set(spellings.filter((word) => addsDiacritics(typed, word)).map((word) => word.normalize("NFC")));
    if (found.size === 1) return [...found][0];
    if (found.size) return null;
  }
  return null;
}

export function sameLabel(first, second) {
  return first.normalize("NFC").toLowerCase() === second.normalize("NFC").toLowerCase();
}
