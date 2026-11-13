// Signed words -> a written Vietnamese sentence (port of sentence.py; pure logic: no camera, no DOM).
//
// Vietnamese Sign Language (VSL) has its own grammar, which differs from written Vietnamese:
//   - the verb tends to come last (subject - object - verb: "tôi + cơm + ăn" = "tôi ăn cơm");
//   - question words are signed at the end ("bạn + tên + gì" = "bạn tên là gì?"); a yes/no question is shown by
//     the face, not by a sign;
//   - negation is signed last ("tôi + ăn cơm + không" = "tôi không ăn cơm");
//   - function words are left out: là, thì, có, của, đi, ở, đến, về, đã, đang...
// (Ngôn ngữ 2014, "Mấy vấn đề về cú pháp của ngôn ngữ kí hiệu ở Việt Nam"; HUFO lecture notes on VSL sentences.)
//
// compose(words) writes the recognised words of one sentence as Vietnamese, in these steps:
//   1. fingerspelled letters (and tone signs) become a capitalised name, one syllable per tone sign / syllable
//      boundary ("t h u h a dấu huyền" -> "Thu Hà"); a greeting that opens the sentence is its own clause;
//   2. question signs that frame the sentence ("có … không?", "muốn không?", "đói không?", "phải không?",
//      "còn bạn?", a taught "X không?"); a modal signed last moves after the subject;
//   3. a final "không" / "chưa": a yes/no question when the sentence is about the person addressed or theirs,
//      otherwise negation before the predicate ("không phải là" before a role, "không có" before a thing);
//   4. a final question word gets the words Vietnamese needs ("giá là bao nhiêu?", "nhà bạn ở đâu?",
//      "bạn bao nhiêu tuổi?", "bạn làm nghề gì?"); "vì sao" moves to the front;
//   5. numbers go before what they count; an object, a person or a place signed before its verb moves after it;
//   6. an activity that ends the predicate gets "đi", a place that is the predicate gets "đi / đến / về";
//   7. "là" between a person and a role or a name.
// Words the lexicon does not know (signs the user taught) stay in place. Lexicon look-ups use lower-case, modern
// tone placement (khỏe = khoẻ); the output keeps how a word is written (môn Toán). composeText() splits a stream
// of words into sentences at long pauses and after a final question.
//
// Python details kept: str.split()/strip()/\s use Python's whitespace set (PY_SPACE, not JS \s), \d is any
// Unicode decimal digit (\p{Nd}), lengths and str[:1] count code points, plain() uses casefold() (CASEFOLD
// table) and unicodedata.combining() (combining()), dicts are Maps (no prototype keys), sets are Sets.

// ------------------------------------------------------------------ Python string semantics
const SPACE_CLASS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const SPACE_RUN = new RegExp(`[${SPACE_CLASS}]+`, "g");          // Python re \s+ / str.isspace() runs
const SPACE_EDGES = new RegExp(`^[${SPACE_CLASS}]+|[${SPACE_CLASS}]+$`, "g");
const DIGITS_OR_SPACES = /^[\p{Nd} ]+$/u;                       // re.fullmatch(r"[\d ]+", text)
const NUMBER = /^\p{Nd}[\p{Nd} ]*$/u;                           // re.fullmatch(r"\d[\d ]*", text)
const ASCII = /^[\x00-\x7f]*$/;

// str.split() without arguments: runs of Python whitespace, no empty parts.
export function pySplit(text) {
  return text.split(SPACE_RUN).filter((part) => part !== "");
}

// str.strip() without arguments.
export function pyStrip(text) {
  return text.replace(SPACE_EDGES, "");
}

function replaceAllText(text, target, replacement) {
  return text.split(target).join(replacement);
}

function codePointLength(text) {
  let count = 0;
  for (const _ of text) count++;  // eslint-disable-line no-unused-vars
  return count;
}

// text[:1].upper() + text[1:] (first code point).
function upperFirst(text) {
  if (!text) return text;
  const first = String.fromCodePoint(text.codePointAt(0));
  return first.toUpperCase() + text.slice(first.length);
}

function lowerFirst(text) {
  if (!text) return text;
  const first = String.fromCodePoint(text.codePointAt(0));
  return first.toLowerCase() + text.slice(first.length);
}

function last(items) {
  return items[items.length - 1];
}

// Python str.casefold() differs from per-character lower() only for these characters (generated from Python's
// unicodedata 13.0, restricted to characters that survive NFD and are not combining: the only ones plain() folds).
const CASEFOLD = new Map();
for (const [first, end, delta] of [[0xb5, 0xb5, 775], [0x17f, 0x17f, -268], [0x3c2, 0x3c2, 1], [0x3d0, 0x3d0, -30],
  [0x3d1, 0x3d1, -25], [0x3d5, 0x3d5, -15], [0x3d6, 0x3d6, -22], [0x3f0, 0x3f0, -54], [0x3f1, 0x3f1, -48],
  [0x3f5, 0x3f5, -64], [0x13a0, 0x13f5, 0], [0x13f8, 0x13fd, -8], [0x1c80, 0x1c80, -6222], [0x1c81, 0x1c81, -6221],
  [0x1c82, 0x1c82, -6212], [0x1c83, 0x1c84, -6210], [0x1c85, 0x1c85, -6211], [0x1c86, 0x1c86, -6204],
  [0x1c87, 0x1c87, -6180], [0x1c88, 0x1c88, 35267], [0xab70, 0xabbf, -38864]]) {
  for (let code = first; code <= end; code++) CASEFOLD.set(code, String.fromCodePoint(code + delta));
}
for (const [code, folded] of [[0xdf, "ss"], [0x149, "ʼn"], [0x587, "եւ"], [0x1e9a, "aʾ"],
  [0x1e9e, "ss"], [0xfb00, "ff"], [0xfb01, "fi"], [0xfb02, "fl"], [0xfb03, "ffi"], [0xfb04, "ffl"], [0xfb05, "st"],
  [0xfb06, "st"], [0xfb13, "մն"], [0xfb14, "մե"], [0xfb15, "մի"],
  [0xfb16, "վն"], [0xfb17, "մխ"]]) {
  CASEFOLD.set(code, folded);
}

// str.casefold() of text whose characters are already decomposed (NFD) and free of combining marks.
function casefold(text) {
  if (ASCII.test(text)) return text.toLowerCase();
  let out = "";
  for (const char of text) out += CASEFOLD.get(char.codePointAt(0)) ?? char.toLowerCase();
  return out;
}

const combiningCache = new Map();

// unicodedata.combining(char) != 0 for one character of an NFD string. JS has no canonical combining class
// property, so it is read off canonical ordering: a mark of class > 1 moves before U+0334 (class 1), a mark of
// class 1..6 moves before U+093C (class 7). Cached per code point.
export function combining(char) {
  const code = char.codePointAt(0);
  if (code < 0x300) return false;
  let known = combiningCache.get(code);
  if (known === undefined) {
    known = ("a" + char + "̴").normalize("NFD") !== "a" + char + "̴"
      || ("a़" + char).normalize("NFD") !== "a़" + char;
    combiningCache.set(code, known);
  }
  return known;
}

// ------------------------------------------------------------------ lexicon (lower-case, modern tone placement)
export const SENTENCE_PAUSE_SECONDS = 2.5;   // hands down this long between two signs: a new sentence

export const PRON1 = new Set(["tôi", "mình", "tớ", "tao", "chúng tôi", "chúng mình", "chúng ta", "em"]);
export const PRON2 = new Set(["bạn", "các bạn", "cậu", "anh", "chị", "ông", "bà", "cô", "chú", "bác", "thầy", "mày"]);
export const PRON3 = new Set(["nó", "họ", "anh ấy", "chị ấy", "ông ấy", "bà ấy", "cô ấy", "bạn ấy", "người ấy"]);
// also family / friend nouns: "chị tôi"
export const KIN_PRON2 = new Set(["anh", "chị", "ông", "bà", "cô", "chú", "bác", "em", "bạn"]);
export const DEMONSTRATIVE = new Set(["đây", "đó", "kia", "này", "cái này", "cái đó", "người này", "người kia"]);
export const KIN = new Set(["bố", "mẹ", "cha", "má", "ba", "bố mẹ", "cha mẹ", "ông bà", "ông nội", "bà nội",
  "ông ngoại", "bà ngoại", "anh trai", "chị gái", "em trai", "em gái", "em út", "anh ruột", "con trai", "con gái",
  "cháu trai", "cháu gái", "thím", "dì", "chồng", "vợ", "con", "anh em", "anh chị em", "gia đình", "dòng họ",
  "bạn bè", "người quen", "phụ huynh", "cháu"]);
export const ROLE = new Set(["cô giáo", "thầy giáo", "thầy cô", "giáo viên", "học trò", "học sinh", "sinh viên",
  "bác sĩ", "y tá", "công nhân", "nông dân", "kỹ sư", "phiên dịch viên", "người điếc", "người nghe", "bộ đội",
  "công an", "lái xe", "đầu bếp", "ca sĩ", "nhân viên"]);
export const MODAL = new Set(["muốn", "thích", "cần", "phải", "nên", "sẽ", "đang", "đã", "vừa", "sắp", "có thể",
  "biết", "hay", "định", "thường", "vẫn", "cũng", "từng", "không thể", "được"]);
// also the main verb when nothing follows: "tôi phở thích"
export const MODAL_VERBS = new Set(["muốn", "thích", "cần", "biết"]);
export const ABILITY = new Set(["biết", "có thể", "không thể"]);   // "tôi biết bơi": no "đi" after them
export const NEGATION = new Set(["không", "chưa", "chẳng"]);
export const QUESTION_WORDS = new Set(["gì", "ai", "đâu", "bao nhiêu", "mấy", "mấy giờ", "khi nào", "bao giờ",
  "lúc nào", "tại sao", "vì sao", "sao", "thế nào", "nào", "là gì", "tên là gì", "làm gì", "ở đâu", "đi đâu",
  "còn bạn"]);
export const WHY = new Set(["tại sao", "vì sao", "sao"]);
export const TAIL_TAGS = new Set(["phải không", "đúng không", "được không"]);   // written after a comma
export const TAIL_MODALS = new Map([["muốn", ""], ["cần", ""], ["nên", "có "], ["thích", "có "], ["biết", "có "]]);
export const TAIL_STATES = new Set(["mệt", "đói", "khát", "đẹp", "sợ", "ngon", "vui", "buồn", "đau", "khoẻ", "no",
  "thích", "xa", "đắt"]);
export const TRANSITIVE_STATES = new Set(["sợ", "thích"]);   // "bạn chó sợ không?" -> "bạn có sợ chó không?"
export const GOING = new Set(["đá bóng", "bơi", "ngủ", "chơi", "câu cá", "dạo", "du lịch", "khám bệnh", "mua sắm",
  "xem phim", "làm", "làm việc", "học", "tập thể dục", "chạy bộ", "dã ngoại", "cấp cứu", "học bơi", "ăn cưới", "tắm",
  "ra ngoài"]);
// already says where the subject goes
export const MOVES = Object.freeze(["đi", "đến", "về", "ra", "lên", "vào", "sang", "tới"]);
export const PLACES = new Map([["trường", "đến"], ["nhà trường", "đến"], ["lớp", "đến"], ["sân trường", "ra"],
  ["nhà", "về"], ["quê", "về"], ["nhà cửa", "về"], ["chợ", "đi"], ["siêu thị", "đi"], ["bệnh viện", "đi"],
  ["ngân hàng", "đi"], ["công viên", "đi"], ["rạp chiếu phim", "đi"], ["khách sạn", "đến"], ["cửa hàng", "đến"],
  ["hiệu thuốc", "đến"], ["trạm y tế", "đến"], ["ga tàu", "ra"], ["bến xe", "ra"], ["trạm xe buýt", "ra"],
  ["nhà vệ sinh", "đi"], ["phòng thư viện", "lên"], ["thư viện", "lên"], ["phố", "ra"], ["biển", "đi"],
  ["núi", "lên"], ["chùa", "đi"], ["nhà hàng", "đến"], ["quán", "ra"], ["sân bay", "ra"], ["công ty", "đến"],
  ["cơ quan", "đến"]]);
// "bạn nhà đâu": a place asked about
export const OWN_PLACES = new Set(["nhà", "quê", "trường", "lớp", "nhà trường", "công ty", "cơ quan"]);
export const STAY = new Set(["sống", "ở", "ngủ", "nghỉ ngơi", "nằm", "ngồi", "làm việc", "học", "chơi", "đi chơi",
  "đợi", "chờ"]);
export const COPULA_NOUNS = new Set(["tên", "giá", "nghề", "nghề nghiệp", "số điện thoại", "địa chỉ", "quê",
  "ngày sinh", "sở thích", "tiền", "giá tiền", "môn", "lớp", "ngôn ngữ", "tuổi", "sức khỏe", "sức khoẻ"]);
// "bạn tên là gì" keeps its order
export const TOPIC_NOUNS = new Set(["tên", "quê", "tuổi", "giá", "giá tiền", "tiền", "môn", "lớp"]);
export const COUNT_NOUNS = new Map([["tuổi", ""], ["giờ", ""], ["anh chị em", "có "], ["anh em", "có "],
  ["con", "có "], ["người", "có "], ["quyển vở", "có "], ["sách", "có "], ["tiền", "có "], ["bạn", "có "],
  ["cháu", "có "]]);
export const ORDINAL_NOUNS = new Set(["tháng", "năm", "lớp", "phòng", "số", "tuần", "giá", "giá tiền", "tiền",
  "trang", "bài", "tầng"]);
export const BODY = new Set(["tay", "chân", "bàn chân", "bàn tay", "bụng", "đầu", "răng", "mắt", "tai", "cổ", "vai",
  "mặt", "tim", "lưng", "họng"]);
export const VEHICLES = new Set(["xe máy", "xe đạp", "ô tô", "taxi", "tàu hoả", "xe buýt", "máy bay", "tàu"]);
export const OBJECTS = new Set(["cơm", "phở", "bánh mì", "xôi", "trứng", "rau", "sữa", "trà", "nước", "mỳ tôm", "mì",
  "kẹo", "quả cam", "quả chuối", "cam", "chuối", "táo", "đồ ăn", "đồ uống", "bún chả", "bữa cơm", "chai nước",
  "nước ép trái cây", "cà phê", "bia", "sách", "bút", "quyển vở", "vở", "vé", "báo", "phim", "bóng", "điện thoại",
  "máy tính", "quần áo", "bát", "đũa", "thuốc", "tiền", "bài tập", "ngôn ngữ kí hiệu", "tiếng việt", "tiếng anh",
  "nhà", "đồng hồ", "cặp sách", "bàn ghế", "bảng đen", "thịt", "cá", "gà", "mèo", "chó", "đèn đỏ", "môn toán", "áo",
  "quần", "giày", "dép", "mũ", "tranh", "nhạc", "bài hát", "trò chơi", ...BODY, ...VEHICLES]);
export const VERBS = new Set(["ăn", "uống", "đọc", "xem", "mua", "bán", "nấu", "nấu nướng", "gặp", "gặp gỡ", "yêu",
  "thương", "học", "dạy", "viết", "gọi", "hỏi", "giúp", "giúp đỡ", "nghe", "nhìn", "đợi", "chờ", "khen", "mời",
  "đếm", "hiểu", "tiêm", "kiểm tra", "rửa", "lau", "quét", "phơi", "mặc", "cầm", "lấy", "cho", "tặng", "gửi", "nhớ",
  "quên", "biết", "làm", "chơi", "đi", "đến", "về", "ra", "vào", "chạy", "ngồi", "nằm", "đứng", "ngủ", "nói",
  "nói chuyện", "trả lời", "giới thiệu", "làm quen", "kết bạn", "chia tay", "hẹn", "tha thứ", "xin lỗi", "cảm ơn",
  "chào", "đá", "đánh", "rửa bát", "rửa tay", "rửa mặt", "đánh răng", "quét nhà", "lau nhà", "phơi quần áo",
  "mặc quần áo", "đọc sách", "làm bài tập", "tập viết", "học tập", "làm việc", "đi học", "đi dạo", "đi lạc",
  "đi chơi", "nghỉ ngơi", "nghỉ học", "giải lao", "giao tiếp", "hỏi thăm", "khóc", "cười", "thích", "muốn", "cần",
  "giơ tay", "khám bệnh", "đá bóng", "bơi", "tắm", "hát", "nhảy", "mua sắm", "sống", "nuôi", "thương yêu",
  "yêu thương", "yêu mến", "ăn uống", "lái", "đi xe", "phiên dịch", "đi bộ", "dùng", "thăm", "chụp", "vẽ"]);
export const PERSON_VERBS = new Set(["gặp", "gặp gỡ", "yêu", "thương", "thương yêu", "yêu thương", "yêu mến", "dạy",
  "gọi", "hỏi", "giúp", "giúp đỡ", "đợi", "chờ", "khen", "mời", "nhớ", "quên", "hiểu", "tha thứ", "chào", "đánh",
  "giới thiệu", "hỏi thăm", "cảm ơn", "xin lỗi", "nhìn", "thăm", "phiên dịch", "kết bạn", "làm quen", "nuôi"]);
export const TIME = new Set(["hôm nay", "hôm qua", "ngày mai", "bây giờ", "hiện nay", "sáng", "chiều", "tối", "trưa",
  "đêm", "buổi sáng", "buổi chiều", "buổi tối", "buổi trưa", "ban ngày", "ban đêm", "tuần này", "tuần sau",
  "tuần trước", "tháng này", "tháng sau", "năm nay", "năm ngoái", "năm tới", "mùa xuân", "mùa hè", "mùa thu",
  "mùa đông", "cuối tuần", "hàng ngày", "mỗi ngày", "hôm kia", "ngày kia", "một tiếng"]);
export const STATES = new Set(["hạnh phúc", "vui vẻ", "buồn ngủ", "biết ơn", "mệt", "mệt mỏi", "đói", "khát",
  "khát nước", "no", "vui", "buồn", "đẹp", "xấu", "ngon", "không ngon", "ngon miệng", "khoẻ", "khoẻ mạnh", "ốm",
  "ốm đau", "đau", "đau bụng", "đau chân", "đau mắt", "đau tay", "sốt", "buồn nôn", "lo sợ", "hoảng sợ", "sợ",
  "giận", "giận dữ", "giận dỗi", "nổi giận", "bực mình", "ghen tị", "hồi hộp", "hài lòng", "yên tâm", "tự hào",
  "cô đơn", "ngạc nhiên", "vui mừng", "vui sướng", "thích thú", "buồn thảm", "e thẹn", "tươi cười", "tiếc", "bận",
  "rảnh", "lạnh", "nóng", "cao", "thấp", "to", "nhỏ", "nhiều", "ít", "tốt", "giỏi", "chăm chỉ", "lười", "đắt", "rẻ",
  "xa", "gần", "lâu", "muộn", "sớm"]);
// known, so never "restored"
export const FUNCTION_WORDS = new Set(["có", "là", "của", "ở", "thì", "và", "với", "rất", "cho", "nhé", "ạ"]);
export const GREETINGS = new Set(["chào", "xin chào", "chào mừng", "tạm biệt", "cảm ơn", "xin lỗi", "không có chi",
  "chúc mừng", "rất vui được gặp bạn", "hello"]);
// their own clause: "Xin chào, ..."
export const CLAUSE_GREETINGS = new Set(["chào", "xin chào", "tạm biệt", "cảm ơn", "xin lỗi"]);
export const TONE_SIGNS = new Map([["dấu sắc", "́"], ["dấu huyền", "̀"], ["dấu hỏi", "̉"],
  ["dấu ngã", "̃"], ["dấu nặng", "̣"]]);
export const TONE_MARKS = new Set(TONE_SIGNS.values());
export const LETTERS = new Set([..."abcdefghijklmnopqrstuvwxyz", ..."ăâđêôơư"]);
export const VOWELS = "aăâeêioôơuưy";
export const ONSETS = Object.freeze(["ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr", "b", "c",
  "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x", ""]);
export const CODAS = Object.freeze(["ng", "nh", "ch", "c", "m", "n", "p", "t", ""]);
const CODAS_SHORTEST_FIRST = Object.freeze([...CODAS].sort((a, b) => a.length - b.length));  // stable, like sorted()
const MARKED_VOWELS = "ăâêôơư";
const KHONG = " không";

export class Token {
  // text: lower-case, modern tone placement (what the lexicon looks up); written: how the word is written in the
  // output ("" = text); spelled: a fingerspelled name; question: a sign named as a question ("hiểu không?").
  constructor(text, { written = "", spelled = false, tags = null, question = false } = {}) {
    this.text = text;
    this.written = written;
    this.spelled = spelled;
    this.tags = tags ?? new Set();
    this.question = question;
  }

  out() {
    return this.written || this.text;
  }
}

// A word a rule adds.
export function word(text, ...tags) {
  return new Token(text, { tags: new Set(tags) });
}

function anyTag(token, names) {
  for (const name of names) if (token.tags.has(name)) return true;
  return false;
}

function startsWithMove(text) {
  for (const move of MOVES) if (text.startsWith(move + " ")) return true;
  return false;
}

// ------------------------------------------------------------------ spelling and tones
// Index of the vowel that carries the tone in a syllable given as code points (modern spelling: hoà, khoẻ, thuý).
function tonePosition(chars) {
  const vowels = [];
  for (let index = 0; index < chars.length; index++) if (VOWELS.includes(chars[index])) vowels.push(index);
  if (!vowels.length) return -1;
  let start = vowels[0];
  const head = chars.slice(0, 2).join("");
  if ((head === "qu" || head === "gi") && vowels.length > 1 && start === 1) start = vowels[1];  // not vowels here
  const block = [];  // the first run of contiguous vowels
  for (const index of vowels) {
    if (index < start) continue;
    if (block.length && index !== last(block) + 1) break;
    block.push(index);
  }
  const marked = block.filter((index) => MARKED_VOWELS.includes(chars[index]));
  if (marked.length) return last(marked);
  const chunk = block.map((index) => chars[index]).join("");
  if (chunk === "oa" || chunk === "oe" || chunk === "uy") return block[1];
  if (block.length === 3) return block[1];
  if (block.length === 2) return last(block) < chars.length - 1 ? block[1] : block[0];
  return block[0];
}

// _tone_position: index (in code points) of the vowel that carries the tone, -1 without a vowel.
export function _tonePosition(letters) {
  return tonePosition(Array.from(letters));
}

// A syllable with its tone mark where modern spelling puts it (khỏe -> khoẻ, hòa -> hoà).
export function _retone(syllable) {
  const decomposed = syllable.normalize("NFD");
  const tones = [];
  let rest = "";
  for (const char of decomposed) {
    if (TONE_MARKS.has(char)) tones.push(char);
    else rest += char;
  }
  if (!tones.length) return syllable;
  const base = Array.from(rest.normalize("NFC"));
  const position = tonePosition(base);
  if (position < 0) return syllable;
  return (base.slice(0, position + 1).join("") + last(tones) + base.slice(position + 1).join("")).normalize("NFC");
}

// A recognised word as the lexicon spells it: lower-case, modern tones, no ?/…, single spaces.
export function normalise(text) {
  let value = String(text).normalize("NFC").toLowerCase();
  value = replaceAllText(replaceAllText(value, "…", " "), "?", " ");
  return pySplit(value).map(_retone).join(" ");
}

// A recognised word as it is written in the output: its own capitals, no ?/…, single spaces.
export function _written(text) {
  const value = replaceAllText(replaceAllText(String(text).normalize("NFC"), "…", " "), "?", " ");
  return pyStrip(value.replace(SPACE_RUN, " "));
}

// Split letters into syllables (onset + vowels + coda), preferring a consonant as the next onset; null: no split.
// (Onsets, vowels and codas are BMP characters, so UTF-16 indices equal Python's code point indices here.)
export function _parseSyllables(letters) {
  if (!letters) return [];
  for (const onset of ONSETS) {
    if (!letters.startsWith(onset)) continue;
    const middle = onset.length;
    let end = middle;
    while (end < letters.length && VOWELS.includes(letters[end])) end++;
    if (end === middle) continue;
    for (const coda of CODAS_SHORTEST_FIRST) {  // the shortest coda that leaves a valid rest: thu|ha, lan|mai
      if (letters.startsWith(coda, end)) {
        const rest = _parseSyllables(letters.slice(end + coda.length));
        if (rest !== null) return [letters.slice(0, end + coda.length), ...rest];
      }
    }
  }
  return null;
}

// One fingerspelled syllable group (letters and the tone sign signed after them) -> capitalised syllables:
// ['l', 'a', 'n'] -> 'Lan', ['t', 'h', 'u', 'h', 'a'] + huyền -> 'Thu Hà' (the tone goes on the last syllable).
export function spell(letters, tones) {
  const raw = [...letters].join("");
  const parsed = _parseSyllables(raw);
  const syllables = parsed && parsed.length ? parsed : [raw];
  if (tones.length) {
    const chars = Array.from(last(syllables));
    const position = tonePosition(chars);
    if (position >= 0) {
      const mark = TONE_SIGNS.get(last(tones));
      if (mark === undefined) throw new Error(`KeyError: ${last(tones)}`);
      syllables[syllables.length - 1] = chars.slice(0, position + 1).join("") + mark + chars.slice(position + 1).join("");
    }
  }
  return syllables.map((part) => upperFirst(part).normalize("NFC")).join(" ");
}

// [lookup text, written text, named as a question] entries -> tokens; runs of letters become one name, a tone
// sign ends a syllable.
export function _mergeSpelling(words) {
  const tokens = [];
  const parts = [];
  const letters = [];

  const endSyllable = (tone = null) => {
    if (letters.length) {
      parts.push(spell([...letters], tone ? [tone] : []));
      letters.length = 0;
    }
  };

  const flush = () => {
    endSyllable();
    if (!parts.length) return;
    const name = parts.join(" ");
    if (codePointLength(replaceAllText(name, " ", "")) >= 2) {
      tokens.push(new Token(name.toLowerCase(), { written: name, spelled: true, tags: new Set(["name"]) }));
    } else {  // a single letter: kept as a letter
      tokens.push(new Token(name.toLowerCase().normalize("NFC")));
    }
    parts.length = 0;
  };

  for (const [text, written, question] of words) {
    if (LETTERS.has(text)) {
      letters.push(text);
    } else if (TONE_SIGNS.has(text) && letters.length) {
      endSyllable(text);
    } else {
      flush();
      tokens.push(new Token(text, { written: written !== text ? written : "", question }));
    }
  }
  flush();
  return tokens;
}

// ------------------------------------------------------------------ tagging
export const LEXICON = Object.freeze([["pron1", PRON1], ["pron2", PRON2], ["pron3", PRON3],
  ["demonstrative", DEMONSTRATIVE], ["kin", KIN], ["role", ROLE], ["modal", MODAL], ["negation", NEGATION],
  ["question", QUESTION_WORDS], ["greeting", GREETINGS], ["time", TIME], ["state", STATES], ["object", OBJECTS],
  ["verb", VERBS], ["going", GOING], ["copula_noun", COPULA_NOUNS], ["body", BODY], ["vehicle", VEHICLES],
  ["function", FUNCTION_WORDS]]);
const LEXICON_BY_NAME = new Map(LEXICON);
const LEXICON_UNION = new Set(LEXICON.flatMap(([, words]) => [...words]));

// Lower-case text without Vietnamese diacritics ("Tôi" -> "toi", "đi" -> "di"): how words typed without a
// Vietnamese keyboard look (Python: NFD, combining marks removed, casefold()).
export function plain(text) {
  const decomposed = replaceAllText(replaceAllText(String(text), "đ", "d"), "Đ", "D").normalize("NFD");
  if (ASCII.test(decomposed)) return decomposed.toLowerCase();
  let kept = "";
  for (const char of decomposed) if (!combining(char)) kept += char;
  return casefold(kept);
}

let lexiconWordsCache = null;

// Every word the lexicon knows, question signs included (for accent restoration and suggestions); sorted by code
// point like Python's sorted() (the lexicon is BMP-only, where UTF-16 order is code point order).
export function lexiconWords() {
  if (!lexiconWordsCache) {
    const questions = new Set(["có không", ...TAIL_TAGS,
      ...[...TAIL_MODALS.keys(), ...TAIL_STATES].map((core) => `${core} không`)]);
    const all = new Set(LEXICON_UNION);
    for (const words of [PLACES.keys(), COPULA_NOUNS, COUNT_NOUNS.keys(), questions]) {
      for (const text of words) all.add(text);
    }
    lexiconWordsCache = Object.freeze([...all].sort());
  }
  return [...lexiconWordsCache];
}

const PLAIN_INDEX = new Map();
export const RESTORE_PRIORITY = Object.freeze(["function", "pron1", "pron2", "pron3", "kin", "question", "negation",
  "modal", "greeting", "time", "verb", "state", "object", "role", "copula_noun"]);

// In the lexicon, or a question sign recognised by its form ("đói không", "có không").
export function _known(text) {
  return LEXICON_UNION.has(text) || PLACES.has(text) || TAIL_TAGS.has(text) || text === "có không"
    || (text.endsWith(KHONG) && text !== "không");
}

// A word typed without (some of) its diacritics, written as the lexicon word it stands for: "toi" -> "tôi",
// "xin chao" -> "xin chào", "ban" -> "bạn" (several candidates: pronouns first). Unknown words stay; so does a
// single letter, unless it is a word of a phrase ("toi song o").
export function _restore(text, inPhrase = false) {
  if (_known(text) || DIGITS_OR_SPACES.test(text) || (!inPhrase && (codePointLength(text) < 2 || LETTERS.has(text)))) {
    return text;
  }
  if (!PLAIN_INDEX.size) {
    for (const known of lexiconWords()) {
      const key = plain(known);
      if (!PLAIN_INDEX.has(key)) PLAIN_INDEX.set(key, []);
      PLAIN_INDEX.get(key).push(known);
    }
  }
  const candidates = PLAIN_INDEX.get(plain(text)) ?? [];
  if (candidates.length === 1) return candidates[0];
  for (const category of RESTORE_PRIORITY) {
    const words = LEXICON_BY_NAME.get(category);
    const chosen = candidates.filter((candidate) => words.has(candidate));
    if (chosen.length) return chosen[0];
  }
  return text;
}

// One recognised word -> lexicon entries [text, written, question]: diacritics restored, and a taught phrase made
// only of known words ("toi tuoi") split into them so the rules see its parts.
export function _entries(text, written, question) {
  const restored = _restore(text);
  if (restored !== text) return [[restored, "", question]];
  if (text.includes(" ") && !_known(text)) {
    const parts = pySplit(text).map((part) => _restore(part, true));
    if (parts.every((part) => _known(part))) {
      return parts.map((part, index) => [part, "", question && index === parts.length - 1]);
    }
  }
  return [[text, written, question]];
}

// Adds the lexicon tags of a token (in place) and returns it.
export function _tag(token) {
  if (token.spelled) {
    token.tags.add("name");
    return token;
  }
  const { text, tags } = token;
  for (const [name, words] of LEXICON) if (words.has(text)) tags.add(name);
  if (PLACES.has(text)) tags.add("place");
  if (NUMBER.test(text)) tags.add("number");
  if (text === "có không") {
    tags.add("co_khong");
  } else if (TAIL_TAGS.has(text)) {
    tags.add("tail_tag");
  } else if (text.endsWith(KHONG) && text !== "không") {
    const core = text.slice(0, -KHONG.length);
    if (TAIL_MODALS.has(core)) tags.add("tail_modal");
    else if (TAIL_STATES.has(core)) tags.add("tail_state");
    else tags.add("tail_other");  // a taught "X không?" sign
  }
  return token;
}

const PERSON_TAGS = ["pron1", "pron2", "pron3", "kin", "name", "demonstrative", "role"];
const PRONOUNS = ["pron1", "pron2", "pron3"];
const OWNER_TAGS = ["pron1", "pron2", "pron3", "name"];

export function _isPerson(token) {
  return anyTag(token, PERSON_TAGS);
}

export function _owner(token) {
  return anyTag(token, OWNER_TAGS) || KIN.has(token.text);
}

// Index just after the subject: leading time words, then a person (and its possessor: "bố tôi", "chị tôi",
// "tên tôi").
export function _subjectEnd(tokens) {
  let index = 0;
  while (index < tokens.length && tokens[index].tags.has("time")) index++;
  if (index < tokens.length && (_isPerson(tokens[index]) || tokens[index].tags.has("copula_noun"))) {
    index++;
    const head = tokens[index - 1];
    if (index < tokens.length && (anyTag(head, ["kin", "copula_noun", "role"]) || KIN_PRON2.has(head.text))
        && anyTag(tokens[index], OWNER_TAGS) && !head.tags.has("pron1")) {
      index++;
    }
  }
  return index;
}

// Some word says what the subject does (a verb that is not only a modal).
export function _acts(tokens) {
  return tokens.some((token) => anyTag(token, ["verb", "going"]) && !token.tags.has("modal"));
}

// _subjectEnd, or, for a sentence about a thing ("phở ngon", "nhà bạn xa"), the words before its first state.
export function _topicEnd(tokens) {
  const start = _subjectEnd(tokens);
  let lead = 0;
  while (lead < tokens.length && tokens[lead].tags.has("time")) lead++;
  if (start === lead && !_acts(tokens)) {
    for (let index = start; index < tokens.length; index++) if (tokens[index].tags.has("state")) return index;
  }
  return start;
}

// The sentence is about the person addressed or theirs (bạn, anh..., bố bạn, nhà bạn): a final "không" asks.
export function _addressee(tokens) {
  const end = Math.max(_topicEnd(tokens), _subjectEnd(tokens));
  if (tokens.slice(0, end).some((token) => token.tags.has("pron2"))) return true;
  for (const token of tokens) if (!token.tags.has("time")) return token.tags.has("pron2");
  return false;
}

// ------------------------------------------------------------------ rules
// Question signs that frame the sentence: [tokens, is a question, a tail written after a comma].
export function _frameQuestions(tokens) {
  if (!tokens.length) return [tokens, false, []];
  const lastToken = last(tokens);
  const head = tokens.slice(0, -1);
  const start = _topicEnd(head);
  const subject = head.slice(0, start);
  const predicate = head.slice(start);
  if (lastToken.tags.has("co_khong")) {  // "bạn đi học có … không?" -> "bạn có đi học không?"
    return [[...subject, word("có"), ...predicate, word("không")], true, []];
  }
  if (lastToken.tags.has("tail_modal")) {  // "bạn ăn muốn không?" -> "bạn muốn ăn không?"
    const core = lastToken.text.slice(0, -KHONG.length);
    return [[...subject, word(TAIL_MODALS.get(core) + core, "modal"), ...predicate, word("không")], true, []];
  }
  if (lastToken.tags.has("tail_state")) {
    const core = lastToken.text.slice(0, -KHONG.length);
    if (predicate.length && TRANSITIVE_STATES.has(core)) {  // "bạn chó sợ không?" -> "bạn có sợ chó không?"
      return [[...subject, word("có " + core), ...predicate, word("không")], true, []];
    }
    const times = predicate.filter((token) => token.tags.has("time"));  // "bạn bây giờ đói không?"
    const rest = predicate.filter((token) => !token.tags.has("time"));
    if (!rest.length) return [[...times, ...subject, word(core), word("không")], true, []];
    return [[...times, ...subject, ...rest, word("có " + core), word("không")], true, []];
  }
  if (head.length && (lastToken.tags.has("tail_tag") || lastToken.text === "còn bạn")) {  // ", phải không?"
    return [head, true, [lastToken]];
  }
  if (lastToken.tags.has("tail_other") || (lastToken.question && lastToken.text.endsWith(KHONG))) {
    const core = lastToken.text.slice(0, -KHONG.length);  // a taught "hiểu không?" sign
    if (predicate.length || !subject.length) return [tokens, true, []];
    return [[...subject, word("có " + core, "verb"), word("không")], true, []];
  }
  return [tokens, false, []];
}

// A modal signed after what it modifies ("tôi trà uống muốn", "tôi giúp đỡ cần") goes right after the subject;
// a final "có" (have) too ("tôi em gái có").
export function _modalLast(tokens) {
  const end = tokens.length && last(tokens).tags.has("negation") ? tokens.length - 1 : tokens.length;
  let cut = end;
  while (cut > 0 && ((tokens[cut - 1].tags.has("modal") && !tokens[cut - 1].tags.has("state"))
                     || tokens[cut - 1].text === "có")) {
    cut--;
  }
  const start = _subjectEnd(tokens.slice(0, cut));
  if (cut === end || !start || cut <= start) return tokens;
  const moved = tokens.slice(cut, end);
  if (moved.length === 1 && MODAL_VERBS.has(moved[0].text) && cut - start === 1
      && anyTag(tokens[start], ["object", "place"]) && !anyTag(tokens[start], ["verb", "going"])) {
    return tokens;  // "tôi phở thích": the modal is the verb (moved with its object later)
  }
  return [...tokens.slice(0, start), ...moved, ...tokens.slice(start, cut), ...tokens.slice(end)];
}

// A final "không"/"chưa": a yes/no question to the person addressed, else negation before the predicate.
// Returns [tokens, is a question].
export function _finalNegation(tokens) {
  if (tokens.length < 2 || !last(tokens).tags.has("negation")) return [tokens, false];
  const negation = last(tokens).text;
  const body = tokens.slice(0, -1);
  const start = _topicEnd(body);
  const subject = body.slice(0, start);
  const predicate = body.slice(start);
  if (!predicate.length) return [tokens, _addressee(tokens)];
  const asking = _addressee(body);
  if (predicate.every((token) => anyTag(token, ["role", "name"]))) {  // a role or a name: "phải là"
    if (asking) return [[...subject, word("có phải là"), ...predicate, word("không")], true];
    return [[...subject, word(negation + " phải là"), ...predicate], false];
  }
  const possession = predicate.every((token) => anyTag(token, ["object", "kin"])
    && !anyTag(token, ["verb", "going", "state", "place"]));
  if (asking) {
    if (negation === "chưa") return [[...subject, word("đã"), ...predicate, word("chưa")], true];
    if (predicate.every((token) => token.tags.has("state"))) {  // "bạn mệt không?", "nhà bạn có xa không?"
      const head = subject.find((token) => !token.tags.has("time")) ?? null;
      if (head !== null && !_isPerson(head)) return [[...subject, word("có"), ...predicate, word("không")], true];
      return [[...body, word("không")], true];
    }
    return [[...subject, word("có"), ...predicate, word("không")], true];
  }
  if (possession) return [[...subject, word(negation + " có"), ...predicate], false];  // "tôi không có tiền"
  return [[...subject, word(negation), ...predicate], false];
}

// "vì sao" goes first; a negation before it is a statement: "vì sao bạn không đi học".
export function _why(tokens, why) {
  if (tokens.length >= 2 && last(tokens).tags.has("negation")) {
    const body = tokens.slice(0, -1);
    const start = _subjectEnd(body);
    if (body.length > start) tokens = [...body.slice(0, start), last(tokens), ...body.slice(start)];
  }
  return [why, ...tokens];
}

const OWN_PLACE_QUESTIONS = ["đâu", "ở đâu", "mấy", "nào", "gì", "là gì"];

// A final question word: add what Vietnamese needs around it. Returns [tokens, is a question].
export function _finalQuestion(tokens) {
  if (!tokens.length || !last(tokens).tags.has("question")) return [tokens, false];
  const lastToken = last(tokens);
  const before = tokens.slice(0, -1);
  const previous = before.length ? last(before) : null;
  const asked = lastToken.text;
  const texts = before.map((token) => token.text);
  if (WHY.has(asked)) return [_why(before, lastToken), true];
  const pronoun = previous !== null && anyTag(previous, PRONOUNS);
  // "bạn nhà đâu" -> "nhà bạn ở đâu?"; "bạn lớp mấy" -> "bạn học lớp mấy?"
  if (before.length === 2 && anyTag(before[0], PRONOUNS) && OWN_PLACES.has(before[1].text)
      && OWN_PLACE_QUESTIONS.includes(asked)) {
    const [owner, place] = before;
    if ((place.text === "lớp" || place.text === "trường") && (asked === "mấy" || asked === "nào")) {
      return [[owner, word("học", "verb"), place, lastToken], true];
    }
    if (asked === "đâu" || asked === "ở đâu") return [[place, owner, word("ở đâu")], true];
    return [[place, owner, word("là"), word("gì")], true];
  }
  // counted: "bạn tuổi bao nhiêu" -> "bạn bao nhiêu tuổi", "bạn anh chị em mấy" -> "bạn có mấy anh chị em"
  if ((asked === "bao nhiêu" || asked === "mấy") && previous !== null && COUNT_NOUNS.has(previous.text)
      && !(pronoun && before.length >= 2 && !_isPerson(before[before.length - 2]))) {  // not "số điện thoại bạn"
    const has = COUNT_NOUNS.get(previous.text);
    return [[...before.slice(0, -1), word(pyStrip(`${has}${asked} ${previous.text}`))], true];
  }
  const acted = before.some((token) => anyTag(token, ["verb", "going", "modal"]));
  // "bạn nghề gì" -> "bạn làm nghề gì?", "bạn địa chỉ gì" -> "địa chỉ của bạn là gì?"
  const owners = before.slice(0, -1);
  if (previous !== null && COPULA_NOUNS.has(previous.text) && !TOPIC_NOUNS.has(previous.text)
      && (asked === "gì" || asked === "bao nhiêu" || asked === "thế nào")
      && owners.length && owners.every((token) => _isPerson(token)) && !acted) {
    if ((previous.text === "nghề" || previous.text === "nghề nghiệp") && asked === "gì") {
      return [[...owners, word("làm"), previous, lastToken], true];
    }
    if (asked === "thế nào") return [[previous, word("của"), ...owners, lastToken], true];
    return [[previous, word("của"), ...owners, word("là"), lastToken], true];
  }
  if (previous !== null && (asked === "gì" || asked === "bao nhiêu" || asked === "ai") && !texts.includes("là")
      && !acted) {
    if (anyTag(previous, ["copula_noun", "demonstrative"])
        || (asked === "ai" && anyTag(previous, ["pron1", "pron2", "pron3", "kin", "role"]))
        || (pronoun && before.length >= 2 && before[before.length - 2].tags.has("copula_noun"))) {
      return [[...before, word("là"), lastToken], true];  // giá là bao nhiêu, tên bạn là gì, bạn là ai
    }
  }
  if (asked === "đâu" && previous !== null && !texts.includes("ở")
      && !MOVES.includes(previous.text) && !startsWithMove(previous.text)) {
    return [[...before, word("ở đâu")], true];  // nhà bạn ở đâu, bạn làm việc ở đâu
  }
  return [tokens, true];
}

const NUMBER_BLOCKERS = ["verb", "going", "modal", "negation", "question", "time", "pron1", "pron2", "pron3", "name",
  "demonstrative"];

// noun + number (VSL) -> number + noun ("tôi con 2" -> "tôi có 2 con"); "tháng 10", "lớp 5" stay.
export function _numbers(tokens) {
  const out = [];
  let index = 0;
  while (index < tokens.length) {
    const token = tokens[index];
    const following = index + 1 < tokens.length ? tokens[index + 1] : null;
    if (following !== null && following.tags.has("number") && !token.tags.has("number")
        && !anyTag(token, NUMBER_BLOCKERS) && !ORDINAL_NOUNS.has(token.text)) {
      out.push(following, token);
      index += 2;
      continue;
    }
    out.push(token);
    index += 1;
  }
  const start = _subjectEnd(out);
  const rest = out.slice(start);
  if (start && rest.length === 2 && rest[0].tags.has("number") && rest[1].text !== "tuổi"
      && !_acts(rest) && !out.some((token) => token.text === "có")) {
    return [...out.slice(0, start), word("có", "verb"), ...rest];  // "gia đình tôi 4 người" -> "... có 4 người"
  }
  return out;
}

// subject - object - (modal) - verb (VSL) -> subject - (modal) - verb - object. The object may be a thing (with its
// number), a person in front of a verb that takes one, a vehicle or place before its motion verb, or a body part
// before "đau".
export function _objectBeforeVerb(tokens) {
  const start = _subjectEnd(tokens);
  const rest = tokens.slice(start);
  for (let index = 0; index < rest.length - 1; index++) {
    const obj = rest[index];
    const person = _isPerson(obj) && !obj.tags.has("object");
    const movable = anyTag(obj, ["object", "place", "vehicle"]) || person;
    if (!movable || (anyTag(obj, ["verb", "going"]) && !obj.tags.has("object"))) continue;
    const first = index && rest[index - 1].tags.has("number") ? index - 1 : index;
    let ownerEnd = index + 1;
    if (ownerEnd < rest.length - 1 && _owner(rest[ownerEnd]) && anyTag(obj, ["place", "object"])) {
      ownerEnd += 1;  // "nhà bạn", "nhà bố"
    }
    const verbs = rest.slice(ownerEnd);
    let cut = 0;
    while (cut < verbs.length && anyTag(verbs[cut], ["modal", "negation"])) cut++;
    if (cut >= verbs.length) {  // no other verb: a final muốn / thích / cần / biết is the verb ("tôi phở thích")
      const main = [];
      verbs.forEach((token, position) => { if (MODAL_VERBS.has(token.text)) main.push(position); });
      if (!main.length) continue;
      cut = last(main);
    }
    const verb = verbs[cut];
    if (obj.tags.has("body") && verb.text === "đau") {
      // "tôi tay đau" -> "tôi đau tay"
    } else if (person && !PERSON_VERBS.has(verb.text)) {
      continue;
    } else if (obj.tags.has("place") && !obj.tags.has("object")
               && !(MOVES.includes(verb.text) || STAY.has(verb.text) || startsWithMove(verb.text))) {
      continue;
    } else if (!anyTag(verb, ["verb", "going"]) && verb.text !== "đau") {
      continue;
    }
    let moved;
    if (obj.tags.has("place") && (STAY.has(verb.text) || (verb.text.startsWith("đi ") && verb.text !== "đi học"))) {
      moved = [verb, word("ở"), ...rest.slice(first, ownerEnd)];  // "ông bà quê sống" -> "ông bà sống ở quê"
    } else {
      moved = [verb, ...rest.slice(first, ownerEnd)];
    }
    return [...tokens.slice(0, start), ...rest.slice(0, first), ...verbs.slice(0, cut), ...moved,
      ...verbs.slice(cut + 1)];
  }
  return tokens;
}

// An activity signed before "đi" (VSL order): "tôi khám bệnh đi" -> "tôi đi khám bệnh".
export function _activityBeforeGo(tokens) {
  const start = _subjectEnd(tokens);
  for (let index = start; index < tokens.length - 1; index++) {
    if (tokens[index].tags.has("going") && !tokens[index].tags.has("place") && tokens[index + 1].text === "đi") {
      return [...tokens.slice(0, index), tokens[index + 1], tokens[index], ...tokens.slice(index + 2)];
    }
  }
  return tokens;
}

// An activity that ends the predicate gets "đi" right after the subject (or a modal / negation / "có"); a place
// that is the predicate gets its verb (đi chợ, đến trường, về nhà); after "đi + vehicle" too.
export function _going(tokens) {
  const start = _subjectEnd(tokens);
  if (start === 0) return tokens;
  const out = tokens.slice(0, start);
  for (let position = start; position < tokens.length; position++) {
    const token = tokens[position];
    const previous = last(out);
    const following = position + 1 < tokens.length ? tokens[position + 1] : null;
    let rightAfter = position === start || anyTag(previous, ["modal", "negation"]) || previous.text === "có";
    rightAfter = rightAfter && !ABILITY.has(previous.text);
    const moving = MOVES.includes(previous.text) || startsWithMove(token.text);
    const vehicleBefore = previous.tags.has("vehicle") && out.length >= 2 && MOVES.includes(out[out.length - 2].text);
    if (token.tags.has("place") && !moving && (vehicleBefore || (
      rightAfter && !_acts(tokens.slice(start, position))
      && (following === null || !following.tags.has("question"))))) {
      out.push(word(PLACES.get(token.text), "verb"));
    } else if (rightAfter && !moving && token.tags.has("going") && !token.tags.has("place") && (
      following === null || anyTag(following, ["negation", "tail_tag", "time"]))) {
      out.push(word("đi", "verb"));
    }
    out.push(token);
  }
  return out;
}

const COPULA_BLOCKERS = ["verb", "going", "modal", "negation", "state", "question"];

// A person followed only by a role (with what qualifies it) or a name gets "là": "tôi cô giáo", "tên tôi Lan",
// "bố tôi bác sĩ", "tôi học sinh lớp 5".
export function _copula(tokens) {
  const start = _subjectEnd(tokens);
  const rest = tokens.slice(start);
  if (!start || !rest.length || tokens.some((token) => token.text === "là")) return tokens;
  if (anyTag(rest[0], ["role", "name"]) && rest.slice(1).every((token) => !anyTag(token, COPULA_BLOCKERS))) {
    return [...tokens.slice(0, start), word("là"), ...rest];
  }
  return tokens;
}

// One clause: [tokens, is a question, tail]. `question` (true / false) overrides the rules' decision.
export function _clause(tokens, question = null) {
  const marked = tokens.length > 0 && last(tokens).question;  // the last sign is named as a question ("… ?")
  let [out, asked, tail] = _frameQuestions(tokens);
  if (!asked) {
    out = _modalLast(out);
    [out, asked] = _finalNegation(out);
  }
  if (!asked) [out, asked] = _finalQuestion(out);
  out = out.map((token) => (token.tags.size ? token : _tag(token)));
  out = _numbers(out);
  out = _activityBeforeGo(out);
  out = _objectBeforeVerb(out);
  out = _going(out);
  out = _copula(out);
  if (question !== null && question !== undefined) asked = question;
  if (marked || (out.length && QUESTION_WORDS.has(out[0].text))) asked = true;
  return [out, asked, tail];
}

export function _render(tokens) {
  const text = pyStrip(tokens.map((token) => token.out()).join(" ").replace(SPACE_RUN, " "));
  return upperFirst(text);
}

// One sentence of recognised words (display names, in signing order) -> written Vietnamese.
export function compose(words, question = null) {
  const entries = [];
  for (const item of words) {
    const written = _written(item);
    const text = normalise(item);
    if (text) entries.push(..._entries(text, written !== text ? written : "", pyStrip(String(item)).endsWith("?")));
  }
  if (!entries.length) return "";
  let tokens = _mergeSpelling(entries).map(_tag);
  let lead = "";
  if (CLAUSE_GREETINGS.has(tokens[0].text) && tokens.length > 1) {
    const size = tokens.length > 2 && anyTag(tokens[1], ["pron2", "kin", "name"]) && _subjectEnd(tokens.slice(2))
      ? 2 : 1;
    lead = _render(tokens.slice(0, size)) + ", ";
    tokens = tokens.slice(size);
  }
  const [out, asked, tail] = _clause(tokens, question);
  let text = _render(out);
  if (tail.length) text += ", " + tail.map((token) => token.out()).join(" ");
  if (lead) text = !out.length || !out[0].spelled ? lead + lowerFirst(text) : lead + text;
  return text + (asked || tail.length ? "?" : ".");
}

export function _endsSentence(item) {
  const text = normalise(item);
  return QUESTION_WORDS.has(text) || TAIL_TAGS.has(text) || (text.endsWith(KHONG) && text !== "không")
    || text === "có không";
}

// A word's time: [start, end] (Python: a tuple), or one time for both.
function span(stamp) {
  return Array.isArray(stamp) ? stamp : [stamp, stamp];
}

// Split a stream of recognised words into sentences: where the hands rested longer than `pause` seconds (from one
// sign's end to the next one's start; `times` holds [start, end] per word, or one time), where `breaks` (one bool
// per word, decided by the caller) says a new sentence starts, and after a final question sign (not after a
// question word that opens the sentence: "ai khóc", "vì sao bạn khóc").
export function sentences(words, times = null, pause = SENTENCE_PAUSE_SECONDS, breaks = null) {
  const groups = [];
  let current = [];
  let index = 0;
  for (const item of words) {
    let rested = false;
    if (times !== null && times !== undefined && index < times.length && times[index] != null) {
      const before = times[index - 1 < 0 ? times.length - 1 : index - 1];  // Python's times[-1] at the first word
      rested = before != null && span(times[index])[0] - span(before)[1] > pause;
    }
    if (current.length && (rested || (breaks !== null && breaks !== undefined && index < breaks.length
                                      && breaks[index]))) {
      groups.push(current);
      current = [];
    }
    current.push(item);
    const opening = current.length === 1 && QUESTION_WORDS.has(normalise(item)) && normalise(item) !== "còn bạn";
    if (_endsSentence(item) && !opening) {
      groups.push(current);
      current = [];
    }
    index += 1;
  }
  if (current.length) groups.push(current);
  return groups;
}

// A stream of recognised words -> written sentences, each composed on its own.
export function composeText(words, times = null, pause = SENTENCE_PAUSE_SECONDS) {
  return sentences(words, times, pause).map((group) => compose(group)).filter((text) => text).join(" ");
}
