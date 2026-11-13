// The phone's data store (IndexedDB), with dataset.py's meaning: the built-in samples (bundle.js, read-only) plus
// what was done on this phone — new samples, renamed and deleted signs and samples — and the personal accounts.
// Everything stays on the phone; exportSync() writes the file the PC imports (mobile_sync.py).
//
// Sync rules (the bundle is the PC's truth; the phone's edits are pending until the PC has taken them in):
// - every sync file has an id; the PC records the files it imported and the bundle lists them (`imported`): an edit is
//   dropped only once a file carrying its latest state was imported (a lost or unimported file can be sent again);
// - a built-in sample's rename / deletion is an override {id, key (content key), base, sent, label?, display?,
//   deleted?}; it follows the sample by its content key when the PC renames or renumbers it (only when no other
//   sample holds the same keypoints) and is dropped when the PC deleted the sample;
// - a rename is sent with its bases (the names the PC may have: the bundle's, and every name already sent): the PC
//   applies it only while it still has one of them, so its own later renames win;
// - a phone sample is dropped when the bundle has it (its mobile id) or lists it as retired (the PC deleted or
//   refused it); a rename made meanwhile becomes an override of the PC's copy; deleting one that was ever exported
//   leaves a tombstone override "mobile:<id>" for the PC.
//
// Accounts: salted PBKDF2-SHA256 password hashes (WebCrypto) and face embeddings, never sent anywhere and never
// part of the sync file (identity.py's rule: biometric data stays on the device it was made on).

import { KEYPOINT_SIZE } from "./layout.js";
import { canonicalLabel, sameLabel } from "./naming.js";

export const FRAME_COUNT = 30;
export const PASSWORD_ITERATIONS = 200_000;
export const MIN_PASSWORD_LENGTH = 4;
export const MAX_NAME_LENGTH = 40;
const DB_NAME = "vsl-studio";
const DB_VERSION = 1;

export class DatasetError extends Error {}
export class DuplicateSampleError extends DatasetError {}
export class IdentityError extends Error {}

function request(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function done(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error("transaction aborted"));
  });
}

export function openDatabase(name = DB_NAME) {
  const req = indexedDB.open(name, DB_VERSION);
  req.onupgradeneeded = () => {
    const db = req.result;
    if (!db.objectStoreNames.contains("samples")) db.createObjectStore("samples", { keyPath: "id" });
    if (!db.objectStoreNames.contains("overrides")) db.createObjectStore("overrides", { keyPath: "id" });
    if (!db.objectStoreNames.contains("profiles")) db.createObjectStore("profiles", { keyPath: "id" });
    if (!db.objectStoreNames.contains("settings")) db.createObjectStore("settings");
  };
  return request(req);
}

function uuid() {
  return crypto.randomUUID ? crypto.randomUUID().replaceAll("-", "")
    : Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

function nowIso() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "+00:00");
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

function flatten(frames) {
  const out = new Float32Array(frames.length * KEYPOINT_SIZE);
  frames.forEach((frame, i) => out.set(frame, i * KEYPOINT_SIZE));
  return out;
}

function unflatten(buffer) {
  const values = new Float32Array(buffer);
  const frames = [];
  for (let i = 0; i < values.length / KEYPOINT_SIZE; i++) frames.push(values.subarray(i * KEYPOINT_SIZE, (i + 1) * KEYPOINT_SIZE));
  return frames;
}

function mostCommon(texts) {
  const votes = new Map();
  for (const text of texts) votes.set(text, (votes.get(text) ?? 0) + 1);
  let best = null, count = 0;
  for (const [text, n] of votes) if (n > count) { best = text; count = n; }
  return best;
}

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

export class Store {
  constructor(db) {
    this.db = db;
    this.bundle = { samples: [], labels: [], vocabulary: [] };
    this.phone = new Map(); // id -> phone sample record
    // built-in sample id -> {id, key, base: {label, display}, sent: [{label, display}], label?, display?, deleted?, at,
    // exported, files: [sync file ids carrying this state]};
    // "mobile:<id>" -> {id, mobileId, deleted: true, at, exported, files} (a phone sample deleted after an export)
    this.overrides = new Map();
  }

  static async open(bundle, name = DB_NAME) {
    const store = new Store(await openDatabase(name));
    await store.load(bundle);
    return store;
  }

  async load(bundle) {
    this.bundle = bundle;
    const tx = this.db.transaction(["samples", "overrides"], "readwrite");
    const phone = await request(tx.objectStore("samples").getAll());
    const overrides = await request(tx.objectStore("overrides").getAll());
    const synced = new Map(bundle.samples.filter((s) => s.mobile).map((s) => [s.mobile, s]));
    const retired = new Set(bundle.retired ?? []);
    const imported = new Set(bundle.imported ?? []);
    const byId = new Map(bundle.samples.map((s) => [s.id, s]));
    const byKey = new Map(); // content key -> the one sample holding it (null: twins in several signs)
    for (const sample of bundle.samples) if (sample.key) byKey.set(sample.key, byKey.has(sample.key) ? null : sample);
    const carried = [];
    this.phone.clear();
    for (const record of phone) {
      const copy = synced.get(record.mobileId);
      const own = copy ? copy.own ?? copy.display : null; // the PC copy's own display (not its sign's)
      const pending = !record.exported || !(record.files ?? []).some((file) => imported.has(file));
      if (copy && pending && (record.label !== copy.label || record.display !== own)
        && (record.synced ?? []).some((name) => name.label === copy.label && name.display === own)) {
        // Renamed here after the PC imported it (not exported yet, or in a file the PC has not imported), and the
        // PC kept a name it was sent: the rename goes on, still for this one sample (phoneSample).
        const unique = Boolean(copy.key) && bundle.samples.filter((s) => s.key === copy.key).length === 1;
        carried.push({ id: copy.id, key: copy.key, unique, base: { label: copy.label, display: own },
          sent: record.synced ?? [], phoneSample: record.mobileId, label: record.label, display: record.display,
          at: record.edited ?? nowIso(), edit: record.edit ?? 0, exported: Boolean(record.exported),
          files: record.exported ? record.files ?? [] : [] });
      }
      // The PC has it now (synced), or deleted / refused it (retired): the bundle is the truth.
      if (copy || retired.has(record.mobileId)) tx.objectStore("samples").delete(record.id);
      else this.phone.set(record.id, record);
    }
    this.overrides.clear();
    for (const override of [...overrides, ...carried]) {
      if (override.exported && (override.files ?? []).some((file) => imported.has(file))) {
        tx.objectStore("overrides").delete(override.id); // the PC took it in: the bundle shows what it made of it
        continue;
      }
      if (override.id.startsWith("mobile:")) {
        this.overrides.set(override.id, override);
        continue;
      }
      let sample = byId.get(override.id);
      if (override.key && sample?.key !== override.key) {
        // Renamed / renumbered on the PC: found again by its keypoints, only when they were this sample's alone when
        // the change was made (else a twin in another sign could be taken for it once the PC deleted it).
        sample = override.unique ? byKey.get(override.key) ?? null : null;
      }
      if (!sample) { // the PC deleted it
        tx.objectStore("overrides").delete(override.id);
        continue;
      }
      if (sample.id !== override.id) tx.objectStore("overrides").delete(override.id);
      override.id = sample.id;
      tx.objectStore("overrides").put(override);
      this.overrides.set(override.id, override);
    }
    await done(tx);
  }

  // Every sample as it is now: {id, label, display, keypoints, aspect, seconds, source, origin, created, ...}.
  samples() {
    const out = [];
    for (const sample of this.bundle.samples) {
      const override = this.overrides.get(sample.id);
      if (override?.deleted) continue;
      if (sample.mobile && this.overrides.get(`mobile:${sample.mobile}`)?.deleted) continue; // deleted here meanwhile
      out.push({ ...sample, label: override?.label ?? sample.label, display: override?.display ?? sample.display,
        origin: "builtin" });
    }
    for (const sample of this.phone.values()) {
      out.push({ id: sample.id, label: sample.label, display: sample.display, keypoints: unflatten(sample.keypoints),
        aspect: sample.aspect, seconds: sample.seconds, source: sample.metadata?.source ?? "webcam", origin: "phone",
        created: sample.created, preview: sample.preview ?? null, key: sample.hash, metadata: sample.metadata ?? {} });
    }
    return out;
  }

  // Signs with their samples, sorted by display name: [{label, display, samples: [...]}].
  signs() {
    const groups = new Map();
    for (const sample of this.samples()) {
      if (!groups.has(sample.label)) groups.set(sample.label, []);
      groups.get(sample.label).push(sample);
    }
    return [...groups].map(([label, samples]) => ({ label, display: mostCommon(samples.map((s) => s.display)), samples }))
      .sort((a, b) => a.display.localeCompare(b.display, "vi"));
  }

  displayOf(label) {
    return this.signs().find((sign) => sameLabel(sign.label, label))?.display ?? label.replaceAll("_", " ");
  }

  // dataset.save_sample: FRAME_COUNT x KEYPOINT_SIZE finite values; refuses an identical sample in the sign.
  // revive: the user confirmed saving a segment they deleted before (the PC then saves it despite its tombstone).
  async saveSample(text, frames, { display = null, metadata = {}, preview = null, aspect, seconds, revive = false } = {}) {
    if (frames.length !== FRAME_COUNT || frames.some((f) => f.length !== KEYPOINT_SIZE)) {
      throw new DatasetError(`Mẫu cần đúng ${FRAME_COUNT} frame x ${KEYPOINT_SIZE} tọa độ.`);
    }
    const values = flatten(frames);
    if (!values.every(Number.isFinite)) throw new DatasetError("Tọa độ chứa giá trị không hợp lệ (NaN/Inf).");
    const label = canonicalLabel(text);
    const shown = (display ?? String(text)).trim();
    const hash = await sha256Hex(values.buffer);
    for (const sample of this.phone.values()) {
      if (sample.label === label && sample.hash === hash) {
        throw new DuplicateSampleError(`Ký hiệu “${shown}” đã có một mẫu giống hệt.`);
      }
    }
    const id = `m:${uuid()}`;
    const record = { id, label, display: shown, keypoints: values.buffer, aspect, seconds, hash, created: nowIso(),
      mobileId: id.slice(2), metadata: { ...metadata }, preview, exported: false, ...(revive ? { revive: true } : {}) };
    if (revive && metadata.media_id) { // saved again on purpose: this phone no longer counts the segment as deleted
      const { provenanceKeys } = await import("./collectbatch.js");
      const keys = new Set((await provenanceKeys(metadata)).filter((key) => /^(seg|src):/.test(key)));
      const local = await this.setting("deletedProvenance", []);
      if (local.some((key) => keys.has(key))) await this.setSetting("deletedProvenance", local.filter((key) => !keys.has(key)));
      const pending = [...this.overrides.values()].filter((override) => (override.prov ?? []).some((key) => keys.has(key)));
      if (pending.length) await this._override(pending.map((override) => ({ ...override, prov: override.prov.filter((key) => !keys.has(key)) })));
    }
    const tx = this.db.transaction("samples", "readwrite");
    tx.objectStore("samples").put(record);
    await done(tx);
    this.phone.set(id, record);
    return { label, id, count: this.samples().filter((s) => s.label === label).length };
  }

  async _override(entries) {
    const tx = this.db.transaction("overrides", "readwrite");
    for (const entry of entries) tx.objectStore("overrides").put(entry);
    await done(tx);
    for (const entry of entries) this.overrides.set(entry.id, entry);
  }

  async _putPhone(records) {
    const tx = this.db.transaction("samples", "readwrite");
    for (const record of records) tx.objectStore("samples").put(record);
    await done(tx);
    for (const record of records) this.phone.set(record.id, record);
  }

  // The next edit number of this phone (persisted): the PC applies an older edit of a sample never after a newer one.
  async _nextEdit() {
    this.editCounter = (this.editCounter ?? (await this.setting("editCounter", 0))) + 1;
    await this.setSetting("editCounter", this.editCounter);
    return this.editCounter;
  }

  // This phone's random id (sync files carry it with the edit numbers; never anything about the person).
  async phoneId() {
    let id = await this.setting("phoneId", null);
    if (!id) {
      id = uuid();
      await this.setSetting("phoneId", id);
    }
    return id;
  }

  // An override of built-in sample `sample` with `changes`, remembering its content key, what the PC called it and
  // what was already sent. Only files carrying this new state can retire it (files: []).
  _builtinOverride(sample, changes, at) {
    const original = this.bundle.samples.find((s) => s.id === sample.id);
    const previous = this.overrides.get(sample.id);
    const unique = Boolean(original?.key) && this.bundle.samples.filter((s) => s.key === original.key).length === 1;
    return { id: sample.id, key: original?.key ?? null, unique, base: { label: original?.label, display: original?.display },
      sent: [], ...(previous ?? {}), ...changes, at, exported: false, files: [] };
  }

  // dataset.rename_label: rename a sign (or only its display text), merging into an existing sign of that name.
  async renameLabel(label, newText) {
    const display = String(newText).trim();
    const target = canonicalLabel(display);
    const at = nowIso();
    const edit = await this._nextEdit();
    const builtins = this.samples().filter((s) => s.label === label && s.origin === "builtin");
    await this._override(builtins.map((s) => this._builtinOverride(s, { label: target, display, edit }, at)));
    // Every sample of the target sign shows the new name too (rename_label rewrites them all).
    const targetBuiltins = this.samples().filter((s) => s.label === target && s.origin === "builtin" && s.display !== display);
    await this._override(targetBuiltins.map((s) => this._builtinOverride(s, { label: target, display, edit }, at)));
    // A new state: only a file carrying it can show the PC took it in (files: []).
    const phone = [...this.phone.values()].filter((s) => s.label === label || s.label === target)
      .map((s) => ({ ...s, label: target, display, edited: at, edit, exported: false, files: [] }));
    await this._putPhone(phone);
    return target;
  }

  // dataset.remove_samples / remove_label. A phone sample the PC may already have (exported) leaves a tombstone.
  async removeSamples(ids) {
    const at = nowIso();
    const edit = await this._nextEdit();
    const builtins = this.bundle.samples.filter((s) => ids.includes(s.id));
    const phone = ids.filter((id) => this.phone.has(id));
    // Video samples deleted here keep their segment / source keys (prov) so the collector knows the segment was
    // deleted: on the deletion waiting for the PC (gone with it once the PC decided: the bundle then tells), or, for a
    // sample the PC never had, in the setting "deletedProvenance" (until the segment is saved again on purpose).
    const { provenanceKeys } = await import("./collectbatch.js");
    const keysOf = async (record) => (record.metadata?.media_id
      ? (await provenanceKeys(record.metadata)).filter((key) => /^(seg|src):/.test(key)) : []);
    const records = phone.map((id) => this.phone.get(id));
    const sent = records.filter((record) => record.exported || record.everExported);
    const local = [];
    for (const record of records) if (!sent.includes(record)) local.push(...(await keysOf(record)));
    if (local.length) await this.setSetting("deletedProvenance", [...new Set([...(await this.setting("deletedProvenance", [])), ...local])]);
    const tombstones = [];
    for (const record of sent) {
      tombstones.push({ id: `mobile:${record.mobileId}`, mobileId: record.mobileId, deleted: true, at, exported: false,
        files: [], prov: await keysOf(record) });
    }
    await this._override([...builtins.map((s) => this._builtinOverride(s, { deleted: true, edit, prov: s.prov ?? [] }, at)),
      ...tombstones]);
    const tx = this.db.transaction("samples", "readwrite");
    for (const id of phone) tx.objectStore("samples").delete(id);
    await done(tx);
    for (const id of phone) this.phone.delete(id);
  }

  // seg: / src: keys of the video segments deleted on this phone: deletions still waiting for the PC, and samples the
  // PC never had (the PC's own deletions come with the bundle: segments / sources).
  async deletedProvenance() {
    const keys = new Set(await this.setting("deletedProvenance", []));
    for (const override of this.overrides.values()) if (override.deleted) for (const key of override.prov ?? []) keys.add(key);
    return [...keys];
  }

  async removeLabel(label) {
    await this.removeSamples(this.samples().filter((s) => s.label === label).map((s) => s.id));
  }

  // What the PC has not got yet: {count, file: Blob, id, included} (JSON: new samples with keypoints and preview,
  // renamed and deleted built-in samples by their PC id and content key, phone samples deleted after an earlier export
  // by their mobile id; renames with their bases). Pass the result to markExported() once the file is saved. `all`:
  // everything still pending, exported or not (the file was lost before the PC imported it).
  async exportSync(appVersion = "", { all = false } = {}) {
    const id = uuid();
    const included = { samples: [], overrides: [] };
    const bases = (names) => names.map((name) => [name.label, name.display]);
    const samples = [];
    for (const sample of this.phone.values()) {
      if (sample.exported && !all) continue;
      included.samples.push(sample.id);
      const sent = sample.synced ?? [];
      samples.push({
        mobile_id: sample.mobileId, label: sample.label, display: sample.display, created: sample.created,
        edited: sample.edited ?? null, metadata: sample.metadata, keypoints: toBase64(sample.keypoints),
        preview: sample.preview ? toBase64(await sample.preview.arrayBuffer()) : null,
        // The names the PC may have for it (sent before): a rename applies only while the PC still has one of them.
        ...(sent.length ? { base_label: sent[0].label, base_display: sent[0].display, bases: bases(sent) } : {}),
        edit: sample.edit ?? 0,
        ...(sample.revive ? { revive: true } : {}),
      });
    }
    const renamed = [], deleted = [];
    for (const override of this.overrides.values()) {
      if (override.exported && !all) continue;
      if (override.mobileId) {
        included.overrides.push(override.id);
        deleted.push({ mobile_id: override.mobileId, at: override.at, ...(override.prov?.length ? { prov: override.prov } : {}) });
        continue;
      }
      const sample = this.bundle.samples.find((s) => s.id === override.id);
      if (!sample) continue;
      included.overrides.push(override.id);
      if (override.phoneSample && !override.deleted) { // a phone sample renamed after the PC imported it
        const names = [override.base, ...(override.sent ?? [])].filter(Boolean);
        samples.push({ mobile_id: override.phoneSample, label: override.label, display: override.display,
          edited: override.at, base_label: names[0]?.label, base_display: names[0]?.display, bases: bases(names),
          edit: override.edit ?? 0 });
        continue;
      }
      // The content key the override was made for (the PC reuses sample numbers), else the bundle's.
      const key = override.key ?? sample.key;
      const unique = Boolean(override.unique); // the PC may look it up by its keypoints
      if (override.deleted) {
        deleted.push({ id: sample.id, key, unique, at: override.at });
      } else {
        const names = [override.base ?? { label: sample.label, display: sample.display }, ...(override.sent ?? [])];
        renamed.push({ id: sample.id, key, unique, label: override.label, display: override.display, at: override.at,
          base_label: names[0].label, base_display: names[0].display, bases: bases(names), edit: override.edit ?? 0 });
      }
    }
    const document = { format: "vsl-mobile-sync", version: 2, id, phone: await this.phoneId(), created: nowIso(), app: appVersion,
      bundle: this.bundle.sha256 ?? null, samples, renamed, deleted };
    return { count: samples.length + renamed.length + deleted.length, id, included,
      file: new Blob([JSON.stringify(document)], { type: "application/json" }) };
  }

  // The sync file `exported` (exportSync's result) was saved: its changes wait for the PC, which records the file as
  // imported; the names sent become bases the PC may now have.
  async markExported(exported) {
    const name = (item) => ({ label: item.label, display: item.display });
    const known = (list, item) => list.some((entry) => entry.label === item.label && entry.display === item.display);
    await this._putPhone(exported.included.samples.filter((id) => this.phone.has(id)).map((id) => {
      const record = this.phone.get(id);
      const synced = record.synced ?? [];
      return { ...record, exported: true, everExported: true, files: [...(record.files ?? []), exported.id],
        synced: known(synced, record) ? synced : [...synced, name(record)] };
    }));
    await this._override(exported.included.overrides.filter((id) => this.overrides.has(id)).map((id) => {
      const override = this.overrides.get(id);
      const sent = override.sent ?? [];
      const update = override.mobileId || override.deleted || known(sent, override) ? sent : [...sent, name(override)];
      return { ...override, exported: true, files: [...(override.files ?? []), exported.id], sent: update };
    }));
  }

  // Changes exported but not taken in by the PC yet (no imported file carries them): they can be exported again.
  pendingChanges() {
    return [...this.phone.values()].filter((s) => s.exported).length
      + [...this.overrides.values()].filter((o) => o.exported).length;
  }

  // ---------------------------------------------------------------- accounts (identity.py)

  async profiles() {
    const tx = this.db.transaction("profiles", "readonly");
    const list = await request(tx.objectStore("profiles").getAll());
    return list.map((p) => ({ ...p, embeddings: new Float32Array(p.embeddings) }))
      .sort((a, b) => a.name.localeCompare(b.name, "vi"));
  }

  async validateNewAccount(name, password, confirm) {
    const clean = String(name).trim().normalize("NFC");
    if (!clean) throw new IdentityError("Hãy nhập tên tài khoản.");
    if ([...clean].length > MAX_NAME_LENGTH) throw new IdentityError(`Tên tài khoản tối đa ${MAX_NAME_LENGTH} ký tự.`);
    const same = (await this.profiles()).some((p) => p.name.normalize("NFC").toLowerCase() === clean.toLowerCase());
    if (same) throw new IdentityError("Tên tài khoản này đã tồn tại.");
    if (password.length < MIN_PASSWORD_LENGTH) throw new IdentityError(`Mật khẩu cần ít nhất ${MIN_PASSWORD_LENGTH} ký tự.`);
    if (password !== confirm) throw new IdentityError("Mật khẩu nhập lại không khớp.");
    return clean;
  }

  async createProfile(name, password, enrollment) {
    const clean = await this.validateNewAccount(name, password, password);
    if (!enrollment?.done) throw new IdentityError("Chưa chụp đủ khuôn mặt để tạo tài khoản.");
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const hash = await passwordHash(password, salt, PASSWORD_ITERATIONS);
    const embeddings = new Float32Array(enrollment.embeddings.length * 128);
    enrollment.embeddings.forEach((e, i) => embeddings.set(e, i * 128));
    const profile = { id: uuid(), name: clean, created: nowIso(), salt: Array.from(salt), iterations: PASSWORD_ITERATIONS,
      hash, embeddings: embeddings.buffer, avatar: enrollment.avatar ?? null };
    const tx = this.db.transaction("profiles", "readwrite");
    tx.objectStore("profiles").put(profile);
    await done(tx);
    return { ...profile, embeddings };
  }

  async verifyPassword(profile, password) {
    const stored = (await this.profiles()).find((p) => p.id === profile.id);
    if (!stored) return false;
    const actual = await passwordHash(password, Uint8Array.from(stored.salt), stored.iterations);
    return actual === stored.hash;
  }

  async deleteProfile(profile, password) {
    if (!(await this.verifyPassword(profile, password))) throw new IdentityError("Sai mật khẩu.");
    const tx = this.db.transaction("profiles", "readwrite");
    tx.objectStore("profiles").delete(profile.id);
    await done(tx);
  }

  // ---------------------------------------------------------------- settings

  async setting(key, fallback = null) {
    const tx = this.db.transaction("settings", "readonly");
    const value = await request(tx.objectStore("settings").get(key));
    return value === undefined ? fallback : value;
  }

  async setSetting(key, value) {
    const tx = this.db.transaction("settings", "readwrite");
    tx.objectStore("settings").put(value, key);
    await done(tx);
  }
}

export async function passwordHash(password, salt, iterations) {
  const material = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations }, material, 256);
  return Array.from(new Uint8Array(bits), (b) => b.toString(16).padStart(2, "0")).join("");
}
