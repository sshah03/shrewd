// Run a model exported by shrewd. No dependencies: Node 18+, Deno, Bun or a browser.
//
//   import { Model } from "./shrewd_model.js";
//   const model = await Model.load("path/to/export");      // Node, Deno, Bun
//   const model = await Model.fetch("/models/tickets/");    // a browser
//   model.predict("I was charged twice");                   // a classifier: "billing"
//   model.decide("I was charged twice");                    // a panel: {question: answer}

const EPS = 1e-12;
const PY_SPACE = "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";
const BERT_SPACE = "\t\n\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";
const ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const WORD = /[\p{L}\p{N}_]/u, CONTROL = /^[\p{Cc}\p{Cf}\p{Co}]$/u, MN = /^\p{Mn}$/u, PUNCT = /^\p{P}$/u;

function float16(bits) {
  const sign = bits & 0x8000 ? -1 : 1, exp = (bits >> 10) & 0x1f, frac = bits & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export function readSafetensors(buffer) {
  const view = new DataView(buffer), size = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, size)));
  const out = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const start = 8 + size + info.data_offsets[0], end = 8 + size + info.data_offsets[1];
    const bytes = new DataView(buffer, start, end - start);
    let array;
    if (info.dtype === "F32") { array = new Float32Array(buffer.slice(start, end)); }
    else if (info.dtype === "F64") { array = new Float64Array(buffer.slice(start, end)); }
    else if (info.dtype === "F16") {
      array = new Float32Array((end - start) / 2);
      for (let i = 0; i < array.length; i++) array[i] = float16(bytes.getUint16(i * 2, true));
    } else throw new Error(`unsupported dtype ${info.dtype}`);
    array.shape = info.shape;
    out[name] = array;
  }
  return out;
}

function pySplit(text) {
  const out = []; let current = "";
  for (const c of text) {
    if (PY_SPACE.includes(c)) { if (current) { out.push(current); current = ""; } }
    else current += c;
  }
  if (current) out.push(current);
  return out;
}

function tokens(text) {
  const out = []; let current = "", n = 0;
  for (const c of text) {
    if (WORD.test(c)) { current += c; n++; }
    else { if (n >= 2) out.push(current); current = ""; n = 0; }
  }
  if (n >= 2) out.push(current);
  return out;
}

function wordTerms(text, low, high) {
  const t = tokens(text), out = [];
  for (let n = low; n <= high; n++) for (let i = 0; i + n <= t.length; i++) out.push(t.slice(i, i + n).join(" "));
  return out;
}

function charWbTerms(text, low, high) {
  const out = [];
  for (const word of pySplit(text)) {
    const w = [" ", ...word, " "];
    for (let n = low; n <= high; n++) {
      for (let i = 0; i < Math.max(1, w.length - n + 1); i++) out.push(w.slice(i, i + n).join(""));
      if (w.length <= n) break;
    }
  }
  return out;
}

class WordPiece {
  constructor(spec) {
    this.vocab = new Map(spec.vocab.map((t, i) => [t, i]));
    this.unk = spec.unk; this.prefix = spec.prefix; this.maxChars = spec.max_chars; this.cjk = spec.cjk;
    this.overrides = new Map(Object.entries(spec.overrides).map(([k, v]) => [Number(k), v]));
    this.punctAdd = new Set(spec.punct_add); this.punctRemove = new Set(spec.punct_remove);
    this.added = Object.entries(spec.added || {}).sort((a, b) => b[0].length - a[0].length);
  }

  char(ch) {
    const c = ch.codePointAt(0);
    if (this.overrides.has(c)) return this.overrides.get(c);
    if (c === 0 || c === 0xfffd || (!"\t\n\r".includes(ch) && CONTROL.test(ch))) return "";
    if (BERT_SPACE.includes(ch)) return " ";
    if (this.cjk.some(([a, b]) => a <= c && c <= b)) return ` ${ch} `;
    let out = "";
    for (const x of ch.normalize("NFD")) if (!MN.test(x)) out += x.toLowerCase();
    return out;
  }

  punct(ch) {
    const c = ch.codePointAt(0);
    if (this.punctAdd.has(c)) return true;
    if (this.punctRemove.has(c)) return false;
    return ASCII_PUNCT.includes(ch) || PUNCT.test(ch);
  }

  words(text) {
    let normal = "";
    for (const ch of text) normal += this.char(ch);
    const out = []; let current = "";
    for (const ch of normal) {
      if (BERT_SPACE.includes(ch)) { if (current) out.push(current); current = ""; }
      else if (this.punct(ch)) { if (current) out.push(current); out.push(ch); current = ""; }
      else current += ch;
    }
    if (current) out.push(current);
    return out;
  }

  pieces(word) {
    const chars = Array.from(word);
    if (chars.length > this.maxChars) return [this.unk];
    const ids = []; let start = 0;
    while (start < chars.length) {
      let end = chars.length, found;
      while (start < end) {
        const piece = (start === 0 ? "" : this.prefix) + chars.slice(start, end).join("");
        if (this.vocab.has(piece)) { found = this.vocab.get(piece); break; }
        end--;
      }
      if (found === undefined) return [this.unk];
      ids.push(found); start = end;
    }
    return ids;
  }

  encode(text) {
    const ids = []; let plain = "", i = 0;
    const flush = () => { for (const w of this.words(plain)) ids.push(...this.pieces(w)); plain = ""; };
    while (i < text.length) {
      const hit = this.added.find(([content]) => text.startsWith(content, i));
      if (hit) { flush(); ids.push(hit[1]); i += hit[0].length; }
      else { const ch = String.fromCodePoint(text.codePointAt(i)); plain += ch; i += ch.length; }
    }
    flush();
    return ids;
  }
}

const sum = (a) => a.reduce((s, v) => s + v, 0);
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), t = sum(e); return e.map((v) => v / t); };
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const clip = (v, low, high) => Math.min(Math.max(v, low), high);
const argmax = (p) => { let best = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i; return best; };

function interp(x, xs, ys) {
  const n = xs.length;
  if (x < xs[0]) return ys[0];
  if (x > xs[n - 1]) return ys[n - 1];
  let low = 0, high = n - 1;
  while (low < high) { const mid = (low + high + 1) >> 1; if (xs[mid] <= x) low = mid; else high = mid - 1; }
  if (low === n - 1 || xs[low] === x) return ys[low];
  return ys[low] + (ys[low + 1] - ys[low]) * (x - xs[low]) / (xs[low + 1] - xs[low]);
}

function rescaleTop(p, newTop) {
  const win = argmax(p), rest = sum(p) - p[win];
  let other = 0; p.forEach((v, i) => { if (i !== win && v > other) other = v; });
  const floor = rest > EPS ? other / Math.max(rest + other, EPS) : 0;
  newTop = Math.max(newTop, Math.min(floor + 1e-9, 1 - EPS));
  const scale = rest > EPS ? (1 - newTop) / Math.max(rest, EPS) : 0;
  const out = p.map((v) => v * scale); out[win] = newTop;
  const total = Math.max(sum(out), EPS);
  return out.map((v) => v / total);
}

function calibrate(p, cal) {
  if (!cal) return p;
  p = p.map((v) => clip(v, EPS, 1));
  const prm = cal.params;
  if (cal.method === "none") { const t = sum(p); return p.map((v) => v / t); }
  if (cal.method === "temperature") return softmax(p.map((v) => Math.log(v) / prm.temperature));
  if (cal.method === "vector" || cal.method === "matrix") {
    const l = p.map(Math.log);
    return softmax(prm.W.map((row, i) => row.reduce((s, w, j) => s + w * l[j], prm.b[i])));
  }
  const binary = Boolean(prm.binary) && p.length === 2, score = binary ? p[1] : Math.max(...p);
  let mapped;
  if (cal.method === "platt") mapped = sigmoid(prm.a * Math.log(clip(score, EPS, 1 - 1e-9) / clip(1 - score, EPS, 1)) + prm.b);
  else if (cal.method === "isotonic") mapped = interp(score, prm.x, prm.y);
  else throw new Error(`unknown calibration ${cal.method}`);
  mapped = clip(mapped, EPS, 1 - EPS);
  return binary ? [1 - mapped, mapped] : rescaleTop(p, mapped);
}

function shape(p) {
  if (p.length < 2) return 1;
  const q = p.map((v) => clip(v, 1e-12, 1));
  return 1 - -sum(q.map((v) => v * Math.log(v))) / Math.log(p.length);
}

export class Model {
  constructor(spec, weights) {
    if (spec.format !== "shrewd-model") throw new Error("not a shrewd model export");
    this.spec = spec; this.weights = weights; this.task = spec.task; this.outputs = spec.outputs;
    this.labels = this.task === "classifier" ? this.outputs[0].options : null;
    this.threshold = spec.threshold || null;
    const f = spec.features;
    if (f.type === "tfidf") {
      this.blocks = f.blocks; this.width = weights.idf.length;
      this.index = f.blocks.map((b) => new Map(b.vocab.map((t, i) => [t, i])));
    } else {
      this.wordpiece = new WordPiece(f.tokenizer); this.width = weights.embedding.shape[1];
    }
  }

  static async load(folder) {
    const { readFile } = await import("node:fs/promises");
    const spec = JSON.parse(await readFile(`${folder}/model.json`, "utf8"));
    const raw = await readFile(`${folder}/weights.safetensors`);
    return new Model(spec, readSafetensors(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));
  }

  static async fetch(base) {
    const root = base.endsWith("/") ? base : `${base}/`;
    const [spec, raw] = await Promise.all([
      fetch(`${root}model.json`).then((r) => r.json()),
      fetch(`${root}weights.safetensors`).then((r) => r.arrayBuffer()),
    ]);
    return new Model(spec, readSafetensors(raw));
  }

  features(text) {
    const x = new Float64Array(this.width);
    if (this.wordpiece) {
      const f = this.spec.features, d = this.width, table = this.weights.embedding;
      const ids = this.wordpiece.encode(Array.from(text).slice(0, f.max_input_chars).join(""))
        .filter((i) => i !== f.drop_token).slice(0, f.max_tokens);
      for (const id of ids) for (let j = 0; j < d; j++) x[j] += table[id * d + j];
      if (ids.length) for (let j = 0; j < d; j++) x[j] /= ids.length;
      if (f.normalize) { let sq = 0; for (let j = 0; j < d; j++) sq += x[j] * x[j]; const n = Math.sqrt(sq) + 1e-32; for (let j = 0; j < d; j++) x[j] /= n; }
      return x;
    }
    const idf = this.weights.idf;
    this.blocks.forEach((b, k) => {
      const doc = b.lowercase ? text.toLowerCase() : text;
      for (const t of (b.analyzer === "word" ? wordTerms : charWbTerms)(doc, b.ngram[0], b.ngram[1])) {
        const i = this.index[k].get(t);
        if (i !== undefined) x[b.offset + i] += 1;
      }
      let sq = 0;
      for (let i = b.offset; i < b.offset + b.size; i++) { x[i] *= idf[i]; sq += x[i] * x[i]; }
      if (sq > 0) { const n = Math.sqrt(sq); for (let i = b.offset; i < b.offset + b.size; i++) x[i] /= n; }
    });
    return x;
  }

  head(output, x, nonzero) {
    const k = output.options.length, h = output.head;
    if (h.type === "constant") { const p = Array(k).fill(1e-6); p[h.index] = 1 - 1e-6 * (k - 1); return p; }
    const coef = this.weights[`${output.id}.coef`], icpt = this.weights[`${output.id}.intercept`];
    const z = Array.from(icpt, (b, c) => { let s = b; for (const i of nonzero) s += coef[c * this.width + i] * x[i]; return s; });
    const q = h.binary ? [1 - sigmoid(z[0]), sigmoid(z[0])] : softmax(z);
    if (!h.classes) return q;
    const p = Array(k).fill(1e-6); h.classes.forEach((c, j) => { p[c] = q[j]; });
    const t = sum(p); return p.map((v) => v / t);
  }

  probabilities(text) {
    const x = this.features(text), nonzero = [];
    for (let i = 0; i < x.length; i++) if (x[i]) nonzero.push(i);
    return Object.fromEntries(this.outputs.map((o) => [o.id, calibrate(this.head(o, x, nonzero), o.calibration)]));
  }

  predictProba(texts) {
    const one = (t) => (this.task === "classifier" ? this.probabilities(t).label : this.probabilities(t));
    return Array.isArray(texts) ? texts.map(one) : one(texts);
  }

  predict(texts) {
    if (this.task !== "classifier") throw new Error("this is a decision panel; use decide()");
    const one = (t) => this.labels[argmax(this.probabilities(t).label)];
    return Array.isArray(texts) ? texts.map(one) : one(texts);
  }

  decide(texts) {
    if (this.task !== "panel") throw new Error("this is a classifier; use predict()");
    const one = (text) => {
      const probs = this.probabilities(text), answers = {};
      for (const o of this.outputs) {
        const p = probs[o.id], probabilities = Object.fromEntries(o.options.map((name, i) => [name, p[i]]));
        if (o.type === "choice") answers[o.id] = { choice: o.options[argmax(p)], confidence: shape(p), probabilities };
        else if (o.type === "score") answers[o.id] = { score: p.reduce((s, v, i) => s + v * i, 0), confidence: shape(p), probabilities };
        else answers[o.id] = { noul: p[o.options.indexOf("yes")], probabilities };
      }
      return answers;
    };
    return Array.isArray(texts) ? texts.map(one) : one(texts);
  }

  confident(texts) {
    if (!this.threshold) throw new Error("this export has no threshold");
    const one = (text) => {
      const probs = this.probabilities(text), keys = this.task === "classifier" ? ["label"] : this.threshold.questions;
      const top = Math.min(...keys.map((k) => Math.max(...probs[k])));
      return this.threshold.value !== null && top >= this.threshold.value;
    };
    return Array.isArray(texts) ? texts.map(one) : one(texts);
  }
}
