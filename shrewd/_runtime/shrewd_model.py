"""Run a model exported by shrewd. Needs numpy and nothing else.

    from shrewd_model import Model

    model = Model("path/to/export")
    model.predict(["I was charged twice"])      # a classifier: ["billing"]
    model.decide("I was charged twice")         # a panel: {question: answer}
"""

import json
import re
import struct
import unicodedata
from pathlib import Path

import numpy as np

__all__ = ["Model"]

EPS = 1e-12
_DTYPES = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
_RUNS = re.compile(r"\s\s+")
_ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_SPACE = set("\t\n\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
             "\u200a\u2028\u2029\u202f\u205f\u3000")


def read_safetensors(path):
    raw = Path(path).read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header, body = json.loads(raw[8:8 + size]), raw[8 + size:]
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        array = np.frombuffer(body[start:end], dtype=_DTYPES[info["dtype"]])
        out[name] = array.reshape(info["shape"]).astype(np.float64)
    return out


def _word_terms(text, low, high):
    tokens = _TOKEN.findall(text)
    return [" ".join(tokens[i:i + n]) for n in range(low, high + 1)
            for i in range(len(tokens) - n + 1)]


def _char_wb_terms(text, low, high):
    out = []
    for word in _RUNS.sub(" ", text).split():
        word = " " + word + " "
        for n in range(low, high + 1):
            out += [word[i:i + n] for i in range(max(1, len(word) - n + 1))]
            if len(word) <= n:
                break
    return out


class _WordPiece:
    def __init__(self, spec):
        self.vocab = {token: i for i, token in enumerate(spec["vocab"])}
        self.unk, self.prefix, self.max_chars = spec["unk"], spec["prefix"], spec["max_chars"]
        self.cjk = [tuple(r) for r in spec["cjk"]]
        self.overrides = {int(k): v for k, v in spec["overrides"].items()}
        self.punct_add, self.punct_remove = set(spec["punct_add"]), set(spec["punct_remove"])
        self.added = sorted(spec.get("added", {}).items(), key=lambda kv: -len(kv[0]))

    def _char(self, ch):
        code = ord(ch)
        if code in self.overrides:
            return self.overrides[code]
        category = unicodedata.category(ch)
        if code in (0, 0xFFFD) or (ch not in "\t\n\r" and category in ("Cc", "Cf", "Co")):
            return ""
        if ch in _SPACE:
            return " "
        if any(a <= code <= b for a, b in self.cjk):
            return " " + ch + " "
        return "".join(x.lower() for x in unicodedata.normalize("NFD", ch)
                       if unicodedata.category(x) != "Mn")

    def _punct(self, ch):
        code = ord(ch)
        if code in self.punct_add:
            return True
        if code in self.punct_remove:
            return False
        return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith("P")

    def _words(self, text):
        out, current = [], ""
        for ch in "".join(self._char(ch) for ch in text):
            if ch in _SPACE:
                if current:
                    out.append(current)
                current = ""
            elif self._punct(ch):
                if current:
                    out.append(current)
                out.append(ch)
                current = ""
            else:
                current += ch
        if current:
            out.append(current)
        return out

    def _pieces(self, word):
        if len(word) > self.max_chars:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def _segments(self, text):
        i, plain = 0, ""
        while i < len(text):
            for content, token_id in self.added:
                if text.startswith(content, i):
                    if plain:
                        yield plain, None
                    yield content, token_id
                    plain, i = "", i + len(content)
                    break
            else:
                plain += text[i]
                i += 1
        if plain:
            yield plain, None

    def encode(self, text):
        ids = []
        for segment, token_id in self._segments(text):
            if token_id is not None:
                ids.append(token_id)
            else:
                ids += [i for word in self._words(segment) for i in self._pieces(word)]
        return ids


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rescale_top(p, new_top):
    win = int(p.argmax())
    rest = p.sum() - p[win]
    other = max((v for i, v in enumerate(p) if i != win), default=0.0)
    floor = other / max(rest + other, EPS) if rest > EPS else 0.0
    new_top = max(new_top, min(floor + 1e-9, 1.0 - EPS))
    out = p * ((1.0 - new_top) / max(rest, EPS) if rest > EPS else 0.0)
    out[win] = new_top
    return out / max(out.sum(), EPS)


def _calibrate(p, calibration):
    if calibration is None:
        return p
    p = np.clip(p, EPS, 1.0)
    method, params = calibration["method"], calibration["params"]
    if method == "none":
        return p / p.sum()
    if method == "temperature":
        return _softmax(np.log(p) / params["temperature"])
    if method in ("vector", "matrix"):
        return _softmax(np.asarray(params["W"]) @ np.log(p) + np.asarray(params["b"]))
    binary = bool(params.get("binary")) and len(p) == 2
    score = p[1] if binary else p.max()
    if method == "platt":
        x = np.log(np.clip(score, EPS, 1 - 1e-9) / np.clip(1 - score, EPS, 1.0))
        mapped = _sigmoid(params["a"] * x + params["b"])
    elif method == "isotonic":
        mapped = np.interp(score, params["x"], params["y"])
    else:
        raise ValueError(f"unknown calibration {method!r}")
    mapped = float(np.clip(mapped, EPS, 1.0 - EPS))
    return np.array([1.0 - mapped, mapped]) if binary else _rescale_top(p, mapped)


def _shape(p):
    p = np.clip(p, 1e-12, 1.0)
    if len(p) < 2:
        return 1.0
    return float(1.0 - float(-(p * np.log(p)).sum()) / np.log(len(p)))


class Model:
    """A classifier or decision panel exported by shrewd."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        if self.spec.get("format") != "shrewd-model":
            raise ValueError(f"{folder} is not a shrewd model export")
        self.task = self.spec["task"]
        self.weights = read_safetensors(folder / "weights.safetensors")
        features = self.spec["features"]
        if features["type"] == "tfidf":
            self._blocks = features["blocks"]
            self._index = [{t: i for i, t in enumerate(b["vocab"])} for b in self._blocks]
        else:
            self._wordpiece = _WordPiece(features["tokenizer"])
        self.outputs = self.spec["outputs"]
        self.labels = self.outputs[0]["options"] if self.task == "classifier" else None
        self.threshold = self.spec.get("threshold")

    def _tfidf(self, text):
        idf = self.weights["idf"]
        x = np.zeros(idf.shape[0])
        for block, index in zip(self._blocks, self._index, strict=True):
            doc = text.lower() if block["lowercase"] else text
            terms = (_word_terms if block["analyzer"] == "word" else _char_wb_terms)(
                doc, *block["ngram"])
            start, size = block["offset"], block["size"]
            part = np.zeros(size)
            for term in terms:
                i = index.get(term)
                if i is not None:
                    part[i] += 1
            part *= idf[start:start + size]
            norm = np.sqrt(part @ part)
            x[start:start + size] = part / norm if norm > 0 else part
        return x

    def _static(self, text):
        f = self.spec["features"]
        ids = [i for i in self._wordpiece.encode(text[:f["max_input_chars"]])
               if i != f["drop_token"]][:f["max_tokens"]]
        table = self.weights["embedding"]
        x = table[ids].mean(axis=0) if ids else np.zeros(table.shape[1])
        return x / (np.sqrt(x @ x) + 1e-32) if f["normalize"] else x

    def _head(self, output, x):
        k, head = len(output["options"]), output["head"]
        if head["type"] == "constant":
            p = np.full(k, 1e-6)
            p[head["index"]] = 1.0 - 1e-6 * (k - 1)
            return p
        z = self.weights[f"{output['id']}.coef"] @ x + self.weights[f"{output['id']}.intercept"]
        q = np.array([1 - _sigmoid(z[0]), _sigmoid(z[0])]) if head["binary"] else _softmax(z)
        if head["classes"] is None:
            return q
        p = np.full(k, 1e-6)
        for j, c in enumerate(head["classes"]):
            p[c] = q[j]
        return p / p.sum()

    def _one(self, text):
        x = self._tfidf(text) if self.spec["features"]["type"] == "tfidf" else self._static(text)
        return {o["id"]: _calibrate(self._head(o, x), o["calibration"]) for o in self.outputs}

    def predict_proba(self, texts):
        """A classifier: (texts x labels). A panel: {question: (texts x options)}."""
        rows = [self._one(t) for t in ([texts] if isinstance(texts, str) else texts)]
        if self.task == "classifier":
            return np.array([r["label"] for r in rows]).reshape(len(rows), -1)
        return {o["id"]: np.array([r[o["id"]] for r in rows]) for o in self.outputs}

    def predict(self, texts):
        """A classifier's labels, one per text."""
        if self.task != "classifier":
            raise ValueError("this is a decision panel; use decide()")
        single = isinstance(texts, str)
        labels = [self.labels[i] for i in self.predict_proba(texts).argmax(axis=1)]
        return labels[0] if single else labels

    def decide(self, texts):
        """A panel's answers. One text in gives one {question: answer} back."""
        if self.task != "panel":
            raise ValueError("this is a classifier; use predict()")
        single = isinstance(texts, str)
        rows = []
        for text in [texts] if single else texts:
            probs, answers = self._one(text), {}
            for o in self.outputs:
                p, options = probs[o["id"]], o["options"]
                answer = {"probabilities": dict(zip(options, map(float, p), strict=True))}
                if o["type"] == "choice":
                    answer.update(choice=options[int(p.argmax())], confidence=_shape(p))
                elif o["type"] == "score":
                    answer.update(score=float((p * np.arange(len(p))).sum()), confidence=_shape(p))
                else:
                    answer.update(noul=float(p[options.index("yes")]))
                answers[o["id"]] = answer
            rows.append(answers)
        return rows[0] if single else rows

    def confident(self, texts):
        """Whether each text clears the threshold saved with the model, if one was."""
        if not self.threshold:
            raise ValueError("this export has no threshold")
        single = isinstance(texts, str)
        proba = self.predict_proba(texts)
        if self.task == "classifier":
            top = proba.max(axis=1)
        else:
            top = np.min([proba[k].max(axis=1) for k in self.threshold["questions"]], axis=0)
        value = self.threshold["value"]
        sure = [False] * len(top) if value is None else [bool(t >= value) for t in top]
        return sure[0] if single else sure
