import json
import shutil
import struct
import unicodedata
from pathlib import Path

import numpy as np

from shrewd import __version__

RUNTIME = Path(__file__).parent / "_runtime"
RUNTIME_FILES = ("shrewd_model.py", "shrewd_model.js", "shrewd_model.d.ts")
FORMATS = ("auto", "plain", "onnx", "hf")
PRECISIONS = ("float32", "float16")
CJK = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF),
       (0x2A700, 0x2B81F), (0x2B920, 0x2CEAF), (0x2F800, 0x2FA1F)]
ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
BERT_SPACE = set("\t\n\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
                 "\u2009\u200a\u2028\u2029\u202f\u205f\u3000")
CHECK_TEXTS = ["", "ok", "I was charged twice for my order", "The app crashes when I log in!",
               "Café résumé naïve İstanbul straße ΣΊΣΥΦΟΣ", "emoji 😀👍🏽 and 日本語 text",
               "tabs\tand\nnewlines  and\xa0spaces", "[CLS] [SEP] [UNK] [MASK]", "x" * 400]


def export(project_dir, out_dir, format="auto", precision="float32", check=True,
           overwrite=False):
    """Write a project's shipped model to `out_dir` so it runs without shrewd.

    `format="plain"` (tf-idf and embedding models, classifiers and panels) writes the weights
    with a small runtime for Python (numpy only) and JavaScript (no dependencies).
    `format="onnx"` (embedding, encoder and SetFit classifiers) writes a `model.onnx` and the
    `tokenizer.json` it expects. `format="hf"` (encoder classifiers) writes a Hugging Face
    folder that `transformers` loads directly. `"auto"` picks plain where it can and ONNX
    otherwise. `precision="float16"` stores an embedding model's table at half size. With
    `check=True` the export is compared with `load()` on the project's own documents and
    raises if they disagree. Teacher backup is not part of an export; a routing threshold
    travels as metadata.
    """
    from shrewd.students import fetch_project

    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, not {format!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
    if isinstance(project_dir, str) and project_dir.startswith(("https://", "http://")):
        project_dir = fetch_project(project_dir)
    project_dir, out_dir = Path(project_dir), Path(out_dir)
    student_dir = project_dir / "student"
    if not (project_dir / "manifest.json").exists():
        raise FileNotFoundError(f"{project_dir} is not a shrewd project (no manifest.json)")
    if not (student_dir / "meta.json").exists():
        raise FileNotFoundError(f"no trained student in {project_dir}; run distill() first")
    meta = json.loads((student_dir / "meta.json").read_text())
    kind = _kind(meta)
    format = _pick_format(kind, format)
    if precision != "float32" and (format != "plain" or kind[1] != "embed"):
        raise ValueError("precision='float16' is for plain embedding exports, whose table is "
                         "most of their size")
    _prepare(out_dir, overwrite)
    if format == "plain":
        _plain(student_dir, meta, kind, out_dir, precision)
    elif format == "onnx":
        _onnx(student_dir, meta, kind, out_dir)
    else:
        _hf(student_dir, meta, out_dir)
    _readme(out_dir, format, kind)
    report = {"path": str(out_dir), "format": format,
              "files": sorted(p.name for p in out_dir.iterdir()),
              "checked": 0, "max_difference": None}
    if check:
        report.update(_check(project_dir, out_dir, format, precision))
    return report


def _kind(meta):
    if meta["type"] == "decisions":
        return ("panel", meta.get("features", "tfidf"))
    return ("classifier", meta["type"])


def _pick_format(kind, format):
    task, features = kind
    plain = features in ("tfidf", "embed")
    onnx = task == "classifier" and features in ("embed", "encoder", "setfit")
    hf = task == "classifier" and features == "encoder"
    if format == "auto":
        format = "plain" if plain else "onnx" if onnx else None
    allowed = {"plain": plain, "onnx": onnx, "hf": hf}
    if format is None or not allowed[format]:
        options = [f for f, ok in allowed.items() if ok]
        name = {"tfidf": "tf-idf", "embed": "embedding", "setfit": "SetFit"}.get(features, features)
        what = f"{'an' if name[0] in 'aeiou' else 'a'} {name} {task}"
        if not options:
            raise ValueError(f"{what} can't be exported yet")
        raise ValueError(f"{what} exports as {' or '.join(options)}, not {format}")
    return format


def _prepare(out_dir, overwrite):
    if out_dir.exists() and any(out_dir.iterdir()):
        previous = (out_dir / "model.json").exists() or (out_dir / "model.onnx").exists()
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty; pass overwrite=True to replace it")
        if not previous:
            raise FileExistsError(f"{out_dir} is not empty and holds no earlier export")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_safetensors(path, tensors, dtypes=None):
    little = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        dtype = (dtypes or {}).get(name, "F32")
        array = np.ascontiguousarray(tensors[name]).astype(little[dtype])
        data = array.tobytes(order="C")
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for data in chunks:
            f.write(data)


def _tfidf_blocks(union):
    from sklearn.pipeline import FeatureUnion

    if not isinstance(union, FeatureUnion) or union.transformer_weights is not None:
        raise ValueError("only shrewd's built-in tf-idf features can be exported")
    blocks, idfs, offset = [], [], 0
    for name, vec in union.transformer_list:
        simple = (vec.lowercase and vec.strip_accents is None and vec.preprocessor is None
                  and vec.tokenizer is None and vec.use_idf and vec.norm == "l2"
                  and not vec.sublinear_tf and not vec.binary and vec.stop_words is None)
        word_ok = vec.analyzer == "word" and vec.token_pattern == r"(?u)\b\w\w+\b"
        if not simple or not (word_ok or vec.analyzer == "char_wb"):
            raise ValueError(f"tf-idf block {name!r} uses settings the runtime doesn't support")
        vocab = [None] * len(vec.vocabulary_)
        for term, i in vec.vocabulary_.items():
            vocab[i] = term
        blocks.append({"name": name, "analyzer": vec.analyzer, "ngram": list(vec.ngram_range),
                       "lowercase": True, "offset": offset, "size": len(vocab), "vocab": vocab})
        idfs.append(vec.idf_)
        offset += len(vocab)
    features = {"type": "tfidf", "unicode": unicodedata.unidata_version, "blocks": blocks}
    return features, {"idf": np.concatenate(idfs)}


def _rule_normalize(ch):
    code, category = ord(ch), unicodedata.category(ch)
    if code in (0, 0xFFFD) or (ch not in "\t\n\r" and category in ("Cc", "Cf", "Co")):
        return ""
    if ch in BERT_SPACE:
        return " "
    if any(a <= code <= b for a, b in CJK):
        return " " + ch + " "
    return "".join(x.lower() for x in unicodedata.normalize("NFD", ch)
                   if unicodedata.category(x) != "Mn")


def _rule_punct(ch):
    return ch in ASCII_PUNCT or unicodedata.category(ch).startswith("P")


def _wordpiece_spec(tokenizer):
    model = tokenizer.model
    if type(model).__name__ != "WordPiece":
        raise ValueError(f"the embedding model's tokenizer is {type(model).__name__}, "
                         "and only WordPiece can be exported")
    normalizer = json.loads(tokenizer.to_str())["normalizer"] or {}
    expected = {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True,
                "lowercase": True}
    if any(normalizer.get(k) != v for k, v in expected.items()) or normalizer.get(
            "strip_accents") not in (None, True):
        raise ValueError("the embedding model's tokenizer normalizes text in a way the runtime "
                         "doesn't support")
    vocab = [None] * tokenizer.get_vocab_size()
    for token, i in tokenizer.get_vocab().items():
        vocab[i] = token
    added = {}
    for token in tokenizer.get_added_tokens_decoder().values():
        if token.lstrip or token.rstrip or token.single_word or token.normalized:
            raise ValueError(f"added token {token.content!r} has matching rules the runtime "
                             "doesn't support")
        added[token.content] = tokenizer.token_to_id(token.content)
    overrides, punct_add, punct_remove = {}, [], []
    norm, pre = tokenizer.normalizer, tokenizer.pre_tokenizer
    for code in range(0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        ch = chr(code)
        got = norm.normalize_str(ch)
        if got != _rule_normalize(ch):
            overrides[code] = got
        if any(a <= code <= b for a, b in CJK):
            continue
        split = [w for w, _ in pre.pre_tokenize_str("a" + ch + "b")] == ["a", ch, "b"]
        if split != _rule_punct(ch):
            (punct_add if split else punct_remove).append(code)
    return {"type": "wordpiece", "vocab": vocab, "unk": tokenizer.token_to_id(model.unk_token),
            "prefix": model.continuing_subword_prefix,
            "max_chars": model.max_input_chars_per_word, "cjk": CJK, "added": added,
            "overrides": overrides, "punct_add": punct_add, "punct_remove": punct_remove}


def _static_features(name):
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(name)
    if model.weights is not None or model.token_mapping is not None:
        raise ValueError(f"{name} is a quantized or weighted model2vec model, which the runtime "
                         "doesn't support")
    features = {"type": "static_embeddings", "source": name,
                "unicode": unicodedata.unidata_version, "max_tokens": 512,
                "max_input_chars": 512 * model.median_token_length,
                "normalize": bool(model.normalize), "drop_token": model.unk_token_id,
                "tokenizer": _wordpiece_spec(model.tokenizer)}
    return features, {"embedding": model.embedding}


def _logistic(model, key, tensors, classes=None):
    tensors[f"{key}.coef"], tensors[f"{key}.intercept"] = model.coef_, model.intercept_
    return {"type": "logistic", "binary": model.coef_.shape[0] == 1,
            "classes": None if classes is None else [int(c) for c in classes]}


def _calibration(calibrator):
    if calibrator is None:
        return None
    params = {k: (np.asarray(v).tolist() if isinstance(v, (list, tuple, np.ndarray)) else v)
              for k, v in calibrator.params.items()}
    return {"method": calibrator.method, "params": params}


def _threshold(routing, questions=None):
    if not routing:
        return None
    out = {"value": routing.get("threshold"), "budget": routing.get("budget"),
           "delta": routing.get("delta")}
    if questions is not None:
        out["questions"] = list(routing.get("questions") or questions)
    return out


def _plain(student_dir, meta, kind, out_dir, precision):
    import joblib

    from shrewd import decide
    from shrewd.students import ROUTING_FILE, load_student

    task, features_kind = kind
    tensors, outputs, threshold = {}, [], None
    if task == "classifier":
        model = joblib.load(student_dir / "model.joblib")
        if features_kind == "tfidf":
            features, extra = _tfidf_blocks(model.named_steps["features"])
            head = model.named_steps["lr"]
        else:
            features, extra = _static_features(meta["embedder"])
            head = model
        outputs.append({"id": "label", "type": "label", "options": [str(c) for c in head.classes_],
                        "head": _logistic(head, "label", tensors), "calibration": None})
        if (student_dir / ROUTING_FILE).exists():
            threshold = _threshold(json.loads((student_dir / ROUTING_FILE).read_text()))
    else:
        from shrewd.decisions import _ConstantHead, _Expanded

        student = load_student(student_dir)
        if student.stacks:
            raise ValueError("this panel stacks a zero-shot model, which runs a language model "
                             "at prediction time and can't be exported")
        if features_kind == "tfidf":
            features, extra = _tfidf_blocks(student._features)
        else:
            features, extra = _static_features(student._features.model_name)
        for key, question in student.questions.items():
            head = student.heads.get(key)
            if head is None:
                continue
            if isinstance(head, _ConstantHead):
                spec = {"type": "constant", "index": int(head.index)}
            elif isinstance(head, _Expanded):
                spec = _logistic(head.model, key, tensors, head.model.classes_)
            else:
                raise ValueError(f"question {key!r} has a head the runtime doesn't support")
            outputs.append({"id": key, "type": question.kind, "options": question.options(),
                            "question": decide.to_dict(question), "head": spec,
                            "calibration": _calibration(student.calibrators.get(key))})
        threshold = _threshold(student.routing, [o["id"] for o in outputs])
    tensors.update(extra)
    half = {"embedding": "F16"} if precision == "float16" else None
    write_safetensors(out_dir / "weights.safetensors", tensors, half)
    spec = {"format": "shrewd-model", "version": 1, "task": task,
            "source": {"shrewd": __version__, "student": features_kind,
                       "n_train": meta.get("n_train")},
            "features": features, "outputs": outputs}
    if threshold:
        spec["threshold"] = threshold
    (out_dir / "model.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    for name in RUNTIME_FILES:
        shutil.copyfile(RUNTIME / name, out_dir / name)


def _need(module, extra):
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise ImportError(f'this export needs {module}: pip install "shrewd[{extra}]"')


def _save_onnx(model, path):
    import onnx

    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _onnx_static(student_dir, meta, out_dir):
    import joblib
    from model2vec import StaticModel
    from onnx import TensorProto
    from onnx import helper as h
    from onnx import numpy_helper as nh

    head = joblib.load(student_dir / "model.joblib")
    static = StaticModel.from_pretrained(meta["embedder"])
    coef, icpt = head.coef_.astype(np.float32), head.intercept_.astype(np.float32)
    binary = coef.shape[0] == 1
    nodes = [
        h.make_node("Gather", ["embedding", "input_ids"], ["vectors"]),
        h.make_node("Equal", ["input_ids", "unk"], ["is_unk"]),
        h.make_node("Not", ["is_unk"], ["known"]),
        h.make_node("Cast", ["known"], ["known_f"], to=TensorProto.FLOAT),
        h.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        h.make_node("Mul", ["known_f", "mask_f"], ["kept"]),
        h.make_node("CumSum", ["kept", "axis1_scalar"], ["position"]),
        h.make_node("LessOrEqual", ["position", "max_tokens"], ["in_window"]),
        h.make_node("Cast", ["in_window"], ["in_window_f"], to=TensorProto.FLOAT),
        h.make_node("Mul", ["kept", "in_window_f"], ["mask"]),
        h.make_node("Unsqueeze", ["mask", "axis2"], ["mask3"]),
        h.make_node("Mul", ["vectors", "mask3"], ["masked"]),
        h.make_node("ReduceSum", ["masked", "axis1"], ["summed"], keepdims=0),
        h.make_node("ReduceSum", ["mask", "axis1"], ["count"], keepdims=1),
        h.make_node("Max", ["count", "one"], ["count1"]),
        h.make_node("Div", ["summed", "count1"], ["mean"]),
    ]
    last = "mean"
    if static.normalize:
        nodes += [h.make_node("Mul", ["mean", "mean"], ["squares"]),
                  h.make_node("ReduceSum", ["squares", "axis1"], ["sumsq"], keepdims=1),
                  h.make_node("Sqrt", ["sumsq"], ["norm"]),
                  h.make_node("Add", ["norm", "tiny"], ["norm1"]),
                  h.make_node("Div", ["mean", "norm1"], ["unit"])]
        last = "unit"
    nodes.append(h.make_node("Gemm", [last, "coef", "intercept"], ["logits"], transB=1))
    if binary:
        nodes += [h.make_node("Sigmoid", ["logits"], ["p_yes"]),
                  h.make_node("Sub", ["one", "p_yes"], ["p_no"]),
                  h.make_node("Concat", ["p_no", "p_yes"], ["probabilities"], axis=1)]
    else:
        nodes.append(h.make_node("Softmax", ["logits"], ["probabilities"], axis=1))
    inits = [nh.from_array(static.embedding.astype(np.float32), "embedding"),
             nh.from_array(coef, "coef"), nh.from_array(icpt, "intercept"),
             nh.from_array(np.array(static.unk_token_id, np.int64), "unk"),
             nh.from_array(np.array(512, np.float32), "max_tokens"),
             nh.from_array(np.array(1, np.int64), "axis1_scalar"),
             nh.from_array(np.array([1], np.int64), "axis1"),
             nh.from_array(np.array([2], np.int64), "axis2"),
             nh.from_array(np.array([1.0], np.float32), "one"),
             nh.from_array(np.array([1e-32], np.float32), "tiny")]
    k = len(head.classes_)
    graph = h.make_graph(
        nodes, "shrewd_classifier",
        [h.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
         h.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"])],
        [h.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", k])], inits)
    _save_onnx(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)]), out_dir / "model.onnx")
    static.tokenizer.no_truncation()
    static.tokenizer.no_padding()
    static.tokenizer.save(str(out_dir / "tokenizer.json"))
    return ([str(c) for c in head.classes_],
            {"add_special_tokens": False, "max_input_chars": 512 * static.median_token_length})


def _probabilities(trunk, logits, binary=False):
    import torch

    class Probabilities(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk

        def forward(self, input_ids, attention_mask):
            z = logits(self.trunk, input_ids, attention_mask)
            if binary:
                p = torch.sigmoid(z)
                return torch.cat([1 - p, p], dim=1)
            return torch.softmax(z, dim=-1)

    return Probabilities().eval()


def _encoder_max_length():
    import inspect

    from shrewd.students import EncoderStudent

    return inspect.signature(EncoderStudent).parameters["max_length"].default


def _torch_export(module, tokenizer, path):
    import warnings

    import onnx
    import torch

    sample = tokenizer(["export", "a longer sample sentence for tracing"], padding=True,
                       return_tensors="pt")
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module, (sample["input_ids"], sample["attention_mask"]), str(path),
            input_names=["input_ids", "attention_mask"], output_names=["probabilities"],
            dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                          "attention_mask": {0: "batch", 1: "tokens"},
                          "probabilities": {0: "batch"}},
            opset_version=18, dynamo=False)
    _save_onnx(onnx.load(str(path)), path)


def _onnx_encoder(student_dir, meta, out_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    max_length = _encoder_max_length()
    folder = student_dir / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(
        folder, attn_implementation="eager").eval()
    module = _probabilities(
        model, lambda m, ids, mask: m(input_ids=ids, attention_mask=mask).logits)
    _torch_export(module, tokenizer, out_dir / "model.onnx")
    backend = tokenizer.backend_tokenizer
    backend.enable_truncation(max_length)
    backend.no_padding()
    backend.save(str(out_dir / "tokenizer.json"))
    return list(meta["classes"]), {"add_special_tokens": True, "max_tokens": max_length}


def _pooling_mode(config):
    if "pooling_mode" in config:
        return config["pooling_mode"]
    modes = [m for m in ("cls_token", "mean_tokens", "max_tokens", "mean_sqrt_len_tokens",
                         "weightedmean_tokens", "lasttoken")
             if config.get(f"pooling_mode_{m}")]
    return {"cls_token": "cls", "mean_tokens": "mean"}.get(modes[0]) if len(modes) == 1 else None


def _onnx_setfit(student_dir, meta, out_dir):
    import torch
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(student_dir / "setfit"))
    body, head = model.model_body.to("cpu"), model.model_head
    if not hasattr(head, "coef_"):
        raise ValueError("this SetFit model has a differentiable head, which can't be exported yet")
    modules = list(body)
    pooling = next((m for m in modules if type(m).__name__ == "Pooling"), None)
    if pooling is None:
        raise ValueError("this SetFit model has no pooling layer to export")
    mode = _pooling_mode(pooling.get_config_dict())
    if mode not in ("mean", "cls"):
        raise ValueError("only mean or first-token pooling can be exported")
    normalize = any(type(m).__name__ == "Normalize" for m in modules)
    trunk = modules[0].auto_model.eval()
    coef = torch.tensor(head.coef_, dtype=torch.float32)
    icpt = torch.tensor(head.intercept_, dtype=torch.float32)

    def logits(trunk, ids, mask):
        hidden = trunk(input_ids=ids, attention_mask=mask).last_hidden_state
        if mode == "cls":
            vector = hidden[:, 0]
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            vector = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
        if normalize:
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector @ coef.T + icpt

    module = _probabilities(trunk, logits, binary=coef.shape[0] == 1)
    tokenizer = body.tokenizer
    _torch_export(module, tokenizer, out_dir / "model.onnx")
    backend = tokenizer.backend_tokenizer
    backend.enable_truncation(body.max_seq_length)
    backend.no_padding()
    backend.save(str(out_dir / "tokenizer.json"))
    return ([str(c) for c in meta["classes"]],
            {"add_special_tokens": True, "max_tokens": body.max_seq_length})


def _onnx(student_dir, meta, kind, out_dir):
    from shrewd.students import ROUTING_FILE

    _need("onnx", "onnx")
    features = kind[1]
    if features == "embed":
        labels, tokenizing = _onnx_static(student_dir, meta, out_dir)
    elif features == "encoder":
        labels, tokenizing = _onnx_encoder(student_dir, meta, out_dir)
    else:
        labels, tokenizing = _onnx_setfit(student_dir, meta, out_dir)
    spec = {"format": "shrewd-onnx", "version": 1, "task": "classifier",
            "source": {"shrewd": __version__, "student": features, "n_train": meta.get("n_train")},
            "labels": labels, "inputs": ["input_ids", "attention_mask"],
            "output": "probabilities", "tokenizer": "tokenizer.json", "tokenizing": tokenizing}
    if (student_dir / ROUTING_FILE).exists():
        spec["threshold"] = _threshold(json.loads((student_dir / ROUTING_FILE).read_text()))
    (out_dir / "model.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _hf(student_dir, meta, out_dir):
    from shrewd.students import ROUTING_FILE

    shutil.copytree(student_dir / "encoder", out_dir, dirs_exist_ok=True)
    config = json.loads((out_dir / "config.json").read_text())
    config["id2label"] = {str(i): c for i, c in enumerate(meta["classes"])}
    config["label2id"] = {c: i for i, c in enumerate(meta["classes"])}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    tokenizer_config = out_dir / "tokenizer_config.json"
    settings = json.loads(tokenizer_config.read_text())
    settings["model_max_length"] = _encoder_max_length()
    tokenizer_config.write_text(json.dumps(settings, indent=2))
    spec = {"format": "shrewd-hf", "version": 1, "task": "classifier",
            "source": {"shrewd": __version__, "student": "encoder", "n_train": meta.get("n_train")},
            "labels": list(meta["classes"]), "max_tokens": _encoder_max_length()}
    if (student_dir / ROUTING_FILE).exists():
        spec["threshold"] = _threshold(json.loads((student_dir / ROUTING_FILE).read_text()))
    (out_dir / "model.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _readme(out_dir, format, kind):
    task = kind[0]
    if format == "plain":
        call_py = ('model.predict(["I was charged twice"])' if task == "classifier"
                   else 'model.decide("I was charged twice")')
        call_js = ('model.predict("I was charged twice")' if task == "classifier"
                   else 'model.decide("I was charged twice")')
        text = f"""# Exported shrewd model

Runs without shrewd. Copy this folder into your project.

Python (needs numpy only):

```python
from shrewd_model import Model

model = Model("path/to/this/folder")
{call_py}
```

JavaScript (no dependencies; Node 18+, Deno, Bun or a browser):

```js
import {{ Model }} from "./shrewd_model.js";

const model = await Model.load("path/to/this/folder");   // in a browser: Model.fetch(url)
{call_js}
```

`model.json` holds the settings and `weights.safetensors` the weights. If the model was
shipped with a routing threshold, `model.confident(text)` says whether a text clears it.
"""
    elif format == "onnx":
        text = """# Exported shrewd model (ONNX)

`model.onnx` takes token ids from `tokenizer.json` and returns `probabilities`, one per label,
in the order `model.json` lists them. Any ONNX runtime works.

```python
import json, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

spec = json.load(open("model.json"))
tokenizer = Tokenizer.from_file("tokenizer.json")
session = ort.InferenceSession("model.onnx")

enc = tokenizer.encode("I was charged twice",
                       add_special_tokens=spec["tokenizing"]["add_special_tokens"])
probs = session.run(None, {"input_ids": np.array([enc.ids], dtype=np.int64),
                           "attention_mask": np.array([enc.attention_mask], dtype=np.int64)})[0]
print(spec["labels"][int(probs[0].argmax())])
```

In JavaScript, `@huggingface/transformers` reads `tokenizer.json` and `onnxruntime-node` or
`onnxruntime-web` runs the model. Batch by padding with zeros and setting `attention_mask` to 0.
Check `model.json` under `tokenizing` for whether to add special tokens and any length limit.
"""
    else:
        text = """# Exported shrewd model (Hugging Face)

A standard `transformers` sequence classification model with its tokenizer.

```python
from transformers import pipeline

classify = pipeline("text-classification", model="path/to/this/folder", truncation=True)
classify("I was charged twice")
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _check_texts(project_dir, limit=200):
    import pandas as pd

    texts = []
    for name in ("seed_test.csv", "seed_dev.csv", "pool_input.csv", "pool_judged.csv",
                 "pool_labeled.csv"):
        path = project_dir / name
        if path.exists() and len(texts) < limit:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            if "text" in frame:
                texts += frame["text"].tolist()[:limit - len(texts)]
    return list(dict.fromkeys(texts)) + CHECK_TEXTS


def _load_runtime(out_dir):
    import importlib.util

    path = out_dir / "shrewd_model.py"
    spec = importlib.util.spec_from_file_location("_shrewd_model_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(project_dir, out_dir, format, precision):
    import importlib.util

    from shrewd.students import load

    texts = _check_texts(project_dir)
    model = load(project_dir)
    want = model.predict_proba(texts)
    want = want if isinstance(want, dict) else {"label": np.asarray(want)}
    if format == "plain":
        exported = _load_runtime(out_dir).Model(out_dir)
        got = exported.predict_proba(texts)
        got = got if isinstance(got, dict) else {"label": got}
    elif format == "hf":
        got = {"label": _run_hf(out_dir, texts)}
    elif all(importlib.util.find_spec(m) for m in ("onnxruntime", "tokenizers")):
        got = {"label": _run_onnx(out_dir, texts)}
    else:
        return {"checked": 0, "max_difference": None,
                "note": "install onnxruntime to check the export"}
    difference = max(float(np.abs(got[k] - w).max()) for k, w in want.items())
    flips = sum(int((got[k].argmax(1) != w.argmax(1)).sum()) for k, w in want.items())
    tolerance = 1e-4 if format == "plain" and precision == "float32" else 1e-3
    if difference > tolerance:
        raise RuntimeError(f"the export in {out_dir} differs from the model by up to "
                           f"{difference:.2g} ({flips} answers changed on {len(texts)} texts)")
    return {"checked": len(texts), "max_difference": difference, "changed_answers": flips}


def _run_onnx(out_dir, texts):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    spec = json.loads((out_dir / "model.json").read_text())
    tokenizing = spec["tokenizing"]
    tokenizer = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(out_dir / "model.onnx"))
    out = []
    for text in texts:
        if "max_input_chars" in tokenizing:
            text = text[:tokenizing["max_input_chars"]]
        enc = tokenizer.encode(text, add_special_tokens=tokenizing["add_special_tokens"])
        ids = enc.ids or [0]
        mask = enc.attention_mask if enc.ids else [0]
        out.append(session.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                                      "attention_mask": np.array([mask], dtype=np.int64)})[0][0])
    return np.array(out)


def _run_hf(out_dir, texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            enc = tokenizer(texts[start:start + 32], truncation=True, padding=True,
                            return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(out)
