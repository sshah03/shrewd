import importlib.util
import json
import random
import shutil
import subprocess
import unicodedata

import numpy as np
import pytest
from conftest import QUESTIONS, SNIPPETS, make_docs

import shrewd
from shrewd.calibrate import Calibrator
from shrewd.decide import Noul
from shrewd.decisions import DecisionStudent
from shrewd.students import TfidfStudent

HAS_NODE = shutil.which("node") is not None
TEXTS = ["I was charged twice", "the app crashes on start", "please cancel my plan", "",
         "Café İstanbul ΣΊΣΥΦΟΣ straße", "emoji 😀👍🏽 日本語", "tabs\tand\nnew  lines\xa0here",
         "x" * 300, "refund refund refund invoice"]


def _onehot(values, options):
    out = np.zeros((len(values), len(options)))
    for i, value in enumerate(values):
        out[i, options.index(value)] = 1.0
    return out


def _runtime(folder):
    spec = importlib.util.spec_from_file_location("exported_runtime", folder / "shrewd_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model(folder)


def _node(folder, texts, call):
    inputs = folder.parent / f"{folder.name}-texts.json"
    inputs.write_text(json.dumps(texts), encoding="utf-8")
    script = (f'import {{ Model }} from "{(folder / "shrewd_model.js").as_posix()}";\n'
              'import { readFileSync } from "node:fs";\n'
              f'const m = await Model.load("{folder.as_posix()}");\n'
              f'const texts = JSON.parse(readFileSync("{inputs.as_posix()}", "utf8"));\n'
              f'console.log(JSON.stringify(texts.map((t) => m.{call}(t))));')
    return json.loads(subprocess.check_output(["node", "--input-type=module", "-e", script],
                                              text=True))


@pytest.fixture
def classifier_project(tmp_path):
    texts, labels = [], []
    for label, snippets in SNIPPETS.items():
        for i, snippet in enumerate(snippets * 8):
            texts.append(f"{snippet} #{i}")
            labels.append(label)
    student = TfidfStudent(seed=0)
    student.fit(texts, labels)
    project = tmp_path / "tickets"
    student.save(project / "student")
    (project / "student" / "routing.json").write_text(json.dumps(
        {"budget": 0.05, "delta": 0.05, "threshold": 0.4}))
    (project / "manifest.json").write_text(json.dumps({"labels": sorted(SNIPPETS)}))
    return project


@pytest.fixture
def panel_project(tmp_path):
    docs = make_docs(160)
    questions = dict(QUESTIONS, always=Noul(instructions="Is this a ticket?"))
    docs["always"] = "yes"
    student = DecisionStudent(questions, seed=0)
    targets = {key: _onehot(docs[key].tolist(), q.options()) for key, q in questions.items()}
    student.fit(docs["text"].tolist(), targets)
    student.calibrators = {
        "department": Calibrator("isotonic", {"x": [0.0, 0.3, 0.3, 0.6, 1.0],
                                              "y": [0.05, 0.2, 0.4, 0.7, 0.98]}),
        "angry": Calibrator("platt", {"a": 1.3, "b": -0.2, "binary": True}),
        "severity": Calibrator("vector", {"W": np.diag([1.2, 0.9, 1.1]).tolist(),
                                          "b": [0.1, 0.0, -0.1]}),
    }
    student.routing = {"threshold": 0.3, "budget": 0.02, "delta": 0.05,
                       "questions": ["department", "angry"]}
    project = tmp_path / "panel"
    student.save(project / "student")
    (project / "manifest.json").write_text(json.dumps({"kind": "decisions"}))
    return project, docs["text"].tolist()[:40] + TEXTS


def test_a_tfidf_classifier_exports_and_matches(classifier_project, tmp_path):
    out = tmp_path / "export"
    report = shrewd.export(classifier_project, out)
    assert report["format"] == "plain" and report["checked"] > 0
    assert report["max_difference"] < 1e-6
    assert {"model.json", "weights.safetensors", "shrewd_model.py", "shrewd_model.js",
            "shrewd_model.d.ts", "README.md"} <= set(report["files"])
    assert not (out / "__pycache__").exists()
    model, exported = shrewd.load(classifier_project), _runtime(out)
    assert exported.predict(TEXTS) == model.predict(TEXTS)
    assert exported.predict("I was charged twice") == model.predict(["I was charged twice"])[0]
    np.testing.assert_allclose(exported.predict_proba(TEXTS), model.predict_proba(TEXTS),
                               atol=1e-6)
    top = model.predict_proba(TEXTS).max(axis=1)
    assert exported.confident(TEXTS) == [bool(t >= 0.4) for t in top]


def test_a_panel_exports_with_every_calibration(panel_project, tmp_path):
    project, texts = panel_project
    out = tmp_path / "export"
    shrewd.export(project, out)
    spec = json.loads((out / "model.json").read_text())
    heads = {o["id"]: o["head"]["type"] for o in spec["outputs"]}
    assert heads["always"] == "constant"
    model, exported = shrewd.load(project), _runtime(out)
    want, got = model.predict_proba(texts), exported.predict_proba(texts)
    for key in want:
        np.testing.assert_allclose(got[key], want[key], atol=1e-6)
    for theirs, ours in zip(model.decide(texts), exported.decide(texts), strict=True):
        assert ours["department"]["choice"] == theirs["department"].choice
        assert ours["angry"]["noul"] == pytest.approx(theirs["angry"].noul, abs=1e-4)
        assert ours["severity"]["score"] == pytest.approx(theirs["severity"].score, abs=1e-4)
        assert ours["severity"]["confidence"] == pytest.approx(
            theirs["severity"].payload["confidence"], abs=1e-4)
    assert exported.confident(texts) == list(model.local(texts))


def test_temperature_and_matrix_calibration_export_too(panel_project, tmp_path):
    project, texts = panel_project
    meta = json.loads((project / "student" / "meta.json").read_text())
    meta["calibration"]["department"] = {"method": "temperature", "params": {"temperature": 1.7}}
    meta["calibration"]["severity"] = {"method": "matrix", "params": {
        "W": [[1.1, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.0]], "b": [0.0, 0.1, -0.1]}}
    meta["calibration"]["angry"] = {"method": "none", "params": {}}
    (project / "student" / "meta.json").write_text(json.dumps(meta))
    shrewd.export(project, tmp_path / "export")
    want = shrewd.load(project).predict_proba(texts)
    got = _runtime(tmp_path / "export").predict_proba(texts)
    for key in want:
        np.testing.assert_allclose(got[key], want[key], atol=1e-6)


@pytest.mark.skipif(not HAS_NODE, reason="needs node")
def test_the_javascript_runtime_matches(classifier_project, panel_project, tmp_path):
    shrewd.export(classifier_project, tmp_path / "c")
    model = shrewd.load(classifier_project)
    assert _node(tmp_path / "c", TEXTS, "predict") == model.predict(TEXTS)
    np.testing.assert_allclose(_node(tmp_path / "c", TEXTS, "predictProba"),
                               model.predict_proba(TEXTS), atol=1e-6)
    project, texts = panel_project
    shrewd.export(project, tmp_path / "p")
    want = shrewd.load(project).predict_proba(texts)
    rows = _node(tmp_path / "p", texts, "predictProba")
    for key in want:
        np.testing.assert_allclose([r[key] for r in rows], want[key], atol=1e-6)
    assert _node(tmp_path / "p", texts, "confident") == list(shrewd.load(project).local(texts))


@pytest.mark.skipif(not HAS_NODE, reason="needs node")
def test_javascript_splits_text_like_scikit_learn(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = random.Random(0)
    pool = [chr(c) for c in range(0x30000)
            if unicodedata.category(chr(c)) not in ("Cs", "Cn", "Co")]
    spaces = [chr(c) for c in range(0x3001) if chr(c).isspace()] + ["\u200b", "\ufeff"]
    texts = TEXTS + ["".join(rng.choice(pool if rng.random() < 0.7 else spaces + list("ab "))
                             for _ in range(rng.randint(1, 30))) for _ in range(2000)]
    word = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(texts)
    student = TfidfStudent(seed=0)
    student._pipe.named_steps["features"].transformer_list = [("word", word), ("char", char)]
    student.fit(texts, [str(i % 2) for i in range(len(texts))])
    project = tmp_path / "unicode"
    student.save(project / "student")
    (project / "manifest.json").write_text(json.dumps({"labels": ["0", "1"]}))
    shrewd.export(project, tmp_path / "export")
    np.testing.assert_allclose(_node(tmp_path / "export", texts, "predictProba"),
                               shrewd.load(project).predict_proba(texts), atol=1e-6)


def test_the_weights_are_standard_safetensors(classifier_project, tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")
    import joblib

    shrewd.export(classifier_project, tmp_path / "export")
    tensors = safetensors.load_file(tmp_path / "export" / "weights.safetensors")
    pipe = joblib.load(classifier_project / "student" / "model.joblib")
    np.testing.assert_allclose(tensors["label.coef"], pipe.named_steps["lr"].coef_, rtol=1e-6)


def test_an_export_refuses_what_it_cannot_carry(panel_project, classifier_project, tmp_path):
    project, _ = panel_project
    with pytest.raises(ValueError, match="exports as plain, not onnx"):
        shrewd.export(project, tmp_path / "a", format="onnx")
    with pytest.raises(ValueError, match="float16"):
        shrewd.export(classifier_project, tmp_path / "b", precision="float16")
    with pytest.raises(ValueError, match="format must be"):
        shrewd.export(classifier_project, tmp_path / "c", format="tflite")
    meta = json.loads((project / "student" / "meta.json").read_text())
    meta["stacks"] = {"angry": {"bias": 0.0, "weights": [1.0, 0.0]}}
    meta["zero_shot"] = {"model": "x"}
    (project / "student" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="zero-shot"):
        shrewd.export(project, tmp_path / "d")


def test_an_export_only_replaces_an_earlier_export(classifier_project, tmp_path):
    out = tmp_path / "export"
    shrewd.export(classifier_project, out)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        shrewd.export(classifier_project, out)
    shrewd.export(classifier_project, out, overwrite=True)
    other = tmp_path / "notes"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="no earlier export"):
        shrewd.export(classifier_project, other, overwrite=True)
    assert (other / "keep.txt").read_text() == "mine"


def _static_model():
    pytest.importorskip("model2vec")
    from model2vec import StaticModel

    try:
        StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception:
        pytest.skip("potion-base-8M is not available offline")


def test_an_embedding_panel_exports_and_matches(tmp_path):
    _static_model()
    docs = make_docs(120)
    student = DecisionStudent(QUESTIONS, features="embed", seed=0)
    targets = {key: _onehot(docs[key].tolist(), q.options()) for key, q in QUESTIONS.items()}
    student.fit(docs["text"].tolist(), targets)
    project = tmp_path / "panel"
    student.save(project / "student")
    (project / "manifest.json").write_text(json.dumps({"kind": "decisions"}))
    texts = TEXTS + ["[CLS] literal [SEP] tokens", "x" * 4000]
    for precision, tolerance in (("float32", 1e-6), ("float16", 1e-3)):
        out = tmp_path / precision
        shrewd.export(project, out, precision=precision)
        want, got = shrewd.load(project).predict_proba(texts), _runtime(out).predict_proba(texts)
        for key in want:
            np.testing.assert_allclose(got[key], want[key], atol=tolerance)
        if HAS_NODE:
            rows = _node(out, texts, "predictProba")
            for key in want:
                np.testing.assert_allclose([r[key] for r in rows], want[key], atol=tolerance)


def test_an_embedding_classifier_exports_to_onnx(tmp_path):
    _static_model()
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from shrewd.students import EmbedStudent

    texts, labels = [], []
    for label, snippets in SNIPPETS.items():
        texts += [f"{s} {i}" for i, s in enumerate(snippets * 5)]
        labels += [label] * len(snippets) * 5
    student = EmbedStudent(seed=0)
    student.fit(texts, labels)
    project = tmp_path / "embed"
    student.save(project / "student")
    (project / "manifest.json").write_text(json.dumps({"labels": sorted(SNIPPETS)}))
    report = shrewd.export(project, tmp_path / "onnx", format="onnx")
    assert report["checked"] > 0 and report["max_difference"] < 1e-4
    assert {"model.onnx", "tokenizer.json", "model.json"} <= set(report["files"])
