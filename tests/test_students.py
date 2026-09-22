import sys
import types

import numpy as np
import pytest
from conftest import make_seed

from shrewd import students

seed_df = make_seed(20)
TEXTS, LABELS_COL = seed_df["text"].tolist(), seed_df["label"].tolist()


def roundtrip(student, tmp_path):
    student.fit(TEXTS, LABELS_COL)
    before = student.predict(TEXTS)
    assert all(type(label) is str for label in before)  # plain str, not np.str_
    proba = student.predict_proba(TEXTS)
    assert proba.shape == (len(TEXTS), len(set(LABELS_COL)))
    assert np.allclose(proba.sum(axis=1), 1.0)
    student.save(tmp_path / "student")
    loaded = students.load_student(tmp_path / "student")
    assert type(loaded) is type(student)
    assert loaded.classes_ == student.classes_
    assert loaded.predict(TEXTS) == before
    assert loaded.n_train_ == len(TEXTS)


def test_tfidf_roundtrip(tmp_path):
    roundtrip(students.TfidfStudent(), tmp_path)


def test_tfidf_learns_the_task():
    student = students.TfidfStudent()
    student.fit(TEXTS, LABELS_COL)
    assert student.predict(["I was charged double"]) == ["billing"]
    assert student.classes_ == sorted(set(LABELS_COL))


def test_tfidf_kwargs_reach_logistic_regression():
    student = students.TfidfStudent(C=0.5)
    assert student._pipe.named_steps["lr"].C == 0.5
    assert student._pipe.named_steps["lr"].class_weight == "balanced"


class FakeStaticModel:
    """Deterministic stand-in for a model2vec embedder: letter-frequency vectors."""

    @staticmethod
    def from_pretrained(name):
        return FakeStaticModel()

    def encode(self, texts):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return np.array(
            [[text.lower().count(letter) for letter in alphabet] for text in texts],
            dtype=float,
        )


@pytest.fixture
def fake_model2vec(monkeypatch):
    module = types.ModuleType("model2vec")
    module.StaticModel = FakeStaticModel
    monkeypatch.setitem(sys.modules, "model2vec", module)


def test_embed_roundtrip(fake_model2vec, tmp_path):
    roundtrip(students.EmbedStudent(), tmp_path)


def test_embed_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "model2vec", None)
    with pytest.raises(ImportError, match=r"shrewd\[embed\]"):
        students.EmbedStudent()


def test_setfit_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "setfit", None)
    with pytest.raises(ImportError, match=r"shrewd\[setfit\]"):
        students.SetFitStudent()


def test_setfit_roundtrip(tmp_path):
    pytest.importorskip("setfit")
    student = students.SetFitStudent(train_args={"max_steps": 5}, max_seq_length=64)
    roundtrip(student, tmp_path)
    assert student._model.model_body.max_seq_length == 64


def test_encoder_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    student = students.EncoderStudent(model="prajjwal1/bert-tiny", epochs=1, max_length=64)
    roundtrip(student, tmp_path)


def test_encoder_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"shrewd\[encoder\]"):
        students.EncoderStudent()


def test_predict_abstains_below_min_confidence():
    from shrewd.students import TfidfStudent

    texts = ["refund my charge", "app crashes", "cancel my plan", "office hours"] * 10
    labels = ["billing", "bug", "cancellation", "other"] * 10
    student = TfidfStudent()
    student.fit(texts, labels)
    assert student.predict(["refund my charge"]) == ["billing"]
    out = student.predict(["refund my charge", "zzz qqq"], min_confidence=0.99)
    assert out[1] is None  # nonsense text is not confidently anything
    assert len(out) == 2
