"""An interrupted encoder fine-tune resumes from its last epoch and ends where an
uninterrupted one would. Needs shrewd[encoder] and a tiny model from the Hugging Face hub."""

import types

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from shrewd import encoder as enc  # noqa: E402
from shrewd import students  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-bert"
TEXTS = [f"ticket {i} about {'billing' if i % 2 else 'login'}" for i in range(32)]
LABELS = ["billing" if i % 2 else "account" for i in range(32)]


@pytest.fixture(autouse=True)
def tiny_model_available():
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        pytest.skip("the tiny test model is not cached and the hub is unreachable")


def interrupt_after(monkeypatch, epoch):
    real = enc.Checkpoint.save

    def save_then_stop(self, state):
        real(self, state)
        if state["epoch"] == epoch:
            raise KeyboardInterrupt

    monkeypatch.setattr(enc.Checkpoint, "save", save_then_stop)


def classifier(directory):
    student = students.EncoderStudent(model=MODEL, epochs=3, batch_size=8,
                                      checkpoint_dir=directory)
    student._pick_device = lambda: "cpu"
    return student


def test_classifier_resumes_to_the_same_model(tmp_path, monkeypatch):
    straight = classifier(tmp_path / "a")
    straight.fit(TEXTS, LABELS)
    with monkeypatch.context() as m:
        interrupt_after(m, 1)
        with pytest.raises(KeyboardInterrupt):
            classifier(tmp_path / "b").fit(TEXTS, LABELS)
    assert list((tmp_path / "b").glob("*.pt"))
    resumed = classifier(tmp_path / "b")
    resumed.fit(TEXTS, LABELS)
    assert not list((tmp_path / "b").glob("*.pt"))  # cleared once training finishes
    assert np.allclose(straight.predict_proba(TEXTS[:6]), resumed.predict_proba(TEXTS[:6]),
                       atol=1e-5)


def test_panel_features_resume_to_the_same_model(tmp_path, monkeypatch):
    questions = {"dept": types.SimpleNamespace(kind="choice"),
                 "angry": types.SimpleNamespace(kind="noul")}
    targets = {"dept": np.eye(3)[[i % 3 for i in range(32)]],
               "angry": np.eye(2)[[i % 2 for i in range(32)]]}

    def features(directory):
        return enc.EncoderFeatures(model=MODEL, epochs=3, batch_size=8, device="cpu",
                                   checkpoint_dir=directory)

    straight = features(tmp_path / "a").fit_with_targets(TEXTS, targets, questions)
    with monkeypatch.context() as m:
        interrupt_after(m, 2)
        with pytest.raises(KeyboardInterrupt):
            features(tmp_path / "b").fit_with_targets(TEXTS, targets, questions)
    resumed = features(tmp_path / "b").fit_with_targets(TEXTS, targets, questions)
    assert np.allclose(straight, resumed, atol=1e-4)


def test_saved_features_leave_the_checkpoint_setting_out(tmp_path):
    import json

    f = enc.EncoderFeatures(model=MODEL, epochs=1, batch_size=8, device="cpu",
                            checkpoint_dir=tmp_path / "ck")
    f.fit_with_targets(TEXTS, {"dept": np.eye(2)[[i % 2 for i in range(32)]]},
                       {"dept": types.SimpleNamespace(kind="choice")})
    f.save_to(tmp_path / "saved")
    params = json.loads((tmp_path / "saved" / "features.json").read_text())
    assert "checkpoint_dir" not in params  # older versions load it with cls(**params)
