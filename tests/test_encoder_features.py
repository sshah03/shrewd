import numpy as np
import pytest
from conftest import QUESTIONS, make_docs

from shrewd.decisions import DecisionStudent


class CountingTrainableFeatures:
    """A featurizer that learns from the targets, the hook a fine-tuned encoder uses."""

    expensive = True

    def __init__(self):
        self.seen_targets = None
        self.seen_questions = None
        self.fit_calls = 0

    def fit_with_targets(self, texts, targets, questions):
        self.fit_calls += 1
        self.seen_targets = {k: v.shape for k, v in targets.items()}
        self.seen_questions = list(questions)
        return self.transform(texts)

    def transform(self, texts):
        out = np.zeros((len(texts), 8))
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                out[i, hash(tok) % 8] += 1.0
        return out

    def fit_transform(self, texts, y=None):  # pragma: no cover - must not be used
        raise AssertionError("fit_transform must not be called on a trainable featurizer")


def _targets(docs):
    def onehot(values, options):
        out = np.zeros((len(values), len(options)))
        for i, v in enumerate(values):
            out[i, options.index(str(v))] = 1.0
        return out
    return {
        "department": onehot(docs["department"], QUESTIONS["department"].options()),
        "angry": onehot(docs["angry"], QUESTIONS["angry"].options()),
        "severity": onehot(docs["severity"], QUESTIONS["severity"].options()),
    }


def test_trainable_featurizer_receives_targets_and_questions():
    docs = make_docs(120)
    feats = CountingTrainableFeatures()
    student = DecisionStudent(QUESTIONS, features=lambda: feats, seed=0)
    student.fit(docs["text"].tolist(), _targets(docs))
    assert feats.fit_calls == 1
    assert feats.seen_questions == list(QUESTIONS)
    assert feats.seen_targets["department"] == (120, 4)
    assert student.predict_proba(docs["text"].tolist()[:3])["angry"].shape == (3, 2)


def test_expensive_featurizer_gets_fewer_calibration_folds():
    docs = make_docs(150)
    built = []

    def factory():
        f = CountingTrainableFeatures()
        built.append(f)
        return f

    student = DecisionStudent(QUESTIONS, features=factory, seed=0)
    targets = _targets(docs)
    student.fit(docs["text"].tolist(), targets)
    student.calibrate(docs["text"].tolist(), targets, n_splits=5)
    # 1 for the student itself + 3 folds (not 5), each a fresh featurizer
    assert len(built) == 1 + 3


def test_encoder_name_resolves_to_the_fine_tuned_featurizer():
    torch = pytest.importorskip("torch")  # noqa: F841
    from shrewd.decisions import _featurizer
    from shrewd.encoder import EncoderFeatures

    assert isinstance(_featurizer("encoder", 42), EncoderFeatures)


def test_proper_loss_is_minimized_at_the_target():
    torch = pytest.importorskip("torch")
    from shrewd.encoder import proper_loss

    t = torch.tensor([[0.1, 0.6, 0.3]])
    exact = proper_loss(t, t, ordinal=True)
    for other in ([0.6, 0.1, 0.3], [0.1, 0.3, 0.6], [0.34, 0.33, 0.33]):
        assert proper_loss(torch.tensor([other]), t, ordinal=True) > exact


def test_ranked_probability_charges_for_distance_on_ordinal_questions():
    torch = pytest.importorskip("torch")
    from shrewd.encoder import proper_loss

    t = torch.tensor([[1.0, 0.0, 0.0, 0.0]])            # gold is level 0
    one_off = torch.tensor([[0.05, 0.85, 0.05, 0.05]])
    three_off = torch.tensor([[0.05, 0.05, 0.05, 0.85]])
    # same log score and spherical score, so only the ranked term separates them
    assert proper_loss(three_off, t, ordinal=True) > proper_loss(one_off, t, ordinal=True)
    assert proper_loss(three_off, t, ordinal=False) == pytest.approx(
        proper_loss(one_off, t, ordinal=False).item(), abs=1e-6
    )


def test_encoder_features_fine_tune_and_round_trip(tmp_path):
    """Real fine-tune, tiny: 48 docs, one epoch, short sequences."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from shrewd import load
    from shrewd.encoder import EncoderFeatures

    docs = make_docs(48)
    student = DecisionStudent(
        {"angry": QUESTIONS["angry"]},
        features=lambda: EncoderFeatures(epochs=1, batch_size=16, max_length=32, seed=0),
        seed=0,
    )
    targets = {"angry": _targets(docs)["angry"]}
    student.fit(docs["text"].tolist(), targets)
    before = student.predict_proba(docs["text"].tolist()[:4])["angry"]
    assert before.shape == (4, 2)

    student.features_kind = "encoder"
    student.save(tmp_path / "s")
    assert (tmp_path / "s" / "features" / "trunk").exists()
    import json

    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["features_saved_separately"] is True
    reloaded = DecisionStudent._load(tmp_path / "s", meta)
    after = reloaded.predict_proba(docs["text"].tolist()[:4])["angry"]
    assert np.allclose(before, after, atol=1e-4)
    del load
