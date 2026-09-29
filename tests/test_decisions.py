import json

import numpy as np
import pandas as pd
import pytest
from conftest import QUESTIONS, fake_answer, make_docs

from shrewd import decide
from shrewd.calibrate import (
    Calibrator,
    brier,
    brier_decomposition,
    calibration_metrics,
    cross_fit_proba,
    ece,
)
from shrewd.decide import Choice, Noul, Score
from shrewd.decisions import Decisions, DecisionStudent, gold_index, question_metrics

# ---------------------------------------------------------------- question types


def test_question_options():
    assert QUESTIONS["department"].options() == ["billing", "bug", "cancellation", "other"]
    assert QUESTIONS["angry"].options() == ["no", "yes"]
    assert QUESTIONS["severity"].options() == ["0", "1", "2"]


def test_score_rejects_silly_level_counts():
    with pytest.raises(ValueError, match="between 2 and 10"):
        Score(instructions="x", criteria=["only one"])
    with pytest.raises(ValueError, match="between 2 and 10"):
        Score(instructions="x", criteria=[str(i) for i in range(11)])


def test_choice_needs_two_options():
    with pytest.raises(ValueError, match="at least 2 options"):
        decide.validate({"q": Choice(instructions="x", criteria={"only": "one"})})


def test_question_ids_must_be_identifiers():
    with pytest.raises(ValueError, match="plain identifier"):
        decide.validate({"not an id": QUESTIONS["angry"]})


def test_questions_round_trip():
    for question in QUESTIONS.values():
        assert decide.from_dict(decide.to_dict(question)) == question


def test_score_answer_is_a_weighted_mean():
    answer = decide.make_answer("severity", QUESTIONS["severity"], [0.0, 0.7, 0.3])
    assert answer.score == pytest.approx(1.3)
    assert answer.probabilities == {"0": 0.0, "1": 0.7, "2": 0.3}


def test_noul_answer_is_one_number():
    answer = decide.make_answer("angry", QUESTIONS["angry"], [0.2, 0.8])
    assert answer.noul == pytest.approx(0.8)
    assert not hasattr(answer, "confidence")


def test_confidence_is_a_spread_statistic():
    peaked = decide.make_answer("department", QUESTIONS["department"], [0.97, 0.01, 0.01, 0.01])
    flat = decide.make_answer("department", QUESTIONS["department"], [0.25] * 4)
    assert peaked.confidence > 0.85
    assert flat.confidence == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- parsing


def test_parse_answers_reads_probabilities():
    content = fake_answer("the charge problem again", QUESTIONS)
    answers = decide.parse_answers(content, QUESTIONS)
    assert set(answers) == set(QUESTIONS)
    assert answers["department"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload",
    [
        '{"answers": {"angry": {"noul": 0.8}}}',
        '{"answers": {"angry": 0.8}}',
        '```json\n{"answers": {"angry": {"probabilities": {"yes": 0.8, "no": 0.2}}}}\n```',
        'chatter before {"answers": {"angry": {"probabilities": {"YES": 4, "no": 1}}}} after',
    ],
)
def test_parse_answers_is_lenient(payload):
    answers = decide.parse_answers(payload, {"angry": QUESTIONS["angry"]})
    assert answers["angry"][1] == pytest.approx(0.8)


def test_parse_answers_skips_what_it_cannot_read():
    answers = decide.parse_answers('{"answers": {"angry": "maybe?"}}', QUESTIONS)
    assert answers == {}
    assert decide.parse_answers("not json at all", QUESTIONS) == {}


def test_bare_label_becomes_a_one_hot():
    answers = decide.parse_answers(
        '{"answers": {"department": {"choice": "billing"}}}', QUESTIONS
    )
    assert answers["department"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------- calibration


def test_temperature_recovers_an_underconfident_student():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, size=3000)
    sharp = np.full((3000, 3), 0.05)
    sharp[np.arange(3000), truth] = 0.9
    # flatten toward uniform: the exact failure the gold holdouts show
    flat = sharp**0.25
    flat /= flat.sum(axis=1, keepdims=True)
    classes = ["a", "b", "c"]
    labels = [classes[i] for i in truth]
    before = calibration_metrics(flat, labels, classes)
    cal = Calibrator.fit(flat, labels, classes, method="temperature")
    after = calibration_metrics(cal.transform(flat), labels, classes)
    assert before["ece"] > 0.15
    assert after["ece"] < 0.03
    assert after["accuracy"] == before["accuracy"]


def test_calibration_never_changes_a_multiclass_prediction():
    rng = np.random.default_rng(1)
    proba = rng.dirichlet([1, 1, 1, 1], size=500)
    classes = list("abcd")
    labels = [classes[i] for i in proba.argmax(axis=1)]
    for method in ("temperature", "platt", "isotonic"):
        cal = Calibrator.fit(proba, labels, classes, method=method)
        assert (cal.transform(proba).argmax(axis=1) == proba.argmax(axis=1)).all()


def test_a_lopsided_yes_no_question_is_allowed_to_cross_half():
    """The deliberate exception to argmax preservation. A teacher that answers 0.55
    about something happening 3% of the time must be pulled below 0.5, or the whole
    point of calibrating a rare decision is lost."""
    rng = np.random.default_rng(7)
    score = rng.uniform(0.15, 0.85, size=800)
    proba = np.column_stack([1 - score, score])
    labels = ["yes" if rng.uniform() < 0.10 else "no" for _ in range(800)]
    cal = Calibrator.fit(proba, labels, ["no", "yes"], method="platt")
    calibrated = cal.transform(proba)[:, 1]
    assert cal.transform(np.array([[0.45, 0.55]]))[0, 1] < 0.5
    # the calibrated mean lands on the true rate, which is the property being bought
    assert abs(calibrated.mean() - np.mean([v == "yes" for v in labels])) < 0.02


def test_binary_ece_is_measured_on_p_yes():
    """With a 3% base rate, top-label ECE is dominated by easy "no"s and looks great
    while P(yes) is off by a factor of ten."""
    rng = np.random.default_rng(8)
    truth = (rng.uniform(size=4000) < 0.03).astype(int)
    inflated = np.where(truth == 1, 0.45, 0.30)  # always predicts "no", wildly overstated
    proba = np.column_stack([1 - inflated, inflated])
    labels = ["yes" if t else "no" for t in truth]
    stats = calibration_metrics(proba, labels, ["no", "yes"])
    assert stats["accuracy"] > 0.95          # argmax is right nearly every time
    assert stats["ece"] > 0.20               # and the probability is still nonsense


def test_calibrator_round_trips_through_json():
    rng = np.random.default_rng(2)
    proba = rng.dirichlet([2, 2], size=400)
    labels = ["yes" if p > 0.5 else "no" for p in proba[:, 1]]
    for method in ("temperature", "platt", "isotonic", "none"):
        cal = Calibrator.fit(proba, labels, ["no", "yes"], method=method)
        clone = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
        assert np.allclose(clone.transform(proba), cal.transform(proba))


def _biased_three_way(n, seed):
    """True distribution softmax(z); the head reports softmax(2z + bias): overconfident and
    tilted toward option 1, which a map on the top score alone can't undo."""
    from shrewd.calibrate import _softmax

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3)) * 1.5
    y = np.array([rng.choice(3, p=row) for row in _softmax(z)])
    return _softmax(2.0 * z + np.array([0.0, 1.0, -1.0])), y


def test_full_vector_maps_undo_a_per_option_bias():
    from shrewd.calibrate import nll

    proba, y = _biased_three_way(6000, 12)
    labels, classes = [str(v) for v in y], ["0", "1", "2"]
    fit, held = slice(0, 4000), slice(4000, None)
    losses, rates = {}, {}
    for method in ("temperature", "platt", "vector", "matrix"):
        cal = Calibrator.fit(proba[fit], labels[fit], classes, method=method)
        out = cal.transform(proba[held])
        losses[method], rates[method] = nll(out, y[held]), out.mean(axis=0)
    truth = np.bincount(y[held], minlength=3) / len(y[held])
    for method in ("vector", "matrix"):
        assert losses[method] < losses["temperature"] - 0.02
        assert np.abs(rates[method] - truth).max() < 0.02
    # the top-score map leaves option 1 over-stated
    assert np.abs(rates["platt"] - truth).max() > 0.05


def test_auto_never_picks_a_full_vector_map():
    """`auto` is what existing projects use; widening it would change their predictions."""
    proba, y = _biased_three_way(3000, 13)
    cal = Calibrator.fit(proba, [str(v) for v in y], ["0", "1", "2"], method="auto")
    assert cal.method in ("temperature", "isotonic", "platt", "none")
    wide = Calibrator.fit(proba, [str(v) for v in y], ["0", "1", "2"], method="auto-full")
    assert wide.method in ("vector", "matrix")


def test_full_vector_maps_round_trip_and_stay_sane_on_a_rare_option():
    rng = np.random.default_rng(14)
    y = np.array([0] * 900 + [1] * 98 + [2] * 2)       # option 2: two rows
    proba = np.clip(np.eye(3)[y] * 0.6 + rng.dirichlet([1, 1, 1], size=1000) * 0.4, 1e-6, 1)
    proba /= proba.sum(axis=1, keepdims=True)
    for method in ("vector", "matrix"):
        cal = Calibrator.fit(proba, [str(v) for v in y], ["0", "1", "2"], method=method)
        out = cal.transform(proba)
        assert np.isfinite(out).all() and np.allclose(out.sum(axis=1), 1.0)
        assert out[:, 2].mean() < 0.02                   # a rare option stays rare
        clone = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
        assert np.allclose(clone.transform(proba), out, atol=1e-5)


def test_loading_an_unknown_calibrator_says_to_upgrade():
    with pytest.raises(ValueError, match="newer shrewd"):
        Calibrator.from_dict({"method": "from-the-future", "params": {}})


def test_auto_declines_to_calibrate_on_too_little():
    rng = np.random.default_rng(3)
    proba = rng.dirichlet([1, 1], size=10)
    cal = Calibrator.fit(proba, ["no"] * 5 + ["yes"] * 5, ["no", "yes"], method="auto")
    assert cal.method == "none"


def test_cross_fit_proba_is_out_of_fold():
    docs = make_docs(200)
    oof, labels = cross_fit_proba(
        lambda: _TinyStudent(), docs["text"].tolist(), docs["department"].tolist(),
        ["billing", "bug", "cancellation", "other"], n_splits=4,
    )
    assert oof.shape == (200, 4)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert len(labels) == 200


class _TinyStudent:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self._pipe = Pipeline(
            [("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))]
        )

    @property
    def classes_(self):
        return [str(c) for c in self._pipe.classes_]

    def fit(self, texts, labels):
        self._pipe.fit(texts, labels)

    def predict_proba(self, texts):
        return self._pipe.predict_proba(texts)


def test_ece_of_a_perfectly_calibrated_signal_is_near_zero():
    rng = np.random.default_rng(4)
    conf = rng.uniform(0.5, 1.0, size=20000)
    correct = (rng.uniform(size=20000) < conf).astype(float)
    assert ece(conf, correct) < 0.02


# ---------------------------------------------------------------- metrics


def test_noul_metrics_report_the_base_rate():
    gold = np.array([1] * 5 + [0] * 95)
    proba = np.column_stack([np.full(100, 0.9), np.full(100, 0.1)])
    stats = question_metrics(QUESTIONS["angry"], proba, gold)
    assert stats["base_rate"] == pytest.approx(0.05)
    assert stats["type"] == "noul"


def test_brier_decomposition_adds_up_to_brier():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 3, size=5000)
    proba = rng.dirichlet([1, 1, 1], size=5000)
    proba[np.arange(5000), y] += 0.5
    proba /= proba.sum(axis=1, keepdims=True)
    parts = brier_decomposition(proba, y)
    total = parts["reliability"] - parts["resolution"] + parts["uncertainty"]
    # equal-mass bins drop the within-bin spread, so the identity is close, not exact
    assert total == pytest.approx(brier(proba, y), abs=0.02)


def test_base_rate_forecaster_has_no_resolution():
    """Perfectly calibrated and useless: the case ECE can't flag and resolution can."""
    rng = np.random.default_rng(10)
    y = rng.integers(0, 3, size=3000)
    rates = np.bincount(y, minlength=3) / len(y)
    flat = np.tile(rates, (3000, 1))
    stats = calibration_metrics(flat, [str(v) for v in y], ["0", "1", "2"])
    assert stats["ece"] < 0.05          # bin sampling noise on a constant forecast
    assert stats["resolution_share"] < 0.01
    sharp = np.eye(3)[y] * 0.8 + 0.2 / 3
    assert calibration_metrics(sharp, [str(v) for v in y], ["0", "1", "2"])[
        "resolution_share"] > 0.9


def test_noul_decomposition_is_on_p_yes():
    gold = np.array([1] * 20 + [0] * 80)
    p_yes = np.where(gold == 1, 0.8, 0.1)
    proba = np.column_stack([1 - p_yes, p_yes])
    stats = question_metrics(QUESTIONS["angry"], proba, gold)
    assert stats["brier_unc"] == pytest.approx(0.2 * 0.8, abs=1e-4)
    assert stats["brier_rel"] - stats["brier_res"] + stats["brier_unc"] == \
        pytest.approx(stats["brier"], abs=1e-3)


def test_uninformative_choice_head_is_a_finding():
    from shrewd.decisions import decision_findings

    rng = np.random.default_rng(11)
    gold = rng.integers(0, 3, size=600)
    flat = np.tile(np.bincount(gold, minlength=3) / 600, (600, 1))
    stats = question_metrics(QUESTIONS["department"], np.pad(flat, ((0, 0), (0, 1)),
                                                             constant_values=1e-6), gold)
    findings = decision_findings({"questions": {"department": {"student": stats}}})
    assert any("uninformative" in f.title for f in findings)


def test_score_metrics_use_distance_not_just_accuracy():
    gold = np.array([0, 1, 2, 1])
    near = np.array([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8], [0.1, 0.8, 0.1]])
    far = np.array([[0.0, 0.0, 1.0], [0.1, 0.8, 0.1], [1.0, 0.0, 0.0], [0.1, 0.8, 0.1]])
    assert question_metrics(QUESTIONS["severity"], near, gold)["mae"] < \
        question_metrics(QUESTIONS["severity"], far, gold)["mae"]


def test_gold_index_accepts_the_shapes_people_actually_write():
    noul = QUESTIONS["angry"]
    assert gold_index(noul, "yes") == 1
    assert gold_index(noul, True) == 1
    assert gold_index(noul, 0) == 0
    assert gold_index(noul, "FALSE") == 0
    assert gold_index(noul, "") is None
    assert gold_index(QUESTIONS["severity"], 2) == 2
    assert gold_index(QUESTIONS["severity"], "blocking") == 2      # by description
    assert gold_index(QUESTIONS["severity"], "Minor") == 0
    assert gold_index(QUESTIONS["severity"], "9") is None


# ---------------------------------------------------------------- the student


def test_student_shares_one_featurization_across_heads():
    docs = make_docs(160)
    student = DecisionStudent(QUESTIONS, seed=0)
    targets = {
        "department": _onehot(docs["department"], QUESTIONS["department"].options()),
        "angry": _onehot(docs["angry"], QUESTIONS["angry"].options()),
        "severity": _onehot(docs["severity"], QUESTIONS["severity"].options()),
    }
    student.fit(docs["text"].tolist(), targets)
    assert set(student.heads) == set(QUESTIONS)
    answers = student.decide(docs["text"].iloc[0])
    assert set(answers) == set(QUESTIONS)
    assert 0.0 <= answers["angry"].noul <= 1.0


def test_student_handles_a_question_with_one_answer():
    docs = make_docs(80)
    student = DecisionStudent({"angry": QUESTIONS["angry"]}, seed=0)
    targets = {"angry": _onehot(["no"] * 80, ["no", "yes"])}
    student.fit(docs["text"].tolist(), targets)
    proba = student.predict_proba(docs["text"].tolist()[:3])["angry"]
    assert proba.shape == (3, 2)
    assert (proba.argmax(axis=1) == 0).all()


def _onehot(values, options):
    index = {o: i for i, o in enumerate(options)}
    out = np.zeros((len(values), len(options)))
    for i, value in enumerate(list(values)):
        out[i, index[str(value)]] = 1.0
    return out


# ---------------------------------------------------------------- end to end


def test_full_pipeline_offline(tmp_path, decision_teacher):
    docs = make_docs(240)
    seed, pool = docs.iloc[:120], docs.iloc[120:]

    d = Decisions(
        tmp_path / "panel",
        questions=QUESTIONS,
        instructions="Customer support tickets.",
        teacher="test/fake-model",
    )
    d.add_seed(seed, test_frac=0.4)
    frame = d.judge(pool[["text"]])
    assert len(frame) == len(pool)
    for key, question in QUESTIONS.items():
        columns = [f"{key}__{o}" for o in question.options()]
        assert np.allclose(frame[columns].sum(axis=1), 1.0)

    result = d.distill()
    text = result.report()
    assert "shrewd decisions" in text
    assert "department" in text and "angry" in text and "severity" in text

    from shrewd import load

    loaded = load(tmp_path / "panel")
    answers = loaded.decide("ticket 999: the refund problem again this is unacceptable")
    assert answers["department"].choice in QUESTIONS["department"].options()
    assert 0.0 <= answers["angry"].noul <= 1.0
    assert 0.0 <= answers["severity"].score <= 2.0


def test_distill_with_full_vector_calibration(tmp_path, decision_teacher):
    docs = make_docs(240)
    d = Decisions(tmp_path / "panel", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:120], test_frac=0.4)
    d.judge(docs.iloc[120:][["text"]])
    d.distill(features="tfidf", calibration="auto-full")

    from shrewd import load

    loaded = load(tmp_path / "panel")
    assert set(loaded.calibrators) == set(QUESTIONS)
    proba = loaded.predict_proba(["ticket 7: charged twice for the refund"])
    for values in proba.values():
        assert np.allclose(values.sum(axis=1), 1.0)


def test_one_teacher_call_per_document(tmp_path, decision_teacher):
    docs = make_docs(100)
    d = Decisions(
        tmp_path / "panel", questions=QUESTIONS, teacher="test/fake-model",
    )
    d.add_seed(docs.iloc[:40], test_frac=0.4)
    before = decision_teacher.calls
    d.judge(docs.iloc[40:][["text"]])
    # 60 documents, 3 questions each: one call per document, not per question
    assert decision_teacher.calls - before == 60


def test_judging_resumes_from_cache(tmp_path, decision_teacher):
    docs = make_docs(60)
    d = Decisions(tmp_path / "panel", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:30], test_frac=0.4)
    d.judge(docs.iloc[30:][["text"]])
    calls = decision_teacher.calls
    d.judge(docs.iloc[30:][["text"]], overwrite=True)
    assert decision_teacher.calls == calls  # every answer came from the sqlite cache


def test_changing_a_question_is_refused(tmp_path, decision_teacher):
    d = Decisions(tmp_path / "panel", questions=QUESTIONS, teacher="test/fake-model")
    assert d.questions.keys() == QUESTIONS.keys()
    changed = dict(QUESTIONS, angry=Noul(instructions="Is the customer furious?"))
    with pytest.raises(ValueError, match="do not match"):
        Decisions(tmp_path / "panel", questions=changed, teacher="test/fake-model")


def test_decisions_project_rejects_a_classification_directory(tmp_path, project):
    with pytest.raises(ValueError, match="classification project"):
        Decisions(project.dir, questions=QUESTIONS, teacher="test/fake-model")


# ---------------------------------------------------------------- backends


def test_ensemble_averages_distributions_rather_than_voting(monkeypatch):
    from shrewd.judge import EnsembleBackend

    class Stub:
        name = "stub"

        def __init__(self, table):
            self.table = table

        def judge(self, state, questions, prompt, model, conn):
            return {"angry": np.array(self.table[model])}, 0.0

    ensemble = EnsembleBackend(["a/one", "b/two"])
    ensemble.backend = Stub({"a/one": [0.4, 0.6], "b/two": [0.1, 0.9]})
    answers, _ = ensemble.judge("x", {"angry": QUESTIONS["angry"]}, "p", None, None)
    # both members would have voted "yes"; averaging keeps that one was far less sure
    assert answers["angry"][1] == pytest.approx(0.75)


def test_ensemble_needs_two_models():
    from shrewd.judge import EnsembleBackend

    with pytest.raises(ValueError, match="at least 2 models"):
        EnsembleBackend(["only/one"])


def test_unknown_backend_is_refused():
    from shrewd.judge import resolve_backend

    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("telepathy")
    with pytest.raises(ValueError, match="needs models"):
        resolve_backend("ensemble")


def test_logprob_extraction_renormalizes_over_the_options():
    import types

    from shrewd.judge import _extract_logprobs

    entries = [
        types.SimpleNamespace(token="yes", logprob=np.log(0.6)),
        types.SimpleNamespace(token="no", logprob=np.log(0.2)),
        types.SimpleNamespace(token="maybe", logprob=np.log(0.2)),
    ]
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            logprobs=types.SimpleNamespace(
                content=[types.SimpleNamespace(top_logprobs=entries)]
            )
        )]
    )
    blob = json.loads(_extract_logprobs(resp, ["no", "yes"]))
    assert blob["probabilities"]["yes"] == pytest.approx(0.75)
    assert blob["probabilities"]["no"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "payload",
    [
        '{"answers": {"angry": {"true": 0.8, "false": 0.2}}}',
        '{"answers": {"angry": {"probabilities": {"true": 0.8, "false": 0.2}}}}',
        '{"answers": {"angry": {"1": 0.8, "0": 0.2}}}',
    ],
)
def test_yes_no_questions_answer_to_true_false(payload):
    """Models write true/false whatever the prompt asks for; a dropped answer costs
    far more than accepting the alias."""
    answers = decide.parse_answers(payload, {"angry": QUESTIONS["angry"]})
    assert answers["angry"][1] == pytest.approx(0.8)


def test_noul_prompt_names_yes_and_no_not_true_and_false():
    rendered = decide.build_prompt({"angry": QUESTIONS["angry"]})
    assert "probability of yes" in rendered
    assert '"<noul id>": <0-1>' in rendered


def test_head_survives_a_question_nothing_wins():
    """Calibrating an inflated teacher can leave a rare question with probability mass
    on yes everywhere and a winning yes nowhere. That must not crash the fit."""
    docs = make_docs(120)
    student = DecisionStudent({"angry": QUESTIONS["angry"]}, seed=0)
    target = np.column_stack([np.full(120, 0.93), np.full(120, 0.07)])
    student.fit(docs["text"].tolist(), {"angry": target})
    proba = student.predict_proba(docs["text"].tolist()[:4])["angry"]
    assert proba.shape == (4, 2)
    assert (proba.argmax(axis=1) == 0).all()


def test_a_head_that_predicts_the_base_rate_is_flagged_not_praised():
    """The trap proper scoring rules exist to catch: answering the base rate on every
    document scores a beautiful ECE and tells you nothing."""
    from shrewd.decisions import decision_findings

    metrics = {"questions": {"annoyance": {"student": {
        "n": 500, "type": "noul", "ece": 0.009, "base_rate": 0.03, "auroc": 0.47,
        "accuracy": 0.97, "mean_confidence": 0.03, "overconfidence": 0.0,
    }, "teacher": None}}}
    findings = decision_findings(metrics)
    assert [f.severity for f in findings] == ["fail"]
    assert "uninformative" in findings[0].title


def test_add_seed_warns_about_rare_answers_before_any_money_is_spent(tmp_path):
    """A rare question is limited by its minority answers, not its row count."""
    docs = make_docs(300)
    docs["angry"] = ["yes" if i < 6 else "no" for i in range(300)]  # 2% yes
    d = Decisions(tmp_path / "p", questions={"angry": QUESTIONS["angry"]},
                  teacher="test/fake-model")
    with pytest.warns(UserWarning, match="rarest answer"):
        d.add_seed(docs, test_frac=0.35)


class _CountFeatures:
    """A deliberately non-sklearn transformer, to prove the slot is genuinely open."""

    def __init__(self, width=64):
        self.width = width
        self.fitted = 0

    def fit_transform(self, texts, y=None):
        self.fitted += 1
        return self.transform(texts)

    def transform(self, texts):
        out = np.zeros((len(texts), self.width))
        for i, text in enumerate(texts):
            for token in str(text).lower().split():
                out[i, hash(token) % self.width] += 1.0
        return out


def test_a_custom_featurizer_can_be_passed_as_an_instance():
    docs = make_docs(160)
    targets = {"angry": _onehot(docs["angry"], ["no", "yes"])}
    student = DecisionStudent({"angry": QUESTIONS["angry"]}, features=_CountFeatures(), seed=0)
    student.fit(docs["text"].tolist(), targets)
    assert student.features_kind == "_CountFeatures"
    assert student.predict_proba(docs["text"].tolist()[:5])["angry"].shape == (5, 2)


def test_a_custom_featurizer_can_be_passed_as_a_factory():
    docs = make_docs(160)
    targets = {"angry": _onehot(docs["angry"], ["no", "yes"])}
    student = DecisionStudent(
        {"angry": QUESTIONS["angry"]}, features=lambda: _CountFeatures(32), seed=0
    )
    student.fit(docs["text"].tolist(), targets)
    assert student.predict_proba(docs["text"].tolist()[:5])["angry"].shape == (5, 2)


def test_cross_fit_calibration_gives_each_fold_its_own_featurizer():
    """Folds sharing one fitted featurizer would leak held-out rows into the fit, which
    is the whole thing cross-fitting exists to prevent."""
    docs = make_docs(200)
    targets = {"angry": _onehot(docs["angry"], ["no", "yes"])}
    shared = _CountFeatures()
    student = DecisionStudent({"angry": QUESTIONS["angry"]}, features=shared, seed=0)
    student.fit(docs["text"].tolist(), targets)
    student.calibrate(docs["text"].tolist(), targets, n_splits=4)
    # the instance handed in is cloned, never fitted itself
    assert shared.fitted == 0


def test_a_featurizer_without_transform_is_refused():
    with pytest.raises(TypeError, match="needs a fit_transform"):
        DecisionStudent({"angry": QUESTIONS["angry"]}, features=object())


def test_unknown_featurizer_name_says_what_is_allowed():
    with pytest.raises(ValueError, match="pass your own transformer"):
        DecisionStudent({"angry": QUESTIONS["angry"]}, features="wishful")


def test_the_student_is_bit_for_bit_deterministic(tmp_path):
    """Same input, same output, including across a save/load round trip and whatever
    batch the row happens to land in."""
    docs = make_docs(160)
    targets = {"angry": _onehot(docs["angry"], ["no", "yes"])}
    student = DecisionStudent({"angry": QUESTIONS["angry"]}, seed=0)
    student.fit(docs["text"].tolist(), targets)
    texts = docs["text"].tolist()[:40]
    first = student.predict_proba(texts)["angry"]
    assert np.array_equal(first, student.predict_proba(texts)["angry"])
    one_at_a_time = np.vstack([student.predict_proba([t])["angry"] for t in texts])
    assert np.array_equal(first, one_at_a_time)
    student.save(tmp_path / "s")
    import json as _json

    reloaded = DecisionStudent._load(
        tmp_path / "s", _json.loads((tmp_path / "s" / "meta.json").read_text())
    )
    assert np.array_equal(first, reloaded.predict_proba(texts)["angry"])


def test_asking_an_uncompiled_question_raises(tmp_path, decision_teacher):
    docs = make_docs(240)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:120], test_frac=0.4)
    d.judge(docs.iloc[120:][["text"]])
    d.distill()
    from shrewd import load

    dec = load(tmp_path / "p")
    with pytest.raises(KeyError, match="no head for"):
        dec.decide("some ticket", questions=["sarcasm"])
    # selecting a subset of compiled questions is fine
    assert set(dec.decide("some ticket", questions=["angry"])) == {"angry"}


def test_adding_a_question_only_pays_for_the_new_question(tmp_path, decision_teacher):
    """Keying the cache on the whole prompt would re-judge the entire pool to learn one
    new thing. For a real pool that is the difference between $0.30 and $5."""
    from shrewd.judge import judge_texts
    from shrewd.teacher import open_cache

    docs = make_docs(60)
    conn = open_cache(tmp_path / "cache.db")
    eight = QUESTIONS
    judge_texts(docs["text"].tolist(), eight, "test/fake-model", conn=conn)
    calls_after_first = decision_teacher.calls
    assert calls_after_first == 60

    # re-judging the same panel is free
    judge_texts(docs["text"].tolist(), eight, "test/fake-model", conn=conn)
    assert decision_teacher.calls == calls_after_first

    # adding a question costs one pass, and the eight already answered are reused
    nine = dict(eight, sarcasm=Noul(instructions="Is this sarcastic?"))
    decision_teacher.respond = lambda messages: json.dumps(
        {"answers": {"sarcasm": 0.3}}
    )
    frame, _ = judge_texts(docs["text"].tolist(), nine, "test/fake-model", conn=conn)
    assert decision_teacher.calls == calls_after_first + 60      # not 120
    assert np.allclose(frame["sarcasm__yes"], 0.3)
    # and the original eight survived untouched
    assert frame["angry__yes"].notna().all()


def test_judge_writes_a_review_queue_of_unsure_answers(tmp_path, decision_teacher):
    """The human-in-the-loop step is editing pool_judged.csv; needs_review.csv says where."""
    docs = make_docs(120)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:40], test_frac=0.4)
    # make the fake teacher unsure about severity on every document
    decision_teacher.respond = lambda messages: json.dumps({"answers": {
        "department": {"billing": 0.9, "bug": 0.05, "cancellation": 0.03, "other": 0.02},
        "angry": 0.95, "severity": {"0": 0.45, "1": 0.40, "2": 0.15}}})
    d.judge(docs.iloc[40:][["text"]])
    review = pd.read_csv(tmp_path / "p" / "needs_review.csv")
    assert set(review["question"]) == {"severity"}
    assert len(review) == 80
    assert review["teacher_prob"].max() < 0.6
    assert review["why"].str.startswith("unsure").all()


def test_a_human_answer_in_the_review_queue_overrules_the_teacher(tmp_path, decision_teacher):
    docs = make_docs(120)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:40], test_frac=0.4)
    decision_teacher.respond = lambda messages: json.dumps({"answers": {
        "department": {"billing": 0.9, "bug": 0.05, "cancellation": 0.03, "other": 0.02},
        "angry": 0.95, "severity": {"0": 0.45, "1": 0.40, "2": 0.15}}})
    d.judge(docs.iloc[40:][["text"]])
    review = pd.read_csv(tmp_path / "p" / "needs_review.csv", dtype=str, keep_default_na=False)
    assert "human_answer" in review.columns and (review["human_answer"] == "").all()
    fixed_text = review["text"].iloc[0]
    review.loc[0, "human_answer"] = "blocking"           # level 2 by name
    review.loc[1, "human_answer"] = "2"                   # level 2 by number
    extra = {"text": docs["text"].iloc[41], "question": "angry", "teacher_answer": "",
             "teacher_prob": "", "why": "", "human_answer": "no"}
    review = pd.concat([review, pd.DataFrame([extra])])   # a row the queue never listed
    review.to_csv(tmp_path / "p" / "needs_review.csv", index=False)

    applied = d.apply_review()
    assert applied == 3
    pool = pd.read_csv(tmp_path / "p" / "pool_judged.csv")
    row = pool[pool["text"] == fixed_text].iloc[0]
    assert row["severity__2"] == 1.0 and row["severity__0"] == 0.0
    added = pool[pool["text"] == docs["text"].iloc[41]].iloc[0]
    assert added["angry__no"] == 1.0 and added["angry__yes"] == 0.0
    ledger = pd.read_csv(tmp_path / "p" / "reviewed.csv")
    assert len(ledger) == 3

    # a re-judge with overwrite=True cannot lose the fixes
    d.judge(docs.iloc[40:][["text"]], overwrite=True)
    pool = pd.read_csv(tmp_path / "p" / "pool_judged.csv")
    assert pool[pool["text"] == fixed_text].iloc[0]["severity__2"] == 1.0

    result = d.distill(features="tfidf")
    assert result.metrics["human_reviewed"] == 3
    assert "overruled by hand" in result.report()


def test_an_unusable_human_answer_is_reported_not_applied(tmp_path, decision_teacher):
    docs = make_docs(100)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:40], test_frac=0.4)
    d.judge(docs.iloc[40:][["text"]])
    pd.DataFrame([{"text": docs["text"].iloc[41], "question": "severity", "teacher_answer": "",
                   "teacher_prob": "", "why": "", "human_answer": "catastrophic"}]
                 ).to_csv(tmp_path / "p" / "needs_review.csv", index=False)
    with pytest.warns(UserWarning, match="not one of"):
        assert d.apply_review() == 0
    assert not (tmp_path / "p" / "reviewed.csv").exists()


def test_an_ensemble_of_teachers_survives_reload(tmp_path, decision_teacher):
    """Averaging several frontier models' distributions is how a reference label is built
    when no single model is trusted; the setup has to come back when the project reopens."""
    from shrewd.judge import EnsembleBackend

    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model",
                  backend=EnsembleBackend(["test/a", "test/b"], weights=[2, 1]))
    assert json.loads((d.dir / "manifest.json").read_text())["backend"] == {
        "name": "ensemble", "models": ["test/a", "test/b"], "weights": [2 / 3, 1 / 3]}
    reopened = Decisions(tmp_path / "p")
    assert reopened.backend.name == "ensemble"
    assert reopened.backend.models == ["test/a", "test/b"]
    assert reopened.backend.weights == pytest.approx([2 / 3, 1 / 3])
    # and it is exercised: two models -> two calls per document
    docs = make_docs(60)
    reopened.add_seed(docs.iloc[:30], test_frac=0.4)
    before = decision_teacher.calls
    reopened.judge(docs.iloc[30:][["text"]])
    assert decision_teacher.calls - before == 2 * 30


def test_pending_human_answers_survive_a_rejudge(tmp_path, decision_teacher):
    """Filling in needs_review.csv and then re-running judge() must not lose the edit."""
    docs = make_docs(100)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:40], test_frac=0.4)
    decision_teacher.respond = lambda messages: json.dumps({"answers": {
        "department": {"billing": 0.9, "bug": 0.05, "cancellation": 0.03, "other": 0.02},
        "angry": 0.95, "severity": {"0": 0.45, "1": 0.40, "2": 0.15}}})
    d.judge(docs.iloc[40:][["text"]])
    review = pd.read_csv(tmp_path / "p" / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_answer"] = "blocking"
    text = review.loc[0, "text"]
    review.to_csv(tmp_path / "p" / "needs_review.csv", index=False)
    d.judge(docs.iloc[40:][["text"]], overwrite=True)        # no apply_review() in between
    pool = pd.read_csv(tmp_path / "p" / "pool_judged.csv")
    assert pool[pool["text"] == text].iloc[0]["severity__2"] == 1.0
    assert (tmp_path / "p" / "reviewed.csv").exists()


def test_auto_features_is_picked_on_gold_dev(tmp_path, decision_teacher, capsys):
    """Neither tf-idf nor embeddings is the right default; the dev split decides."""
    pytest.importorskip("model2vec")
    docs = make_docs(240)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:120], test_frac=0.4)
    d.judge(docs.iloc[120:][["text"]])
    result = d.distill(features="auto")
    out = capsys.readouterr().out
    assert 'features="auto":' in out and "on gold dev" in out
    assert result.metrics["student_type"].split()[0] in ("tfidf", "embed")


def test_a_bare_student_treats_auto_as_the_safe_choice():
    from shrewd.decisions import resolve_features

    assert resolve_features("auto") == "tfidf"



def test_loading_a_model_without_its_extra_names_the_install(monkeypatch):
    import importlib.util

    from shrewd import decisions

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r'pip install "shrewd\[embed\]"'):
        decisions._check_extra("embed")
    decisions._check_extra("tfidf")  # needs nothing extra


def _routed_panel(tmp_path, budget):
    docs = make_docs(400)
    d = Decisions(tmp_path / "panel", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:120], test_frac=0.4)
    d.judge(docs.iloc[120:][["text"]])
    return d, d.distill(features="tfidf", route_budget=budget)


def test_routing_threshold_is_set_on_documents_the_model_never_saw(tmp_path, decision_teacher):
    d, result = _routed_panel(tmp_path, budget=0.9)
    routing = result.metrics["routing"]
    # 30% of the 280 judged documents are held out; the rest plus 72 dev rows train
    assert routing["n"] == 84 and result.metrics["n_train"] == 196 + 72
    assert routing["threshold"] is not None and routing["bound"] <= 0.9
    assert routing["test"]["n_local"] > 0
    assert "routing: answer locally" in result.report()

    from shrewd import load

    loaded = load(tmp_path / "panel")
    assert loaded.routing["threshold"] == routing["threshold"]
    texts = ["ticket 1: the refund problem again", "ticket 2: the crash problem again"] * 20
    local = loaded.local(texts)
    routed = loaded.decide(texts, route=True)
    plain = loaded.decide(texts)
    for ok, r, p in zip(local, routed, plain, strict=True):
        assert (r is None) == (not ok)
        if ok:
            assert {k: a.payload for k, a in r.items()} == {k: a.payload for k, a in p.items()}


def test_fallback_answers_what_the_student_does_not(tmp_path, decision_teacher):
    d, _ = _routed_panel(tmp_path, budget=0.9)
    student = __import__("shrewd").load(tmp_path / "panel")
    student.routing["threshold"] = 2.0          # nothing clears it: everything falls back
    texts = ["ticket 1: the refund problem again", "ticket 2: the crash problem again"]
    rows = student.decide(texts, fallback=d.ask)
    assert rows == d.ask(texts)
    assert set(rows[0]) == set(QUESTIONS)
    single = student.decide(texts[0], fallback=d.ask)
    assert single["department"].choice == rows[0]["department"].choice
    with pytest.raises(ValueError, match="fallback returned"):
        student.decide(texts, fallback=lambda ts: [])


def test_a_budget_too_tight_for_the_data_answers_nothing_locally(tmp_path, decision_teacher):
    _, result = _routed_panel(tmp_path, budget=0.001)
    assert result.metrics["routing"]["threshold"] is None
    assert "could not certify" in result.report()
    loaded = __import__("shrewd").load(tmp_path / "panel")
    assert loaded.decide(["ticket 1: the refund problem again"], route=True) == [None]


def test_routing_without_a_threshold_says_how_to_get_one(tmp_path, decision_teacher):
    docs = make_docs(240)
    d = Decisions(tmp_path / "panel", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:120], test_frac=0.4)
    d.judge(docs.iloc[120:][["text"]])
    d.distill(features="tfidf")
    loaded = __import__("shrewd").load(tmp_path / "panel")
    assert loaded.routing is None
    with pytest.raises(ValueError, match="route_budget"):
        loaded.decide(["ticket 1"], route=True)
