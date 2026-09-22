import json

import numpy as np
import pandas as pd
from conftest import KEYWORDS, QUESTIONS, make_docs

from shrewd import Choice, Noul, Score
from shrewd.decisions import Decisions, DecisionStudent, gold_frame
from shrewd.zeroshot import combine, fit_bias, fit_weights, hypotheses


class KeywordScorer:
    """Stands in for the NLI model: an informative but imperfect second opinion.

    Sees the keyword the fake teacher keys on, adds noise, and never touches the label.
    """

    def __init__(self, strength=3.0, seed=0):
        self.strength = strength
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def score(self, texts, question, instructions=None):
        self.calls += 1
        options = question.options()
        out = np.full((len(texts), len(options)), 1.0)
        for i, text in enumerate(texts):
            low = str(text).lower()
            if question.kind == "choice":
                for keyword, label in KEYWORDS:
                    if keyword in low and label in options:
                        out[i, options.index(label)] += self.strength
            elif question.kind == "noul":
                out[i, 1] += self.strength if "unacceptable" in low else 0.0
        out += self.rng.uniform(0, 0.5, size=out.shape)
        return out / out.sum(axis=1, keepdims=True)

    def to_dict(self):
        return {"model": "keyword-fake"}


class NoiseScorer(KeywordScorer):
    """Pure noise: the gate must throw this away."""

    def score(self, texts, question, instructions=None):
        self.calls += 1
        k = len(question.options())
        out = self.rng.uniform(0.2, 1.0, size=(len(texts), k))
        return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- hypotheses


def test_hypotheses_never_use_raw_label_names():
    q = Choice(
        instructions="Which group?",
        criteria={"rec.autos": "cars: buying and maintaining", "sci.electronics": "circuit design"},
    )
    for hyp in hypotheses(q):
        assert "rec.autos" not in hyp and "sci.electronics" not in hyp
    assert hypotheses(q)[0] == "This text is about cars: buying and maintaining."


def test_noul_hypotheses_are_the_criteria_statements():
    q = Noul(
        instructions="Does this express anger?",
        criteria={"true": "The comment expresses anger.", "false": "The comment is calm."},
    )
    assert hypotheses(q) == ["The comment is calm.", "The comment expresses anger."]


def test_hypothesis_template_overrides_the_frame():
    q = Score(
        instructions="How severe?", criteria=["cosmetic", "blocking"],
        hypothesis="This issue is {description}.",
    )
    assert hypotheses(q) == ["This issue is cosmetic.", "This issue is blocking."]


def test_structured_descriptions_are_flattened():
    q = Choice(
        instructions="x",
        criteria={"a": {"what": "refunds", "examples": ["wrong size", "double charge"]},
                  "b": "shipping"},
    )
    assert hypotheses(q)[0] == "This text is about refunds wrong size, double charge."


# ---------------------------------------------------------------- the arithmetic


def test_fit_bias_recovers_a_known_offset():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=2000)
    true_logits = np.full((2000, 3), -1.0)
    true_logits[np.arange(2000), y] = 1.0
    skew = np.array([0.0, 0.0, 2.0])          # option 2 is over-scored by 2 nats
    observed = true_logits + skew
    bias = fit_bias(observed, y)
    assert bias[2] < bias[0] - 1.5 and bias[2] < bias[1] - 1.5
    assert abs(bias.mean()) < 1e-6


def test_combine_with_zero_stack_weight_is_the_head_alone():
    head = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    zs = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    out = combine(head, zs, bias=[0, 0, 0], weights=[1.0, 0.0])
    assert np.allclose(out, head, atol=1e-6)


def test_fit_weights_trusts_the_informative_source():
    """An informative-but-imperfect source gets the weight; pure noise gets ~none.
    Imperfect on purpose: a perfectly separable source has no finite optimum."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=1500)
    flipped = np.where(rng.uniform(size=1500) < 0.2, 1 - y, y)      # 80% right
    good = np.where(flipped[:, None] == np.arange(2), 0.75, 0.25)
    noise = rng.uniform(0.3, 0.7, size=(1500, 2))
    w = fit_weights(np.log(good), np.log(noise), y)
    assert w[0] > 0.5
    assert abs(w[1]) < 0.3


# ---------------------------------------------------------------- the gate


def _panel(tmp_path, decision_teacher, n=300):
    docs = make_docs(n)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[: n // 2], test_frac=0.4)
    d.judge(docs.iloc[n // 2 :][["text"]])
    return d


def test_informative_zero_shot_is_kept_and_used(tmp_path, decision_teacher):
    d = _panel(tmp_path, decision_teacher)
    scorer = KeywordScorer()
    result = d.distill(features="tfidf", zero_shot=scorer, stack_margin=-1.0)
    stacking = result.metrics["stacking"]
    assert set(stacking) <= set(QUESTIONS)
    assert any(s["kept"] for s in stacking.values())
    # a kept stack is exercised at prediction time
    from shrewd import load

    dec = load(tmp_path / "p")
    dec.zero_shot = scorer                       # the fake is not reconstructible from meta
    calls = scorer.calls
    dec.decide("ticket 1: the refund problem again this is unacceptable")
    assert scorer.calls > calls
    assert "zero-shot stack" in result.report()


def test_noise_zero_shot_is_rejected_by_the_gold_gate(tmp_path, decision_teacher):
    d = _panel(tmp_path, decision_teacher)
    result = d.distill(features="tfidf", zero_shot=NoiseScorer(), stack_margin=0.02)
    stacking = result.metrics["stacking"]
    assert stacking and not any(s.get("kept") for s in stacking.values())
    from shrewd import load

    dec = load(tmp_path / "p")
    assert dec.stacks == {} and dec.zero_shot is None
    section = result.report().split("zero-shot stack")[1].split("findings")[0]
    assert "rejected" in section and "kept" not in section


def test_the_gate_never_sees_dev_through_the_heads(tmp_path, decision_teacher):
    """The gate compares candidates on dev; the heads used for that must not have
    trained on dev, or the plain head is scored on memorized rows."""
    d = _panel(tmp_path, decision_teacher)
    dev = pd.read_csv(d.dir / "seed_dev.csv")
    pool = pd.read_csv(d.dir / "pool_judged.csv")
    student = DecisionStudent(QUESTIONS, features="tfidf", seed=0)
    student.zero_shot = KeywordScorer()
    targets = d._targets_from_pool(pool)
    student.fit(pool["text"].astype(str).tolist(), targets)
    seen = []
    original_fit = DecisionStudent.fit

    def spy(self, texts, targets_):
        seen.append(set(map(str, texts)))
        return original_fit(self, texts, targets_)

    DecisionStudent.fit = spy
    try:
        student.stack(pool["text"].astype(str).tolist(), targets,
                      dev["text"].astype(str).tolist(), gold_frame(dev, QUESTIONS))
    finally:
        DecisionStudent.fit = original_fit
    dev_texts = set(dev["text"].astype(str))
    # every refit during stack() (the pool-only twin and the cross-fit folds) excludes dev
    assert seen and all(not (s & dev_texts) for s in seen)


def test_stack_params_round_trip_through_meta(tmp_path, decision_teacher):
    d = _panel(tmp_path, decision_teacher)
    d.distill(features="tfidf", zero_shot=KeywordScorer(), stack_margin=-1.0)
    meta = json.loads((d.dir / "student" / "meta.json").read_text())
    assert meta["stacks"] and meta["zero_shot"] == {"model": "keyword-fake"}
    for params in meta["stacks"].values():
        assert len(params["weights"]) == 2 and len(params["bias"]) >= 2


def test_gate_abstains_without_enough_gold(tmp_path, decision_teacher):
    docs = make_docs(120)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:40], test_frac=0.4)          # 24 dev rows: below MIN_GATE_ROWS
    d.judge(docs.iloc[40:][["text"]])
    result = d.distill(features="tfidf", zero_shot=KeywordScorer())
    assert all(s.get("reason") for s in result.metrics["stacking"].values())
    assert not any(s.get("kept") for s in result.metrics["stacking"].values())


class OracleScorer(KeywordScorer):
    """Knows the answer from a side table; the text itself carries no lexical signal."""

    def __init__(self, truth):
        super().__init__()
        self.truth = truth

    def score(self, texts, question, instructions=None):
        self.calls += 1
        k = len(question.options())
        out = np.full((len(texts), k), 0.1)
        for i, text in enumerate(texts):
            out[i, self.truth.get(str(text), 0)] = 0.9
        return out / out.sum(axis=1, keepdims=True)


def test_rare_yes_no_questions_are_gated_on_auroc_not_f1():
    """With a 4% base rate there are a handful of positives on dev. Macro-F1 over the
    argmax cannot see a large ranking improvement there; AUROC can."""
    rng = np.random.default_rng(5)
    words = ["alpha", "beta", "gamma", "delta", "kappa", "sigma"]
    texts = [" ".join(rng.choice(words, size=6)) + f" #{i}" for i in range(400)]
    truth = {t: int(i % 25 == 0) for i, t in enumerate(texts)}          # 4% positive
    q = {"rare": Noul(instructions="Is this the rare thing?")}
    pool, dev = texts[:300], texts[300:]
    target = np.zeros((300, 2))
    target[np.arange(300), [truth[t] for t in pool]] = 1.0
    dev_gold = np.array([[truth[t]] for t in dev])
    student = DecisionStudent(q, features="tfidf", seed=0)
    student.zero_shot = OracleScorer(truth)
    student.fit(pool, {"rare": target})
    student.stack(pool, {"rare": target}, dev, dev_gold, margin=0.02)
    report = student.stack_report["rare"]
    assert report["metric"] == "auroc"
    assert report["kept"] and report["dev_gain"] > 0.3
    assert "rare" in student.stacks


def test_multiclass_questions_are_still_gated_on_macro_f1(tmp_path, decision_teacher):
    d = _panel(tmp_path, decision_teacher)
    result = d.distill(features="tfidf", zero_shot=KeywordScorer(), stack_margin=-1.0)
    stacking = result.metrics["stacking"]
    assert stacking["department"]["metric"] == "macro_f1"
    assert stacking["severity"]["metric"] == "macro_f1"
    assert stacking["angry"]["metric"] == "auroc"
    assert "AUROC" in result.report() and "F1" in result.report()

