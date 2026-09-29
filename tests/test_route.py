import numpy as np
import pytest

from shrewd.route import cp_lower, cp_upper, doc_disagree, doc_scores, pick_threshold


def test_clopper_pearson_with_no_events_is_the_closed_form():
    # k = 0: the upper bound is 1 - delta ** (1 / n)
    assert cp_upper(0, 150, 0.05) == pytest.approx(1 - 0.05 ** (1 / 150))
    assert cp_upper(3, 3, 0.05) == 1.0
    assert cp_upper(0, 0, 0.05) == 1.0
    assert cp_lower(0, 10, 0.05) == 0.0
    assert cp_lower(5, 100, 0.05) < 0.05 < cp_upper(5, 100, 0.05)


def test_a_document_is_as_sure_as_its_least_sure_answer():
    proba = {"a": np.array([[0.9, 0.1], [0.6, 0.4]]), "b": np.array([[0.2, 0.8], [0.99, 0.01]])}
    assert doc_scores(proba).tolist() == pytest.approx([0.8, 0.6])


def test_disagreement_ignores_questions_the_teacher_did_not_answer():
    proba = {"a": np.array([[0.9, 0.1], [0.9, 0.1]]), "b": np.array([[0.9, 0.1], [0.9, 0.1]])}
    targets = {"a": np.array([[1.0, 0.0], [0.0, 0.0]]), "b": np.array([[0.0, 1.0], [0.0, 1.0]])}
    assert doc_disagree(proba, targets, ["a"]).tolist() == [False, False]
    assert doc_disagree(proba, targets).tolist() == [True, True]


def test_too_few_rows_answer_nothing_locally():
    scores, disagree = np.linspace(0.5, 1, 100), np.zeros(100, dtype=bool)
    t, stats = pick_threshold(scores, disagree, budget=0.02)
    assert t == np.inf and stats["coverage"] == 0.0 and stats["threshold"] is None


def test_threshold_stops_at_the_first_failure():
    rng = np.random.default_rng(0)
    n = 2000
    scores = rng.uniform(size=n)
    # disagreement only among the least confident half
    disagree = (scores < 0.5) & (rng.uniform(size=n) < 0.3)
    t, stats = pick_threshold(scores, disagree, budget=0.02)
    assert 0.3 < t < 0.5
    assert stats["bound"] <= 0.02
    assert stats["disagreements"] == int((disagree & (scores >= t)).sum())


def test_tied_scores_route_together():
    scores = np.array([1.0] * 400 + [0.9] * 400)
    disagree = np.zeros(800, dtype=bool)
    disagree[400:420] = True
    t, stats = pick_threshold(scores, disagree, budget=0.02)
    assert t == 1.0 and stats["coverage"] == 0.5


def test_the_guarantee_holds_on_fresh_draws():
    # true risk at any threshold is known here, so check the 95% claim directly
    rng = np.random.default_rng(1)
    broke = 0
    for _ in range(300):
        s = rng.uniform(size=400)
        d = rng.uniform(size=400) < 0.12 * (1 - s)   # risk grows as confidence falls
        t, _ = pick_threshold(s, d, budget=0.02)
        # true risk of routing s >= t: integral of 0.12 (1 - s) over [t, 1]
        true = 0.0 if t == np.inf else 0.06 * (1 - t) ** 2
        broke += true > 0.02
    assert broke / 300 <= 0.08
