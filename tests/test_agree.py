import numpy as np
import pytest

from shrewd.agree import MIN_ROWS, QuestionCombiner


def simulate(rng, n, k, accuracies, ordinal=False, sure=(0.6, 0.99)):
    """Teachers of given accuracy; a wrong answer is a random other option, or one level off
    when `ordinal`."""
    gold = rng.integers(0, k, n)
    votes = np.zeros((n, len(accuracies), k))
    for j, accuracy in enumerate(accuracies):
        right = rng.random(n) < accuracy
        if ordinal:
            wrong = np.clip(gold + rng.choice([-1, 1], n), 0, k - 1)
        else:
            wrong = (gold + rng.integers(1, k, n)) % k
        said = np.where(right, gold, wrong)
        conf = rng.uniform(*sure, n)
        votes[:, j] = ((1 - conf) / (k - 1))[:, None]
        votes[np.arange(n), j, said] = conf
    return votes, gold


def nll(proba, gold):
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(gold)), gold], 1e-12, None))))


@pytest.mark.parametrize("kind,k,ordinal", [("choice", 5, False), ("noul", 2, False),
                                            ("score", 5, True)])
def test_fitted_combination_beats_a_plain_average(kind, k, ordinal):
    rng = np.random.default_rng(0)
    dev_votes, dev_gold = simulate(rng, 300, k, [0.85, 0.75, 0.55], ordinal)
    test_votes, test_gold = simulate(rng, 4000, k, [0.85, 0.75, 0.55], ordinal)
    combined = QuestionCombiner(kind, k).fit(dev_votes, dev_gold).combine(test_votes)
    average = test_votes.mean(axis=1)
    assert np.allclose(combined.sum(axis=1), 1.0)
    assert nll(combined, test_gold) < nll(average, test_gold)
    accuracy = (combined.argmax(1) == test_gold).mean()
    assert accuracy >= (average.argmax(1) == test_gold).mean() - 0.01


def test_a_teacher_with_no_answer_drops_out_of_that_row():
    rng = np.random.default_rng(1)
    dev_votes, dev_gold = simulate(rng, 200, 4, [0.9, 0.6])
    combiner = QuestionCombiner("choice", 4).fit(dev_votes, dev_gold)
    rows = dev_votes[:10].copy()
    rows[:5, 0] = np.nan                       # the good teacher missed these
    rows[9] = np.nan                           # nobody answered this one
    out = combiner.combine(rows)
    assert np.isnan(out[9]).all()
    assert np.allclose(out[:9].sum(axis=1), 1.0)
    alone = QuestionCombiner("choice", 4).fit(dev_votes[:, 1:], dev_gold).combine(rows[:5, 1:])
    assert np.allclose(out[:5], alone)         # the same as a combiner of the poor one alone


def test_too_few_dev_rows_fall_back_to_averaging():
    rng = np.random.default_rng(2)
    votes, gold = simulate(rng, MIN_ROWS - 1, 3, [0.9, 0.7])
    combiner = QuestionCombiner("choice", 3).fit(votes, gold)
    assert combiner.method(2) == ("average", None)
    assert np.allclose(combiner.combine(votes), votes.mean(axis=1))


def test_unlabeled_dev_rows_are_ignored_when_fitting():
    rng = np.random.default_rng(3)
    votes, gold = simulate(rng, 200, 3, [0.9, 0.7])
    masked = gold.copy()
    masked[100:] = -1
    a = QuestionCombiner("choice", 3).fit(votes, masked).combine(votes)
    b = QuestionCombiner("choice", 3).fit(votes[:100], gold[:100]).combine(votes)
    assert np.allclose(a, b)


def test_cross_fit_never_grades_a_row_on_itself():
    rng = np.random.default_rng(4)
    votes, gold = simulate(rng, 150, 3, [0.8, 0.7])
    combiner = QuestionCombiner("choice", 3)
    out_of_fold = combiner.cross_fit(votes, gold)
    in_sample = combiner.fit(votes, gold).combine(votes)
    assert not np.allclose(out_of_fold, in_sample)
    assert nll(out_of_fold, gold) >= nll(in_sample, gold) - 1e-9


def test_correlated_mistakes_get_tempered():
    """Two copies of the same teacher are one opinion. Multiplying their evidence would be
    overconfident, so the fitted temperature should discount it."""
    rng = np.random.default_rng(5)
    votes, gold = simulate(rng, 400, 4, [0.75])
    twins = np.concatenate([votes, votes, votes], axis=1)
    method, tripled = QuestionCombiner("choice", 4).fit(twins, gold).method(3)
    _, single = QuestionCombiner("choice", 4).fit(votes, gold).method(1)
    assert method == "dawid-skene" and tripled < single
