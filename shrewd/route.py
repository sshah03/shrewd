"""Pick a confidence threshold with a guarantee on how often the student disagrees with the
teacher, and answer locally only above it.

The guarantee is on disagreement as a share of *all* requests: with probability 1 - delta over
the rows the threshold was set on, the expected share of requests that are answered locally
and disagree with the teacher is at most `budget`. It holds for traffic like those rows, scored
by the same model. Rows the model trained on do not count, and nor do out-of-fold scores from
other models: in testing both broke the budget in 10-30% of runs instead of at most 5%.

The threshold is chosen by fixed-sequence testing (Learn then Test): walk the candidate
thresholds from strict to loose, test each with a Clopper-Pearson upper bound at level delta,
and stop at the first that fails. Because the sequence is fixed before looking at the
outcomes, no multiplicity correction is needed.
"""

import numpy as np
from scipy.stats import beta


def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound on a rate after `k` events in `n` trials."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def cp_lower(k, n, delta):
    """One-sided Clopper-Pearson lower bound on a rate after `k` events in `n` trials."""
    if k <= 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))


def doc_scores(proba, keys=None):
    """A document's routing score: its lowest top probability over the questions.

    One teacher call answers the whole panel, so a document is routed as a unit and is only
    as sure as its least sure answer.
    """
    keys = list(proba) if keys is None else list(keys)
    return np.min([np.asarray(proba[k]).max(axis=1) for k in keys], axis=0)


def doc_disagree(proba, targets, keys=None):
    """True where any question the teacher answered gets a different top option."""
    keys = list(proba) if keys is None else list(keys)
    out = np.zeros(len(np.asarray(targets[keys[0]])), dtype=bool)
    for k in keys:
        t = np.asarray(targets[k])
        answered = t.max(axis=1) > 1e-9
        out |= answered & (np.asarray(proba[k]).argmax(axis=1) != t.argmax(axis=1))
    return out


def pick_threshold(scores, disagree, budget=0.02, delta=0.05):
    """The loosest threshold whose disagreement bound stays within `budget`.

    Returns (threshold, stats). The threshold is `inf` when not even the most confident row
    can be answered locally with the guarantee, which is the usual outcome on a few hundred
    rows: with no disagreements at all, n rows bound the rate at 1 - delta ** (1 / n), so at
    least 150 rows are needed to certify 2%.
    """
    scores = np.asarray(scores, dtype=float)
    disagree = np.asarray(disagree, dtype=bool)
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, cum = scores[order], np.cumsum(disagree[order])
    best, bound, k_best, m_best = np.inf, 0.0, 0, 0
    for i in range(n):
        if i + 1 < n and s_sorted[i + 1] == s_sorted[i]:
            continue          # ties route together, so only a distinct score is a cut point
        upper = cp_upper(int(cum[i]), n, delta)
        if upper > budget:
            break
        best, bound, k_best, m_best = float(s_sorted[i]), upper, int(cum[i]), i + 1
    stats = {
        "budget": budget,
        "delta": delta,
        "n": n,
        "threshold": None if best == np.inf else round(best, 6),
        "coverage": round(m_best / n, 4) if n else 0.0,
        "disagreements": k_best,
        "bound": round(bound, 4),
    }
    return best, stats
