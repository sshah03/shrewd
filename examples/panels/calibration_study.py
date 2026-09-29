"""`calibration="auto"` vs `"auto-full"` on the pre-built panels. No API calls.

    python examples/panels/fetch.py
    python examples/panels/build.py <panel> --from-judged          # for each panel
    python examples/panels/calibration_study.py [panel ...] [--reps 200]

For every question it refits both calibrators on the same out-of-fold student
probabilities over the judged pool (what distill() does) and reports:

  picks         which method each setting chose for each question
  vs teacher    for the multi-option heads, by 5-fold nested cross-validation on the pool:
                log loss against the teacher's answers, mean error of the predicted option
                rates, ECE of the top option's stated confidence, agreement with the
                teacher's answer, and how often auto-full picks a different option than auto
  gold          holdout Brier for the yes/no questions with human labels
  derived gold  holdout Brier of a multi-option answer that is logically a yes/no gold
                question, e.g. P(kind in {promotional, scam}) against gold `unsolicited`.
                No multi-option question has human labels, so this is the only direct
                check against people rather than the teacher
  consistency   mean violation of the rules in consistency.py

Intervals are paired bootstraps that resample both the pool rows the calibrators are fit
on and the holdout rows they're scored on. Negative differences favor auto-full.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consistency import RULES, frames_from_student  # noqa: E402
from panels import GOLD, PANELS  # noqa: E402

from shrewd import load  # noqa: E402
from shrewd.calibrate import Calibrator, ece, nll  # noqa: E402
from shrewd.decisions import gold_frame  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS = ("auto", "auto-full")

# a gold yes/no question that is (approximately) a set of options of a multi-option one.
# pii's are loose: `high` risk also covers identity numbers and device identifiers
DERIVED = {
    "messages": [("unsolicited", "kind", ["promotional", "scam"])],
    "email": [("spam", "intent", ["credential_theft", "payment_fraud", "malware_lure",
                                  "other_scam"])],
    "guardrail": [("harmful", "handling", ["refuse"]), ("jailbreak", "handling", ["refuse"])],
    "pii": [("financial", "share_risk", ["high"]), ("credentials", "share_risk", ["high"]),
            ("contact", "share_risk", ["moderate", "high"])],
}


def pool_targets(pool, questions):
    out = {}
    for key, q in questions.items():
        v = np.nan_to_num(pool[[f"{key}__{o}" for o in q.options()]].to_numpy(float))
        total = v.sum(axis=1, keepdims=True)
        out[key] = np.divide(v, total, out=np.zeros_like(v), where=total > 1e-9)
    return out


def rule_gap(rule, frames):
    _, kind, lhs, rhs = rule
    a, b = lhs(frames), rhs(frames)
    return (np.maximum(0.0, a - b) if kind == "implies" else np.abs(a - b)).mean()


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    if hi < 0:
        verdict = "auto-full better"
    else:
        verdict = "auto-full worse" if lo > 0 else "no clear difference"
    return f"{np.mean(values):+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}"


def study(name, reps, rng):
    run = ROOT / "runs" / f"panel-{name}"
    questions = PANELS[name]["questions"]
    keys = list(questions)
    student = load(run)
    pool = pd.read_csv(run / "pool_judged.csv")
    pool["text"] = pool["text"].astype(str)
    targets = pool_targets(pool, questions)
    oof = student._oof(pool["text"].tolist(), targets)
    hold = pd.read_csv(ROOT / "data" / name / "holdout.csv")
    for key in questions:
        if key not in hold:
            hold[key] = None
    gold = gold_frame(hold, questions)
    raw = student.predict_proba(hold["text"].astype(str).tolist(), calibrated=False)

    rows = {}
    for key, q in questions.items():
        keep = np.where(targets[key].max(axis=1) > 1e-9)[0]
        rows[key] = (oof[key][keep], targets[key][keep].argmax(axis=1), q.options())

    def fit(key, setting, idx=None):
        P, y, options = rows[key]
        idx = np.arange(len(y)) if idx is None else idx
        return Calibrator.fit(P[idx], [options[i] for i in y[idx]], options, method=setting)

    print(f"\n=== {name} ({student.features_kind} student)")
    picks = {s: {k: fit(k, s).method for k in keys} for s in SETTINGS}
    for s in SETTINGS:
        print(f"  picks, {s:9}: " + "  ".join(f"{k}={m}" for k, m in picks[s].items()))

    multi = [k for k, q in questions.items() if q.kind != "noul"]
    print("  vs teacher, nested 5-fold on the pool "
          "(log loss / option-rate error / top-option ECE / agreement):")
    for key in multi:
        P, y, options = rows[key]
        cells, winners = [], []
        for s in SETTINGS:
            held = np.zeros_like(P)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(P, y):
                held[te] = fit(key, s, tr).transform(P[te])
            rates = np.bincount(y, minlength=P.shape[1]) / len(y)
            rate_err = np.abs(held.mean(axis=0) - rates).mean()
            top = held.argmax(axis=1)
            winners.append(top)
            cells.append(f"{s} {nll(held, y):.4f} / {rate_err:.4f} / "
                         f"{ece(held.max(axis=1), top == y):.4f} / {np.mean(top == y):.3f}")
        changed = np.mean(winners[0] != winners[1])
        print(f"    {key:12} " + "   ".join(cells) + f"   winner changed on {changed:.1%}")

    diffs = {}
    n_hold = gold.shape[0]
    for _ in range(reps):
        h = rng.integers(0, n_hold, n_hold)
        # one pool resample per question, shared by both settings, so the comparison is paired
        idx = {key: rng.integers(0, len(rows[key][1]), len(rows[key][1])) for key in keys}
        probs = {s: {key: fit(key, s, idx[key]).transform(raw[key][h]) for key in keys}
                 for s in SETTINGS}
        for key in GOLD[name]:
            j = keys.index(key)
            m = gold[h, j] >= 0
            t = (gold[h, j][m] == 1).astype(float)
            err = [np.mean((probs[s][key][m][:, 1] - t) ** 2) for s in SETTINGS]
            diffs.setdefault(f"gold {key}", []).append(err[1] - err[0])
        for gold_q, key, opts in DERIVED.get(name, []):
            j = keys.index(gold_q)
            m = gold[h, j] >= 0
            t = (gold[h, j][m] == 1).astype(float)
            oi = [questions[key].options().index(o) for o in opts]
            err = [np.mean((probs[s][key][m][:, oi].sum(axis=1) - t) ** 2) for s in SETTINGS]
            label = f"derived {gold_q} <- {key} in {{{', '.join(opts)}}}"
            diffs.setdefault(label, []).append(err[1] - err[0])
        for rule in RULES[name]:
            gaps = [rule_gap(rule, frames_from_student(probs[s], questions)) for s in SETTINGS]
            diffs.setdefault(f"rule {rule[0]}", []).append(gaps[1] - gaps[0])
    print(f"  holdout, auto-full minus auto (Brier or mean violation), {reps} paired "
          "bootstrap reps:")
    for label, values in diffs.items():
        print(f"    {label:66} {interval(np.array(values))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="*", default=list(PANELS))
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    for name in args.panels:
        study(name, args.reps, rng)


if __name__ == "__main__":
    main()
