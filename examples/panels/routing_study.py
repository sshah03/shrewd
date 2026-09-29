"""`distill(route_budget=...)` on the pre-built panels: does the guarantee hold, what coverage
does it leave, and what are the local answers worth against human labels. No API calls.

    python examples/panels/fetch.py
    python examples/panels/build.py <panel> --from-judged          # for each panel
    python examples/panels/routing_study.py [panel ...] [--part validity|grid|both]

The budget throughout: at most 2% of all requests answered locally and disagreeing with the
teacher, with a 95% bound. A document is routed as a unit (one teacher call answers the
whole panel), scored by its least confident answer.

  validity  one student per panel, fit on 60% of the judged pool, scores the other 40% once.
            500 times, draw a threshold set from those scored documents, set the threshold
            on it, and measure the true risk on the scored documents outside the draw. A
            valid guarantee exceeds the budget in at most 5% of draws. Also the shortcut of
            setting the threshold on out-of-fold scores of the training documents.
  grid      the whole distill() recipe at 200, 500 and 1,000 judged documents, 10 random
            draws each: train on 70%, set the threshold on 30%, then measure coverage and
            disagreement on the rest of the pool and error against human labels on the
            holdout. Compared with the naive threshold, the loosest whose observed
            disagreement is within budget.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panels import GOLD, PANELS  # noqa: E402

from shrewd import load  # noqa: E402
from shrewd.calibrate import Calibrator  # noqa: E402
from shrewd.decisions import DecisionStudent, gold_frame  # noqa: E402
from shrewd.route import doc_disagree, doc_scores, pick_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
BUDGET, DELTA = 0.02, 0.05


def pool_targets(pool, questions):
    out = {}
    for key, q in questions.items():
        v = np.nan_to_num(pool[[f"{key}__{o}" for o in q.options()]].to_numpy(float))
        t = v.sum(axis=1, keepdims=True)
        out[key] = np.divide(v, t, out=np.zeros_like(v), where=t > 1e-9)
    return out


def calibrated_oof(texts, T, questions, features, seed):
    """Out-of-fold probabilities, each fold calibrated by a calibrator fit on the others.
    Returns (calibrated, raw)."""
    raw = DecisionStudent(questions, features=features, seed=seed)._oof(texts, T)
    out = {}
    folds = list(KFold(5, shuffle=True, random_state=seed).split(texts))
    for k, q in questions.items():
        opts = q.options()
        answered = T[k].max(axis=1) > 1e-9
        res = np.array(raw[k], copy=True)
        for tr, te in folds:
            tr = tr[answered[tr]]
            labels = [opts[i] for i in T[k][tr].argmax(axis=1)]
            res[te] = Calibrator.fit(raw[k][tr], labels, opts, method="auto").transform(raw[k][te])
        out[k] = res
    return out, raw


def fit_student(texts, T, raw_oof, questions, features, seed):
    """What distill() saves: heads on all the rows, calibrated on their out-of-fold scores."""
    st = DecisionStudent(questions, features=features, seed=seed)
    st.fit(texts, T)
    for k, q in questions.items():
        answered = np.where(T[k].max(axis=1) > 1e-9)[0]
        opts = q.options()
        st.calibrators[k] = Calibrator.fit(raw_oof[k][answered],
                                           [opts[i] for i in T[k][answered].argmax(axis=1)],
                                           opts, method="auto")
    return st


def naive_threshold(score, disagree):
    order = np.argsort(-score, kind="mergesort")
    s_sorted, cum = score[order], np.cumsum(disagree[order])
    ok = np.where(cum / len(score) <= BUDGET)[0]
    return s_sorted[ok[-1]] if len(ok) else np.inf


def load_panel(name):
    questions = PANELS[name]["questions"]
    run = ROOT / "runs" / f"panel-{name}"
    pool = pd.read_csv(run / "pool_judged.csv")
    pool["text"] = pool["text"].astype(str)
    return questions, load(run).features_kind, pool, pool_targets(pool, questions)


def validity(name, draws=500):
    questions, features, pool, T = load_panel(name)
    perm = np.random.default_rng(0).permutation(len(pool))
    cut = int(0.6 * len(pool))
    fit_i, rest = perm[:cut], perm[cut:]
    texts_fit = [pool["text"][i] for i in fit_i]
    T_fit = {k: v[fit_i] for k, v in T.items()}
    cal, raw = calibrated_oof(texts_fit, T_fit, questions, features, 0)
    student = fit_student(texts_fit, T_fit, raw, questions, features, 0)
    P = student.predict_proba([pool["text"][i] for i in rest])
    score, dis = doc_scores(P), doc_disagree(P, {k: v[rest] for k, v in T.items()})
    print(f"\n=== {name} ({features}): fit on {cut}, scored {len(rest)}", flush=True)

    t, stats = pick_threshold(doc_scores(cal), doc_disagree(cal, T_fit), BUDGET, DELTA)
    print(f"  out-of-fold threshold: claims {stats['coverage']:.1%} local, gets "
          f"{(score >= t).mean():.1%} with true risk {(dis & (score >= t)).mean():.4f}")
    rng = np.random.default_rng(1)
    for n_cal in (150, 300):
        broke, cov, risk = [], [], []
        for _ in range(draws):
            idx = rng.permutation(len(rest))
            c, e = idx[:n_cal], idx[n_cal:]
            t, _ = pick_threshold(score[c], dis[c], BUDGET, DELTA)
            r = float((dis[e] & (score[e] >= t)).mean())
            broke.append(r > BUDGET)
            cov.append((score[e] >= t).mean())
            risk.append(r)
        print(f"  threshold set of {n_cal}: over budget in {np.mean(broke):.1%} of {draws} draws, "
              f"mean risk {np.mean(risk):.4f}, {np.mean(cov):.1%} local", flush=True)


def grid(name, sizes=(200, 500, 1000), reps=10):
    questions, features, pool, T = load_panel(name)
    keys = list(questions)
    hold = pd.read_csv(ROOT / "data" / name / "holdout.csv")
    for k in questions:
        if k not in hold:
            hold[k] = None
    hold_gold = gold_frame(hold, questions)
    hold_texts = hold["text"].astype(str).tolist()
    print(f"\n=== {name} ({features})", flush=True)
    rows = []
    for size in sizes:
        for rep in range(reps):
            perm = np.random.default_rng(rep).permutation(len(pool))
            tr, live = perm[:size], perm[size:]
            cut = int(0.7 * size)
            fit_i, thr_i = tr[:cut], tr[cut:]
            texts_fit = [pool["text"][i] for i in fit_i]
            T_fit = {k: v[fit_i] for k, v in T.items()}
            raw = DecisionStudent(questions, features=features, seed=rep)._oof(texts_fit, T_fit)
            student = fit_student(texts_fit, T_fit, raw, questions, features, rep)
            P_thr = student.predict_proba([pool["text"][i] for i in thr_i])
            s_thr = doc_scores(P_thr)
            d_thr = doc_disagree(P_thr, {k: v[thr_i] for k, v in T.items()})
            P_lv = student.predict_proba([pool["text"][i] for i in live])
            s_lv, d_lv = doc_scores(P_lv), doc_disagree(P_lv, {k: v[live] for k, v in T.items()})
            P_h = student.predict_proba(hold_texts)
            s_h = doc_scores(P_h)
            for how, t in (("bound", pick_threshold(s_thr, d_thr, BUDGET, DELTA)[0]),
                           ("naive", naive_threshold(s_thr, d_thr))):
                local, local_h = s_lv >= t, s_h >= t
                err_local, err_all = [], []
                for j, k in enumerate(keys):
                    if k not in GOLD[name]:
                        continue
                    g = hold_gold[:, j]
                    wrong = (P_h[k].argmax(axis=1) != g) & (g >= 0)
                    err_local.append(wrong[local_h].sum() / max(((g >= 0) & local_h).sum(), 1))
                    err_all.append(wrong.sum() / (g >= 0).sum())
                risk = (d_lv & local).mean()
                rows.append(dict(size=size, how=how, coverage=local.mean(), risk=risk,
                                 broke=risk > BUDGET, err_local=np.mean(err_local),
                                 err_all=np.mean(err_all)))
        df = pd.DataFrame([r for r in rows if r["size"] == size])
        for how, g in df.groupby("how", sort=False):
            print(f"  {size:5} judged, {how:5}: {g.coverage.mean():.1%} local, disagreement "
                  f"{g.risk.mean():.4f}, over budget in {g.broke.mean():.0%} of {reps} | "
                  f"gold error local {g.err_local.mean():.1%} vs {g.err_all.mean():.1%} on all",
                  flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("panels", nargs="*", default=["messages", "email", "guardrail", "pii"])
    parser.add_argument("--part", choices=["validity", "grid", "both"], default="both")
    args = parser.parse_args()
    for name in args.panels:
        if args.part in ("validity", "both"):
            validity(name)
        if args.part in ("grid", "both"):
            grid(name)
