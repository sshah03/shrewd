"""Do a panel's answers agree with each other? No API calls.

    python examples/panels/consistency.py [panel ...]     # default: all four, after build.py

Several questions in a panel are logically tied: a phishing email is spam, a message the
panel calls a scam is unsolicited, text with a password in it is high risk to share. Each
head is trained and calibrated on its own, so nothing stops the student from saying
P(phishing) = 0.9 and P(spam) = 0.2 about the same email. This measures how often that
happens, for the teacher (its answers on the judged pool) and the student (on the holdout,
raw and calibrated). The pool and holdout are disjoint draws from one shuffle, so the
rates are comparable even though the documents differ.

An `implies` rule A => B counts max(0, P(A) - P(B)) as the violation. A `same` rule counts
|P(A) - P(B)|. `>0.3` is the share of documents where the gap exceeds 0.3.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panels import PANELS  # noqa: E402

from shrewd import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GAP = 0.3


def p(q, *options):
    return lambda P: P[q][list(options)].sum(axis=1).to_numpy()


def any_of(*fns):
    """Upper bound on P(A or B or ...), which is the most a rule on a union can ask for."""
    return lambda P: np.minimum(1.0, sum(f(P) for f in fns))


def most_of(*fns):
    """Lower bound on P(A or B or ...)."""
    return lambda P: np.max([f(P) for f in fns], axis=0)


def none_of(*fns):
    """Upper bound on P(none of A, B, ...)."""
    lower = most_of(*fns)
    return lambda P: 1.0 - lower(P)


RULES = {
    "messages": [
        ("unsolicited == kind in {promotional, scam}", "same",
         p("unsolicited", "yes"), p("kind", "promotional", "scam")),
        ("kind=scam => risk >= high", "implies", p("kind", "scam"), p("risk", "3", "4")),
        ("kind=scam => asks_action", "implies", p("kind", "scam"), p("asks_action", "yes")),
    ],
    "email": [
        ("phishing => spam", "implies", p("phishing", "yes"), p("spam", "yes")),
        ("intent in {credential, payment, malware} => phishing", "implies",
         p("intent", "credential_theft", "payment_fraud", "malware_lure"),
         p("phishing", "yes")),
        # not "spam == not legitimate": a promotion nobody asked for is spam but its intent
        # is legitimate_marketing, so only this direction holds
        ("intent is a scam => spam", "implies",
         p("intent", "credential_theft", "payment_fraud", "malware_lure", "other_scam"),
         p("spam", "yes")),
    ],
    "guardrail": [
        ("harmful => handling=refuse", "implies", p("harmful", "yes"), p("handling", "refuse")),
        ("jailbreak => handling=refuse", "implies", p("jailbreak", "yes"),
         p("handling", "refuse")),
        ("refuse => harmful or jailbreak or injection", "implies", p("handling", "refuse"),
         any_of(p("harmful", "yes"), p("jailbreak", "yes"), p("injection", "yes"))),
    ],
    "pii": [
        ("financial or credentials or device => share_risk=high", "implies",
         most_of(p("financial", "yes"), p("credentials", "yes"), p("device", "yes")),
         p("share_risk", "high")),
        ("contact => share_risk >= moderate", "implies", p("contact", "yes"),
         p("share_risk", "moderate", "high")),
        ("share_risk=low => no personal data", "implies", p("share_risk", "low"),
         none_of(*(p(q, "yes") for q in
                   ("contact", "identity", "financial", "credentials", "device")))),
    ],
}


def frames_from_pool(pool, questions):
    """judged pool columns `q__option` -> {q: DataFrame with one column per option}."""
    return {q: pool[[f"{q}__{o}" for o in spec.options()]].set_axis(spec.options(), axis=1)
            for q, spec in questions.items()}


def frames_from_student(proba, questions):
    return {q: pd.DataFrame(proba[q], columns=spec.options()) for q, spec in questions.items()}


def score(rule, P):
    _, kind, lhs, rhs = rule
    a, b = lhs(P), rhs(P)
    gap = np.maximum(0.0, a - b) if kind == "implies" else np.abs(a - b)
    return {"mean": round(float(gap.mean()), 4), "over": round(float((gap > GAP).mean()), 4),
            "support": round(float(a.mean()), 4)}


def main(names):
    out = {}
    for name in names:
        run = ROOT / "runs" / f"panel-{name}"
        questions = PANELS[name]["questions"]
        pool = pd.read_csv(run / "pool_judged.csv")
        hold = pd.read_csv(ROOT / "data" / name / "holdout.csv")
        dec = load(run)
        texts = hold["text"].astype(str).tolist()
        sources = {
            "teacher (pool)": frames_from_pool(pool, questions),
            "student raw (holdout)": frames_from_student(
                dec.predict_proba(texts, calibrated=False), questions),
            "student cal (holdout)": frames_from_student(
                dec.predict_proba(texts, calibrated=True), questions),
        }
        print(f"\n{name}  (mean violation / share of docs with gap > {GAP})")
        print(f"  {'rule':56}" + "".join(f"{s:>24}" for s in sources))
        out[name] = {}
        for rule in RULES[name]:
            row = {s: score(rule, P) for s, P in sources.items()}
            out[name][rule[0]] = row
            cells = "".join(f"{r['mean']:>14.3f} / {r['over']:<7.3f}" for r in row.values())
            print(f"  {rule[0]:56}{cells}")
    path = ROOT / "runs" / "consistency.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(RULES))
