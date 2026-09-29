"""Do the option descriptions help a small student? A research script, not a library feature.

    python examples/panels/fetch.py
    python examples/panels/build.py <panel> --from-judged          # for each panel
    python examples/panels/description_prior.py [panel ...] [--pools 100 300 1200]
                                                [--devs 325] [--seeds 5]

Needs `pip install "shrewd[embed]"`. No API calls.

The shipped heads learn each question from the teacher's answers alone. This head also
reads the text of the question: each option gets a key, the static embedding of its
description, and the utility of option o of question q for document x is

    u = s * cos(x, e_qo) + x^T dW e_qo + x^T A_q[:, o] + b_qo

s, dW (shared by the panel's questions) and b learn how far to trust the descriptions,
A_q is the usual per-question head. For a yes/no question only "yes" gets a key: static
embeddings can't negate, "Not the case: X" lands on top of "X", and yes-minus-no cancels.

Per question, a gate keeps the description head only where it beats the plain head on gold
dev AUROC by more than --margin (the same shape as the zero-shot stack's gate), so the
comparison that matters is `plain` vs `gated`. Both train on the teacher's soft answers for
a random subsample of the judged pool; regularization is picked on gold dev; the score is
gold holdout AUROC for the yes/no questions with public labels.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panels import GOLD, PANELS  # noqa: E402

from shrewd.decisions import DecisionStudent, _auroc, _Model2VecFeatures, gold_frame  # noqa: E402
from shrewd.zeroshot import hypotheses  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)
C_GRID = (0.1, 1.0, 10.0)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


class DescriptionHead:
    """The utility above, fit by L-BFGS on soft cross-entropy summed over questions."""

    def __init__(self, keys, lam):
        self.keys = {q: unit(E) for q, E in keys.items()}   # q -> (options, d)
        self.lam = lam

    def _shapes(self, d):
        shapes = [("s", (1,)), ("dW", (d, d))]
        for q, E in self.keys.items():
            shapes += [(f"b:{q}", (len(E),)), (f"A:{q}", (d, len(E)))]
        return shapes

    def _unpack(self, theta, d):
        out, i = {}, 0
        for label, shape in self._shapes(d):
            size = int(np.prod(shape))
            out[label] = theta[i:i + size].reshape(shape)
            i += size
        return out

    def _logits(self, Xn, p):
        XW = Xn @ p["dW"]
        return {q: p["s"][0] * (Xn @ E.T) + XW @ E.T + Xn @ p[f"A:{q}"] + p[f"b:{q}"]
                for q, E in self.keys.items()}

    def fit(self, X, targets):
        Xn, d = unit(X), X.shape[1]
        answered = {q: np.where(targets[q].sum(axis=1) > 1e-9)[0] for q in self.keys}

        def objective(theta):
            p = self._unpack(theta, d)
            Z = self._logits(Xn, p)
            penalty = (p["dW"] ** 2).sum() + sum((p[f"A:{q}"] ** 2).sum() for q in self.keys)
            loss = self.lam * penalty
            g = {label: np.zeros(shape) for label, shape in self._shapes(d)}
            g["dW"] += 2 * self.lam * p["dW"]
            for q, E in self.keys.items():
                r = answered[q]
                g[f"A:{q}"] += 2 * self.lam * p[f"A:{q}"]
                if not len(r):
                    continue
                z = Z[q][r] - Z[q][r].max(axis=1, keepdims=True)
                logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
                T = targets[q][r]
                loss += -np.mean((T * logp).sum(axis=1))
                G = (np.exp(logp) - T) / len(r)
                Xr = Xn[r]
                g["s"] += ((Xr @ E.T) * G).sum()
                g["dW"] += Xr.T @ (G @ E)
                g[f"A:{q}"] += Xr.T @ G
                g[f"b:{q}"] += G.sum(axis=0)
            return loss, np.concatenate([g[label].ravel() for label, _ in self._shapes(d)])

        theta = np.zeros(sum(int(np.prod(s)) for _, s in self._shapes(d)))
        theta[0] = 10.0
        self.d = d
        self.theta = minimize(objective, theta, jac=True, method="L-BFGS-B",
                              options={"maxiter": 500}).x
        return self

    def predict_proba(self, X):
        Z = self._logits(unit(X), self._unpack(self.theta, self.d))
        out = {}
        for q, z in Z.items():
            e = np.exp(z - z.max(axis=1, keepdims=True))
            out[q] = e / e.sum(axis=1, keepdims=True)
        return out


class Lookup:
    """Hands DecisionStudent embeddings computed once up front."""

    def __init__(self, table):
        self.table = table

    def fit_transform(self, texts, y=None):
        return self.transform(texts)

    def transform(self, texts):
        return np.vstack([self.table[t] for t in texts])


def aurocs(P, gold, keys, gold_keys):
    out = {}
    for j, key in enumerate(keys):
        m = gold[:, j] >= 0
        if key in gold_keys and m.sum() >= 10 and len(np.unique(gold[m, j])) == 2:
            out[key] = _auroc(P[key][m][:, 1], (gold[m, j] == 1).astype(float))
    return out


def best_by_dev(candidates, dev_score):
    scored = [(np.mean(list(dev_score(c).values()) or [np.nan]), c) for c in candidates]
    return max(scored, key=lambda sc: -np.inf if sc[0] != sc[0] else sc[0])[1]


def run_panel(name, args, embed):
    spec = PANELS[name]
    questions, keys = spec["questions"], list(spec["questions"])
    gold_keys = set(GOLD[name])
    run = ROOT / "runs" / f"panel-{name}"
    pool = pd.read_csv(run / "pool_judged.csv")
    dev = pd.read_csv(run / "seed_dev.csv")
    hold = pd.read_csv(ROOT / "data" / name / "holdout.csv")
    for df in (pool, dev, hold):
        df["text"] = df["text"].astype(str)
    for key in questions:
        if key not in hold:
            hold[key] = None
    dev_gold, hold_gold = gold_frame(dev, questions), gold_frame(hold, questions)

    texts = list(dict.fromkeys([*pool["text"], *dev["text"], *hold["text"]]))
    table = dict(zip(texts, embed.transform(texts), strict=True))
    X_pool = np.vstack([table[t] for t in pool["text"]])
    X_dev = np.vstack([table[t] for t in dev["text"]])
    X_hold = np.vstack([table[t] for t in hold["text"]])
    option_keys = {}
    for q, question in questions.items():
        E = embed.transform(hypotheses(question, spec["instructions"]))
        if question.kind == "noul":
            E[0] = 0.0
        option_keys[q] = E
    targets = {}
    for key, q in questions.items():
        v = np.nan_to_num(pool[[f"{key}__{o}" for o in q.options()]].to_numpy(float))
        total = v.sum(axis=1, keepdims=True)
        targets[key] = np.divide(v, total, out=np.zeros_like(v), where=total > 1e-9)

    print(f"\n=== {name}  (gold: {', '.join(sorted(gold_keys))})")
    for size in args.pools:
        for dev_n in args.devs:
            plain_all, gated_all, kept = [], [], {}
            for seed in range(args.seeds):
                idx = np.random.default_rng(seed).permutation(len(pool))[:size]
                didx = np.random.default_rng(100 + seed).permutation(len(dev))[:dev_n]
                T = {q: t[idx] for q, t in targets.items()}
                sub = [pool["text"][i] for i in idx]
                dev_texts = [dev["text"][i] for i in didx]

                def plain_dev(st, dev_texts=dev_texts, didx=didx):
                    return aurocs(st.predict_proba(dev_texts, calibrated=False), dev_gold[didx],
                                  keys, gold_keys)

                plain = best_by_dev(
                    [DecisionStudent(questions, features=lambda: Lookup(table), seed=seed, C=c)
                     .fit(sub, T) for c in C_GRID], plain_dev)

                def desc_dev(h, didx=didx):
                    return aurocs(h.predict_proba(X_dev[didx]), dev_gold[didx], keys, gold_keys)

                desc = best_by_dev([DescriptionHead(option_keys, lam).fit(X_pool[idx], T)
                                    for lam in LAMBDAS], desc_dev)
                p_hold = aurocs(plain.predict_proba(hold["text"].tolist(), calibrated=False),
                                hold_gold, keys, gold_keys)
                d_hold = aurocs(desc.predict_proba(X_hold), hold_gold, keys, gold_keys)
                p_dev, d_dev = plain_dev(plain), desc_dev(desc)
                gated = {}
                for k in p_hold:
                    use = k in p_dev and k in d_dev and d_dev[k] - p_dev[k] > args.margin
                    kept.setdefault(k, []).append(use)
                    gated[k] = d_hold[k] if use else p_hold[k]
                plain_all.append(p_hold)
                gated_all.append(gated)
            mean = lambda runs: np.mean([np.mean(list(r.values())) for r in runs])  # noqa: E731
            pairs = list(zip(plain_all, gated_all, strict=True))
            gains = sum(g[k] > p[k] + 0.01 for p, g in pairs for k in p)
            losses = sum(g[k] < p[k] - 0.01 for p, g in pairs for k in p)
            per_q = "  ".join(
                f"{k} {np.mean([r[k] for r in plain_all]):.3f}->"
                f"{np.mean([r[k] for r in gated_all]):.3f} (kept {sum(kept[k])}/{len(kept[k])})"
                for k in sorted(kept))
            print(f"  pool {size:5}, dev {dev_n:3}: plain {mean(plain_all):.3f}  gated "
                  f"{mean(gated_all):.3f}  | per-question gains >.01: {gains}, "
                  f"losses >.01: {losses}")
            print(f"      {per_q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="*", default=list(PANELS))
    ap.add_argument("--pools", type=int, nargs="+", default=[100, 300, 1200])
    ap.add_argument("--devs", type=int, nargs="+", default=[325])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    embed = _Model2VecFeatures()
    for name in args.panels:
        run_panel(name, args, embed)


if __name__ == "__main__":
    main()
