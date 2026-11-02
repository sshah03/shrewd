"""Combine several teachers' answers, weighted by how well each matched the hand labels on
the dev split.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

MIN_ROWS = 30
PRIOR_STRENGTH = 4.0
FLOOR = 0.02
TEMPERATURES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LOGIT_CLIP = 0.005
N_SPLITS = 5


def _normalize(values):
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.full_like(values, np.nan), where=total > 0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(proba, gold):
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(gold)), gold], 1e-12, None))))


def _shrunk(votes, gold, k, shape):
    counts = np.zeros((k, k))
    np.add.at(counts, gold, votes)
    theta = _normalize(counts + PRIOR_STRENGTH * shape)
    return (1 - FLOOR) * theta + FLOOR / k


def _confusion_choice(votes, gold, k):
    accuracy = (votes[np.arange(len(gold)), gold].sum() + 1.0) / (len(gold) + 2.0)
    one_coin = np.where(np.eye(k, dtype=bool), accuracy, (1.0 - accuracy) / (k - 1))
    return _shrunk(votes, gold, k, one_coin)


def _confusion_score(votes, gold, k):
    levels = np.arange(k, dtype=float)

    def theta_of(params):
        bias, log_spread = params
        spread = np.exp(log_spread)
        z = -((levels[None, :] - levels[:, None] - bias) ** 2) / (2 * spread ** 2)
        theta = _softmax(z)
        return (1 - FLOOR) * theta + FLOOR / k

    def objective(params):
        theta = theta_of(params)
        ridge = 0.5 * params[0] ** 2 + 0.5 * params[1] ** 2
        return -(votes * np.log(theta[gold])).sum() + ridge

    fit = minimize(objective, x0=np.array([0.0, 0.0]), method="L-BFGS-B",
                   bounds=[(-k, k), (np.log(0.2), np.log(k))])
    return _shrunk(votes, gold, k, theta_of(fit.x))


def _logits(yes):
    p = np.clip(yes, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


class _Fitted:

    def __init__(self, kind, k, votes, gold, seed=0, pick_temperature=True):
        self.kind, self.k = kind, k
        n = len(gold)
        present = np.bincount(gold, minlength=k) > 0
        if n < MIN_ROWS or present.sum() < 2:
            self.method = "average"
            return
        if kind == "noul":
            self.method = "stacking"
            self.model = LogisticRegression(C=1.0, max_iter=1000)
            self.model.fit(_logits(votes[:, :, 1]), gold)
            return
        self.method = "dawid-skene"
        confusion = _confusion_score if kind == "score" else _confusion_choice
        self.thetas = [confusion(votes[:, j], gold, k) for j in range(votes.shape[1])]
        self.log_prior = np.log((np.bincount(gold, minlength=k) + 1.0) / (n + k))
        self.temperature = 1.0
        if pick_temperature:
            self.temperature = self._pick_temperature(votes, gold, seed)

    def _loglik(self, votes):
        return sum(votes[:, j] @ np.log(theta).T for j, theta in enumerate(self.thetas))

    def _pick_temperature(self, votes, gold, seed):
        folds = KFold(N_SPLITS, shuffle=True, random_state=seed)
        losses = np.zeros(len(TEMPERATURES))
        for train, held in folds.split(votes):
            inner = _Fitted(self.kind, self.k, votes[train], gold[train], pick_temperature=False)
            if inner.method != "dawid-skene":
                continue
            loglik = inner._loglik(votes[held])
            for t, temperature in enumerate(TEMPERATURES):
                losses[t] += _nll(_softmax(inner.log_prior + temperature * loglik), gold[held])
        return float(TEMPERATURES[int(np.argmin(losses))])

    def apply(self, votes):
        if self.method == "average":
            return _normalize(votes.mean(axis=1))
        if self.method == "stacking":
            yes = self.model.predict_proba(_logits(votes[:, :, 1]))[:, 1]
            return np.column_stack([1 - yes, yes])
        return _softmax(self.log_prior + self.temperature * self._loglik(votes))


class QuestionCombiner:
    """Combines the teachers' answers to one question."""

    def __init__(self, kind, n_options, seed=0):
        self.kind, self.k, self.seed = kind, n_options, seed
        self._fits = {}

    def fit(self, votes, gold):
        """`votes` is (rows, teachers, options), NaN where a teacher didn't answer; `gold` is the
        true option per row, -1 where it isn't labeled.
        """
        answered = (gold >= 0) & ~np.isnan(votes).all(axis=2).all(axis=1)
        self._votes = _normalize(votes[answered])
        self._valid = ~np.isnan(votes[answered]).any(axis=2)
        self._gold = gold[answered]
        self._fits = {}
        return self

    def _for(self, subset):
        if subset not in self._fits:
            cols = list(subset)
            rows = self._valid[:, cols].all(axis=1)
            self._fits[subset] = _Fitted(
                self.kind, self.k, self._votes[rows][:, cols], self._gold[rows], self.seed
            )
        return self._fits[subset]

    def method(self, n_teachers):
        fit = self._for(tuple(range(n_teachers)))
        return fit.method, getattr(fit, "temperature", None)

    def combine(self, votes):
        """(rows, teachers, options) to (rows, options)."""
        valid = ~np.isnan(votes).any(axis=2)
        out = np.full((votes.shape[0], self.k), np.nan)
        for pattern in {tuple(np.where(row)[0]) for row in valid if row.any()}:
            rows = (valid == np.isin(np.arange(votes.shape[1]), pattern)).all(axis=1)
            sub = _normalize(votes[rows][:, list(pattern)])
            out[rows] = self._for(pattern).apply(sub)
        return out

    def cross_fit(self, votes, gold):
        """Out-of-fold combined answers for the dev rows."""
        out = self.fit(votes, gold).combine(votes)
        labeled = np.where(gold >= 0)[0]
        if len(labeled) < 2 * N_SPLITS:
            return out
        for _, held in KFold(N_SPLITS, shuffle=True, random_state=self.seed).split(labeled):
            masked = gold.copy()
            masked[labeled[held]] = -1
            inner = QuestionCombiner(self.kind, self.k, self.seed).fit(votes, masked)
            out[labeled[held]] = inner.combine(votes[labeled[held]])
        self.fit(votes, gold)
        return out


def vote_tensor(frames, texts, key, options):
    """The teachers' answer frames as a (rows, teachers, options) array."""
    columns = [f"{key}__{o}" for o in options]
    out = np.full((len(texts), len(frames), len(options)), np.nan)
    for j, frame in enumerate(frames):
        by_text = frame.drop_duplicates("text").set_index("text")
        out[:, j] = by_text.reindex(texts)[columns].to_numpy(dtype=float)
    return out


class Agreement:
    """One combiner per question, for a list of teachers."""

    def __init__(self, questions, teachers, seed=0):
        self.questions, self.teachers, self.seed = questions, list(teachers), seed
        self.combiners = {
            key: QuestionCombiner(q.kind, len(q.options()), seed) for key, q in questions.items()
        }

    def fit(self, dev_frames, dev_texts, dev_gold):
        """`dev_frames` follows the teacher order; `dev_gold` is gold_frame() of the dev split."""
        self._dev = (dev_frames, list(dev_texts), dev_gold)
        for j, (key, question) in enumerate(self.questions.items()):
            votes = vote_tensor(dev_frames, dev_texts, key, question.options())
            self.combiners[key].fit(votes, dev_gold[:, j])
        return self

    def combine(self, frames, texts):
        """The combined answer frame, in the shape judge_texts() returns."""
        import pandas as pd

        texts = list(texts)
        data = {"text": texts}
        for key, question in self.questions.items():
            options = question.options()
            combined = self.combiners[key].combine(vote_tensor(frames, texts, key, options))
            for i, option in enumerate(options):
                data[f"{key}__{option}"] = combined[:, i]
        return pd.DataFrame(data)

    def cross_fit_dev(self, only=None):
        """The dev split combined out-of-fold. `only` uses just those teachers."""
        import pandas as pd

        frames, texts, gold = self._dev
        data = {"text": texts}
        for j, (key, question) in enumerate(self.questions.items()):
            options = question.options()
            votes = vote_tensor(frames, texts, key, options)
            if only is not None:
                votes[:, [t for t in range(len(frames)) if t not in only]] = np.nan
            combined = QuestionCombiner(question.kind, len(options), self.seed).cross_fit(
                votes, gold[:, j])
            for i, option in enumerate(options):
                data[f"{key}__{option}"] = combined[:, i]
        return pd.DataFrame(data)

    def summary(self, pool_frames=None, pool_texts=None, floor=0.6):
        """What agreement.json holds."""
        frames, texts, gold = self._dev
        crossed = self.cross_fit_dev()
        out = {"teachers": self.teachers, "n_dev": len(texts), "questions": {}}
        for j, (key, question) in enumerate(self.questions.items()):
            options = question.options()
            votes = vote_tensor(frames, texts, key, options)
            labeled = gold[:, j] >= 0
            method, temperature = self.combiners[key].method(len(self.teachers))
            entry = {"method": method, "n_dev": int(labeled.sum()), "dev_accuracy": {}}
            if temperature is not None:
                entry["temperature"] = temperature
            for t, name in enumerate(self.teachers):
                usable = labeled & ~np.isnan(votes[:, t]).any(axis=1)
                if usable.any():
                    hits = votes[usable, t].argmax(axis=1) == gold[usable, j]
                    entry["dev_accuracy"][name] = round(float(hits.mean()), 4)
            combined = crossed[[f"{key}__{o}" for o in options]].to_numpy(dtype=float)
            usable = labeled & ~np.isnan(combined).any(axis=1)
            if usable.any():
                hits = combined[usable].argmax(axis=1) == gold[usable, j]
                entry["dev_accuracy"]["combined"] = round(float(hits.mean()), 4)
            entry["influence"] = self._influence(key, votes)
            out["questions"][key] = entry
        if pool_frames is not None:
            out["pool"] = pool_status(self, pool_frames, pool_texts, floor)["summary"]
            out["pairwise"] = pairwise_agreement(self.questions, self.teachers,
                                                 pool_frames, pool_texts)
        return out

    def gold_disagreements(self, top=5, examples=3):
        """Where the teachers disagree with your dev labels, per question, with examples."""
        frames, texts, gold = self._dev
        crossed = self.cross_fit_dev()
        out = {}
        for j, (key, question) in enumerate(self.questions.items()):
            options = question.options()
            votes = vote_tensor(frames, texts, key, options)
            combined = crossed[[f"{key}__{o}" for o in options]].to_numpy(dtype=float)
            labeled = (gold[:, j] >= 0) & ~np.isnan(combined).any(axis=1)
            pairs, unanimous = {}, 0
            for i in np.where(labeled)[0]:
                said = int(combined[i].argmax())
                if said == gold[i, j]:
                    continue
                answered = votes[i][~np.isnan(votes[i]).any(axis=1)]
                tops = {int(v.argmax()) for v in answered}
                if len(answered) > 1 and len(tops) == 1:
                    unanimous += 1
                pairs.setdefault((int(gold[i, j]), said), []).append(i)
            ranked = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:top]
            out[key] = {
                "n": int(labeled.sum()),
                "wrong": int(sum(len(v) for v in pairs.values())),
                "unanimous_wrong": unanimous,
                "pairs": [{"yours": options[g], "theirs": options[s], "count": len(rows),
                           "examples": [texts[i] for i in rows[:examples]]}
                          for (g, s), rows in ranked],
            }
        return out

    def _influence(self, key, votes):
        full_rows = ~np.isnan(votes).any(axis=2).any(axis=1)
        if len(self.teachers) < 2 or not full_rows.any():
            return {}
        rows = votes[full_rows]
        combiner = self.combiners[key]
        full = combiner.combine(rows)
        moves = []
        for t in range(len(self.teachers)):
            without = rows.copy()
            without[:, t] = np.nan
            shift = 0.5 * np.abs(full - combiner.combine(without)).sum(axis=1)
            moves.append(float(np.mean(shift)))
        total = sum(moves)
        if total <= 0:
            return {name: round(1 / len(self.teachers), 4) for name in self.teachers}
        return {name: round(m / total, 4) for name, m in zip(self.teachers, moves, strict=True)}


def pool_status(agreement, frames, texts, floor=0.6):
    """Per document and question, whether the teachers agree and how sure the combination is.
    Returns {"cells": {(text, key): state}, "summary": counts per document}.
    """
    combined = agreement.combine(frames, texts)
    asked = np.array([[text in set(frame["text"]) for frame in frames] for text in texts])
    cells = {}
    doc_state = {}
    rank = {"split": 3, "missing": 2, "unsure": 1, "agree": 0}
    for key, question in agreement.questions.items():
        options = question.options()
        votes = vote_tensor(frames, texts, key, options)
        values = combined[[f"{key}__{o}" for o in options]].to_numpy(dtype=float)
        for i, text in enumerate(texts):
            answered = ~np.isnan(votes[i]).any(axis=1)
            tops = {int(v.argmax()) for v in votes[i][answered]}
            if len(tops) > 1:
                state = "split"
            elif (asked[i] & ~answered).any():
                state = "missing"
            elif np.isnan(values[i]).any() or values[i].max() < floor:
                state = "unsure"
            else:
                state = "agree"
            cells[(text, key)] = state
            doc_state[text] = max(doc_state.get(text, "agree"), state, key=rank.get)
    counts = {state: 0 for state in rank}
    for state in doc_state.values():
        counts[state] += 1
    return {"cells": cells, "summary": {"documents": len(doc_state), **counts}}


def pairwise_agreement(questions, teachers, frames, texts):
    """For each pair of teachers, how often their top answers match."""
    out = {}
    tensors = [vote_tensor(frames, texts, key, q.options()) for key, q in questions.items()]
    for a in range(len(teachers)):
        for b in range(a + 1, len(teachers)):
            same = total = 0
            for votes in tensors:
                both = ~np.isnan(votes[:, a]).any(axis=1) & ~np.isnan(votes[:, b]).any(axis=1)
                total += int(both.sum())
                same += int((votes[both, a].argmax(axis=1) == votes[both, b].argmax(axis=1)).sum())
            out[f"{teachers[a]} | {teachers[b]}"] = round(same / total, 4) if total else None
    return out


def write_votes(path, teachers, frames):
    """Save each teacher's own answers next to pool_judged.csv."""
    import pandas as pd

    parts = [frame.assign(teacher=name) for name, frame in zip(teachers, frames, strict=True)]
    table = pd.concat(parts, ignore_index=True)
    table[["teacher", *[c for c in table.columns if c != "teacher"]]].to_csv(path, index=False)


def read_votes(path, teachers):
    """Read write_votes() back, in teacher order."""
    import pandas as pd

    table = pd.read_csv(path)
    table["text"] = table["text"].astype(str)
    return [table[table["teacher"] == name].drop(columns="teacher").reset_index(drop=True)
            for name in teachers]


def short_name(model):
    """"anthropic/claude-fable-5-1" -> "claude-fable-5-1"."""
    return str(model).rsplit("/", 1)[-1]
