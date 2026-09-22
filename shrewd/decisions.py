"""Many typed questions about one document, answered in a single pass.

Define the questions once, a teacher answers all of them per document in one call, and
the result is distilled into one featurization with a calibrated head per question.

    d = Decisions("runs/tickets", questions={...}, teacher="anthropic")
    d.add_seed(labeled_df)
    d.judge(pool_df)
    print(d.distill().report())

    dec = load("runs/tickets")
    dec.decide("I was charged twice and nobody will call me back")
"""

import json
import shutil
import warnings
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from shrewd import decide
from shrewd.calibrate import Calibrator, calibration_metrics
from shrewd.decide import NO, YES
from shrewd.evaluate import Finding
from shrewd.zeroshot import combine, fit_bias, fit_weights, hypotheses

MIN_CALIBRATION_ROWS = 60
MIN_MINORITY_ROWS = 12
STACK_MARGIN = 0.02      # macro-F1 the stack must gain on gold dev to be kept
REVIEW_FLOOR = 0.6       # a teacher answer below this lands in needs_review.csv
MIN_GATE_ROWS = 40       # fewer gold dev rows than this and the gate abstains


# ---------------------------------------------------------------- gold answers


def gold_index(question, value):
    """A hand-labeled answer to an option index. None when it cannot be read."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    options = question.options()
    text = str(value).strip()
    if not text:
        return None
    lookup = {o.lower(): i for i, o in enumerate(options)}
    if text.lower() in lookup:
        return lookup[text.lower()]
    if question.kind == "noul":
        truthy = {"1", "true", "t", "y", "yes"}
        falsy = {"0", "false", "f", "n", "no"}
        if text.lower() in truthy:
            return options.index(YES)
        if text.lower() in falsy:
            return options.index(NO)
    if question.kind == "score":
        # a level number, or the level's description. A reviewer writes "blocking",
        # not "2", and the ledger should read that way too
        for i, description in enumerate(question.criteria):
            if str(description).strip().lower() == text.lower():
                return i
        try:
            level = int(round(float(text)))
        except ValueError:
            return None
        if 0 <= level < len(options):
            return level
    return None


def gold_frame(df, questions):
    """Seed answers to an integer matrix (rows x questions), -1 where unanswered."""
    out = np.full((len(df), len(questions)), -1, dtype=int)
    for j, (key, question) in enumerate(questions.items()):
        if key not in df.columns:
            raise ValueError(
                f"seed data has no column for question {key!r}; it needs one column per "
                f"question id, holding the hand-labeled answer"
            )
        for i, value in enumerate(df[key].tolist()):
            index = gold_index(question, value)
            if index is not None:
                out[i, j] = index
    return out


# ---------------------------------------------------------------- the student


BUILTIN_FEATURES = ("tfidf", "embed", "encoder", "auto")


def resolve_features(features):
    """`"auto"` at the student level means tf-idf. `Decisions.distill()` resolves it by fitting
    both candidates and keeping the one that wins on the hand-labeled dev split. Neither
    featurization is a safe default on its own (embeddings won by 4 points on one panel and
    lost by 8-11 on two others).
    """
    return "tfidf" if features == "auto" else features


def _embed_available():
    try:
        import model2vec  # noqa: F401
    except ImportError:
        return False
    return True


def _featurizer(features, seed):
    """Build a fresh, unfitted featurizer from a name, a factory, or an instance.

    Instances are cloned rather than used directly: cross-fit calibration refits the student
    K times and the folds must not share fitted state.
    """
    features = resolve_features(features)
    if isinstance(features, str):
        if features == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import FeatureUnion

            return FeatureUnion(
                [
                    ("word", TfidfVectorizer(ngram_range=(1, 2))),
                    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
                ]
            )
        if features == "embed":
            return _Model2VecFeatures()
        if features == "encoder":
            from shrewd.encoder import EncoderFeatures

            return EncoderFeatures(seed=seed)
        raise ValueError(
            f"unknown featurizer {features!r}; pick one of {list(BUILTIN_FEATURES)} or pass "
            "your own transformer (an object with fit_transform/transform, or a factory "
            "returning one)"
        )
    if callable(features) and not hasattr(features, "transform"):
        built = features()
    else:
        built = _clone(features)
    for method in ("fit_transform", "transform"):
        if not hasattr(built, method):
            raise TypeError(
                f"a featurizer needs a {method}(texts) method; {type(built).__name__} has none"
            )
    return built


def _clone(estimator):
    import copy

    try:
        from sklearn.base import clone

        return clone(estimator)
    except (ImportError, TypeError):
        # not a scikit-learn estimator, so a deep copy is the next best guarantee that
        # one fold's fitted state cannot leak into another
        return copy.deepcopy(estimator)


def _kind_name(features):
    if isinstance(features, str):
        return features
    return getattr(features, "__name__", type(features).__name__)


class _Model2VecFeatures:
    """model2vec static embeddings as a transformer, so heads share one encode pass."""

    def __init__(self, model="minishlab/potion-base-8M"):
        self.model_name = model
        self._model = None

    def _load_model(self):
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def fit(self, texts, y=None):
        self._load_model()
        return self

    def transform(self, texts):
        return self._load_model().encode(list(texts))

    def fit_transform(self, texts, y=None):
        return self.fit(texts).transform(texts)


def _check_extra(kind):
    """Fail with an install hint, not a pickle error, when a saved model's extra is missing."""
    import importlib.util

    from shrewd.students import REQUIRES

    for module in REQUIRES.get(kind, ()):
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f'this model uses {kind} features, which need {module}: '
                f'pip install "shrewd[{kind}]"'
            )


class _ConstantHead:
    """Stands in when a question has only one answer in the training data."""

    def __init__(self, index, n_options):
        self.index, self.n_options = index, n_options

    def predict_proba(self, X):
        out = np.full((X.shape[0], self.n_options), 1e-6)
        out[:, self.index] = 1.0 - 1e-6 * (self.n_options - 1)
        return out


class DecisionStudent:
    """One featurization, one calibrated head per question."""

    def __init__(self, questions, features="tfidf", seed=42, soft=True, instructions=None,
                 **lr_kwargs):
        self.questions = questions
        features = resolve_features(features)
        self.features = features                  # the spec, so folds can rebuild it
        self.features_kind = _kind_name(features)
        self.seed = seed
        self.soft = soft
        self.instructions = instructions
        self.lr_kwargs = lr_kwargs
        self._features = _featurizer(features, seed)
        self.heads = {}
        self.calibrators = {}
        self.zero_shot = None      # a ZeroShotScorer, when stacking is on
        self.stacks = {}           # question id -> {"bias", "weights"} for kept stacks
        self.stack_report = {}     # question id -> the gate's decision, kept or not
        self._pool_cache = None    # (texts, oof, zs) from stack(), reused by calibrate()
        self.n_train_ = 0

    # -- training --------------------------------------------------------------

    def _head(self, X, target, n_options):
        """Fit one question's head. `target` is a (rows x options) probability matrix."""
        present = np.where(target.sum(axis=0) > 1e-9)[0]
        if len(present) < 2:
            return _ConstantHead(int(present[0]) if len(present) else 0, n_options)
        if self.soft:
            # one training row per option, weighted by the teacher's probability: the
            # closest thing a linear head has to matching a distribution
            rows, labels, weights = [], [], []
            for option in present:
                mask = target[:, option] > 1e-6
                rows.append(np.where(mask)[0])
                labels.append(np.full(mask.sum(), option))
                weights.append(target[mask, option])
            index = np.concatenate(rows)
            X_fit, y_fit = X[index], np.concatenate(labels)
            sample_weight = np.concatenate(weights)
        else:
            keep = np.where(target.max(axis=1) > 1e-9)[0]
            X_fit, y_fit = X[keep], target[keep].argmax(axis=1)
            sample_weight = None
        # a rare option can carry probability mass without ever winning a row
        winners = np.unique(y_fit)
        if len(winners) < 2:
            return _ConstantHead(int(winners[0]) if len(winners) else 0, n_options)
        # no class reweighting by default: balanced weights skew the probabilities toward
        # minority classes (raw ECE 0.175 vs 0.066). Pass class_weight="balanced" if
        # recall on a rare option matters more than the probabilities.
        params = {"max_iter": 2000, "random_state": self.seed}
        params.update(self.lr_kwargs)
        model = LogisticRegression(**params)
        model.fit(X_fit, y_fit, sample_weight=sample_weight)
        return _Expanded(model, n_options)

    def fit(self, texts, targets):
        """`targets` maps question id to a (rows x options) probability matrix.

        A featurizer that can learn from the targets (a fine-tuned encoder) gets them;
        a frozen one (tf-idf, static embeddings) just sees the texts.
        """
        texts = list(texts)
        if hasattr(self._features, "fit_with_targets"):
            X = self._features.fit_with_targets(texts, targets, self.questions)
        else:
            X = self._features.fit_transform(texts)
        self.heads = {
            key: self._head(X, targets[key], len(q.options()))
            for key, q in self.questions.items()
            if key in targets
        }
        self.n_train_ = len(texts)
        return self

    def _oof(self, texts, targets, n_splits=5):
        """Out-of-fold head probabilities over `texts`: K refits, each scoring the rows it
        never trained on. Shared by calibration and stacking so the pool is refit once."""
        texts = list(texts)
        if getattr(self._features, "expensive", False):
            # each fold is a fine-tune, three is enough to fit a scalar
            n_splits = min(n_splits, 3)
        oof = {key: np.zeros_like(target) for key, target in targets.items()}
        splitter = KFold(n_splits=min(n_splits, len(texts)), shuffle=True, random_state=self.seed)
        for train, test in splitter.split(texts):
            fold = DecisionStudent(
                self.questions, self.features, self.seed, self.soft, **self.lr_kwargs
            )
            fold.fit(
                [texts[i] for i in train],
                {key: target[train] for key, target in targets.items()},
            )
            proba = fold.predict_proba([texts[i] for i in test], calibrated=False)
            for key, values in proba.items():
                oof[key][test] = values
            if getattr(self._features, "expensive", False):
                del fold                     # a fine-tuned trunk per fold, don't let them pile up
                from shrewd.encoder import release_memory
                release_memory()
        return oof

    def stack(self, pool_texts, pool_targets, dev_texts, dev_gold, margin=STACK_MARGIN,
              n_splits=5):
        """Stack the zero-shot scorer onto each head, keeping it only where gold dev says so.

        Per question: fit one bias per hypothesis and two source weights on out-of-fold pool
        predictions, then keep the stack only if it beats the plain head on the gold dev split
        by more than `margin`. The gate uses a twin student fit on the pool alone (the real
        heads train on pool+dev) and gold rather than pool agreement, because the zero-shot
        model and the teacher tend to make the same mistakes.
        """
        if self.zero_shot is None:
            return self
        pool_texts, dev_texts = list(pool_texts), list(dev_texts)
        lg = lambda p: np.log(np.clip(np.asarray(p, dtype=float), 1e-9, 1.0))  # noqa: E731
        oof = self._oof(pool_texts, pool_targets, n_splits)
        twin = DecisionStudent(
            self.questions, self.features, self.seed, self.soft, **self.lr_kwargs
        )
        twin.fit(pool_texts, pool_targets)
        dev_head = twin.predict_proba(dev_texts, calibrated=False)
        zs_pool = {}
        for j, (key, question) in enumerate(self.questions.items()):
            if key not in self.heads or key not in pool_targets:
                continue
            report = {"kept": False}
            target = pool_targets[key]
            keep = np.where(target.max(axis=1) > 1e-9)[0]
            y = target[keep].argmax(axis=1)
            if len(keep) < MIN_CALIBRATION_ROWS or len(np.unique(y)) < 2:
                report["reason"] = f"only {len(keep)} answered pool rows; nothing to fit a stack on"
                self.stack_report[key] = report
                continue
            dev_rows = np.where(dev_gold[:, j] >= 0)[0]
            if len(dev_rows) < MIN_GATE_ROWS:
                report["reason"] = (f"only {len(dev_rows)} gold dev rows; the gate abstains "
                                    f"(needs {MIN_GATE_ROWS})")
                self.stack_report[key] = report
                continue
            zs = self.zero_shot.score(pool_texts, question, self.instructions)
            zs_pool[key] = zs
            bias = fit_bias(lg(zs[keep]), y)
            weights = fit_weights(lg(oof[key][keep]), lg(zs[keep]) + bias, y)
            zs_dev = self.zero_shot.score([dev_texts[i] for i in dev_rows], question,
                                          self.instructions)
            plain = dev_head[key][dev_rows]
            stacked = combine(plain, zs_dev, bias, weights)
            truth = dev_gold[dev_rows, j]
            k = len(question.options())
            # AUROC for yes/no (few dev positives, so F1 barely moves), macro-F1 otherwise
            if question.kind == "noul":
                metric = "auroc"
                before, after = _auroc(plain[:, 1], truth), _auroc(stacked[:, 1], truth)
                if before != before or after != after:  # NaN: one class absent from dev
                    report["reason"] = ("no positive (or no negative) gold dev rows; "
                                        "the gate abstains")
                    self.stack_report[key] = report
                    continue
                gain = float(after - before)
            else:
                metric = "macro_f1"
                gain = float(_macro_f1(truth, stacked, k) - _macro_f1(truth, plain, k))
            report.update(
                kept=bool(gain > margin), metric=metric, dev_gain=round(gain, 4),
                dev_n=int(len(dev_rows)),
                bias=[round(float(b), 4) for b in bias],
                weights=[round(w, 4) for w in weights],
                hypotheses=hypotheses(question, self.instructions),
            )
            if report["kept"]:
                self.stacks[key] = {"bias": report["bias"], "weights": report["weights"]}
            self.stack_report[key] = report
        self._pool_cache = (tuple(pool_texts), oof, zs_pool)
        return self

    def calibrate(self, texts, targets, n_splits=5, method="auto"):
        """Fit one calibrator per question on out-of-fold predictions over these rows (K refits).
        A stacked question is calibrated on its stacked out-of-fold probabilities.
        """
        texts = list(texts)
        if len(texts) < MIN_CALIBRATION_ROWS:
            warnings.warn(
                f"only {len(texts)} rows to calibrate on; skipping calibration rather "
                f"than fitting a scale on too little to support it",
                stacklevel=2,
            )
            return self
        cached = self._pool_cache
        if cached is not None and cached[0] == tuple(texts):
            oof, zs_pool = cached[1], cached[2]
        else:
            oof, zs_pool = self._oof(texts, targets, n_splits), {}
        for key, question in self.questions.items():
            if key not in oof:
                continue
            target = targets[key]
            keep = np.where(target.max(axis=1) > 1e-9)[0]
            if len(keep) < MIN_CALIBRATION_ROWS:
                continue
            source = oof[key]
            stack = self.stacks.get(key)
            if stack is not None:
                zs = zs_pool.get(key)
                if zs is None:
                    zs = self.zero_shot.score(texts, question, self.instructions)
                source = combine(source, zs, stack["bias"], stack["weights"])
            labels = [question.options()[i] for i in target[keep].argmax(axis=1)]
            self.calibrators[key] = Calibrator.fit(
                source[keep], labels, question.options(), method=method, seed=self.seed
            )
        return self

    # -- inference -------------------------------------------------------------

    def _select(self, questions):
        """Which heads to run. Asking for one that was never compiled raises instead of
        quietly leaving the key out, since a missing ninth answer is easy to miss."""
        if questions is None:
            return list(self.heads)
        wanted = [questions] if isinstance(questions, str) else list(questions)
        unknown = [key for key in wanted if key not in self.heads]
        if unknown:
            raise KeyError(
                f"this artifact has no head for {unknown}; it was compiled for "
                f"{sorted(self.heads)}. A question that was not compiled cannot be "
                "answered from the trained weights; add it to the panel and re-run "
                "judge() and distill() (the per-question cache means you pay for the "
                "new question only)."
            )
        return wanted

    def predict_proba(self, texts, calibrated=True, questions=None):
        """{question id: (rows x options) probabilities}. One featurization for all of them."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        keys = self._select(questions)
        X = self._features.transform(texts)
        out = {}
        for key in keys:
            proba = self.heads[key].predict_proba(X)
            stack = self.stacks.get(key)
            if stack is not None:
                zs = self.zero_shot.score(texts, self.questions[key], self.instructions)
                proba = combine(proba, zs, stack["bias"], stack["weights"])
            calibrator = self.calibrators.get(key)
            if calibrated and calibrator is not None:
                proba = calibrator.transform(proba)
            out[key] = proba
        return out

    def decide(self, texts, calibrated=True, questions=None):
        """Typed answers. A single string in gives a single answer map back."""
        single = isinstance(texts, str)
        keys = self._select(questions)
        proba = self.predict_proba(texts, calibrated=calibrated, questions=keys)
        n = 1 if single else len(list(texts))
        rows = [
            {key: decide.make_answer(key, self.questions[key], proba[key][i]) for key in keys}
            for i in range(n)
        ]
        return rows[0] if single else rows

    # -- persistence -----------------------------------------------------------

    def save(self, path):
        import joblib
        import sklearn

        from shrewd import __version__

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        saved_separately = hasattr(self._features, "save_to")
        if saved_separately:
            self._features.save_to(path / "features")
        joblib.dump(
            {"features": None if saved_separately else self._features, "heads": self.heads},
            path / "model.joblib",
        )
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "type": "decisions",
                    "features": self.features_kind,
                    "soft": self.soft,
                    "seed": self.seed,
                    "n_train": self.n_train_,
                    "questions": {k: decide.to_dict(q) for k, q in self.questions.items()},
                    "calibration": {k: c.to_dict() for k, c in self.calibrators.items()},
                    "instructions": self.instructions,
                    "features_saved_separately": saved_separately,
                    "stacks": self.stacks,
                    "stack_report": self.stack_report,
                    "zero_shot": (self.zero_shot.to_dict()
                                  if self.zero_shot is not None and self.stacks else None),
                    "versions": {"shrewd": __version__, "scikit-learn": sklearn.__version__},
                },
                indent=2,
            )
        )

    @classmethod
    def _load(cls, path, meta):
        import joblib

        path = Path(path)
        questions = {k: decide.from_dict(v) for k, v in meta["questions"].items()}
        _check_extra(meta.get("features", "tfidf"))
        blob = joblib.load(path / "model.joblib")
        # the fitted featurizer comes back from joblib whatever it was, so build the
        # student around a placeholder rather than trying to reconstruct a custom
        # class from the name in meta.json
        kind = meta.get("features", "tfidf")
        student = cls(questions, kind if kind in BUILTIN_FEATURES else "tfidf",
                      meta.get("seed", 42), meta.get("soft", False))
        if meta.get("features_saved_separately"):
            if kind != "encoder":
                raise ValueError(f"do not know how to reload featurizer {kind!r}")
            from shrewd.encoder import EncoderFeatures

            features = EncoderFeatures.load_from(path / "features")
        else:
            features = blob["features"]
        student.features = features
        student.features_kind = kind
        student._features = features
        student.heads = blob["heads"]
        student.n_train_ = meta.get("n_train", 0)
        student.calibrators = {
            k: Calibrator.from_dict(v) for k, v in meta.get("calibration", {}).items()
        }
        student.instructions = meta.get("instructions")
        student.stacks = meta.get("stacks", {}) or {}
        student.stack_report = meta.get("stack_report", {}) or {}
        if student.stacks and meta.get("zero_shot"):
            from shrewd.zeroshot import ZeroShotScorer

            # lazy: nothing loads until the first prediction that needs it
            student.zero_shot = ZeroShotScorer.from_dict(meta["zero_shot"])
        return student


def _macro_f1(truth, proba, k):
    return f1_score(truth, proba.argmax(axis=1), average="macro", labels=list(range(k)),
                    zero_division=0)


class _Expanded:
    """A head whose classes may be a subset of the option set. Pads back to full width."""

    def __init__(self, model, n_options):
        self.model, self.n_options = model, n_options

    def predict_proba(self, X):
        out = np.full((X.shape[0], self.n_options), 1e-6)
        proba = self.model.predict_proba(X)
        for j, option in enumerate(self.model.classes_):
            out[:, int(option)] = proba[:, j]
        return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- scoring


def question_metrics(question, proba, gold):
    """Metrics for one question, by type: a Choice is scored like a classifier, a Noul on
    P(yes) with the base rate alongside, and a Score on the distance between expected and true
    level plus exact and within-one hit rates.
    """
    proba = np.asarray(proba, dtype=float)
    gold = np.asarray(gold, dtype=int)
    keep = gold >= 0
    proba, gold = proba[keep], gold[keep]
    if len(gold) == 0:
        return {"n": 0}
    options = question.options()
    labels = [options[i] for i in gold]
    out = calibration_metrics(proba, labels, options)
    out["type"] = question.kind

    if question.kind == "noul":
        p_yes = proba[:, options.index(YES)]
        truth = (gold == options.index(YES)).astype(float)
        out["base_rate"] = round(float(truth.mean()), 4)
        out["brier"] = round(float(np.mean((p_yes - truth) ** 2)), 4)
        out["auroc"] = _auroc(p_yes, truth)
        out["mean_p_yes"] = round(float(p_yes.mean()), 4)
    elif question.kind == "score":
        levels = np.arange(len(options), dtype=float)
        expected = (proba * levels).sum(axis=1)
        out["mae"] = round(float(np.mean(np.abs(expected - gold))), 4)
        out["exact"] = round(float(np.mean(proba.argmax(axis=1) == gold)), 4)
        out["adjacent"] = round(float(np.mean(np.abs(proba.argmax(axis=1) - gold) <= 1)), 4)
        out["mean_score"] = round(float(expected.mean()), 4)
        out["true_mean_score"] = round(float(gold.mean()), 4)
    return out


def _auroc(scores, truth):
    """Rank-based AUROC. NaN when one class is absent, rather than a misleading 0.5."""
    from scipy.stats import rankdata

    n1 = float(truth.sum())
    n0 = float(len(truth) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return round(float((ranks[truth == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


class DecisionResult:
    def __init__(self, metrics, findings):
        self.metrics = metrics
        self.findings = findings

    def report(self):
        m = self.metrics
        keys = list(m["questions"])
        width = max([12, *(len(k) for k in keys)]) + 2
        lines = [
            f"shrewd decisions: {len(keys)} questions, {m['student_type']} student",
            f"trained on {m['n_train']} documents · scored on the locked test set "
            f"({m['n_test']} documents)",
            "",
            f"{'question':{width}} {'type':7} {'':>7} {'ECE':>7} {'Brier':>7} {'acc':>7}",
        ]
        for key in keys:
            q = m["questions"][key]
            for who in ("teacher", "student"):
                s = q.get(who) or {}
                if not s.get("n"):
                    continue
                headline = _headline(s)
                name = key if who == "teacher" else ""
                lines.append(
                    f"{name:{width}} {s['type']:7} {who[:7]:>7} "
                    f"{s['ece']:>7.3f} {s['brier']:>7.3f} {headline:>7}"
                )
        if m.get("uncalibrated"):
            lines += [
                "",
                f"{'before calibration':{width}} student ECE, averaged over questions: "
                f"{m['uncalibrated']:.3f} → {m['calibrated']:.3f} after",
            ]
        if m.get("human_reviewed"):
            lines.append("")
            lines.append(f"{m['human_reviewed']} teacher answers were overruled by hand "
                         "(needs_review.csv → reviewed.csv)")
        if m.get("stacking"):
            lines.append("")
            header = "zero-shot stack (gate on gold dev: AUROC for yes/no, macro-F1 otherwise)"
            lines.append(f"{header:{width}}")
            for key, s in m["stacking"].items():
                verdict = "kept" if s.get("kept") else "rejected"
                if s.get("reason"):
                    lines.append(f"{key:{width}}  {verdict:8} {s['reason']}")
                    continue
                w = s["weights"]
                metric = {"auroc": "AUROC", "macro_f1": "F1"}.get(s.get("metric"), "gain")
                lines.append(
                    f"{key:{width}}  {verdict:8} {metric} {s['dev_gain']:+.3f} on {s['dev_n']} "
                    f"dev rows  weights head {w[0]:+.2f} zero-shot {w[1]:+.2f}"
                )
        lines.append("")
        if not self.findings:
            lines.append("findings: none, nothing looks off")
        else:
            lines.append("findings")
            for f in self.findings:
                lines.append(f"[{f.severity.upper():4}] {f.title}")
                lines.append(f"{' ' * 7}{f.detail}")
                lines.append(f"{' ' * 7}→ {f.suggestion}")
        return "\n".join(lines)


def _headline(stats):
    if stats["type"] == "score":
        return f"{stats['mae']:.2f}mae"
    if stats["type"] == "noul":
        auroc = stats.get("auroc")
        return "  n/a" if auroc != auroc else f"{auroc:.3f}"  # NaN check
    return f"{stats['accuracy']:.3f}"


# ---------------------------------------------------------------- diagnosis


def decision_findings(metrics):
    findings = []
    for key, q in metrics["questions"].items():
        student = q.get("student") or {}
        teacher = q.get("teacher") or {}
        if not student.get("n"):
            continue
        if student.get("type") == "noul":
            rate = student.get("base_rate", 0.5)
            minority = min(rate, 1 - rate) * student["n"]
            if minority < MIN_MINORITY_ROWS:
                findings.append(Finding(
                    "warn",
                    f"{key}: too lopsided to score",
                    f"only {minority:.0f} of {student['n']} test documents are the "
                    f"minority answer (base rate {rate:.1%}), so every number for this "
                    f"question is noise around the base rate.",
                    "collect more documents where the answer is the rare one, or drop "
                    "the question if it is rare because it does not matter.",
                ))
                continue
        auroc = student.get("auroc")
        if auroc is not None and auroc == auroc and auroc < 0.60:
            # a head that answers the base rate on every document is perfectly
            # calibrated and completely useless, and ECE alone rates it well. A
            # ranking metric tells "calibrated" apart from "informative".
            findings.append(Finding(
                "fail",
                f"{key}: calibrated but uninformative",
                f"AUROC {auroc:.3f}; the head barely separates yes from no, so its "
                f"probabilities sit near the {student.get('base_rate', 0):.1%} base rate "
                "on every document. A low ECE here does not mean useful.",
                "this question may not be answerable from the text alone, or the teacher "
                "may be answering it inconsistently; check the teacher's own AUROC for it "
                "before spending more on labels.",
            ))
            continue
        if student["ece"] > 0.10:
            direction = "over" if student["overconfidence"] > 0 else "under"
            findings.append(Finding(
                "warn",
                f"{key}: probabilities are {direction}confident",
                f"stated {student['mean_confidence']:.2f} on average against "
                f"{student['accuracy']:.2f} actually right (ECE {student['ece']:.3f}).",
                "calibration had too few rows to fit, or the pool does not look like the "
                "test set; check the judged pool size and the seed/pool distribution.",
            ))
        if teacher.get("n") and teacher.get("ece", 0) > 0.10:
            findings.append(Finding(
                "info",
                f"{key}: the teacher itself is miscalibrated here",
                f"teacher ECE {teacher['ece']:.3f} on this question. The student is "
                "calibrated against teacher answers, so it inherits this.",
                "rephrase the question, add criteria descriptions that pin down the "
                "boundary, or hand-label more of the seed set for this question.",
            ))
    return findings


# ---------------------------------------------------------------- the project


def _backend_to_manifest(backend):
    """What to write down so the same teacher setup comes back on reload."""
    if backend is None:
        return "verbalized"
    if isinstance(backend, str):
        return backend
    if getattr(backend, "name", None) == "ensemble":
        return {"name": "ensemble", "models": list(backend.models), "weights": backend.weights}
    return getattr(backend, "name", type(backend).__name__)


def _backend_from_manifest(blob):
    if isinstance(blob, dict) and blob.get("name") == "ensemble":
        from shrewd.judge import EnsembleBackend

        return EnsembleBackend(blob["models"], weights=blob.get("weights"))
    return blob



class Decisions:
    """A panel of typed questions asked of one corpus.

    Point it at a fresh directory with `questions` and `teacher` to start, or at an existing
    directory to resume. `teacher` is any litellm model string or a provider alias;
    `backend` decides how its probabilities are obtained (default: asked for in the reply).
    """

    def __init__(self, dir, questions=None, instructions=None, teacher=None,
                 backend=None, seed=42):
        from shrewd.teacher import resolve_model

        self.dir = Path(dir)
        if questions is not None:
            decide.validate(questions)
        if teacher is not None:
            teacher = resolve_model(teacher)
        manifest_path = self.dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("kind") != "decisions":
                raise ValueError(
                    f"{self.dir} is a classification project, not a decisions project; "
                    "use Project() for it or pick a new directory"
                )
            stored = {k: decide.from_dict(v) for k, v in manifest["questions"].items()}
            if questions is not None and {k: decide.to_dict(q) for k, q in questions.items()} != \
                    manifest["questions"]:
                raise ValueError(
                    f"questions do not match the existing project in {self.dir}; "
                    "changing a question invalidates every answer already paid for, so "
                    "use a new directory"
                )
            if teacher is not None and teacher != manifest["teacher"]:
                raise ValueError(f"teacher does not match the existing project in {self.dir}")
            self.questions = stored
            self.instructions = manifest.get("instructions")
            self.teacher = manifest["teacher"]
            self.seed = manifest["seed"]
            self.backend = backend if backend is not None else _backend_from_manifest(
                manifest.get("backend")
            )
            self._manifest = manifest
        else:
            if not questions or not teacher:
                raise ValueError("a new decisions project needs: questions, teacher")
            self.questions, self.instructions, self.teacher, self.seed = (
                questions, instructions, teacher, seed,
            )
            self.backend = backend
            name = _backend_to_manifest(backend)
            self._manifest = {
                "version": 1,
                "kind": "decisions",
                "questions": {k: decide.to_dict(q) for k, q in questions.items()},
                "instructions": instructions,
                "teacher": teacher,
                "backend": name,
                "seed": seed,
                "stages": [],
            }
            self.dir.mkdir(parents=True, exist_ok=True)
            self._save_manifest()
        self._cache_conn = None

    def _save_manifest(self):
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(self._manifest, f, indent=2)
        os.replace(tmp, self.dir / "manifest.json")

    def _record_stage(self, name, cost, **params):
        from datetime import datetime

        self._manifest["stages"].append({
            "name": name,
            "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cost_usd": round(cost, 4),
            "params": params,
        })
        self._save_manifest()

    def _cache(self):
        from shrewd.teacher import open_cache

        if self._cache_conn is None:
            self._cache_conn = open_cache(self.dir / "cache.db")
        return self._cache_conn

    def _header(self):
        """The optimized prompt header, if optimize() has written one."""
        path = self.dir / "prompt_header.txt"
        return path.read_text() if path.exists() else None

    def optimize(self, budget=800, target=None, reflection_model=None, overwrite=False,
                 max_header_words=None):
        """GEPA-optimize the prompt header against the dev split. Never sees the test set.

        The objective is mean 1 - Brier/2 over the panel (a proper scoring rule, not accuracy).
        Only the text above the rendered questions is rewritten. `budget` counts teacher calls.
        `max_header_words` asks the reflection model to keep the header short. The header is
        cached provider-side, but a long one makes the teacher write more per document.
        """
        from shrewd.optimize import run_gepa_decisions
        from shrewd.teacher import resolve_model

        out_path = self.dir / "prompt_header.txt"
        if out_path.exists() and not overwrite:
            print(f"{out_path} exists, skipping optimization (pass overwrite=True to redo)")
            return None
        dev_path = self.dir / "seed_dev.csv"
        if not dev_path.exists():
            raise FileNotFoundError("no seed data; call add_seed() first")
        dev = pd.read_csv(dev_path)
        dev["text"] = dev["text"].astype(str)
        reflection = resolve_model(reflection_model) if reflection_model else self.teacher
        print(f"optimizing the prompt header with GEPA: budget {budget} teacher calls over "
              f"{len(dev)} dev rows, objective 1 - Brier/2")
        header, log, cost = run_gepa_decisions(
            dev, self.questions, decide.default_header(self.instructions), self.teacher,
            budget, target, reflection, self._cache(), self.seed,
            max_header_words=max_header_words,
        )
        out_path.write_text(header)
        (self.dir / "optimize_log.json").write_text(json.dumps(log, indent=2))
        self._record_stage("optimize", cost, budget=budget, target=target)
        cap = f" (cap {max_header_words})" if max_header_words else ""
        print(f"dev score: {log['baseline_score']:.3f} (default header) → "
              f"{log['best_score']:.3f} best of {len(log['candidates'])}; cost ${cost:.2f}; "
              f"header {len(header.split())} words{cap}")
        print(f"wrote {out_path}; judge() and distill() will use it")
        return log

    # -- stages ----------------------------------------------------------------

    def add_seed(self, df, test_frac=0.35, overwrite=False):
        """Split hand-answered documents into dev and a locked test set.

        `df` needs a `text` column plus one column per question id, holding the
        hand-labeled answer. Blanks are allowed: a document may answer some questions
        and not others, and each question is scored on the rows that answer it.
        """
        from sklearn.model_selection import train_test_split

        missing = {"text"} - set(df.columns)
        if missing:
            raise ValueError("seed data needs a 'text' column")
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
        gold = gold_frame(df, self.questions)
        answered = (gold >= 0).sum(axis=0)
        for j, key in enumerate(self.questions):
            if answered[j] < 2 * MIN_MINORITY_ROWS:
                warnings.warn(
                    f"question {key!r} has only {answered[j]} answered seed rows; its "
                    "scores will be noisy and it cannot be calibrated on the seed alone",
                    stacklevel=2,
                )
                continue
            # a rare question is limited by its minority answers, not its row count:
            # 500 documents that are 2% yes carry 10 usable rows, and after the test
            # split, 3. Better to hear that now than after paying to judge a pool.
            column = gold[gold[:, j] >= 0, j]
            counts = np.bincount(column, minlength=len(self.questions[key].options()))
            minority = int(counts[counts > 0].min()) if (counts > 0).any() else 0
            in_test = minority * test_frac
            if in_test < MIN_MINORITY_ROWS:
                warnings.warn(
                    f"question {key!r} has {minority} seed documents with its rarest "
                    f"answer, so only about {in_test:.0f} land in the test split; its "
                    "test scores will be noise. Add seed documents with that answer, or "
                    "score this question on a larger held-out set of your own.",
                    stacklevel=2,
                )
        test_path = self.dir / "seed_test.csv"
        if test_path.exists() and not overwrite:
            raise ValueError(
                f"{test_path} already exists and the test set is locked. Pass "
                "overwrite=True if you really mean to re-split."
            )
        # stratify on the question with the most balanced answers, so the rarest
        # thing we can stratify on still lands on both sides of the split
        strat = None
        spread = [(abs(0.5 - np.mean(gold[:, j][gold[:, j] >= 0] > 0)), j)
                  for j in range(gold.shape[1]) if (gold[:, j] >= 0).sum() == len(df)]
        if spread:
            column = min(spread)[1]
            values = gold[:, column]
            if min(np.bincount(values[values >= 0]).tolist() or [0]) >= 2:
                strat = values
        dev, test = train_test_split(
            df, test_size=test_frac, random_state=self.seed, stratify=strat
        )
        dev.reset_index(drop=True).to_csv(self.dir / "seed_dev.csv", index=False)
        test.reset_index(drop=True).to_csv(test_path, index=False)
        self._manifest["splits"] = {"n_dev": len(dev), "n_test": len(test)}
        self._record_stage("add_seed", 0.0, test_frac=test_frac)
        print(f"seed: {len(dev)} dev / {len(test)} test documents, "
              f"{len(self.questions)} questions")

    def judge(self, df, dry_run=False, concurrency=8, overwrite=False):
        """Have the teacher answer every question about every pooled document.

        One call per document covers the whole panel, which is where the saving is: the
        document is the expensive part of the prompt and asking N questions separately
        pays for it N times.
        """
        from shrewd.judge import judge_texts
        from shrewd.teacher import estimate_calls

        out_path = self.dir / "pool_judged.csv"
        if out_path.exists() and not overwrite:
            existing = pd.read_csv(out_path)
            print(f"pool already judged: {len(existing)} documents (pass overwrite=True to redo)")
            return existing
        # a human may have filled in needs_review.csv since the last run, so ledger those
        # answers before the queue is regenerated, or re-judging would throw them away
        self.apply_review()
        texts = [t for t in df["text"].astype(str) if t.strip()]
        seen = set()
        if (self.dir / "seed_dev.csv").exists():
            for name in ("seed_dev.csv", "seed_test.csv"):
                seen |= set(pd.read_csv(self.dir / name)["text"].astype(str))
        texts = [t for t in dict.fromkeys(texts) if t not in seen]
        prompt = decide.build_prompt(self.questions, self.instructions, header=self._header())
        if dry_run:
            n, cost = estimate_calls(texts, prompt, self.teacher, 1, self._cache())
            per = f"${cost:.2f}" if cost is not None else "unknown"
            print(f"dry run: {n} uncached calls for {len(texts)} documents × "
                  f"{len(self.questions)} questions, about {per}")
            return None
        frame, cost = judge_texts(
            texts, self.questions, self.teacher, instructions=self.instructions,
            backend=self.backend, conn=self._cache(), concurrency=concurrency,
            header=self._header(),
        )
        frame = self._reapply_reviewed(frame)
        frame.to_csv(out_path, index=False)
        review = self._review_queue(frame)
        review.to_csv(self.dir / "needs_review.csv", index=False)
        self._record_stage("judge", cost, n_documents=len(texts))
        print(f"judged {len(texts)} documents × {len(self.questions)} questions "
              f"for ${cost:.2f} (one call per document)")
        if len(review):
            print(f"{len(review)} (document, question) pairs where the teacher was unsure "
                  f"are in needs_review.csv; edit pool_judged.csv to overrule it")
        return frame

    def distill(self, features="auto", soft=True, calibration="auto",
                calibrate_teacher=False, zero_shot=False, stack_margin=STACK_MARGIN):
        """Train the heads on the judged pool, calibrate them, and score against gold.

        `zero_shot=True` (or a `ZeroShotScorer`) stacks an NLI model onto each head and keeps
        it per question only if it beats the plain head on the gold dev split by
        `stack_margin`. Needs `shrewd[encoder]` and costs a transformer pass per (document, option).

        `calibrate_teacher=True` first calibrates the teacher's answers on the dev split and
        rescales the pool answers through it before training. Useful for rare questions where
        the teacher over-states. The locked test set is never used for this.
        """
        from shrewd.judge import judge_texts

        self.apply_review()
        ledger = self.dir / "reviewed.csv"
        # what the report shows is every human answer in effect, not just the ones
        # applied by this call: a fix made before a re-judge is still a fix
        reviewed = len(pd.read_csv(ledger)) if ledger.exists() else 0
        pool = pd.read_csv(self.dir / "pool_judged.csv")
        dev = pd.read_csv(self.dir / "seed_dev.csv")
        test = pd.read_csv(self.dir / "seed_test.csv")
        for frame in (dev, test):
            frame["text"] = frame["text"].astype(str)
        pool["text"] = pool["text"].astype(str)

        pool_targets = self._targets_from_pool(pool)
        dev_gold = gold_frame(dev, self.questions)
        dev_targets = self._targets_from_gold(dev_gold)
        teacher_cost = 0.0
        teacher_cals = {}
        if calibrate_teacher:
            teacher_cals, teacher_cost = self._calibrate_teacher(dev, dev_gold, calibration)
            pool_targets = {
                key: (teacher_cals[key].transform(values) if key in teacher_cals else values)
                for key, values in pool_targets.items()
            }
        texts = pool["text"].tolist() + dev["text"].tolist()
        targets = {
            key: np.vstack([pool_targets[key], dev_targets[key]]) for key in self.questions
        }

        if features == "auto":
            features = self._pick_features(pool["text"].tolist(), pool_targets,
                                           dev["text"].tolist(), dev_gold, soft)
        student = DecisionStudent(self.questions, features=features, seed=self.seed, soft=soft,
                                  instructions=self.instructions)
        print(f"training {student.features_kind} student on {len(texts)} documents, "
              f"{len(self.questions)} heads over one featurization")
        student.fit(texts, targets)
        before = self._score_all(student, test, calibrated=False)
        if zero_shot:
            from shrewd.zeroshot import ZeroShotScorer

            student.zero_shot = zero_shot if hasattr(zero_shot, "score") else ZeroShotScorer()
            print(f"stacking a zero-shot second opinion; gate is gold dev AUROC (yes/no) or "
                  f"macro-F1 (multi-class) > {stack_margin:.2f}")
            student.stack(pool["text"].tolist(), pool_targets, dev["text"].tolist(),
                          dev_gold, margin=stack_margin)
            kept = [k for k, r in student.stack_report.items() if r.get("kept")]
            print(f"  stack kept on {len(kept)}/{len(student.stack_report)} questions"
                  + (f": {', '.join(kept)}" if kept else ""))
        print(f"calibrating on {len(pool)} judged documents (cross-fit, no API calls)")
        student.calibrate(pool["text"].tolist(), pool_targets, method=calibration)
        shutil.rmtree(self.dir / "student", ignore_errors=True)
        student.save(self.dir / "student")

        after = self._score_all(student, test, calibrated=True)
        teacher_frame, cost = judge_texts(
            test["text"].tolist(), self.questions, self.teacher,
            instructions=self.instructions, backend=self.backend, conn=self._cache(),
            desc="teacher on test", header=self._header(),
        )
        teacher_scores = self._score_teacher(teacher_frame, test)

        test_gold = gold_frame(test, self.questions)
        metrics = {
            "student_type": student.features_kind + (" soft" if soft else ""),
            "n_train": len(texts),
            "n_test": len(test),
            "human_reviewed": int(reviewed),
            "questions": {
                key: {
                    "student": after.get(key),
                    "teacher": teacher_scores.get(key),
                    "calibration": (student.calibrators[key].to_dict()
                                    if key in student.calibrators else None),
                    "n_answered_test": int((test_gold[:, j] >= 0).sum()),
                }
                for j, key in enumerate(self.questions)
            },
        }
        scored = [k for k in self.questions if after.get(k, {}).get("n")]
        if scored:
            metrics["uncalibrated"] = round(
                float(np.mean([before[k]["ece"] for k in scored])), 4)
            metrics["calibrated"] = round(
                float(np.mean([after[k]["ece"] for k in scored])), 4)
        if teacher_cals:
            metrics["teacher_calibration"] = {k: c.to_dict() for k, c in teacher_cals.items()}
        if student.stack_report:
            metrics["stacking"] = student.stack_report
        findings = decision_findings(metrics)
        (self.dir / "report.json").write_text(json.dumps(metrics, indent=2))
        self._record_stage("distill", cost + teacher_cost, features=student.features_kind,
                           soft=soft, calibrate_teacher=calibrate_teacher,
                           zero_shot=bool(zero_shot))
        return DecisionResult(metrics, findings)

    def _pick_features(self, pool_texts, pool_targets, dev_texts, dev_gold, soft):
        """tf-idf or static embeddings: fit each on the pool alone, score on gold dev, keep
        the winner. The same shape as the stacking gate, for the same reason: on seven
        datasets neither candidate was the better default more than about half the time,
        and the losses on the wrong side were 8-11 points."""
        candidates = ["tfidf"] + (["embed"] if _embed_available() else [])
        if len(candidates) == 1:
            return "tfidf"
        scores = {}
        for feats in candidates:
            twin = DecisionStudent(self.questions, features=feats, seed=self.seed, soft=soft)
            twin.fit(pool_texts, pool_targets)
            proba = twin.predict_proba(dev_texts, calibrated=False)
            per_question = []
            for j, (key, question) in enumerate(self.questions.items()):
                if key not in proba:
                    continue
                rows = np.where(dev_gold[:, j] >= 0)[0]
                if len(rows) < MIN_GATE_ROWS:
                    continue
                truth = dev_gold[rows, j]
                if question.kind == "noul":
                    value = _auroc(proba[key][rows, 1], truth)
                    if value == value:
                        per_question.append(value)
                else:
                    per_question.append(_macro_f1(truth, proba[key][rows], len(question.options())))
            scores[feats] = float(np.mean(per_question)) if per_question else float("-inf")
        winner = max(candidates, key=lambda f: scores[f])
        shown = "  ".join(f"{f} {scores[f]:.3f}" for f in candidates)
        print(f'features="auto": {shown} on gold dev → {winner}')
        return winner

    def _calibrate_teacher(self, dev, dev_gold, method):
        """Fit a calibrator per question on the teacher's answers over the dev split.

        Uses dev and never test: the teacher's answers on the locked set are what the
        report scores it on, and fitting a scale there would be grading its own work.
        """
        from shrewd.judge import judge_texts

        path = self.dir / "dev_judged.csv"
        if path.exists():
            frame, cost = pd.read_csv(path), 0.0
        else:
            frame, cost = judge_texts(
                dev["text"].tolist(), self.questions, self.teacher,
                instructions=self.instructions, backend=self.backend, conn=self._cache(),
                desc="teacher on dev", header=self._header(),
            )
            frame.to_csv(path, index=False)
        by_text = frame.set_index("text")
        out = {}
        for j, (key, question) in enumerate(self.questions.items()):
            options = question.options()
            values = by_text.reindex(dev["text"])[
                [f"{key}__{o}" for o in options]
            ].to_numpy(dtype=float)
            usable = ~np.isnan(values).any(axis=1) & (dev_gold[:, j] >= 0)
            if usable.sum() < MIN_CALIBRATION_ROWS:
                continue
            labels = [options[i] for i in dev_gold[usable, j]]
            out[key] = Calibrator.fit(
                values[usable], labels, options, method=method, seed=self.seed
            )
        if out:
            print(f"calibrated the teacher on {len(dev)} dev documents "
                  f"({len(out)}/{len(self.questions)} questions) for ${cost:.2f}")
        return out, cost

    # -- helpers ---------------------------------------------------------------

    def apply_review(self):
        """Overrule the teacher wherever a human filled in `human_answer` in needs_review.csv.
        Each fix becomes a certain target and is written to `reviewed.csv`, which judge()
        re-applies after any re-judging. Runs at the start of distill(). Returns the count.
        """
        path = self.dir / "needs_review.csv"
        pool_path = self.dir / "pool_judged.csv"
        if not path.exists() or not pool_path.exists():
            return 0
        review = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "human_answer" not in review.columns:
            return 0
        fixes = review[review["human_answer"].str.strip() != ""]
        if fixes.empty:
            return 0
        pool = pd.read_csv(pool_path)
        pool["text"] = pool["text"].astype(str)
        index = {t: i for i, t in enumerate(pool["text"])}
        applied, rejected = [], []
        for _, row in fixes.iterrows():
            key, answer, text = row["question"], row["human_answer"].strip(), row["text"]
            question = self.questions.get(key)
            if question is None:
                rejected.append(f"{key!r} is not a question in this panel")
                continue
            if text not in index:
                rejected.append(f"{text[:40]!r} is not in the judged pool")
                continue
            g = gold_index(question, answer)
            if g is None:
                rejected.append(f"{key}: {answer!r} is not one of {question.options()}")
                continue
            for j, option in enumerate(question.options()):
                pool.loc[index[text], f"{key}__{option}"] = 1.0 if j == g else 0.0
            applied.append({"text": text, "question": key, "answer": question.options()[g]})
        if applied:
            pool.to_csv(pool_path, index=False)
            ledger_path = self.dir / "reviewed.csv"
            ledger = pd.DataFrame(applied)
            if ledger_path.exists():
                old = pd.read_csv(ledger_path, dtype=str)
                ledger = pd.concat([old, ledger]).drop_duplicates(["text", "question"], keep="last")
            ledger.to_csv(ledger_path, index=False)
            self._record_stage("review", 0.0, applied=len(applied), rejected=len(rejected))
            print(f"applied {len(applied)} human answers over the teacher's; "
                  f"ledger in reviewed.csv")
        if rejected:
            warnings.warn(
                f"{len(rejected)} human answers could not be applied: " + "; ".join(rejected[:5]),
                stacklevel=2,
            )
        return len(applied)

    def _reapply_reviewed(self, frame):
        """Put every ledgered human answer back onto a freshly judged pool."""
        ledger_path = self.dir / "reviewed.csv"
        if not ledger_path.exists():
            return frame
        ledger = pd.read_csv(ledger_path, dtype=str)
        frame = frame.copy()
        frame["text"] = frame["text"].astype(str)
        index = {t: i for i, t in enumerate(frame["text"])}
        n = 0
        for _, row in ledger.iterrows():
            question = self.questions.get(row["question"])
            if question is None or row["text"] not in index:
                continue
            g = gold_index(question, row["answer"])
            if g is None:
                continue
            i = index[row["text"]]
            for j, option in enumerate(question.options()):
                frame.loc[i, f"{row['question']}__{option}"] = 1.0 if j == g else 0.0
            n += 1
        if n:
            print(f"re-applied {n} human answers from reviewed.csv")
        return frame

    def _review_queue(self, frame, floor=REVIEW_FLOOR):
        """The teacher's least confident answers, one row per (document, question): the file a
        human edits. Fill in `human_answer` (an option name, yes/no, or a Score level) and the
        next distill() or apply_review() uses it. Rows land here when the winning option is
        below `floor`. Extra rows for pool documents not in the queue are honored too.
        """
        rows = []
        for _, row in frame.iterrows():
            for key, question in self.questions.items():
                options = question.options()
                probs = np.array([row.get(f"{key}__{o}", np.nan) for o in options], dtype=float)
                if np.isnan(probs).any():
                    rows.append({"text": row["text"], "question": key, "teacher_answer": None,
                                 "teacher_prob": None, "why": "no usable answer",
                                 "human_answer": ""})
                    continue
                top = int(probs.argmax())
                if probs[top] < floor:
                    runner = int(np.argsort(probs)[-2]) if len(options) > 1 else top
                    rows.append({
                        "text": row["text"], "question": key,
                        "teacher_answer": options[top], "teacher_prob": round(float(probs[top]), 3),
                        "why": f"unsure: {options[top]} {probs[top]:.2f} vs "
                               f"{options[runner]} {probs[runner]:.2f}",
                        "human_answer": "",
                    })
        columns = ["text", "question", "teacher_answer", "teacher_prob", "why", "human_answer"]
        out = pd.DataFrame(rows, columns=columns)
        return out.sort_values(["teacher_prob"], na_position="first").reset_index(drop=True)

    def _targets_from_pool(self, pool):
        out = {}
        for key, question in self.questions.items():
            options = question.options()
            columns = [f"{key}__{o}" for o in options]
            missing = [c for c in columns if c not in pool.columns]
            if missing:
                raise ValueError(f"judged pool is missing columns for question {key!r}")
            values = pool[columns].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            total = values.sum(axis=1, keepdims=True)
            out[key] = np.divide(values, total, out=np.zeros_like(values), where=total > 1e-9)
        return out

    def _targets_from_gold(self, gold):
        out = {}
        for j, (key, question) in enumerate(self.questions.items()):
            target = np.zeros((gold.shape[0], len(question.options())))
            answered = gold[:, j] >= 0
            target[np.where(answered)[0], gold[answered, j]] = 1.0
            out[key] = target
        return out

    def _score_all(self, student, test, calibrated):
        gold = gold_frame(test, self.questions)
        proba = student.predict_proba(test["text"].tolist(), calibrated=calibrated)
        return {
            key: question_metrics(self.questions[key], proba[key], gold[:, j])
            for j, key in enumerate(self.questions) if key in proba
        }

    def _score_teacher(self, frame, test):
        gold = gold_frame(test, self.questions)
        by_text = frame.set_index("text")
        out = {}
        for j, (key, question) in enumerate(self.questions.items()):
            options = question.options()
            columns = [f"{key}__{o}" for o in options]
            values = by_text.reindex(test["text"])[columns].to_numpy(dtype=float)
            usable = ~np.isnan(values).any(axis=1)
            column_gold = np.where(usable, gold[:, j], -1)
            values = np.nan_to_num(values, nan=1.0 / len(options))
            total = values.sum(axis=1, keepdims=True)
            values = np.divide(values, total, out=np.zeros_like(values), where=total > 1e-9)
            out[key] = question_metrics(question, values, column_gold)
        return out
