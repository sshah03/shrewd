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

import dataclasses
import json
import shutil
import textwrap
import time
import warnings
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from shrewd import data, decide
from shrewd.calibrate import (
    Calibrator,
    brier_decomposition,
    calibration_metrics,
    resolution_share,
)
from shrewd.decide import NO, YES
from shrewd.evaluate import Finding
from shrewd.zeroshot import combine, fit_bias, fit_weights, hypotheses

MIN_CALIBRATION_ROWS = 60
MIN_MINORITY_ROWS = 12
STACK_MARGIN = 0.02      # macro-F1 the stack must gain on gold dev to be kept
REVIEW_FLOOR = 0.6       # a teacher answer below this lands in needs_review.csv
AUDIT_SHARE = 0.05
SPLIT_SETTLED = 0.9
MIN_GATE_ROWS = 40       # fewer gold dev rows than this and the gate abstains
MIN_RESOLUTION_SHARE = 0.03  # below this a Choice or Score head is ~AUROC 0.6 on a balanced split


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
        self.routing = None        # threshold and its guarantee, from set_routing()
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

    def decide(self, texts, calibrated=True, questions=None, route=False, fallback=None):
        """Typed answers. A single string in gives a single answer map back.

        `route=True` answers only the documents that clear the routing threshold set by
        `distill(route_budget=...)` and gives None for the rest. `fallback` (a callable taking
        a list of texts and returning one answer map per text, such as `Decisions.ask`)
        answers the rest instead, and implies `route=True`.
        """
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        keys = self._select(questions)
        route = route or fallback is not None
        if route:
            local = self.local(texts)
            kept = [t for t, ok in zip(texts, local, strict=True) if ok]
            proba = (self.predict_proba(kept, calibrated=calibrated, questions=keys)
                     if kept else {})
        else:
            local = np.ones(len(texts), dtype=bool)
            proba = self.predict_proba(texts, calibrated=calibrated, questions=keys)
        rows = [None] * len(texts)
        for i, row in enumerate(np.where(local)[0]):
            rows[row] = {key: decide.make_answer(key, self.questions[key], proba[key][i])
                         for key in keys}
        if fallback is not None and not local.all():
            rest = [t for t, ok in zip(texts, local, strict=True) if not ok]
            answers = list(fallback(rest))
            if len(answers) != len(rest):
                raise ValueError(f"fallback returned {len(answers)} answers for {len(rest)} texts")
            answers = iter(answers)
            rows = [row if row is not None else next(answers) for row in rows]
        return rows[0] if single else rows

    # -- routing ---------------------------------------------------------------

    def routing_scores(self, texts, calibrated=True):
        """Each document's lowest top probability over the questions the threshold covers."""
        from shrewd.route import doc_scores

        keys = self.routing["questions"] if self.routing else list(self.heads)
        return doc_scores(self.predict_proba(texts, calibrated=calibrated, questions=keys))

    def local(self, texts):
        """True for the documents this model answers itself under its routing guarantee."""
        if not self.routing:
            raise ValueError(
                "this model has no routing threshold; distill with route_budget= (for "
                "example 0.02) to set one with a guarantee"
            )
        texts = [texts] if isinstance(texts, str) else list(texts)
        threshold = self.routing["threshold"]
        if threshold is None:
            return np.zeros(len(texts), dtype=bool)
        return self.routing_scores(texts) >= threshold

    def set_routing(self, texts, targets, budget=0.02, delta=0.05):
        """Set the threshold on documents this model never trained on, judged by the teacher.

        The guarantee: with probability 1 - delta over these documents, the share of requests
        like them that are answered locally *and* disagree with the teacher is at most
        `budget`. It is a share of all requests, not of the locally answered ones.
        """
        from shrewd.route import doc_disagree, doc_scores, pick_threshold

        keys = [k for k in self.heads if k in targets]
        proba = self.predict_proba(list(texts), questions=keys)
        scores, disagree = doc_scores(proba), doc_disagree(proba, targets, keys)
        _, stats = pick_threshold(scores, disagree, budget, delta)
        self.routing = {**stats, "questions": keys}
        self._routing_held = {"scores": [round(float(s), 6) for s in scores],
                              "disagree": [bool(d) for d in disagree]}
        return self

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
                    "routing": self.routing,
                    "zero_shot": (self.zero_shot.to_dict()
                                  if self.zero_shot is not None and self.stacks else None),
                    "versions": {"shrewd": __version__, "scikit-learn": sklearn.__version__},
                },
                indent=2,
            )
        )
        held = getattr(self, "_routing_held", None)
        if held is not None:
            from shrewd.route import HELD_FILE

            (path / HELD_FILE).write_text(json.dumps(held))

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
        try:
            student.calibrators = {
                k: Calibrator.from_dict(v) for k, v in meta.get("calibration", {}).items()
            }
        except ValueError as exc:
            from shrewd import __version__

            saved = (meta.get("versions") or {}).get("shrewd", "an unknown version")
            raise ValueError(f"{exc} (saved with shrewd {saved}, this is {__version__})") from exc
        student.instructions = meta.get("instructions")
        student.stacks = meta.get("stacks", {}) or {}
        student.stack_report = meta.get("stack_report", {}) or {}
        student.routing = meta.get("routing")
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
        parts = brier_decomposition(p_yes, truth)
        out["brier_rel"] = round(parts["reliability"], 4)
        out["brier_res"] = round(parts["resolution"], 4)
        out["brier_unc"] = round(parts["uncertainty"], 4)
        out["resolution_share"] = round(resolution_share(parts), 4)
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
            # the last column is a different metric per question type, so each value says
            # which: accuracy (choice), AUROC (yes/no), mean absolute error (score)
            f"{'question':{width}} {'type':7} {'':>7} {'ECE':>7} {'Brier':>7} {'resol':>7}  "
            f"headline",
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
                    f"{s['ece']:>7.3f} {s['brier']:>7.3f} {_share(s):>7}  {headline}"
                )
        if m.get("teachers"):
            from shrewd.agree import short_name

            names = [short_name(t) for t in m["teachers"]]
            cell = max(14, *(len(n) for n in names)) + 2
            lines += ["", "each teacher alone on the same test documents (the row above is "
                          "their combination)",
                      f"{'question':{width}} " + "".join(f"{n:>{cell}}" for n in names)]
            for key in keys:
                row = [(m["teachers"][t].get(key) or {}) for t in m["teachers"]]
                if not any(s.get("n") for s in row):
                    continue
                lines.append(f"{key:{width}} " + "".join(
                    f"{_headline(s) if s.get('n') else 'n/a':>{cell}}" for s in row))
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
        routing = m.get("routing")
        if routing:
            lines.append("")
            if routing.get("threshold") is None:
                text = (f"routing: {routing['n']} held-out documents could not certify a "
                        f"{routing['budget']:.0%} disagreement budget, so nothing is answered "
                        "locally")
            else:
                text = (f"routing: answer locally when every question's confidence is at least "
                        f"{routing['threshold']:.3f}. At most {routing['budget']:.0%} of all "
                        f"requests get a local answer the teacher disagrees with "
                        f"({1 - routing['delta']:.0%} bound, set on {routing['n']} held-out "
                        f"documents, {routing['coverage']:.1%} of them local).")
                t = routing.get("test") or {}
                if t.get("answers_checked"):
                    text += (f" On the test set {t['coverage']:.1%} are local, and against "
                             f"{t['answers_checked']} hand-labeled answers those are wrong "
                             f"{t['student_error']:.1%} of the time (the teacher "
                             f"{t['teacher_error']:.1%} on the same documents).")
            lines += textwrap.wrap(text, width=88, subsequent_indent=" " * 9)
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


def _share(stats):
    share = stats.get("resolution_share")
    return "  n/a" if share is None or share != share else f"{share:.3f}"  # NaN check


def _headline(stats):
    if stats["type"] == "score":
        return f"{stats['mae']:.2f} MAE"
    if stats["type"] == "noul":
        auroc = stats.get("auroc")
        return "n/a AUROC" if auroc != auroc else f"{auroc:.3f} AUROC"  # NaN check
    return f"{stats['accuracy']:.3f} acc"


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
        share = student.get("resolution_share")
        if (student.get("type") != "noul" and share is not None and share == share
                and share < MIN_RESOLUTION_SHARE):
            # the multi-option version of the AUROC check above. Yes/no questions keep
            # AUROC because the share shrinks with the base rate at a fixed ranking quality.
            # A warning, not a fail: the threshold comes from a simulation and hasn't been
            # checked against a multi-option question with human labels yet
            findings.append(Finding(
                "warn",
                f"{key}: calibrated but uninformative",
                f"resolution is {share:.1%} of the uncertainty (Brier reliability "
                f"{student['brier_rel']:.3f}, resolution {student['brier_res']:.3f}); the "
                "head's probabilities barely move from the option base rates from one "
                "document to the next. A low ECE here does not mean useful.",
                "this question may not be answerable from the text alone, or the teacher "
                "may be answering it inconsistently; compare the teacher's own resolution "
                "for it before spending more on labels.",
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


def _write_teacher_check(project, agreement, cost):
    from shrewd.agree import short_name

    summary = agreement.summary()
    summary["disagreements"] = agreement.gold_disagreements()
    (project.dir / "teacher_check.json").write_text(json.dumps(summary, indent=2))
    project._record_stage("check_teachers", cost, teachers=len(agreement.teachers))
    print(f"checked {len(agreement.teachers)} teacher(s) on {summary['n_dev']} dev documents "
          f"for ${cost:.2f}")
    for key, entry in summary["questions"].items():
        accuracy = entry["dev_accuracy"]
        shown = "  ".join(f"{short_name(m)} {accuracy[m]:.2f}" for m in agreement.teachers
                          if m in accuracy)
        if len(agreement.teachers) > 1 and "combined" in accuracy:
            shown += f"  combined {accuracy['combined']:.2f}"
        gap = summary["disagreements"][key]
        tail = ""
        if gap["pairs"]:
            first = gap["pairs"][0]
            tail = f"; most often you said {first['yours']} and they said {first['theirs']}"
        print(f"  {key}: {shown}{tail}")
    return summary


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


def _float_or_none(value):
    value = float(value)
    return None if np.isnan(value) else value

class Decisions:
    """A panel of typed questions asked of one corpus.

    Point it at a fresh directory with `questions` and `teacher` to start, or at an existing
    directory to resume. `teacher` is any litellm model string or a provider alias;
    `backend` decides how its probabilities are obtained (default: asked for in the reply).

    `teacher` can also be a list of models. Their answers are combined per question,
    weighted by how well each matched your dev labels.
    """

    def __init__(self, dir, questions=None, instructions=None, teacher=None,
                 backend=None, seed=42):
        from shrewd.teacher import resolve_model

        self.dir = Path(dir)
        if questions is not None:
            decide.validate(questions)
        if isinstance(teacher, (list, tuple)):
            teacher = [resolve_model(t) for t in teacher]
            if len(set(teacher)) != len(teacher) or not teacher:
                raise ValueError("teacher needs distinct models")
            if len(teacher) == 1:
                teacher = teacher[0]
            elif getattr(backend, "name", None) == "ensemble":
                raise ValueError("pass several teachers or an EnsembleBackend, not both")
        elif teacher is not None:
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
        self._agreement_fit = None

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

    def set_teacher(self, teacher):
        """Change the teacher, or pass a list of several. Cached answers are kept."""
        from shrewd.teacher import resolve_model

        if isinstance(teacher, (list, tuple)):
            teacher = [resolve_model(t) for t in teacher]
            if not teacher or len(set(teacher)) != len(teacher):
                raise ValueError("teacher needs distinct models")
            if len(teacher) == 1:
                teacher = teacher[0]
            elif getattr(self.backend, "name", None) == "ensemble":
                raise ValueError("pass several teachers or an EnsembleBackend, not both")
        else:
            teacher = resolve_model(teacher)
        self.teacher = teacher
        self._manifest["teacher"] = teacher
        self._agreement_fit = None
        self._record_stage("set_teacher", 0.0, teacher=teacher)

    def check_teachers(self, dry_run=False):
        """Have each teacher answer the dev split and report how often it matches your labels."""
        dev_path = self.dir / "seed_dev.csv"
        if not dev_path.exists():
            raise FileNotFoundError("no seed data; call add_seed() first")
        if dry_run:
            from shrewd.teacher import estimate_calls

            texts = pd.read_csv(dev_path)["text"].astype(str).tolist()
            prompt = decide.build_prompt(self.questions, self.instructions, header=self._header())
            n, cost = 0, 0.0
            for model in self._teachers():
                calls, price = estimate_calls(texts, prompt, model, 1, self._cache(),
                                              cached_prompt=True)
                n += calls
                cost = None if cost is None or price is None else cost + price
            return {"uncached_calls": n, "estimated_usd": cost}
        agreement, cost = self._agreement()
        return _write_teacher_check(self, agreement, cost)

    def _teachers(self):
        return list(self.teacher) if isinstance(self.teacher, (list, tuple)) else [self.teacher]

    def _several(self):
        return len(self._teachers()) > 1

    def _judge_each(self, texts, desc):
        from shrewd.agree import short_name
        from shrewd.judge import judge_texts

        frames, total = [], 0.0
        for model in self._teachers():
            frame, cost = judge_texts(
                texts, self.questions, model, instructions=self.instructions,
                backend=self.backend, conn=self._cache(), desc=f"{desc} ({short_name(model)})",
                header=self._header(),
            )
            frames.append(frame)
            total += cost
        return frames, total

    def _judge(self, texts, desc):
        if not self._several():
            from shrewd.judge import judge_texts

            frame, cost = judge_texts(
                texts, self.questions, self.teacher, instructions=self.instructions,
                backend=self.backend, conn=self._cache(), desc=desc, header=self._header(),
            )
            return frame, None, cost
        frames, cost = self._judge_each(texts, desc)
        agreement, dev_cost = self._agreement()
        return agreement.combine(frames, texts), frames, cost + dev_cost

    def _agreement(self):
        from shrewd.agree import Agreement

        header = self._header()
        if self._agreement_fit is not None and self._agreement_fit[0] == header:
            return self._agreement_fit[1], 0.0
        dev_path = self.dir / "seed_dev.csv"
        if not dev_path.exists():
            raise FileNotFoundError(
                "several teachers are combined by how well each matches your labels on the "
                "dev split; call add_seed() first"
            )
        dev = pd.read_csv(dev_path)
        dev["text"] = dev["text"].astype(str)
        frames, cost = self._judge_each(dev["text"].tolist(), "teachers on dev")
        agreement = Agreement(self.questions, self._teachers(), seed=self.seed).fit(
            frames, dev["text"].tolist(), gold_frame(dev, self.questions)
        )
        self._agreement_fit = (header, agreement)
        return agreement, cost

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
        primary = self._teachers()[0]
        if self._several():
            print(f"several teachers configured; optimizing the header against {primary}")
        reflection = resolve_model(reflection_model) if reflection_model else primary
        print(f"optimizing the prompt header with GEPA: budget {budget} teacher calls over "
              f"{len(dev)} dev rows, objective 1 - Brier/2")
        header, log, cost = run_gepa_decisions(
            dev, self.questions, decide.default_header(self.instructions), primary,
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

    def set_question(self, key, question):
        """Add a question or change one. Run judge(overwrite=True) afterwards; answers to the
        other questions come from the cache.
        """
        decide.validate({key: question})
        old = self.questions.get(key)
        if old is not None and decide.to_dict(old) == decide.to_dict(question):
            return False
        self.questions = {**self.questions, key: question}
        self._manifest["questions"] = {k: decide.to_dict(q) for k, q in self.questions.items()}
        self._agreement_fit = None
        bad = 0
        for name in ("seed_dev.csv", "seed_test.csv"):
            path = self.dir / name
            if path.exists():
                seed = pd.read_csv(path, dtype=str, keep_default_na=False)
                if key in seed.columns:
                    answered = seed[key].str.strip() != ""
                    bad += sum(gold_index(question, v) is None for v in seed.loc[answered, key])
        if bad:
            warnings.warn(f"{bad} seed answers for {key!r} don't fit the new options; fix "
                          "them with fix_example() or they're left out of scoring",
                          stacklevel=2)
        self._record_stage("set_question", 0.0, question=key, added=old is None)
        return True

    def remove_question(self, key):
        """Remove a question from your examples, the judged pool and the review files."""
        if key not in self.questions:
            raise ValueError(f"{key!r} is not a question in this panel")
        if len(self.questions) == 1:
            raise ValueError("a panel needs at least one question")
        self.questions = {k: q for k, q in self.questions.items() if k != key}
        self._manifest["questions"] = {k: decide.to_dict(q) for k, q in self.questions.items()}
        self._agreement_fit = None
        if (self.dir / "seed_test.csv").exists():
            dev, test = self._seed_frames()
            self._save_seed(dev.drop(columns=[key], errors="ignore"),
                            test.drop(columns=[key], errors="ignore"))
        for name in ("pool_judged.csv", "votes_pool.csv"):
            path = self.dir / name
            if path.exists():
                frame = pd.read_csv(path)
                frame.drop(columns=[c for c in frame.columns if c.startswith(f"{key}__")]
                           ).to_csv(path, index=False)
        for name in ("needs_review.csv", "reviewed.csv"):
            path = self.dir / name
            if path.exists():
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                if "question" in frame.columns:
                    frame[frame["question"] != key].to_csv(path, index=False)
        agreement = self.dir / "agreement.json"
        if agreement.exists():
            blob = json.loads(agreement.read_text())
            if isinstance(blob, dict) and isinstance(blob.get("questions"), dict):
                blob["questions"].pop(key, None)
                agreement.write_text(json.dumps(blob, indent=2))
        self._warn_stale(f"removed {key!r}")
        self._record_stage("remove_question", 0.0, question=key)
        return True

    def merge_options(self, key, mapping):
        """Merge options of a choice question into others, e.g. `{"other": "none"}`."""
        question = self.questions.get(key)
        if question is None:
            raise ValueError(f"{key!r} is not a question in this panel")
        if question.kind != "choice":
            raise ValueError("only a choice question's options can be merged")
        mapping = {str(source): str(target) for source, target in mapping.items()}
        options = question.options()
        for source, target in mapping.items():
            if source not in options:
                raise ValueError(f"{source!r} is not an option of {key!r}")
            if target not in options or target in mapping:
                raise ValueError(f"{source!r} must merge into an option that stays, not {target!r}")
        if not mapping:
            return False
        if len(options) - len(mapping) < 2:
            raise ValueError("a choice question needs at least two options")
        merged = dataclasses.replace(question, criteria={
            o: d for o, d in question.criteria.items() if o not in mapping})
        self.questions = {**self.questions, key: merged}
        self._manifest["questions"] = {k: decide.to_dict(q) for k, q in self.questions.items()}
        self._agreement_fit = None
        rename = lambda v: mapping.get(v, v)  # noqa: E731
        if (self.dir / "seed_test.csv").exists():
            dev, test = self._seed_frames()
            for frame in (dev, test):
                if key in frame.columns:
                    frame[key] = frame[key].map(lambda v: rename(v) if isinstance(v, str) else v)
            self._save_seed(dev, test)
        for name in ("pool_judged.csv", "votes_pool.csv"):
            path = self.dir / name
            if path.exists():
                frame = pd.read_csv(path)
                for source, target in mapping.items():
                    src, dst = f"{key}__{source}", f"{key}__{target}"
                    if src in frame.columns and dst in frame.columns:
                        frame[dst] = frame[dst] + frame[src]
                        frame = frame.drop(columns=src)
                frame.to_csv(path, index=False)
        for name, columns in (("needs_review.csv", ("teacher_answer", "human_answer")),
                              ("reviewed.csv", ("answer",))):
            path = self.dir / name
            if path.exists():
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                if "question" in frame.columns:
                    rows = frame["question"] == key
                    for column in columns:
                        if column in frame.columns:
                            frame.loc[rows, column] = frame.loc[rows, column].map(rename)
                    frame.to_csv(path, index=False)
        self._warn_stale(f"merged on {key!r}")
        self._record_stage("merge_options", 0.0, question=key, mapping=mapping)
        return True

    def _warn_stale(self, what):
        if (self.dir / "student").exists() or (self.dir / "candidates").exists():
            warnings.warn(f"trained models still reflect the options you {what}; compare() or "
                          "distill() again to retrain them", stacklevel=3)

    def add_examples(self, df, test_frac=None):
        """Add hand-answered documents to the existing split. Each one goes to the test set with
        probability `test_frac`, decided by its text. Returns the (dev, test) counts added.
        """
        from shrewd.project import _hash_unit, _last_test_frac

        dev, test = self._seed_frames()
        frame = df.copy()
        frame["text"] = frame["text"].astype(str)
        frame = frame[frame["text"].str.strip() != ""].drop_duplicates("text")
        frame = frame[~frame["text"].isin(set(dev["text"]) | set(test["text"]))]
        for key in self.questions:
            if key not in frame.columns:
                frame[key] = ""
        frame = frame[["text", *self.questions]]
        frac = test_frac if test_frac is not None else _last_test_frac(self._manifest)
        to_test = np.array([_hash_unit(t, self.seed) < frac for t in frame["text"]], dtype=bool)
        dev = pd.concat([dev, frame[~to_test]], ignore_index=True)
        test = pd.concat([test, frame[to_test]], ignore_index=True)
        self._save_seed(dev, test)
        added = (int((~to_test).sum()), int(to_test.sum()))
        self._record_stage("add_examples", 0.0, dev=added[0], test=added[1])
        print(f"added {added[0]} tuning and {added[1]} test documents")
        return added

    def fix_example(self, text, answers):
        """Change the hand answers on one example. `answers` maps question ids to answers
        ("" clears one).
        """
        for key, value in answers.items():
            if key not in self.questions:
                raise ValueError(f"{key!r} is not a question in this panel")
            if str(value).strip() and gold_index(self.questions[key], value) is None:
                raise ValueError(f"{key}: {value!r} is not one of "
                                 f"{self.questions[key].options()}")
        dev, test = self._seed_frames()
        for frame, split in ((dev, "dev"), (test, "test")):
            hit = frame["text"] == text
            if hit.any():
                for key, value in answers.items():
                    if key not in frame.columns:
                        frame[key] = ""
                    frame.loc[hit, key] = str(value)
                self._save_seed(dev, test)
                self._record_stage("fix_example", 0.0, split=split, questions=sorted(answers))
                return split
        raise ValueError(f"{text[:40]!r} is not one of the seed documents")

    def remove_examples(self, texts):
        """Remove documents from your examples, in either split. Returns the (dev, test) counts
        removed.
        """
        texts = {texts} if isinstance(texts, str) else {str(t) for t in texts}
        dev, test = self._seed_frames()
        gone = (int(dev["text"].isin(texts).sum()), int(test["text"].isin(texts).sum()))
        if not any(gone):
            return gone
        self._save_seed(dev[~dev["text"].isin(texts)], test[~test["text"].isin(texts)])
        self._record_stage("remove_examples", 0.0, dev=gone[0], test=gone[1])
        if gone[1]:
            warnings.warn(f"{gone[1]} test example(s) removed; test scores from before aren't "
                          "comparable with the next distill()'s", stacklevel=2)
        return gone

    def _seed_frames(self):
        if not (self.dir / "seed_test.csv").exists():
            raise FileNotFoundError("no seed data; call add_seed() first")
        frames = []
        for name in ("seed_dev.csv", "seed_test.csv"):
            frame = pd.read_csv(self.dir / name, dtype=str, keep_default_na=False)
            frames.append(frame)
        return tuple(frames)

    def _save_seed(self, dev, test):
        dev.to_csv(self.dir / "seed_dev.csv", index=False)
        test.to_csv(self.dir / "seed_test.csv", index=False)
        self._manifest["splits"] = {**self._manifest.get("splits", {}),
                                    "n_dev": len(dev), "n_test": len(test)}
        self._agreement_fit = None
        self._save_manifest()

    def judge(self, df, dry_run=False, concurrency=8, overwrite=False, escalate=None,
              escalate_delta=0.05):
        """Have the teacher answer every question about every pooled document.

        One call per document covers the whole panel, which is where the saving is: the
        document is the expensive part of the prompt and asking N questions separately
        pays for it N times.

        With several teachers, `escalate=0.05` asks the first teacher about every document and
        the others only where its answer might be overruled, so that at most 5% of answers differ
        from the full panel's, with probability 1 - `escalate_delta`.
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
            n, cost = 0, 0.0
            to_price = texts
            if self._several() and (self.dir / "seed_dev.csv").exists():
                to_price = texts + pd.read_csv(self.dir / "seed_dev.csv")["text"].astype(
                    str).tolist()
            for model in self._teachers():
                calls, price = estimate_calls(to_price, prompt, model, 1, self._cache(),
                                              cached_prompt=True)
                n += calls
                cost = None if cost is None or price is None else cost + price
            per = f"${cost:.2f}" if cost is not None else "unknown"
            teachers = f" × {len(self._teachers())} teachers (and the dev split)" \
                if self._several() else ""
            print(f"dry run: {n} uncached calls for {len(texts)} documents{teachers} × "
                  f"{len(self.questions)} questions, about {per}")
            return None
        if self._several():
            return self._judge_several(texts, out_path, escalate, escalate_delta)
        if escalate is not None:
            raise ValueError("escalate needs several teachers: teacher=[cheap, strong, ...]")
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

    def _first_scores(self, frame, texts):
        from shrewd.route import doc_scores

        by_text = frame.drop_duplicates("text").set_index("text").reindex(texts)
        return doc_scores({
            key: np.nan_to_num(by_text[[f"{key}__{o}" for o in q.options()]].to_numpy(
                dtype=float), nan=0.0)
            for key, q in self.questions.items()
        })

    def _escalation_cutoff(self, agreement, budget, delta):
        from shrewd.route import doc_disagree, pick_threshold

        frames, texts, _ = agreement._dev
        first = agreement.cross_fit_dev(only=[0])
        full = agreement.cross_fit_dev()
        alone, panel = {}, {}
        for key, question in self.questions.items():
            columns = [f"{key}__{o}" for o in question.options()]
            alone[key] = np.nan_to_num(first[columns].to_numpy(dtype=float), nan=0.0)
            panel[key] = np.nan_to_num(full[columns].to_numpy(dtype=float), nan=0.0)
        return pick_threshold(self._first_scores(frames[0], texts),
                              doc_disagree(alone, panel), budget=budget, delta=delta)

    def _cascade_preview(self, agreement, budget=0.05, delta=0.05):
        frames, texts, _ = agreement._dev
        threshold, stats = self._escalation_cutoff(agreement, budget, delta)
        scores = self._first_scores(frames[0], texts)
        return {"first": self._teachers()[0], "budget": budget, "delta": delta,
                "threshold": None if threshold == float("inf") else round(float(threshold), 6),
                "escalate_share": round(float(np.mean(~(scores >= threshold))), 4),
                "n_dev": stats.get("n")}

    def _judge_several(self, texts, out_path, escalate=None, escalate_delta=0.05):
        from shrewd.agree import pool_status, short_name, write_votes
        from shrewd.judge import judge_texts

        agreement, cost = self._agreement()
        escalation = None
        if escalate is None:
            frames, pool_cost = self._judge_each(texts, "judging")
            cost += pool_cost
        else:
            threshold, stats = self._escalation_cutoff(agreement, escalate, escalate_delta)
            teachers = self._teachers()
            first, first_cost = judge_texts(
                texts, self.questions, teachers[0], instructions=self.instructions,
                backend=self.backend, conn=self._cache(),
                desc=f"judging ({short_name(teachers[0])})", header=self._header(),
            )
            cost += first_cost
            scores = self._first_scores(first, texts)
            hard = [t for t, s in zip(texts, scores, strict=True) if not s >= threshold]
            frames = [first]
            for model in teachers[1:]:
                frame, more = judge_texts(
                    hard, self.questions, model, instructions=self.instructions,
                    backend=self.backend, conn=self._cache(),
                    desc=f"escalated ({short_name(model)})", header=self._header(),
                )
                frames.append(frame)
                cost += more
            escalation = {"budget": escalate, "delta": escalate_delta,
                          "threshold": None if threshold == float("inf") else threshold,
                          "n_dev": stats.get("n"), "escalated": len(hard),
                          "documents": len(texts)}
            print(f"escalated {len(hard)} of {len(texts)} documents past "
                  f"{short_name(teachers[0])} (at most {escalate:.0%} keep an answer the full "
                  f"panel would change, {1 - escalate_delta:.0%} bound on "
                  f"{stats.get('n')} dev documents)")
        frame = self._reapply_reviewed(agreement.combine(frames, texts))
        frame.to_csv(out_path, index=False)
        write_votes(self.dir / "votes_pool.csv", self._teachers(), frames)
        summary = agreement.summary(frames, texts, floor=REVIEW_FLOOR)
        if escalation:
            summary["escalation"] = escalation
        summary["cascade"] = self._cascade_preview(agreement)
        (self.dir / "agreement.json").write_text(json.dumps(summary, indent=2))
        status = pool_status(agreement, frames, texts, floor=REVIEW_FLOOR)
        review = self._several_review_queue(frame, frames, status["cells"])
        review.to_csv(self.dir / "needs_review.csv", index=False)
        self._record_stage("judge", cost, n_documents=len(texts),
                           teachers=len(self._teachers()),
                           escalated=escalation["escalated"] if escalation else None)
        counts = status["summary"]
        print(f"judged {len(texts)} documents × {len(self.questions)} questions with "
              f"{len(self._teachers())} teachers for ${cost:.2f}")
        print(f"teachers agree on {counts['agree']} documents, split on {counts['split']}, "
              f"unsure on {counts['unsure']}, missing an answer on {counts['missing']}")
        for key, entry in summary["questions"].items():
            accuracy = entry["dev_accuracy"]
            shown = "  ".join(f"{short_name(m)} {accuracy[m]:.2f}" for m in self._teachers()
                              if m in accuracy)
            if "combined" in accuracy:
                shown += f"  combined {accuracy['combined']:.2f}"
            print(f"  {key}: dev accuracy {shown} ({entry['method']})")
        if len(review):
            print(f"{len(review)} (document, question) pairs to look at are in "
                  f"needs_review.csv, most useful first; fill in human_answer to overrule")
        return frame

    def _several_review_queue(self, frame, frames, cells, floor=REVIEW_FLOOR):
        from shrewd.agree import short_name

        ledger_path = self.dir / "reviewed.csv"
        done = set()
        if ledger_path.exists():
            ledger = pd.read_csv(ledger_path, dtype=str)
            done = set(zip(ledger["text"], ledger["question"], strict=True))
        names = [short_name(m) for m in self._teachers()]
        by_text = [f.drop_duplicates("text").set_index("text") for f in frames]
        rows, settled = [], []
        for _, row in frame.iterrows():
            text = row["text"]
            for key, question in self.questions.items():
                if (text, key) in done:
                    continue
                options = question.options()
                columns = [f"{key}__{o}" for o in options]
                probs = row[columns].to_numpy(dtype=float)
                said = []
                for name, table in zip(names, by_text, strict=True):
                    if text not in table.index:
                        continue
                    values = table.loc[text, columns].to_numpy(dtype=float)
                    if np.isnan(values).any():
                        said.append((name, None, None))
                    else:
                        top = int(values.argmax())
                        said.append((name, options[top], float(values[top])))
                teachers = "; ".join(f"{n}: {o} {p:.2f}" if o is not None else f"{n}: no answer"
                                     for n, o, p in said)
                state = cells.get((text, key), "agree")
                entry = {"text": text, "question": key, "teacher_answer": None,
                         "teacher_prob": None, "why": "", "teachers": teachers,
                         "human_answer": ""}
                if np.isnan(probs).any():
                    entry["why"] = "no usable answer"
                    rows.append(entry)
                    continue
                top = int(probs.argmax())
                entry["teacher_answer"] = options[top]
                entry["teacher_prob"] = round(float(probs[top]), 3)
                if state == "split" and probs[top] < SPLIT_SETTLED:
                    entry["why"] = "teachers split"
                elif state == "missing":
                    entry["why"] = "no answer from " + ", ".join(
                        n for n, o, _ in said if o is None)
                elif probs[top] < floor:
                    runner = int(np.argsort(probs)[-2])
                    entry["why"] = (f"unsure: {options[top]} {probs[top]:.2f} vs "
                                    f"{options[runner]} {probs[runner]:.2f}")
                else:
                    entry["why"] = "spot check: settled"
                    settled.append(entry)
                    continue
                rows.append(entry)
        columns = ["text", "question", "teacher_answer", "teacher_prob", "why", "teachers",
                   "human_answer"]
        queue = pd.DataFrame(rows, columns=columns)
        queue = queue.sort_values("teacher_prob", na_position="first").reset_index(drop=True)
        n_audit = min(len(settled), int(round(AUDIT_SHARE * len(queue))))
        if n_audit:
            rng = np.random.default_rng(self.seed)
            picks = [settled[i] for i in sorted(rng.choice(len(settled), n_audit, replace=False))]
            out, every = [], max(1, len(queue) // n_audit)
            for i, record in enumerate(queue.to_dict("records")):
                out.append(record)
                if (i + 1) % every == 0 and picks:
                    out.append(picks.pop(0))
            out += picks
            queue = pd.DataFrame(out, columns=columns)
        return queue

    def compare(self, features=("tfidf", "embed"), soft=True):
        """Train one student per featurization on the judged pool and score each on the dev split.
        The test set isn't used. Each is saved under candidates/<name>/ and listed in
        compare.json. `features` takes the same values as distill().
        """
        pool = pd.read_csv(self.dir / "pool_judged.csv")
        dev = pd.read_csv(self.dir / "seed_dev.csv")
        for frame in (pool, dev):
            frame["text"] = frame["text"].astype(str)
        pool_targets = self._targets_from_pool(pool)
        dev_gold = gold_frame(dev, self.questions)
        dev_texts = dev["text"].tolist()

        rows = []
        for entry in features:
            student = DecisionStudent(self.questions, features=entry, seed=self.seed, soft=soft,
                                      instructions=self.instructions)
            base = name = student.features_kind
            n = 2
            while any(r["features"] == name for r in rows):
                name = f"{base}-{n}"
                n += 1
            print(f"training candidate {name} on {len(pool)} judged documents")
            student.fit(pool["text"].tolist(), pool_targets)
            candidate_dir = self.dir / "candidates" / name
            shutil.rmtree(candidate_dir, ignore_errors=True)
            student.save(candidate_dir)
            student.predict_proba(dev_texts[:1], calibrated=False)
            started = time.perf_counter()
            proba = student.predict_proba(dev_texts, calibrated=False)
            ms_per_item = (time.perf_counter() - started) * 1000 / max(1, len(dev_texts))
            per_question = {}
            for j, key in enumerate(self.questions):
                answered = np.where(dev_gold[:, j] >= 0)[0]
                if key in proba and len(answered):
                    hits = proba[key][answered].argmax(axis=1) == dev_gold[answered, j]
                    per_question[key] = {"dev_accuracy": round(float(hits.mean()), 3),
                                         "n": int(len(answered))}
            size = sum(f.stat().st_size for f in candidate_dir.rglob("*") if f.is_file())
            rows.append({"features": name, "questions": per_question,
                         "ms_per_item": round(ms_per_item, 2), "size_mb": round(size / 1e6, 1)})

        width = max(len(r["features"]) for r in rows) + 2
        keys = list(self.questions)
        print(f"\n{'candidate':{width}}" + "".join(f"  {k[:10]:>10}" for k in keys) + "  ms/item")
        for r in rows:
            scores = [r["questions"].get(k, {}).get("dev_accuracy", float("nan")) for k in keys]
            cells = "".join(f"  {v:>10.3f}" for v in scores)
            print(f"{r['features']:{width}}{cells}  {r['ms_per_item']:>7.2f}")
        (self.dir / "compare.json").write_text(json.dumps({"candidates": rows}, indent=2))
        self._record_stage("compare", 0.0, features=[r["features"] for r in rows], soft=soft)
        return rows

    def distill(self, features="auto", soft=True, calibration="auto",
                calibrate_teacher=False, zero_shot=False, stack_margin=STACK_MARGIN,
                route_budget=None, route_delta=0.05, route_frac=0.3):
        """Train the heads on the judged pool, calibrate them, and score against gold.

        `zero_shot=True` (or a `ZeroShotScorer`) stacks an NLI model onto each head and keeps
        it per question only if it beats the plain head on the gold dev split by
        `stack_margin`. Needs `shrewd[encoder]` and costs a transformer pass per (document, option).

        `calibrate_teacher=True` first calibrates the teacher's answers on the dev split and
        rescales the pool answers through it before training. Useful for rare questions where
        the teacher over-states. The locked test set is never used for this.

        `calibration="auto-full"` lets questions with more than two options use a map that
        moves every option (vector or matrix scaling), not only the top score. It fit the
        teacher better on every multi-option question of the pre-built panels, but it can
        change which option wins, so it's opt-in. `"auto"` is unchanged.

        `route_budget=0.02` holds `route_frac` of the judged pool out of training and sets a
        routing threshold on it with the saved model: with probability 1 - `route_delta`, at
        most 2% of requests like the pool are answered locally and disagree with the teacher.
        Then `decide(texts, route=True)` or `decide(texts, fallback=d.ask)` uses it. The
        student trains on less, and a few hundred held-out documents certify little, so
        this pays off from about a thousand judged documents. The report also checks the
        locally answered test documents against the human labels.
        """
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
        route_pool = None
        if route_budget is not None:
            # the threshold must be set on documents the saved model never saw: out-of-fold
            # scores come from other models, and in testing they broke the budget far more
            # often than delta allows
            order = np.random.default_rng(self.seed).permutation(len(pool))
            held = np.sort(order[: int(round(route_frac * len(pool)))])
            kept = np.sort(order[int(round(route_frac * len(pool))):])
            route_pool = (pool["text"].iloc[held].tolist(),
                          {key: v[held] for key, v in pool_targets.items()})
            pool = pool.iloc[kept].reset_index(drop=True)
            pool_targets = {key: v[kept] for key, v in pool_targets.items()}
            print(f"holding {len(held)} judged documents out of training to set the routing "
                  f"threshold (budget {route_budget:.1%}, {1 - route_delta:.0%} guarantee)")
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
        if route_pool is not None:
            student.set_routing(*route_pool, budget=route_budget, delta=route_delta)
        shutil.rmtree(self.dir / "student", ignore_errors=True)
        student.save(self.dir / "student")

        after = self._score_all(student, test, calibrated=True)
        teacher_frame, frames, cost = self._judge(test["text"].tolist(), "teacher on test")
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
        if frames is not None:
            metrics["teachers"] = {
                model: self._score_teacher(frame, test)
                for model, frame in zip(self._teachers(), frames, strict=True)
            }
        if student.stack_report:
            metrics["stacking"] = student.stack_report
        if student.routing:
            metrics["routing"] = self._routing_check(student, test, teacher_frame)
        findings = decision_findings(metrics)
        (self.dir / "report.json").write_text(json.dumps(metrics, indent=2))
        self._record_stage("distill", cost + teacher_cost, features=student.features_kind,
                           soft=soft, calibrate_teacher=calibrate_teacher,
                           zero_shot=bool(zero_shot))
        return DecisionResult(metrics, findings)

    def _routing_check(self, student, test, teacher_frame):
        """The threshold's guarantee, plus what it means against human labels: on the locked
        test set, the error of the answers the student keeps and of the teacher on the same
        documents, pooled over every answered question."""
        routing = dict(student.routing)
        gold = gold_frame(test, self.questions)
        local = student.local(test["text"].tolist())
        proba = student.predict_proba(test["text"].tolist())
        teacher = teacher_frame.set_index("text").reindex(test["text"])
        wrong = {"student": 0, "teacher": 0}
        answered = 0
        for j, (key, question) in enumerate(self.questions.items()):
            if key not in proba:
                continue
            columns = [f"{key}__{o}" for o in question.options()]
            values = teacher[columns].to_numpy(dtype=float)
            usable = local & (gold[:, j] >= 0) & ~np.isnan(values).any(axis=1)
            answered += int(usable.sum())
            wrong["student"] += int((proba[key].argmax(axis=1) != gold[:, j])[usable].sum())
            wrong["teacher"] += int((np.nan_to_num(values).argmax(axis=1)
                                     != gold[:, j])[usable].sum())
        routing["test"] = {
            "coverage": round(float(local.mean()), 4) if len(local) else 0.0,
            "n_local": int(local.sum()),
            "answers_checked": answered,
            "student_error": round(wrong["student"] / answered, 4) if answered else None,
            "teacher_error": round(wrong["teacher"] / answered, 4) if answered else None,
        }
        return routing

    def ask(self, texts):
        """The teacher's answers, as answer maps in the same shape as `decide()` gives. Pass it
        as `decide(texts, fallback=d.ask)` to send what the student is unsure of to the
        teacher. Answers are cached like `judge()`'s. With several teachers, it's their
        combined answer.
        """
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        frame, _, _ = self._judge(texts, "teacher fallback")
        frame = frame.drop_duplicates("text").set_index("text").reindex(texts)
        rows = []
        for i in range(len(texts)):
            row = {}
            for key, question in self.questions.items():
                values = frame.iloc[i][[f"{key}__{o}" for o in question.options()]]
                values = values.to_numpy(dtype=float)
                if np.isnan(values).any():
                    continue            # the teacher's answer could not be read
                row[key] = decide.make_answer(key, question, values / values.sum())
            rows.append(row)
        return rows[0] if single else rows

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
        if self._several():
            agreement, cost = self._agreement()
            frame = agreement.cross_fit_dev()
        elif path.exists():
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
        return self._apply_answers([
            {"text": row["text"], "question": row["question"], "answer": row["human_answer"]}
            for _, row in fixes.iterrows()
        ])

    def review(self, answers):
        """Record your answers on pool documents, as filling in needs_review.csv and calling
        apply_review() would.
        """
        if not (self.dir / "pool_judged.csv").exists():
            raise FileNotFoundError("no pool_judged.csv; run judge() on your pool first")
        records = answers.to_dict("records") if isinstance(answers, pd.DataFrame) \
            else list(answers)
        applied = self._apply_answers(records)
        path, ledger_path = self.dir / "needs_review.csv", self.dir / "reviewed.csv"
        if applied and path.exists() and ledger_path.exists():
            queue = pd.read_csv(path, dtype=str, keep_default_na=False)
            ledger = pd.read_csv(ledger_path, dtype=str)
            done = set(zip(ledger["text"], ledger["question"], strict=True))
            keep = [(t, q) not in done for t, q in zip(queue["text"], queue["question"],
                                                        strict=True)]
            queue[keep].to_csv(path, index=False)
        return applied

    def _apply_answers(self, records):
        pool_path = self.dir / "pool_judged.csv"
        pool = pd.read_csv(pool_path)
        pool["text"] = pool["text"].astype(str)
        index = {t: i for i, t in enumerate(pool["text"])}
        ledgered = set()
        if (self.dir / "reviewed.csv").exists():
            old = pd.read_csv(self.dir / "reviewed.csv", dtype=str)
            ledgered = set(zip(old["text"], old["question"], strict=True))
        before = data.read_before(self.dir)
        applied, rejected = [], []
        for row in records:
            key, answer, text = row["question"], str(row["answer"]).strip(), str(row["text"])
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
            if (text, key) not in ledgered and key not in before.get(text, {}):
                before.setdefault(text, {})[key] = [
                    _float_or_none(pool.at[index[text], f"{key}__{o}"])
                    for o in question.options()]
            for j, option in enumerate(question.options()):
                pool.loc[index[text], f"{key}__{option}"] = 1.0 if j == g else 0.0
            applied.append({"text": text, "question": key, "answer": question.options()[g]})
        if applied:
            pool.to_csv(pool_path, index=False)
            data.write_before(self.dir, before)
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
        before = data.read_before(self.dir)
        n = 0
        for _, row in ledger.iterrows():
            question = self.questions.get(row["question"])
            if question is None or row["text"] not in index:
                continue
            g = gold_index(question, row["answer"])
            if g is None:
                continue
            i = index[row["text"]]
            before.setdefault(row["text"], {})[row["question"]] = [
                _float_or_none(frame.at[i, f"{row['question']}__{o}"]) for o in question.options()]
            for j, option in enumerate(question.options()):
                frame.loc[i, f"{row['question']}__{option}"] = 1.0 if j == g else 0.0
            n += 1
        if n:
            data.write_before(self.dir, before)
            print(f"re-applied {n} human answers from reviewed.csv")
        return frame

    def unreview(self, texts, questions=None):
        """Undo review() on pool documents, for every question or only `questions`. The teachers'
        answers come back.
        """
        texts = [texts] if isinstance(texts, str) else [str(t) for t in texts]
        ledger_path, pool_path = self.dir / "reviewed.csv", self.dir / "pool_judged.csv"
        if not ledger_path.exists() or not pool_path.exists():
            return 0
        ledger = pd.read_csv(ledger_path, dtype=str)
        wanted = set(texts)
        hit = ledger["text"].isin(wanted) & (
            ledger["question"].isin(list(questions)) if questions is not None else True)
        back = [(t, q) for t, q in zip(ledger.loc[hit, "text"], ledger.loc[hit, "question"],
                                       strict=True) if q in self.questions]
        if not back:
            return 0
        before = data.read_before(self.dir)
        values = {}
        for text, key in back:
            saved = before.get(text, {}).get(key)
            if saved is not None and len(saved) == len(self.questions[key].options()):
                values[(text, key)] = saved
        missing = list(dict.fromkeys(t for t, q in back if (t, q) not in values))
        cost = 0.0
        if missing:
            frame, _, cost = self._judge(missing, "teachers' answers")
            frame = frame.drop_duplicates("text").set_index("text")
            for text, key in back:
                if (text, key) not in values:
                    values[(text, key)] = [_float_or_none(frame.at[text, f"{key}__{o}"])
                                           for o in self.questions[key].options()]
        pool = pd.read_csv(pool_path)
        pool["text"] = pool["text"].astype(str)
        index = {t: i for i, t in enumerate(pool["text"])}
        for (text, key), probs in values.items():
            if text in index:
                for option, p in zip(self.questions[key].options(), probs, strict=True):
                    pool.loc[index[text], f"{key}__{option}"] = np.nan if p is None else p
        pool.to_csv(pool_path, index=False)
        rest = ledger[~hit]
        if rest.empty:
            ledger_path.unlink()
        else:
            rest.to_csv(ledger_path, index=False)
        for text, key in back:
            before.get(text, {}).pop(key, None)
        data.write_before(self.dir, {t: v for t, v in before.items() if v})
        self._record_stage("unreview", cost, items=len(back))
        return len(back)

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
