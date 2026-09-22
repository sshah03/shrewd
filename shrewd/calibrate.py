"""Post-hoc calibration of a student's probabilities.

With more than two options the calibration map runs on the top score, so the predicted
option never changes. For yes/no questions the calibrated quantity is P(yes) itself and it
may cross 0.5: a question with a 3% base rate should not sit at 0.5.
See `cross_fit_proba` for which rows the calibrator is fit on.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-12
METHODS = ("temperature", "isotonic", "platt", "none")


def _logits(proba):
    return np.log(np.clip(np.asarray(proba, dtype=float), EPS, 1.0))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y_idx, k):
    out = np.zeros((len(y_idx), k))
    out[np.arange(len(y_idx)), y_idx] = 1.0
    return out


def encode(labels, classes):
    """Label strings to column indices. Labels outside `classes` come back as -1."""
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index.get(str(y), -1) for y in labels])


# ---------------------------------------------------------------- scoring rules


def nll(proba, y_idx):
    """Mean negative log likelihood of the true class (log loss)."""
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log(proba[np.arange(len(y_idx)), y_idx])))


def brier(proba, y_idx):
    """Multiclass Brier score: mean squared error over the whole probability vector.

    Ranges 0 (perfect) to 2 (confidently wrong). Unlike ECE this is a proper scoring
    rule, so it can't be gamed by reporting the base rate on every row.
    """
    proba = np.asarray(proba, dtype=float)
    return float(np.mean(((proba - _onehot(y_idx, proba.shape[1])) ** 2).sum(axis=1)))


def ece(confidence, correct, bins=10):
    """Expected calibration error with equal-mass bins, so the number does not swing with the
    bin count when most rows sit in one confidence band.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    order = np.argsort(confidence, kind="mergesort")
    conf, corr = confidence[order], correct[order]
    total = 0.0
    for chunk in np.array_split(np.arange(len(conf)), min(bins, len(conf))):
        if len(chunk):
            total += len(chunk) / len(conf) * abs(conf[chunk].mean() - corr[chunk].mean())
    return float(total)


def reliability_table(confidence, correct, bins=10):
    """Per-bin stated confidence vs. measured accuracy: the reliability diagram as rows."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(confidence, kind="mergesort")
    conf, corr = confidence[order], correct[order]
    rows = []
    for chunk in np.array_split(np.arange(len(conf)), min(bins, max(len(conf), 1))):
        if not len(chunk):
            continue
        rows.append(
            {
                "low": round(float(conf[chunk].min()), 4),
                "high": round(float(conf[chunk].max()), 4),
                "n": int(len(chunk)),
                "stated": round(float(conf[chunk].mean()), 4),
                "actual": round(float(corr[chunk].mean()), 4),
            }
        )
    return rows


def aurc(confidence, correct):
    """Area under the risk-coverage curve: mean error rate over every prefix of rows
    sorted most-confident-first. Lower is better. It rewards a confidence signal that
    puts the mistakes at the bottom, independently of whether the scale is calibrated.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    order = np.argsort(-confidence, kind="mergesort")
    errors = 1.0 - correct[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def calibration_metrics(proba, labels, classes, bins=10):
    """Every calibration number for one set of predictions, as a plain dict."""
    proba = np.asarray(proba, dtype=float)
    y_idx = encode(labels, classes)
    keep = y_idx >= 0
    proba, y_idx = proba[keep], y_idx[keep]
    if len(y_idx) == 0:
        return {}
    # for a yes/no question the reliability curve that matters runs on P(yes) across
    # every row, not on the winning side's confidence: with a 3% base rate the latter
    # is dominated by easy "no"s and looks excellent while P(yes) is off by 10x
    if len(classes) == 2:
        confidence, correct = proba[:, 1], (y_idx == 1).astype(float)
    else:
        confidence = proba.max(axis=1)
        correct = (proba.argmax(axis=1) == y_idx).astype(float)
    accuracy = float((proba.argmax(axis=1) == y_idx).mean())
    return {
        "n": int(len(y_idx)),
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "overconfidence": round(float(confidence.mean() - correct.mean()), 4),
        "ece": round(ece(confidence, correct, bins), 4),
        "brier": round(brier(proba, y_idx), 4),
        "nll": round(nll(proba, y_idx), 4),
        "aurc": round(aurc(confidence, correct), 4),
        "reliability": reliability_table(confidence, correct, bins),
    }


# ---------------------------------------------------------------- the calibrator


class Calibrator:
    """A fitted, monotone rescaling of a student's probabilities. Serializes to JSON.

    `temperature` (one scalar), `platt` (two parameters on the log-odds, the right choice
    for a lopsided yes/no question) or `isotonic` (a step function, needs thousands of rows
    and collapses on a few dozen).
    """

    def __init__(self, method, params, n_fit=0, classes=None):
        self.method = method
        self.params = params
        self.n_fit = n_fit
        self.classes = classes

    # -- fitting ---------------------------------------------------------------

    @classmethod
    def fit(cls, proba, labels, classes, method="auto", seed=0):
        """Fit on (probabilities, true labels). `method="auto"` picks by cross-validated
        log loss among temperature/platt/isotonic/none, which keeps a small or degenerate
        calibration set from choosing a flexible method it cannot support."""
        proba = np.asarray(proba, dtype=float)
        y_idx = encode(labels, classes)
        keep = y_idx >= 0
        proba, y_idx = proba[keep], y_idx[keep]
        if len(y_idx) < 20 or len(np.unique(y_idx)) < 2:
            return cls("none", {}, n_fit=int(len(y_idx)), classes=list(classes))
        if method == "auto":
            method = cls._pick(proba, y_idx, classes, seed)
        params = cls._fit_params(method, proba, y_idx, len(classes))
        return cls(method, params, n_fit=int(len(y_idx)), classes=list(classes))

    @staticmethod
    def _fit_params(method, proba, y_idx, k):
        """Fit one method's parameters. Multi-class maps the top score and keeps the argmax;
        two-class maps P(positive) directly and may cross 0.5.
        """
        if method == "none":
            return {}
        if method == "temperature":
            return {"temperature": _fit_temperature(_logits(proba), _onehot(y_idx, k))}
        binary = k == 2
        if binary:
            score, target = proba[:, 1], (y_idx == 1).astype(float)
        else:
            score, target = proba.max(axis=1), (proba.argmax(axis=1) == y_idx).astype(float)
        if method == "platt":
            a, b = _fit_platt(score, target)
            return {"a": a, "b": b, "binary": binary}
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(score, target)
            return {
                "x": [round(float(v), 6) for v in iso.X_thresholds_],
                "y": [round(float(v), 6) for v in iso.y_thresholds_],
                "binary": binary,
            }
        raise ValueError(f"unknown calibration method: {method!r} (expected one of {METHODS})")

    @classmethod
    def _pick(cls, proba, y_idx, classes, seed, n_splits=4):
        """Cross-validated log loss over the calibration rows themselves."""
        n_splits = max(2, min(n_splits, int(np.bincount(y_idx).min()) if len(y_idx) else 2))
        try:
            folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(
                proba, y_idx
            ))
        except ValueError:
            return "temperature"
        scores = {}
        for method in METHODS:
            total, ok = 0.0, True
            for train, test in folds:
                try:
                    params = cls._fit_params(method, proba[train], y_idx[train], len(classes))
                    held = cls(method, params, classes=list(classes)).transform(proba[test])
                except (ValueError, FloatingPointError):
                    ok = False
                    break
                total += nll(held, y_idx[test]) * len(test)
            if ok:
                scores[method] = total / len(y_idx)
        if not scores:
            return "temperature"
        best = min(scores, key=scores.get)
        # a tie goes to the simpler method: temperature generalizes off-distribution
        # better than isotonic and there is no reason to pay for flexibility you don't use
        for method in METHODS:
            if method in scores and scores[method] <= scores[best] + 1e-4:
                return method
        return best

    # -- applying --------------------------------------------------------------

    def transform(self, proba):
        """Rescale probabilities. Row-stochastic in, row-stochastic out."""
        proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
        if self.method == "none":
            return proba / proba.sum(axis=1, keepdims=True)
        if self.method == "temperature":
            return _softmax(_logits(proba) / self.params["temperature"])
        binary = bool(self.params.get("binary")) and proba.shape[1] == 2
        score = proba[:, 1] if binary else proba.max(axis=1)
        if self.method == "platt":
            mapped = _apply_platt(score, self.params["a"], self.params["b"])
        elif self.method == "isotonic":
            mapped = np.interp(score, self.params["x"], self.params["y"])
        else:
            raise ValueError(f"unknown calibration method: {self.method!r}")
        mapped = np.clip(mapped, EPS, 1.0 - EPS)
        if binary:
            return np.column_stack([1.0 - mapped, mapped])
        return _rescale_top(proba, mapped)

    # -- persistence -----------------------------------------------------------

    def to_dict(self):
        return {
            "method": self.method,
            "params": self.params,
            "n_fit": self.n_fit,
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, blob):
        return cls(
            blob["method"], blob.get("params", {}), blob.get("n_fit", 0), blob.get("classes")
        )

    def __repr__(self):
        detail = ""
        if self.method == "temperature":
            detail = f" T={self.params['temperature']:.3f}"
        return f"<Calibrator {self.method}{detail} fit on {self.n_fit} rows>"


def _rescale_top(proba, new_top):
    """Set the winning class to `new_top` and spread the rest proportionally. `new_top` is
    floored where the runner-up would draw level, so the winning option never changes.
    """
    out = np.array(proba, dtype=float, copy=True)
    rows = np.arange(len(out))
    win = out.argmax(axis=1)
    top = out[rows, win]
    rest = out.sum(axis=1) - top
    other_max = np.where(out.shape[1] > 1, (out - np.eye(out.shape[1])[win] * out).max(axis=1), 0.0)
    floor = np.where(rest > EPS, other_max / np.maximum(rest + other_max, EPS), 0.0)
    new_top = np.maximum(new_top, np.minimum(floor + 1e-9, 1.0 - EPS))
    scale = np.where(rest > EPS, (1.0 - new_top) / np.maximum(rest, EPS), 0.0)
    out *= scale[:, None]
    out[rows, win] = new_top
    total = out.sum(axis=1, keepdims=True)
    return out / np.maximum(total, EPS)


def _fit_temperature(logits, onehot, bounds=(-3.0, 3.0)):
    """The temperature minimizing log loss. Optimized over log T so it stays positive."""

    def objective(log_t):
        z = logits / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -np.mean((onehot * log_p).sum(axis=1))

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return round(float(np.exp(result.x)), 6)


def _fit_platt(confidence, correct):
    """Two-parameter logistic on the top score's log-odds.

    Falls back to the identity map when the calibration rows are all right or all
    wrong: there is no slope to estimate from one class, and a logistic fit would
    either refuse outright or run its coefficients off to infinity.
    """
    correct = correct.astype(int)
    if len(np.unique(correct)) < 2:
        return 1.0, 0.0
    x = np.log(np.clip(confidence, EPS, 1 - 1e-9) / np.clip(1 - confidence, EPS, 1.0))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(x.reshape(-1, 1), correct)
    return round(float(model.coef_[0][0]), 6), round(float(model.intercept_[0]), 6)


def _apply_platt(confidence, a, b):
    x = np.log(np.clip(confidence, EPS, 1 - 1e-9) / np.clip(1 - confidence, EPS, 1.0))
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


# ---------------------------------------------------------------- the right rows


def cross_fit_proba(make_student, texts, labels, classes, n_splits=5, seed=0, desc=None):
    """Out-of-fold probabilities over the teacher-labeled pool: K refits, each scoring the
    rows it did not train on. No API calls.

    Fitting on the student's own training rows gives a temperature near 1 (memorized
    scores). Fitting on the seed dev split (80-200 easy rows) made two of three datasets
    worse. The labels are the teacher's, so this calibrates to agreement with the teacher;
    the report says so when the teacher is the weak link.
    """
    texts, labels = list(texts), [str(y) for y in labels]
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index.get(label, -1) for label in labels])
    keep = np.where(y >= 0)[0]
    if len(keep) < 40:
        return None, None
    counts = np.bincount(y[keep], minlength=len(classes))
    n_splits = int(max(2, min(n_splits, counts[counts > 0].min())))
    if n_splits < 2:
        return None, None

    sub_texts = [texts[i] for i in keep]
    sub_labels = [labels[i] for i in keep]
    oof = np.zeros((len(keep), len(classes)))
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(keep)), y[keep]))
    for fold, (train, test) in enumerate(folds, 1):
        if desc:
            print(f"  {desc}: calibration fold {fold}/{len(folds)}", flush=True)
        student = make_student()
        student.fit([sub_texts[i] for i in train], [sub_labels[i] for i in train])
        proba = student.predict_proba([sub_texts[i] for i in test])
        for j, cls in enumerate(student.classes_):
            if cls in index:
                oof[test, index[cls]] = proba[:, j]
    total = oof.sum(axis=1, keepdims=True)
    oof = np.divide(oof, total, out=np.full_like(oof, 1.0 / len(classes)), where=total > EPS)
    return oof, sub_labels
