from dataclasses import dataclass

import numpy as np
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

_SEVERITY_ORDER = {"fail": 0, "warn": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    severity: str  # "fail" | "warn" | "info"
    title: str
    detail: str
    suggestion: str


@dataclass(frozen=True)
class DistillResult:
    metrics: dict
    findings: list[Finding]

    def report(self):
        """Plain-text rendering of metrics and findings."""
        m = self.metrics
        teacher, student = m["teacher"], m["student"]
        width = max(12, *(len(c) for c in m["classes"])) + 2
        lines = [
            f"shrewd report: {m['student_type']} student, {len(m['classes'])} labels",
            f"trained on {m['n_train']} rows · evaluated on the locked test set"
            f" ({m['n_test']} rows)",
            "",
            f"{'':{width}}  teacher   student",
            f"{'accuracy':{width}}  {teacher['accuracy']:>7.3f}   {student['accuracy']:>7.3f}",
            f"{'macro F1':{width}}  {teacher['macro_f1']:>7.3f}   {student['macro_f1']:>7.3f}",
        ]
        if "macro_f1_ci" in teacher:
            t_lo, t_hi = teacher["macro_f1_ci"]
            s_lo, s_hi = student["macro_f1_ci"]
            lines.append(
                f"{'95% interval':{width}}  {t_lo:.2f}-{t_hi:.2f}   {s_lo:.2f}-{s_hi:.2f}"
            )
        lines += [
            "",
            f"{'per-label F1':{width}}  teacher   student   test rows",
        ]
        for cls in m["classes"]:
            t, s = teacher["per_class"][cls], student["per_class"][cls]
            lines.append(
                f"{cls:{width}}  {t['f1']:>7.3f}   {s['f1']:>7.3f}   {t['support']:>9}"
            )
        if student.get("selective"):
            curve = student["selective"]
            label_width = max(width, len("min_confidence for that coverage"))
            lines += [
                "",
                f"{'student accuracy by coverage':{label_width}}"
                + "".join(f"  {row['coverage']:>4.0%} {row['accuracy']:.3f}" for row in curve),
                f"{'min_confidence for that coverage':{label_width}}"
                + "".join(f"{row['threshold']:>12.2f}" for row in curve),
            ]
        if m.get("teacher_invalid_on_test"):
            lines += [
                "",
                f"note: teacher gave unparseable output on {m['teacher_invalid_on_test']}"
                f"/{m['n_test']} test rows (counted as wrong)",
            ]
        if m.get("pool_rows_duplicating_seed"):
            lines += [
                "",
                f"note: {m['pool_rows_duplicating_seed']} pool rows duplicated seed texts and "
                "were kept out of training",
            ]
        lines.append("")
        if not self.findings:
            lines.append("findings: none, nothing looks off")
        else:
            lines.append("findings")
            for f in self.findings:
                pad = " " * 7
                lines.append(f"[{f.severity.upper():4}] {f.title}")
                lines.append(f"{pad}{f.detail}")
                lines.append(f"{pad}→ {f.suggestion}")
        return "\n".join(lines)


def compute_metrics(y_true, y_pred, classes):
    y_true, y_pred = list(y_true), list(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "macro_f1_ci": bootstrap_macro_f1(y_true, y_pred, classes),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)


def selective_curve(y_true, proba, classes, coverages=COVERAGES):
    """Accuracy of the most confident share of rows, at each coverage level.

    Confidence is the top predicted probability. The threshold for each coverage is read
    off the scored rows themselves, so the coverage column describes this sample rather
    than promising the same share on new data.
    """
    y_true = np.asarray(list(y_true))
    proba = np.asarray(proba)
    confidence = proba.max(axis=1)
    predicted = np.asarray(classes)[proba.argmax(axis=1)]
    rows = []
    for coverage in coverages:
        cutoff = np.quantile(confidence, 1 - coverage) if coverage < 1 else 0.0
        keep = confidence >= cutoff
        # the smallest confidence actually kept: predict(min_confidence=threshold)
        # reproduces exactly these rows
        threshold = float(confidence[keep].min()) if coverage < 1 else 0.0
        rows.append(
            {
                "coverage": round(float(keep.mean()), 3),
                "threshold": round(threshold, 3),
                "accuracy": round(float((predicted[keep] == y_true[keep]).mean()), 4),
            }
        )
    return rows


def _encode(y_true, y_pred, classes):
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    # predictions outside the label set (e.g. INVALID) land in an extra column that
    # counts as a false negative for the gold class and nothing else
    true = np.array([index[y] for y in y_true])
    pred = np.array([index.get(y, k) for y in y_pred])
    return true * (k + 1) + pred, k


def _macro_f1_from_codes(codes, k, weights=None):
    cm = np.bincount(codes, weights=weights, minlength=k * (k + 1)).reshape(k, k + 1)
    tp = np.diag(cm[:, :k])
    fp = cm[:, :k].sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return float(f1.mean())


def _resample_weights(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    return [np.bincount(d, minlength=n).astype(float) for d in draws]


def bootstrap_macro_f1(y_true, y_pred, classes, n_boot=2000, seed=0):
    """95% percentile-bootstrap interval for macro-F1, resampling test rows."""
    codes, k = _encode(y_true, y_pred, classes)
    weights = _resample_weights(len(codes), n_boot, seed)
    scores = [_macro_f1_from_codes(codes, k, w) for w in weights]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def paired_gap_interval(y_true, a_pred, b_pred, classes, n_boot=2000, seed=0):
    """95% interval for macro-F1(a) - macro-F1(b), both scored on the same resampled rows.

    Resampling the two models together is what makes the interval meaningful: the
    row-level noise they share cancels, so it is narrower than
    comparing two separate intervals.
    """
    a_codes, k = _encode(y_true, a_pred, classes)
    b_codes, _ = _encode(y_true, b_pred, classes)
    gaps = [
        _macro_f1_from_codes(a_codes, k, w) - _macro_f1_from_codes(b_codes, k, w)
        for w in _resample_weights(len(a_codes), n_boot, seed)
    ]
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _teacher_bottleneck(ctx):
    bar = ctx["target"] if ctx["target"] is not None else 0.75
    score = ctx["teacher"]["macro_f1"]
    if score >= bar:
        return None
    which = "your target" if ctx["target"] is not None else "the default bar"
    detail = f"teacher test macro-F1 is {score:.3f}, below {which} of {bar:.2f}."
    if (ctx["votes"] > 1 and len(ctx["pool_confidence"])
            and float(np.mean(ctx["pool_confidence"])) >= 0.9):
        detail += (
            " The teacher is highly self-consistent, so the target may exceed what this "
            "data's labels support."
        )
    return Finding(
        "fail",
        "teacher is the bottleneck",
        detail,
        "Raise optimize(budget=...), use a stronger teacher model, or refine the label "
        "descriptions. Distilling won't fix this.",
    )


def _student_gap(ctx):
    student, teacher = ctx["student"]["macro_f1"], ctx["teacher"]["macro_f1"]
    if student >= teacher - 0.05:
        return None
    gap_ci = ctx.get("gap_ci")
    if gap_ci and gap_ci[0] <= 0:
        return Finding(
            "info",
            "student trails the teacher, but within test-set noise",
            f"student macro-F1 is {student:.3f} vs teacher {teacher:.3f}; the 95% interval "
            f"for the gap is [{gap_ci[0]:.2f}, {gap_ci[1]:.2f}] and includes zero on "
            f"{len(ctx['test_examples'])} test rows.",
            "Score both on a larger labeled holdout before acting on this gap.",
        )
    stronger = [s for s in ("encoder", "setfit") if s != ctx["student_type"]]
    options = " or ".join(f'student="{s}"' for s in stronger)
    suggestion = f"Try {options}, or label more pool data." if stronger else "Label more pool data."
    return Finding(
        "fail" if student < 0.6 else "warn",
        "student can't match the teacher",
        f"student macro-F1 is {student:.3f} vs teacher {teacher:.3f}.",
        suggestion,
    )


def _selective_parity(ctx):
    curve = ctx.get("student_selective")
    if not curve:
        return None
    teacher_acc = ctx["teacher"]["accuracy"]
    if ctx["student"]["accuracy"] >= teacher_acc:
        return None
    for row in curve:  # highest coverage first
        if row["coverage"] < 1 and row["coverage"] >= 0.5 and row["accuracy"] >= teacher_acc:
            return Finding(
                "info",
                "confident items reach the teacher's overall accuracy",
                f"on the {row['coverage']:.0%} of test rows where the student's top "
                f"probability is at least {row['threshold']:.2f}, its accuracy is "
                f"{row['accuracy']:.3f}; the teacher scores {teacher_acc:.3f} on all rows. "
                "This does not establish parity on the same subset.",
                f"Candidate routing threshold: predict(texts, min_confidence="
                f"{row['threshold']:.2f}), deferring about {1 - row['coverage']:.0%}. "
                "This threshold was read off the test set. Choose a threshold on validation "
                "data, then compare both models on the accepted rows and evaluate the "
                "complete fallback system on a fresh holdout.",
            )
    return None


def _top_confused(confusion, classes, cls):
    row = list(confusion[classes.index(cls)])
    row[classes.index(cls)] = 0
    return classes[int(np.argmax(row))] if max(row) > 0 else None


def _ambiguous_labels(ctx):
    findings = []
    for cls in ctx["classes"]:
        t_f1 = ctx["teacher"]["per_class"][cls]["f1"]
        s_f1 = ctx["student"]["per_class"][cls]["f1"]
        if (
            t_f1 >= ctx["teacher"]["macro_f1"] - 0.15
            or s_f1 >= ctx["student"]["macro_f1"] - 0.15
        ):
            continue
        detail = f"both models score poorly on '{cls}' (teacher F1 {t_f1:.2f}, student {s_f1:.2f})"
        confused = _top_confused(ctx["teacher"]["confusion"], ctx["classes"], cls) or (
            _top_confused(ctx["student"]["confusion"], ctx["classes"], cls)
        )
        if confused:
            detail += f", most often confused with '{confused}'"
        missed = [
            e for e in ctx["test_examples"] if e["gold"] == cls and e["teacher"] != cls
        ][:3]
        if missed:
            quoted = "; ".join(f'"{e["text"][:120]}" → {e["teacher"]}' for e in missed)
            detail += f". Misread test examples: {quoted}"
        suggestion = f"Tighten the description of '{cls}'"
        if confused:
            suggestion += f" (and how it differs from '{confused}')"
        suggestion += ", or merge/split the labels, then re-run optimize(overwrite=True)."
        findings.append(
            Finding("warn", f"'{cls}' may be ambiguously defined", detail + ".", suggestion)
        )
    return findings


def _thin_classes(ctx):
    thin = {cls: ctx["train_counts"].get(cls, 0) for cls in ctx["classes"]}
    thin = {cls: n for cls, n in thin.items() if n < 20}
    if not thin:
        return None
    return Finding(
        "warn",
        "not enough training signal for some labels",
        "training rows after confidence filtering: "
        + ", ".join(f"{cls}={n}" for cls, n in thin.items())
        + ".",
        "Lower min_confidence, add pool data containing these labels, or add seed examples.",
    )


def _unsure_teacher(ctx):
    if ctx["votes"] < 2 or not len(ctx["pool_confidence"]):
        return None
    mean = float(np.mean(ctx["pool_confidence"]))
    if mean >= 0.7:
        return None
    return Finding(
        "warn",
        "teacher is unsure on the pool",
        f"mean vote agreement on the pool is {mean:.2f}; the pool may differ from the "
        "seed data.",
        "Inspect needs_review.csv; consider hand-labeling some pool rows and adding them "
        "to the seed.",
    )


def _distribution_shift(ctx):
    if ctx.get("active"):
        return None  # an actively selected pool is skewed toward hard rows by design
    classes = ctx["classes"]
    seed = np.array([ctx["seed_counts"].get(c, 0) for c in classes], dtype=float)
    pool = np.array([ctx["pool_counts"].get(c, 0) for c in classes], dtype=float)
    if pool.sum() == 0 or (seed == 0).any():
        return None
    seed_share, pool_share = seed / seed.sum(), pool / pool.sum()
    _, p_value = chisquare(pool, seed_share * pool.sum())
    ratio = pool_share / seed_share
    if p_value >= 0.01 or not ((ratio > 2) | (ratio < 0.5)).any():
        return None
    worst = sorted(range(len(classes)), key=lambda i: abs(np.log2(ratio[i] or 0.01)))[-3:]
    deltas = ", ".join(
        f"{classes[i]}: {seed_share[i]:.0%} of seed vs {pool_share[i]:.0%} of pool"
        for i in reversed(worst)
    )
    return Finding(
        "warn",
        "pool distribution differs from the seed data",
        f"label mix shift (chi-square p={p_value:.1g}). Biggest deltas: {deltas}.",
        "Test-set numbers may not reflect production. Add seed examples drawn from the "
        "same source as the pool.",
    )


def _invalid_responses(ctx):
    if not ctx["n_invalid"]:
        return None
    share = ctx["n_invalid"] / ctx["n_pool"]
    return Finding(
        "warn" if share > 0.05 else "info",
        "some pool responses were unparseable",
        f"{ctx['n_invalid']}/{ctx['n_pool']} pool items got no valid label even after a "
        "retry; they were excluded from training.",
        "Spot-check pool_labeled.csv rows with label __invalid__; a stricter output "
        "instruction in prompt.txt usually fixes this.",
    )


_RULES = [
    _teacher_bottleneck,
    _student_gap,
    _selective_parity,
    _ambiguous_labels,
    _thin_classes,
    _unsure_teacher,
    _distribution_shift,
    _invalid_responses,
]


def run_rules(ctx):
    findings = []
    for rule in _RULES:
        result = rule(ctx)
        if isinstance(result, Finding):
            findings.append(result)
        elif result:
            findings.extend(result)
    return sorted(findings, key=lambda f: _SEVERITY_ORDER[f.severity])
