from shrewd import evaluate
from shrewd.evaluate import DistillResult, Finding

CLASSES = ["billing", "bug", "cancellation", "other"]
LABELS = dict.fromkeys(CLASSES, "a description")


def metrics(macro_f1, per_class_f1=None, confusion=None):
    per_class_f1 = per_class_f1 or {}
    return {
        "accuracy": macro_f1,
        "macro_f1": macro_f1,
        "per_class": {
            cls: {
                "precision": 0.9,
                "recall": 0.9,
                "f1": per_class_f1.get(cls, macro_f1),
                "support": 30,
            }
            for cls in CLASSES
        },
        "confusion": confusion
        or [[30 if i == j else 0 for j in range(4)] for i in range(4)],
    }


def make_ctx(**overrides):
    ctx = {
        "teacher": metrics(0.88),
        "student": metrics(0.86),
        "student_type": "tfidf",
        "classes": CLASSES,
        "labels": LABELS,
        "target": None,
        "train_counts": dict.fromkeys(CLASSES, 100),
        "pool_confidence": [0.9] * 400,
        "n_pool": 400,
        "n_invalid": 0,
        "votes": 3,
        "seed_counts": dict.fromkeys(CLASSES, 50),
        "pool_counts": dict.fromkeys(CLASSES, 100),
        "test_examples": [],
    }
    ctx.update(overrides)
    return ctx


def titles(ctx):
    return [f.title for f in evaluate.run_rules(ctx)]


def test_healthy_run_has_no_findings():
    assert titles(make_ctx()) == []


def test_teacher_bottleneck():
    findings = evaluate.run_rules(make_ctx(teacher=metrics(0.70)))
    assert findings[0].severity == "fail"
    assert "bottleneck" in findings[0].title
    assert "default bar" in findings[0].detail


def test_teacher_bottleneck_respects_target():
    assert "teacher is the bottleneck" in titles(make_ctx(teacher=metrics(0.85), target=0.9))
    assert "teacher is the bottleneck" not in titles(make_ctx(teacher=metrics(0.85), target=0.8))


def test_student_gap():
    findings = evaluate.run_rules(make_ctx(student=metrics(0.75)))
    assert [f.severity for f in findings] == ["warn"]
    assert "can't match" in findings[0].title


def test_student_gap_fails_below_060():
    findings = evaluate.run_rules(make_ctx(teacher=metrics(0.80), student=metrics(0.55)))
    assert any(f.severity == "fail" and "can't match" in f.title for f in findings)


def test_no_student_gap_when_close():
    assert titles(make_ctx(student=metrics(0.84))) == []


def test_student_gap_suggestion_skips_current_student():
    findings = evaluate.run_rules(make_ctx(student=metrics(0.75), student_type="setfit"))
    assert "encoder" in findings[0].suggestion
    assert "setfit" not in findings[0].suggestion


def test_bottleneck_notes_self_consistency():
    consistent = make_ctx(teacher=metrics(0.85), target=0.95, pool_confidence=[0.97] * 100)
    assert "self-consistent" in evaluate.run_rules(consistent)[0].detail
    single_vote = make_ctx(teacher=metrics(0.85), target=0.95, votes=1)
    assert "self-consistent" not in evaluate.run_rules(single_vote)[0].detail


def test_ambiguous_label():
    confusion = [[30, 0, 0, 0], [0, 30, 0, 0], [20, 0, 10, 0], [0, 0, 0, 30]]
    weak = {"cancellation": 0.4}
    ctx = make_ctx(
        teacher=metrics(0.88, weak, confusion),
        student=metrics(0.86, weak),
        test_examples=[
            {"text": "please pause my plan for two months", "gold": "cancellation",
             "teacher": "billing"},
        ],
    )
    findings = [f for f in evaluate.run_rules(ctx) if "ambiguously" in f.title]
    assert len(findings) == 1
    assert "cancellation" in findings[0].title
    assert "billing" in findings[0].detail  # top confused pair
    assert "pause my plan" in findings[0].detail  # example included


def test_ambiguous_needs_both_models_weak():
    ctx = make_ctx(teacher=metrics(0.88, {"cancellation": 0.4}))
    assert not [t for t in titles(ctx) if "ambiguously" in t]


def test_thin_classes():
    findings = evaluate.run_rules(make_ctx(train_counts={**dict.fromkeys(CLASSES, 100), "bug": 7}))
    assert any("bug=7" in f.detail for f in findings)


def test_thin_class_when_missing_entirely():
    counts = {cls: 100 for cls in CLASSES if cls != "other"}
    findings = evaluate.run_rules(make_ctx(train_counts=counts))
    assert any("other=0" in f.detail for f in findings)


def test_unsure_teacher():
    assert "teacher is unsure on the pool" in titles(make_ctx(pool_confidence=[0.5] * 100))


def test_unsure_teacher_needs_votes():
    assert titles(make_ctx(pool_confidence=[0.5] * 100, votes=1)) == []


def test_distribution_shift():
    ctx = make_ctx(pool_counts={"billing": 400, "bug": 50, "cancellation": 50, "other": 0})
    findings = [f for f in evaluate.run_rules(ctx) if "distribution" in f.title]
    assert len(findings) == 1
    assert "billing" in findings[0].detail


def test_no_shift_when_matched():
    assert titles(make_ctx(pool_counts=dict.fromkeys(CLASSES, 97))) == []


def test_invalid_responses_info_then_warn():
    info = evaluate.run_rules(make_ctx(n_invalid=5))
    warn = evaluate.run_rules(make_ctx(n_invalid=50))
    assert [f.severity for f in info] == ["info"]
    assert [f.severity for f in warn] == ["warn"]


def test_findings_sorted_by_severity():
    ctx = make_ctx(teacher=metrics(0.70), student=metrics(0.60), n_invalid=5)
    severities = [f.severity for f in evaluate.run_rules(ctx)]
    assert severities == ["fail", "warn", "info"]


def test_compute_metrics_handles_invalid_predictions():
    y_true = ["bug", "bug", "billing", "other"]
    y_pred = ["bug", "__invalid__", "billing", "other"]
    m = evaluate.compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["bug"]["recall"] == 0.5
    assert m["per_class"]["bug"]["support"] == 2


def test_report_renders():
    result = DistillResult(
        metrics={
            "classes": CLASSES,
            "student_type": "tfidf",
            "n_train": 500,
            "n_test": 120,
            "teacher_invalid_on_test": 2,
            "teacher": metrics(0.88),
            "student": metrics(0.86),
        },
        findings=[Finding("warn", "a title", "a detail.", "Do something.")],
    )
    text = result.report()
    assert "shrewd report" in text
    assert "macro F1" in text
    assert "0.880" in text and "0.860" in text
    assert "[WARN] a title" in text
    assert "→ Do something." in text
    assert "unparseable output on 2/120" in text


def test_report_with_no_findings():
    result = DistillResult(
        metrics={
            "classes": CLASSES,
            "student_type": "tfidf",
            "n_train": 500,
            "n_test": 120,
            "teacher_invalid_on_test": 0,
            "teacher": metrics(0.9),
            "student": metrics(0.9),
        },
        findings=[],
    )
    assert "nothing looks off" in result.report()


def test_bootstrap_interval_is_tight_on_perfect_predictions():
    ci = evaluate.bootstrap_macro_f1(["a", "b"] * 20, ["a", "b"] * 20, ["a", "b"])
    assert ci == [1.0, 1.0]


def test_bootstrap_interval_brackets_the_point_estimate():
    y = ["a"] * 30 + ["b"] * 30
    pred = ["a"] * 24 + ["b"] * 6 + ["b"] * 27 + ["a"] * 3
    m = evaluate.compute_metrics(y, pred, ["a", "b"])
    lo, hi = m["macro_f1_ci"]
    assert lo < m["macro_f1"] < hi
    assert hi - lo > 0.05  # 60 rows is not a lot of rows


def test_bootstrap_matches_sklearn_when_predictions_include_invalid():
    y = ["a", "b", "a", "b"]
    pred = ["a", "__invalid__", "b", "b"]
    m = evaluate.compute_metrics(y, pred, ["a", "b"])
    codes, k = evaluate._encode(y, pred, ["a", "b"])
    assert abs(evaluate._macro_f1_from_codes(codes, k) - m["macro_f1"]) < 1e-9


def test_paired_gap_is_zero_for_identical_models():
    y = ["a", "b"] * 20
    pred = ["a", "a"] * 20
    assert evaluate.paired_gap_interval(y, pred, pred, ["a", "b"]) == [0.0, 0.0]


def test_student_gap_downgrades_to_info_inside_noise():
    ctx = make_ctx(student=metrics(0.85), teacher=metrics(0.92))
    ctx["gap_ci"] = [-0.02, 0.16]
    findings = [f for f in evaluate.run_rules(ctx) if "trails" in f.title]
    assert findings and findings[0].severity == "info"
    assert "includes zero" in findings[0].detail


def test_student_gap_stays_a_warning_when_interval_excludes_zero():
    ctx = make_ctx(student=metrics(0.85), teacher=metrics(0.92))
    ctx["gap_ci"] = [0.02, 0.12]
    titles = [f.title for f in evaluate.run_rules(ctx)]
    assert "student can't match the teacher" in titles


def test_selective_curve_is_monotone_when_confidence_tracks_correctness():
    import numpy as np

    y = ["a"] * 40 + ["b"] * 40 + ["a"] * 10 + ["b"] * 10
    # confident and right on the first 80 rows, hesitant and wrong on the last 20
    proba = np.array(
        [[0.9, 0.1]] * 40 + [[0.1, 0.9]] * 40 + [[0.45, 0.55]] * 10 + [[0.55, 0.45]] * 10
    )
    curve = evaluate.selective_curve(y, proba, ["a", "b"])
    accs = [row["accuracy"] for row in curve]
    # tied confidences share a threshold, so 90% is unreachable and lower levels snap to 80%
    assert [row["coverage"] for row in curve] == [1.0, 1.0, 0.8, 0.8, 0.8]
    assert accs[0] == 0.8 and accs[2] == 1.0
    assert curve[2]["threshold"] == 0.9


def test_selective_parity_finding_points_at_a_threshold():
    ctx = make_ctx(
        student={**metrics(0.80), "accuracy": 0.80}, teacher={**metrics(0.86), "accuracy": 0.86}
    )
    ctx["student_selective"] = [
        {"coverage": 1.0, "threshold": 0.0, "accuracy": 0.80},
        {"coverage": 0.9, "threshold": 0.41, "accuracy": 0.84},
        {"coverage": 0.8, "threshold": 0.52, "accuracy": 0.87},
        {"coverage": 0.7, "threshold": 0.60, "accuracy": 0.90},
    ]
    found = [f for f in evaluate.run_rules(ctx) if "confident items" in f.title]
    assert len(found) == 1 and found[0].severity == "info"
    assert "80%" in found[0].detail and "min_confidence=0.52" in found[0].suggestion


def test_no_parity_finding_when_student_already_matches():
    ctx = make_ctx(
        student=metrics(0.9),  # already ahead of the 0.88 teacher on every row
        student_selective=[{"coverage": 0.9, "threshold": 0.4, "accuracy": 0.99}],
    )
    assert not [f for f in evaluate.run_rules(ctx) if "confident items" in f.title]
