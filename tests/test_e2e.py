"""The whole pipeline offline: fake teacher, gepa mocked at its API boundary."""

import json
import os
import types

import pandas as pd
import pytest
from conftest import LABELS, make_pool, make_seed


@pytest.fixture
def fake_gepa(monkeypatch):
    """Evaluates the seed candidate over the valset via the real adapter/evaluator glue."""

    def optimize(seed_candidate, trainset, valset, adapter, **kwargs):
        scores = adapter.evaluate(valset, seed_candidate).scores
        return types.SimpleNamespace(
            candidates=[dict(seed_candidate)],
            val_aggregate_scores=[sum(scores) / len(scores)],
            best_idx=0,
            best_candidate=dict(seed_candidate),
            total_metric_calls=len(scores),
        )

    monkeypatch.setattr("shrewd.optimize.gepa.optimize", optimize)


def test_full_pipeline(project, fake_teacher, fake_gepa):
    project.add_seed(make_seed(30))
    project.optimize(budget=10)
    assert (project.dir / "prompt.txt").exists()
    log = json.loads((project.dir / "optimize_log.json").read_text())
    assert log["best_score"] == 1.0  # fake teacher is a perfect keyword oracle

    calls_after_optimize = fake_teacher.calls
    assert calls_after_optimize > 0

    pool = make_pool(15)
    project.label(pool)
    assert fake_teacher.calls == calls_after_optimize + len(pool)

    # resume: a second run over the same pool is served entirely from cache
    project.label(pool)
    assert fake_teacher.calls == calls_after_optimize + len(pool)

    # dry run agrees that nothing is left to do
    project.label(pool, dry_run=True)

    result = project.distill()
    assert (project.dir / "student" / "model.joblib").exists()
    assert (project.dir / "report.json").exists()
    assert result.metrics["teacher"]["macro_f1"] == 1.0
    assert result.metrics["student"]["macro_f1"] > 0.9
    assert abs(result.metrics["student"]["macro_f1"] - result.metrics["teacher"]["macro_f1"]) < 0.05

    report = result.report()
    assert "shrewd report" in report
    assert "tfidf" in report

    # re-running distill repeats zero teacher calls (test-set eval is cached)
    calls_after_distill = fake_teacher.calls
    project.distill()
    assert fake_teacher.calls == calls_after_distill

    stages = [s["name"] for s in json.loads((project.dir / "manifest.json").read_text())["stages"]]
    assert stages[:3] == ["add_seed", "optimize", "label"]
    assert stages.count("distill") == 2  # every run is recorded, including the cached re-run


def test_resume_after_partial_label(project, fake_teacher):
    """Killing label() mid-run and re-running only calls the API for missing items."""
    (project.dir / "prompt.txt").write_text("p")
    pool = make_pool(10)
    half = pool.head(20)
    project.label(half)
    assert fake_teacher.calls == 20
    project.label(pool)  # "crashed" halfway; run again with the full pool
    assert fake_teacher.calls == 40


def test_min_confidence_filters_and_writes_review_file(project, fake_teacher):
    project.add_seed(make_seed(30))
    (project.dir / "prompt.txt").write_text("p")

    flip = iter(range(10_000))
    real = fake_teacher.respond
    fake_teacher.respond = (
        lambda messages: real(messages) if next(flip) % 3 else '{"label": "other"}'
    )
    project.label(make_pool(15), votes=3)

    # the review queue is written at label time from teacher disagreement, and it is the
    # file a human fills in, so min_confidence, a training-time filter, neither creates
    # nor deletes it
    review = pd.read_csv(project.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    assert (review["confidence"].astype(float) < 1.0).all()
    assert "human_label" in review.columns

    strict = project.distill(min_confidence=0.9)
    assert strict.metrics["n_train"] > 0

    lenient = project.distill(min_confidence=0.0)  # keep everything
    assert lenient.metrics["n_train"] > strict.metrics["n_train"]
    assert (project.dir / "needs_review.csv").exists()   # still there for the human


def test_pool_rows_that_duplicate_seed_texts_never_train(project, fake_teacher):
    """The pool is often the user's whole export, hand-labeled rows included."""
    seed = make_seed(30)
    project.add_seed(seed)
    (project.dir / "prompt.txt").write_text("p")
    project.label(pd.concat([make_pool(10), seed[["text"]]]))

    _, dev, test, _, kept = project._training_data(0.0)
    assert not set(kept["text"]) & set(test["text"])
    assert not set(kept["text"]) & set(dev["text"])
    assert len(kept) == 40  # only the genuinely new pool rows remain


def test_compare_and_promote(project, fake_teacher):
    from shrewd import load
    from shrewd.students import TfidfStudent

    project.add_seed(make_seed(30))
    (project.dir / "prompt.txt").write_text("p")
    project.label(make_pool(15))

    calls_before = fake_teacher.calls
    rows = project.compare(students=["tfidf", TfidfStudent(C=0.5)])
    assert fake_teacher.calls == calls_before  # ranking on dev costs no API calls
    assert [set(r) for r in rows] == [
        {"student", "dev_macro_f1", "dev_accuracy", "dev_accuracy_at_80pct",
         "min_confidence_for_80pct", "ms_per_item", "size_mb"}
    ] * 2
    assert {r["student"] for r in rows} == {"tfidf", "tfidf-2"}
    assert rows[0]["dev_macro_f1"] >= rows[1]["dev_macro_f1"]  # sorted best-first
    assert (project.dir / "candidates" / "tfidf" / "meta.json").exists()
    assert (project.dir / "compare.json").exists()

    project.promote("tfidf")
    clf = load(project.dir)
    assert clf.predict(["the app crashes constantly"]) == ["bug"]

    with pytest.raises(FileNotFoundError, match="no candidate"):
        project.promote("setfit")


def test_compare_rejects_unknown_name(project, fake_teacher):
    project.add_seed(make_seed(30))
    (project.dir / "prompt.txt").write_text("p")
    project.label(make_pool(5))
    with pytest.raises(ValueError, match="unknown student"):
        project.compare(students=["transformer-xxl"])


def test_distill_accepts_custom_student_instance(project, fake_teacher):
    class KeywordStudent:
        classes_ = ["billing", "bug", "cancellation", "other"]
        n_train_ = 0

        def fit(self, texts, labels):
            self.n_train_ = len(texts)

        def predict(self, texts):
            from conftest import keyword_label

            return [keyword_label(t) for t in texts]

        def predict_proba(self, texts):
            import numpy as np

            preds = self.predict(texts)
            return np.array([[1.0 if c == p else 0.0 for c in self.classes_] for p in preds])

        def save(self, path):
            from pathlib import Path

            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "meta.json").write_text('{"type": "keyword", "n_train": 0}')

    project.add_seed(make_seed(30))
    (project.dir / "prompt.txt").write_text("p")
    project.label(make_pool(5))
    result = project.distill(student=KeywordStudent())
    assert result.metrics["student_type"] == "KeywordStudent"
    assert result.metrics["student"]["macro_f1"] == 1.0


@pytest.mark.skipif(not os.environ.get("SHREWD_E2E"), reason="set SHREWD_E2E=1 to run")
def test_real_api_smoke(tmp_path):
    from shrewd import Project, load

    project = Project(
        tmp_path / "smoke",
        instructions="Classify customer support tickets by the customer's primary intent.",
        labels=LABELS,
        teacher="anthropic/claude-haiku-4-5",
    )
    project.add_seed(make_seed(10))
    project.optimize(budget=20)
    project.label(make_pool(5))
    result = project.distill()
    print(result.report())
    clf = load(project.dir)
    assert clf.predict(["I was double charged"]) == ["billing"]


def test_active_labeling_acquires_in_rounds_and_resumes(project, fake_teacher):
    project.add_seed(make_seed(30))
    (project.dir / "prompt.txt").write_text("p")
    pool = make_pool(25)  # 100 rows

    labeled = project.label(pool, n=20, batch=10)
    assert len(labeled) == 20
    assert sorted(labeled["round"].unique()) == [1, 2]
    assert fake_teacher.calls == 20
    log = json.loads((project.dir / "label_log.json").read_text())
    assert [r["n_labeled"] for r in log] == [10, 20]
    assert all(0 <= r["dev_accuracy"] <= 1 for r in log)

    # resume: asking for 30 labels only buys the 10 missing ones
    labeled = project.label(pool, n=30, batch=10)
    assert len(labeled) == 30
    assert fake_teacher.calls == 30
    assert len(json.loads((project.dir / "label_log.json").read_text())) == 3

    # the labeled subset trains a student like any other pool
    result = project.distill()
    assert result.metrics["n_train"] == 30 + 78
    assert "pool distribution differs" not in [f.title for f in result.findings]


def test_active_labeling_stops_at_budget(project, fake_teacher):
    project.add_seed(make_seed(30))
    (project.dir / "prompt.txt").write_text("p")
    # the fake teacher costs $0.001 a call and litellm cannot price "test/fake-model",
    # so the guard uses the previous round's cost: round 1 = $0.010, round 2 would
    # reach $0.020 against a $0.015 budget
    labeled = project.label(make_pool(25), budget_usd=0.015, batch=10)
    assert len(labeled) == 10
    stage = json.loads((project.dir / "manifest.json").read_text())["stages"][-1]
    assert stage["params"]["mode"] == "active" and stage["params"]["rounds"] == 1


def test_active_labeling_never_asks_the_teacher_about_seed_rows(project, fake_teacher):
    seed = make_seed(30)
    project.add_seed(seed)
    (project.dir / "prompt.txt").write_text("p")
    labeled = project.label(pd.concat([make_pool(5), seed[["text"]]]), n=100, batch=10)
    assert len(labeled) == 20  # the 20 genuinely new rows; seed texts were skipped
    assert not set(labeled["text"]) & set(seed["text"])
