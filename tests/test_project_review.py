import pandas as pd
import pytest
from conftest import LABELS, keyword_label, make_pool, make_seed, response


class TwoTeachers:
    """Two fake models that disagree on refund tickets: `a` says billing, `b` says bug."""

    def __init__(self):
        self.calls = 0
        self.by_model = {}

    def __call__(self, model, messages, **kwargs):
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1
        text = messages[1]["content"]
        label = keyword_label(text)
        if model == "test/b" and "refund" in text.lower():
            label = "bug"
        return response(f'{{"label": "{label}"}}')


@pytest.fixture
def two_teachers(monkeypatch):
    fake = TwoTeachers()
    monkeypatch.setattr("shrewd.teacher.litellm.completion", fake)
    monkeypatch.setattr("shrewd.teacher.litellm.completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr("shrewd.teacher.time.sleep", lambda seconds: None)
    return fake


def _project(tmp_path, teacher):
    from shrewd import Project

    p = Project(tmp_path / "proj", instructions="Classify tickets.", labels=LABELS, teacher=teacher)
    p.add_seed(make_seed())
    (p.dir / "prompt.txt").write_text("classify tickets")
    return p


def test_two_teachers_vote_and_disagreements_fill_the_review_queue(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    assert p.teacher == ["test/a", "test/b"]
    pool = make_pool()
    labeled = p.label(pool)
    assert two_teachers.by_model == {"test/a": len(pool), "test/b": len(pool)}
    refund = labeled[labeled["text"].str.contains("refund")]
    assert (refund["confidence"] == 0.5).all()
    assert (refund["label"] == "billing").all()            # tie -> the first teacher listed
    settled = labeled[~labeled["text"].str.contains("refund")]
    assert (settled["confidence"] == 1.0).all()
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    assert set(review["text"]) == set(refund["text"])
    assert (review["human_label"] == "").all()


def test_a_human_label_overrules_the_teachers_and_survives_relabeling(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    pool = make_pool()
    p.label(pool)
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_label"] = "cancellation"
    fixed_text = review.loc[0, "text"]
    review.to_csv(p.dir / "needs_review.csv", index=False)

    assert p.apply_review() == 1
    labeled = pd.read_csv(p.dir / "pool_labeled.csv")
    row = labeled[labeled["text"] == fixed_text].iloc[0]
    assert row["label"] == "cancellation" and row["confidence"] == 1.0
    assert (p.dir / "reviewed.csv").exists()

    p.label(pool)                                            # relabel: the fix comes back
    labeled = pd.read_csv(p.dir / "pool_labeled.csv")
    assert labeled[labeled["text"] == fixed_text].iloc[0]["label"] == "cancellation"

    result = p.distill(student="tfidf")                      # distill applies the queue too
    assert result.metrics["n_train"] > 0


def test_a_bad_human_label_is_reported_not_applied(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    p.label(make_pool())
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_label"] = "not-a-label"
    review.to_csv(p.dir / "needs_review.csv", index=False)
    with pytest.warns(UserWarning, match="not one of"):
        assert p.apply_review() == 0


def test_one_teacher_has_an_empty_queue_and_is_unchanged(tmp_path, fake_teacher):
    p = _project(tmp_path, "test/fake-model")
    labeled = p.label(make_pool())
    assert (labeled["confidence"] == 1.0).all()
    assert not (p.dir / "needs_review.csv").exists()


def test_active_acquisition_refuses_several_teachers(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    with pytest.raises(ValueError, match="several teachers"):
        p.label(make_pool(), n=10)


def test_dry_run_counts_every_teacher(tmp_path, two_teachers, capsys):
    p = _project(tmp_path, ["test/a", "test/b"])
    p.label(make_pool(), dry_run=True)
    assert "× 2 teacher(s) = 80 calls" in capsys.readouterr().out
    assert two_teachers.calls == 0


def test_ensemble_teacher_round_trips_through_the_manifest(tmp_path, two_teachers):
    from shrewd import Project

    _project(tmp_path, ["test/a", "test/b"])
    assert Project(tmp_path / "proj").teacher == ["test/a", "test/b"]


def test_pending_human_labels_survive_a_relabel(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    pool = make_pool()
    p.label(pool)
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_label"] = "other"
    text = review.loc[0, "text"]
    review.to_csv(p.dir / "needs_review.csv", index=False)
    p.label(pool)                                            # no apply_review() in between
    labeled = pd.read_csv(p.dir / "pool_labeled.csv")
    assert labeled[labeled["text"] == text].iloc[0]["label"] == "other"

