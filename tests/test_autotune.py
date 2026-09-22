"""autotune offline: tiered fake teacher, gepa mocked at its API boundary."""

import json
import threading
import types

import pytest
from conftest import LABELS, keyword_label, make_pool, make_seed, response

from shrewd._autotune import autotune


class TieredTeacher:
    """test/strong is a keyword oracle; test/weak mislabels half of every pool class."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, model, messages, **kwargs):
        with self._lock:
            self.calls.append(model)
        text = messages[1]["content"]
        label = keyword_label(text)
        if model == "test/weak" and "pool item" in text and int(text.rsplit(" ", 1)[1]) % 2:
            names = list(LABELS)
            label = names[(names.index(label) + 1) % len(names)]
        return response(json.dumps({"label": label}))


@pytest.fixture
def tiered_teacher(monkeypatch):
    fake = TieredTeacher()
    monkeypatch.setattr("shrewd.teacher.litellm.completion", fake)
    monkeypatch.setattr(
        "shrewd.teacher.litellm.completion_cost", lambda completion_response: 0.001
    )
    monkeypatch.setattr("shrewd.teacher.time.sleep", lambda seconds: None)
    return fake


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


def run(tmp_path, **overrides):
    kwargs = dict(
        budget_usd=50,
        target=0.85,
        teachers=("test/weak", "test/strong"),
        students=("tfidf",),
        optimize_budget=10,
    )
    kwargs.update(overrides)
    return autotune(
        tmp_path / "auto",
        "Classify customer support tickets by the customer's primary intent.",
        LABELS, make_seed(30), make_pool(8), **kwargs,
    )


def stages(tmp_path, tier):
    manifest = json.loads((tmp_path / "auto" / tier / "manifest.json").read_text())
    return [s["name"] for s in manifest["stages"]]


def test_escalates_past_weak_teacher_and_evaluates_test_once(tmp_path, tiered_teacher, fake_gepa):
    out = run(tmp_path)
    assert out["teacher"] == "test/strong"
    assert out["dev_macro_f1"] >= 0.85
    assert out["result"].metrics["student"]["macro_f1"] > 0.9
    # both tiers labeled, but only the winner ever touched the locked test set
    assert "label" in stages(tmp_path, "weak")
    assert "distill" not in stages(tmp_path, "weak")
    assert stages(tmp_path, "strong").count("distill") == 1
    trail = json.loads((tmp_path / "auto" / "autotune_trail.json").read_text())
    assert trail == out["trail"]
    assert trail[-1]["rung"] == "final"
    assert any("stopping early" in entry.get("decision", "") for entry in trail)
    assert out["spent_usd"] > 0


def test_stops_at_cheap_tier_when_target_met(tmp_path, tiered_teacher, fake_gepa):
    out = run(tmp_path, target=0.3)
    assert out["teacher"] == "test/weak"
    assert not (tmp_path / "auto" / "strong").exists()
    assert stages(tmp_path, "weak").count("distill") == 1


def test_skips_tier_the_budget_cannot_cover(tmp_path, tiered_teacher, fake_gepa, monkeypatch):
    monkeypatch.setattr(
        "shrewd._autotune.estimate_calls",
        lambda texts, prompt, model, votes, conn:
            (10, {"test/weak": 1.0, "test/strong": 100.0}[model]),
    )
    out = run(tmp_path, budget_usd=5, target=0.99)
    assert out["teacher"] == "test/weak"
    assert not (tmp_path / "auto" / "strong" / "pool_labeled.csv").exists()
    assert any(entry.get("decision", "").startswith("skipped") for entry in out["trail"])


def test_raises_before_spending_when_first_tier_unaffordable(
    tmp_path, tiered_teacher, fake_gepa, monkeypatch
):
    monkeypatch.setattr(
        "shrewd._autotune.estimate_calls",
        lambda texts, prompt, model, votes, conn: (10, 5.0),
    )
    with pytest.raises(ValueError, match="nothing was spent"):
        run(tmp_path, budget_usd=2)
    assert not (tmp_path / "auto" / "weak" / "pool_labeled.csv").exists()
    assert tiered_teacher.calls == []


def test_package_attribute_is_the_function_not_the_submodule():
    # regression: a submodule named autotune shadowed the function on from-imports
    import shrewd
    from shrewd import autotune as imported

    assert callable(imported)
    assert callable(shrewd.autotune)
    assert callable(shrewd.autotune)  # and stays callable on repeated access


def test_unknown_student_rejected_before_spending(tmp_path, tiered_teacher, fake_gepa):
    with pytest.raises(ValueError, match="unknown student"):
        run(tmp_path, students=("tfidf", "nope"))
    assert tiered_teacher.calls == []


def test_missing_extra_rejected_before_spending(tmp_path, tiered_teacher, fake_gepa, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "model2vec", None)  # simulate a base install
    with pytest.raises(ImportError, match="model2vec"):
        run(tmp_path, students=("tfidf", "embed"))
    assert tiered_teacher.calls == []


def test_collapsed_teacher_raises_instead_of_crashing(tmp_path, monkeypatch, fake_gepa):
    monkeypatch.setattr(
        "shrewd.teacher.litellm.completion",
        lambda model, messages, **kwargs: response(json.dumps({"label": "billing"})),
    )
    monkeypatch.setattr(
        "shrewd.teacher.litellm.completion_cost", lambda completion_response: 0.001
    )
    monkeypatch.setattr("shrewd.teacher.time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="fewer than two distinct labels"):
        run(tmp_path)
    trail = json.loads((tmp_path / "auto" / "autotune_trail.json").read_text())
    assert any("probe skipped" in entry.get("decision", "") for entry in trail)
