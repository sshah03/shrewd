import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import LABELS, make_pool, make_seed

from shrewd import Project


def test_fresh_project_writes_manifest(project):
    manifest = json.loads((project.dir / "manifest.json").read_text())
    assert manifest["labels"] == LABELS
    assert manifest["teacher"] == "test/fake-model"
    assert manifest["seed"] == 42
    assert manifest["stages"] == []


def test_reload_restores_task(project):
    reloaded = Project(project.dir)
    assert reloaded.instructions == project.instructions
    assert reloaded.labels == LABELS
    assert reloaded.teacher == "test/fake-model"
    assert reloaded.seed == 42


def test_reload_rejects_mismatched_task(project):
    with pytest.raises(ValueError, match="does not match"):
        Project(project.dir, instructions="Classify something else entirely.")
    with pytest.raises(ValueError, match="does not match"):
        Project(project.dir, teacher="other/model")


def test_new_project_requires_task_args(tmp_path):
    with pytest.raises(ValueError, match="instructions, labels, teacher"):
        Project(tmp_path / "empty")


def test_labels_list_becomes_descriptionless_dict(tmp_path):
    proj = Project(
        tmp_path / "p", instructions="x", labels=["a", "b"], teacher="test/fake-model"
    )
    assert proj.labels == {"a": "", "b": ""}


def test_provider_aliases_resolve_to_default_models(tmp_path):
    claude = Project(tmp_path / "a", instructions="x", labels=["a", "b"], teacher="claude")
    assert claude.teacher == "anthropic/claude-opus-4-8"
    openai = Project(tmp_path / "o", instructions="x", labels=["a", "b"], teacher="openai")
    assert openai.teacher == "openai/gpt-5.1"
    manifest = json.loads((claude.dir / "manifest.json").read_text())
    assert manifest["teacher"] == "anthropic/claude-opus-4-8"  # resolved form is stored


def test_alias_matches_on_reload(tmp_path):
    Project(tmp_path / "p", instructions="x", labels=["a", "b"], teacher="anthropic")
    reloaded = Project(tmp_path / "p", teacher="claude")  # same default → no mismatch
    assert reloaded.teacher == "anthropic/claude-opus-4-8"
    with pytest.raises(ValueError, match="does not match"):
        Project(tmp_path / "p", teacher="openai")


def test_explicit_model_strings_pass_through(tmp_path):
    proj = Project(
        tmp_path / "p", instructions="x", labels=["a", "b"], teacher="openai/gpt-4o-mini"
    )
    assert proj.teacher == "openai/gpt-4o-mini"


def test_add_seed_writes_splits_and_stage(project):
    project.add_seed(make_seed())
    assert (project.dir / "seed_dev.csv").exists()
    assert (project.dir / "seed_test.csv").exists()
    manifest = json.loads((project.dir / "manifest.json").read_text())
    assert manifest["stages"][0]["name"] == "add_seed"
    assert manifest["splits"]["n_dev"] + manifest["splits"]["n_test"] == 120


def test_test_set_is_locked(project):
    project.add_seed(make_seed())
    hash_before = json.loads((project.dir / "manifest.json").read_text())["splits"]["test_sha256"]
    with pytest.raises(ValueError, match="locked"):
        project.add_seed(make_seed(31))  # different data
    project.add_seed(make_seed(), overwrite=True)
    hash_after = json.loads((project.dir / "manifest.json").read_text())["splits"]["test_sha256"]
    assert hash_before == hash_after  # same data + same seed → same split


def test_optimize_requires_seed_data(project):
    with pytest.raises(FileNotFoundError, match="add_seed"):
        project.optimize()


def test_optimize_skips_when_prompt_exists(project, capsys):
    (project.dir / "prompt.txt").write_text("hand-written prompt")
    project.optimize()
    assert "skipping" in capsys.readouterr().out
    assert (project.dir / "prompt.txt").read_text() == "hand-written prompt"


def test_label_requires_prompt(project):
    with pytest.raises(FileNotFoundError, match="optimize"):
        project.label(make_pool())


def test_label_requires_text_column(project):
    (project.dir / "prompt.txt").write_text("p")
    with pytest.raises(ValueError, match="text"):
        project.label(make_seed()[["label"]])


def test_label_dry_run_makes_no_calls(project, fake_teacher, capsys):
    (project.dir / "prompt.txt").write_text("p")
    project.label(make_pool(), votes=3, dry_run=True)
    out = capsys.readouterr().out
    assert "120 calls" in out
    assert "120 not yet cached" in out
    assert fake_teacher.calls == 0
    assert not (project.dir / "pool_labeled.csv").exists()


def test_label_with_user_written_prompt(project, fake_teacher):
    (project.dir / "prompt.txt").write_text("classify tickets")
    project.label(make_pool())
    labeled = pd.read_csv(project.dir / "pool_labeled.csv")
    assert set(labeled.columns) == {"text", "label", "confidence"}
    assert fake_teacher.calls == 40


def test_distill_requires_labeled_pool(project):
    with pytest.raises(FileNotFoundError, match="label"):
        project.distill()


def test_distill_rejects_unknown_student(project, fake_teacher):
    project.add_seed(make_seed())
    (project.dir / "prompt.txt").write_text("p")
    project.label(make_pool())
    with pytest.raises(ValueError, match="unknown student"):
        project.distill(student="transformer-xxl")


def test_optimize_has_no_path_to_test_set():
    from shrewd import optimize

    assert "seed_test" not in Path(optimize.__file__).read_text()


def test_resplit_removes_trained_artifacts(project):
    project.add_seed(make_seed())
    (project.dir / "prompt.txt").write_text("p")
    (project.dir / "compare.json").write_text("{}")
    (project.dir / "student").mkdir()
    (project.dir / "student" / "meta.json").write_text("{}")
    (project.dir / "candidates" / "tfidf").mkdir(parents=True)
    project.add_seed(make_seed(), overwrite=True)
    for stale in ("prompt.txt", "compare.json", "student", "candidates"):
        assert not (project.dir / stale).exists()


def test_distill_replaces_previous_student_dir(project, fake_teacher):
    project.add_seed(make_seed())
    (project.dir / "prompt.txt").write_text("classify tickets")
    project.label(make_pool())
    project.distill(student="tfidf")
    (project.dir / "student" / "leftover-from-other-type.bin").write_text("x")
    project.distill(student="tfidf")
    assert not (project.dir / "student" / "leftover-from-other-type.bin").exists()
    assert (project.dir / "student" / "meta.json").exists()


def test_repeat_distill_warns_about_test_reuse(project, fake_teacher):
    project.add_seed(make_seed())
    (project.dir / "prompt.txt").write_text("classify tickets")
    project.label(make_pool())
    project.distill(student="tfidf")
    with pytest.warns(UserWarning, match="evaluation #2 of the locked test set"):
        project.distill(student="tfidf")


def test_distill_warns_when_prompt_changed_after_label(project, fake_teacher):
    project.add_seed(make_seed())
    (project.dir / "prompt.txt").write_text("classify tickets")
    project.label(make_pool())
    (project.dir / "prompt.txt").write_text("a different prompt")
    with pytest.warns(UserWarning, match="changed since label"):
        project.distill(student="tfidf")


def test_add_seed_is_a_noop_for_the_same_data(project):
    seed = make_seed(30)
    project.add_seed(seed)
    before = (project.dir / "seed_test.csv").read_text()
    (project.dir / "prompt.txt").write_text("keep me")
    project.add_seed(seed)  # a script re-run, not a re-split
    assert (project.dir / "seed_test.csv").read_text() == before
    assert (project.dir / "prompt.txt").exists()
    with pytest.raises(ValueError, match="locked"):
        project.add_seed(make_seed(31))  # different data still has to be explicit
