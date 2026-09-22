import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import LABELS, make_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def trained_project(tmp_path):
    from shrewd import Project
    from shrewd.students import TfidfStudent

    project = Project(
        tmp_path / "proj", instructions="Classify tickets.", labels=LABELS,
        teacher="test/fake-model",
    )
    seed = make_seed()
    student = TfidfStudent()
    student.fit(seed["text"].tolist(), seed["label"].tolist())
    student.save(project.dir / "student")
    return project.dir


def test_load_returns_working_classifier(tmp_path):
    from shrewd import load

    clf = load(trained_project(tmp_path))
    assert clf.predict(["I was double charged last month"]) == ["billing"]
    assert clf.predict_proba(["anything"]).shape == (1, 4)
    assert clf.classes_ == sorted(LABELS)
    assert clf.labels == LABELS


class TinyStudent:
    """Minimal custom student exercising the extensibility contract."""

    classes_ = ["a", "b"]
    n_train_ = 0
    fallback = "a"

    def fit(self, texts, labels):
        self.n_train_ = len(texts)

    def predict(self, texts):
        return [self.fallback] * len(texts)

    def predict_proba(self, texts):
        import numpy as np

        return np.tile([1.0, 0.0], (len(texts), 1))

    def save(self, path):
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text('{"type": "tiny", "fallback": "a"}')

    @classmethod
    def _load(cls, path, meta):
        student = cls()
        student.fallback = meta["fallback"]
        return student


def test_load_custom_student_class(tmp_path):
    from shrewd import Project, load

    project = Project(
        tmp_path / "proj", instructions="x", labels=["a", "b"], teacher="test/fake-model"
    )
    TinyStudent().save(project.dir / "student")
    with pytest.raises(ValueError, match="student_cls"):
        load(project.dir)
    clf = load(project.dir, student_cls=TinyStudent)
    assert clf.predict(["anything"]) == ["a"]
    assert clf.labels == {"a": "", "b": ""}


def test_load_without_student_raises(tmp_path):
    from shrewd import Project, load

    project = Project(
        tmp_path / "proj", instructions="x", labels=LABELS, teacher="test/fake-model"
    )
    with pytest.raises(FileNotFoundError, match="distill"):
        load(project.dir)


def test_load_works_without_llm_dependencies(tmp_path):
    """Inference must not import litellm or gepa (they may not be installed)."""
    project_dir = trained_project(tmp_path)
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        sys.modules["litellm"] = None
        sys.modules["gepa"] = None
        from shrewd import load
        clf = load({str(project_dir)!r})
        print(clf.predict(["the app crashes on startup"])[0])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "bug"
