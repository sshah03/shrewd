import json
import tempfile
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from tqdm import tqdm

from shrewd import __version__


class Student(Protocol):
    """The duck type distill()/compare() expect. Implement this to bring your own model."""

    classes_: list[str]

    def fit(self, texts: list[str], labels: list[str]) -> None: ...
    def predict(self, texts: list[str]) -> list[str]: ...  # min_confidence= is optional
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


class _Student:
    """predict() shared by every built-in student: argmax of predict_proba, with an
    optional confidence floor. Rows whose top probability is below `min_confidence`
    come back as None so the caller can route them elsewhere."""

    def predict(self, texts, min_confidence=0.0):
        proba = self.predict_proba(texts)
        labels = [self.classes_[i] for i in proba.argmax(axis=1)]
        if min_confidence:
            top = proba.max(axis=1)
            labels = [
                label if p >= min_confidence else None
                for label, p in zip(labels, top, strict=True)
            ]
        return labels


def _write_meta(path, kind, student, **extra):
    meta = {
        "type": kind,
        "n_train": student.n_train_,
        "classes": student.classes_,
        "versions": {"shrewd": __version__, "scikit-learn": sklearn.__version__},
        **extra,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def _lr(seed, overrides):
    params = {"class_weight": "balanced", "max_iter": 2000, "random_state": seed}
    params.update(overrides)
    return LogisticRegression(**params)


class TfidfStudent(_Student):
    """Word + char n-gram tf-idf into logistic regression. No extra dependencies."""

    def __init__(self, seed=42, **lr_kwargs):
        self._pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2))),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
                        ]
                    ),
                ),
                ("lr", _lr(seed, lr_kwargs)),
            ]
        )
        self.n_train_ = 0

    @property
    def classes_(self):
        return [str(c) for c in self._pipe.classes_]

    def fit(self, texts, labels):
        self._pipe.fit(texts, labels)
        self.n_train_ = len(texts)

    def predict_proba(self, texts):
        return self._pipe.predict_proba(texts)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipe, path / "model.joblib")
        _write_meta(path, "tfidf", self)

    @classmethod
    def _load(cls, path, meta):
        student = cls()
        student._pipe = joblib.load(path / "model.joblib")
        student.n_train_ = meta["n_train"]
        return student


class EmbedStudent(_Student):
    """model2vec static embeddings into logistic regression. No torch dependency."""

    def __init__(self, model="minishlab/potion-base-8M", seed=42, **lr_kwargs):
        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError(
                'student="embed" needs model2vec: pip install "shrewd[embed]"'
            ) from None
        self._embedder = StaticModel.from_pretrained(model)
        self._embedder_name = model
        self._head = _lr(seed, lr_kwargs)
        self.n_train_ = 0

    @property
    def classes_(self):
        return [str(c) for c in self._head.classes_]

    def fit(self, texts, labels):
        self._head.fit(self._embedder.encode(texts), labels)
        self.n_train_ = len(texts)

    def predict_proba(self, texts):
        return self._head.predict_proba(self._embedder.encode(texts))

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._head, path / "model.joblib")
        _write_meta(path, "embed", self, embedder=self._embedder_name)

    @classmethod
    def _load(cls, path, meta):
        student = cls(model=meta["embedder"])
        student._head = joblib.load(path / "model.joblib")
        student.n_train_ = meta["n_train"]
        return student


class SetFitStudent(_Student):
    """Contrastive fine-tune of a small sentence-transformer via SetFit. Needs torch.

    SetFit's default pair sampling is intractable beyond a few hundred rows, so pass
    train_args={"max_steps": 2000} (forwarded to setfit.TrainingArguments) on big pools.
    On long documents pass max_seq_length (e.g. 256). It truncates the encoder input
    during the embedding fine-tune, whose memory grows with the model's full window
    otherwise (setfit's train_args max_length only reaches the classifier phase).
    """

    def __init__(self, model="sentence-transformers/paraphrase-mpnet-base-v2", seed=42,
                 train_args=None, max_seq_length=None, **kwargs):
        try:
            import setfit  # noqa: F401
        except ImportError:
            raise ImportError(
                'student="setfit" needs setfit: pip install "shrewd[setfit]"'
            ) from None
        self._base = model
        self._seed = seed
        self._train_args = train_args or {}
        self._max_seq_length = max_seq_length
        self._kwargs = kwargs
        self._model = None
        self.classes_ = []
        self.n_train_ = 0

    def fit(self, texts, labels):
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        self.classes_ = sorted(set(labels))
        index = {label: i for i, label in enumerate(self.classes_)}
        self._model = SetFitModel.from_pretrained(self._base, **self._kwargs)
        if self._max_seq_length is not None:
            self._model.model_body.max_seq_length = self._max_seq_length
        # trainer checkpoints go to a temp dir, cleaned up when training ends
        with tempfile.TemporaryDirectory(prefix="setfit-") as checkpoint_dir:
            params = {"seed": self._seed, "output_dir": checkpoint_dir}
            params.update(self._train_args)
            trainer = Trainer(
                model=self._model,
                args=TrainingArguments(**params),
                train_dataset=Dataset.from_dict(
                    {"text": list(texts), "label": [index[label] for label in labels]}
                ),
            )
            trainer.train()
        self.n_train_ = len(texts)

    def predict_proba(self, texts):
        proba = self._model.predict_proba(list(texts))
        return np.asarray(proba.cpu() if hasattr(proba, "cpu") else proba)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path / "setfit"))
        _write_meta(path, "setfit", self, base_model=self._base)

    @classmethod
    def _load(cls, path, meta):
        from setfit import SetFitModel

        student = cls(model=meta["base_model"])
        student._model = SetFitModel.from_pretrained(str(path / "setfit"))
        student.classes_ = meta["classes"]
        student.n_train_ = meta["n_train"]
        return student


class EncoderStudent(_Student):
    """Full cross-entropy fine-tune of a small encoder. Strongest student at 1k+ rows."""

    def __init__(self, model="answerdotai/ModernBERT-base", seed=42, epochs=5, lr=3e-5,
                 batch_size=16, max_length=256):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                'student="encoder" needs torch and transformers: '
                'pip install "shrewd[encoder]"'
            ) from None
        self._base = model
        self._seed = seed
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None
        self.classes_ = []
        self.n_train_ = 0

    def _pick_device(self):
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def fit(self, texts, labels):
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self._seed)
        self.classes_ = sorted(set(labels))
        index = {label: i for i, label in enumerate(self.classes_)}
        self._tokenizer = AutoTokenizer.from_pretrained(self._base)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._base, num_labels=len(self.classes_)
        )
        self._device = self._pick_device()
        self._model.to(self._device).train()
        y = torch.tensor([index[label] for label in labels])
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr, weight_decay=0.01)
        generator = torch.Generator().manual_seed(self._seed)
        texts = list(texts)
        steps_per_epoch = (len(texts) + self._batch_size - 1) // self._batch_size
        total_steps = self._epochs * steps_per_epoch
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, int(0.1 * total_steps), total_steps
        )
        for epoch in range(self._epochs):
            order = torch.randperm(len(texts), generator=generator)
            batches = range(0, len(texts), self._batch_size)
            for start in tqdm(batches, desc=f"encoder epoch {epoch + 1}/{self._epochs}"):
                idx = order[start:start + self._batch_size]
                encoded = self._tokenizer(
                    [texts[i] for i in idx], truncation=True, max_length=self._max_length,
                    padding=True, return_tensors="pt",
                ).to(self._device)
                loss = self._model(**encoded, labels=y[idx].to(self._device)).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        self._model.eval()
        self.n_train_ = len(texts)

    def predict_proba(self, texts):
        import torch

        texts = list(texts)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                encoded = self._tokenizer(
                    texts[start:start + 64], truncation=True, max_length=self._max_length,
                    padding=True, return_tensors="pt",
                ).to(self._device)
                logits = self._model(**encoded).logits
                chunks.append(torch.softmax(logits, dim=-1).cpu())
        return torch.cat(chunks).numpy()

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path / "encoder"))
        self._tokenizer.save_pretrained(str(path / "encoder"))
        _write_meta(path, "encoder", self, base_model=self._base)

    @classmethod
    def _load(cls, path, meta):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        student = cls(model=meta["base_model"])
        student._tokenizer = AutoTokenizer.from_pretrained(str(path / "encoder"))
        student._model = AutoModelForSequenceClassification.from_pretrained(
            str(path / "encoder")
        )
        student._device = student._pick_device()
        student._model.to(student._device).eval()
        student.classes_ = meta["classes"]
        student.n_train_ = meta["n_train"]
        return student


STUDENTS = {
    "tfidf": TfidfStudent,
    "embed": EmbedStudent,
    "setfit": SetFitStudent,
    "encoder": EncoderStudent,
}

# optional imports each student needs, so callers can fail fast before spending
REQUIRES = {
    "tfidf": (),
    "embed": ("model2vec",),
    "setfit": ("setfit",),
    "encoder": ("torch", "transformers"),
}


def check_students(names):
    """Raise early if a student name is unknown or its optional extra is missing."""
    import importlib

    for name in names:
        if name not in STUDENTS:
            raise ValueError(f"unknown student {name!r}, pick from {sorted(STUDENTS)}")
        for module in REQUIRES[name]:
            try:
                importlib.import_module(module)
            except ImportError:
                raise ImportError(
                    f'student "{name}" needs {module}: pip install "shrewd[{name}]" '
                    f"or drop {name!r} from students"
                ) from None


def load_student(path, student_cls=None):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta["type"] == "decisions":
        from shrewd.decisions import DecisionStudent

        return DecisionStudent._load(path, meta)
    cls = STUDENTS.get(meta["type"], student_cls)
    if cls is None:
        raise ValueError(
            f"unknown student type {meta['type']!r}; pass student_cls=YourStudent "
            "(a class with a _load(path, meta) classmethod)"
        )
    return cls._load(path, meta)


def fetch_project(url, cache_dir=None):
    """Download and unpack a project tarball once and return the directory it lives in."""
    import hashlib
    import os
    import tarfile
    import urllib.request

    root = Path(cache_dir or os.environ.get("SHREWD_CACHE") or Path.home() / ".cache" / "shrewd")
    target = root / hashlib.sha1(url.encode()).hexdigest()[:16]
    if (target / "manifest.json").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "project.tar.gz"
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    archive.unlink()
    if not (target / "manifest.json").exists():        # tarball wrapped in one top directory
        inner = [p for p in target.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        if inner:
            for child in inner[0].iterdir():
                child.rename(target / child.name)
            inner[0].rmdir()
    return target


def load(project_dir, student_cls=None):
    """Load the trained student from a project directory for inference. Needs neither litellm
    nor gepa. For a custom student class pass `student_cls` with a `_load(path, meta)`
    classmethod. `project_dir` may also be an `https://` URL to a `.tar.gz` of a project
    directory. It's downloaded once into `~/.cache/shrewd/` (or `$SHREWD_CACHE`).
    """
    if isinstance(project_dir, str) and project_dir.startswith(("https://", "http://")):
        project_dir = fetch_project(project_dir)
    project_dir = Path(project_dir)
    manifest_path = project_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{project_dir} is not a shrewd project (no manifest.json)")
    student_dir = project_dir / "student"
    if not (student_dir / "meta.json").exists():
        raise FileNotFoundError(f"no trained student in {project_dir}; run distill() first")
    student = load_student(student_dir, student_cls)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") == "decisions":
        return student  # questions already travel in the student's own meta.json
    student.labels = manifest["labels"]
    return student
