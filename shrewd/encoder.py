"""A small encoder fine-tuned on the panel, used as the featurization.

Trained with a strictly proper scoring rule on the teacher's distributions (log score plus
a spherical term, plus an optional ranked-probability term for Score heads), then `transform()`
returns the pooled representation and the student fits its linear heads on top, so
calibration and stacking work unchanged. Defaults (6 epochs, CLS pooling, ranked term off,
clipping on) come from the sweeps in BENCHMARKS.md.

Needs torch and transformers: `pip install "shrewd[encoder]"`.
"""

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "answerdotai/ModernBERT-base"


class EncoderFeatures:
    """Fine-tune a small encoder on the panel, then serve its pooled representation."""

    expensive = True  # cross-fit uses fewer folds, each fold is a fine-tune

    def __init__(self, model=DEFAULT_MODEL, epochs=6, lr=3e-5, batch_size=16, max_length=256,
                 micro_batch=8,
                 seed=42, device=None, spherical=0.5, ranked=0.0, clip=1.0, pooling="cls"):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                'features="encoder" needs torch and transformers: pip install "shrewd[encoder]"'
            ) from None
        self.model_name = model
        self.epochs = epochs
        self.micro_batch = micro_batch
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.device = device
        self.spherical = spherical
        self.ranked = ranked
        self.clip = clip
        self.pooling = pooling
        self._tok = self._trunk = None

    # -- sklearn-ish surface ----------------------------------------------------

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in (
            "model", "epochs", "lr", "batch_size", "max_length", "micro_batch", "seed", "device",
            "spherical", "ranked", "clip", "pooling",
        ) if k != "model"} | {"model": self.model_name}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, "model_name" if key == "model" else key, value)
        return self

    def _pick_device(self):
        import torch

        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_base(self):
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._trunk = AutoModel.from_pretrained(self.model_name)
        self.device = self._pick_device()
        self._trunk.to(self.device)

    def _encode(self, texts):
        return self._tok(
            [str(t) for t in texts], truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt",
        ).to(self.device)

    def _pool(self, enc):
        """First-token or masked-mean representation (ModernBERT has no pooler)."""
        hidden = self._trunk(**enc).last_hidden_state
        if self.pooling == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden[:, 0]

    # -- training ------------------------------------------------------------------

    def fit_with_targets(self, texts, targets, questions):
        """Fine-tune trunk + one temporary head per question on soft targets.

        `targets` maps question id to a (rows x options) probability matrix. A row of
        zeros means that document has no answer for that question and is masked out
        of that head's loss.
        """
        import torch
        import transformers
        from tqdm import tqdm

        torch.manual_seed(self.seed)
        self._load_base()
        texts = [str(t) for t in texts]
        d = self._trunk.config.hidden_size
        keys = [k for k in questions if k in targets]
        heads = torch.nn.ModuleDict(
            {k: torch.nn.Linear(d, targets[k].shape[1]) for k in keys}
        ).to(self.device)
        ordinal = {k: questions[k].kind == "score" for k in keys}
        T = {k: torch.tensor(targets[k], dtype=torch.float32).to(self.device) for k in keys}

        params = list(self._trunk.parameters()) + list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        steps = self.epochs * ((len(texts) + self.batch_size - 1) // self.batch_size)
        sched = transformers.get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        gen = torch.Generator().manual_seed(self.seed)
        self._trunk.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(texts), generator=gen)
            for start in tqdm(range(0, len(texts), self.batch_size),
                              desc=f"encoder epoch {epoch + 1}/{self.epochs}"):
                idx = order[start:start + self.batch_size]
                # the optimizer step sees the whole batch, but the forward/backward passes
                # run in micro-batches so activations, not the model, set the memory peak
                # (a 16 x 256-token batch of ModernBERT-base is ~13 GB resident in fp32)
                stepped = False
                micro = self.micro_batch or self.batch_size
                for m0 in range(0, len(idx), micro):
                    sub = idx[m0:m0 + micro]
                    h = self._pool(self._encode([texts[i] for i in sub]))
                    loss = 0.0
                    for k in keys:
                        t = T[k][sub.to(self.device)]
                        answered = t.sum(dim=1) > 1e-9
                        if not answered.any():
                            continue
                        q = torch.softmax(heads[k](h[answered]).float(), dim=-1)
                        loss = loss + proper_loss(q, t[answered], ordinal[k],
                                                  self.spherical, self.ranked)
                    if isinstance(loss, float):
                        continue
                    (loss * (len(sub) / len(idx))).backward()
                    stepped = True
                if not stepped:
                    continue
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(params, self.clip)
                opt.step()
                sched.step()
                opt.zero_grad()
        self._trunk.eval()
        # the optimizer's moments are twice the trunk's size and the caching allocator
        # keeps every freed block. Without this, three folds plus the final fit grow to
        # ~15 GB resident on a laptop and the run spends its time swapping
        del opt, sched, heads, params, T
        release_memory()
        return self.transform(texts)

    def fit_transform(self, texts, y=None):
        raise TypeError(
            "EncoderFeatures learns from the panel's targets; DecisionStudent calls "
            "fit_with_targets(). Pass features=\"encoder\" rather than an instance."
        )

    def transform(self, texts):
        import torch

        if self._trunk is None:
            raise RuntimeError("EncoderFeatures has not been fit")
        texts = [str(t) for t in texts]
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                out.append(self._pool(self._encode(texts[start:start + 64])).float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, self._trunk.config.hidden_size))

    # -- persistence -----------------------------------------------------------------

    def save_to(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._trunk.save_pretrained(str(path / "trunk"))
        self._tok.save_pretrained(str(path / "trunk"))
        (path / "features.json").write_text(json.dumps(self.get_params(), indent=2))

    @classmethod
    def load_from(cls, path):
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        params = json.loads((path / "features.json").read_text())
        obj = cls(**params)
        obj._tok = AutoTokenizer.from_pretrained(str(path / "trunk"))
        obj._trunk = AutoModel.from_pretrained(str(path / "trunk"))
        obj.device = obj._pick_device()
        obj._trunk.to(obj.device).eval()
        return obj


def release_memory():
    """Return freed tensors to the OS: Python's garbage first, then the device cache."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:            # no torch, or a backend without a cache to empty
        pass


def proper_loss(q, t, ordinal, w_spherical=0.5, w_ranked=1.0):
    """Negative strictly proper score of distribution `q` against soft target `t`.

    Log score plus a spherical term, and for ordinal questions the ranked probability
    score, which charges for distance along the scale rather than only for missing the
    exact level. All three are strictly proper: the unique minimizer is q = t.
    """
    import torch

    logq = torch.log(q.clamp_min(1e-12))
    loss = -(t * logq).sum(dim=-1)
    if w_spherical:
        loss = loss - w_spherical * (t * q).sum(dim=-1) / q.norm(dim=-1).clamp_min(1e-9)
    if ordinal and w_ranked and q.shape[-1] > 1:
        cdf_q, cdf_t = torch.cumsum(q, dim=-1), torch.cumsum(t, dim=-1)
        loss = loss + w_ranked * ((cdf_q - cdf_t) ** 2).sum(dim=-1) / (q.shape[-1] - 1)
    return loss.mean()
