"""A zero-shot NLI model as a second opinion, stacked onto the distilled head.

Hypotheses are built from the option descriptions, not the label names (an NLI model has
no idea what `rec.autos` means). Each hypothesis gets a learned bias so a broad one cannot
absorb probability mass from narrow ones. `DecisionStudent.stack` decides per question
whether to keep the result.

Needs torch and transformers: `pip install "shrewd[encoder]"`.
"""

import numpy as np

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
FRAME = "This text is about {description}."


def hypotheses(question, instructions=None):
    """One NLI hypothesis per option, built from the question.

    Precedence: an explicit `hypothesis` template on the question (with `{option}` and
    `{description}` placeholders), then for a yes/no question its `criteria`, which are
    already statements, then a neutral topic frame around the option description.
    """
    options = question.options()
    template = getattr(question, "hypothesis", None)
    if question.kind == "noul":
        criteria = question.criteria or {}
        yes = criteria.get("true") or question.instructions
        no = criteria.get("false") or f"Not the case: {question.instructions}"
        if template:
            return [template.format(option="no", description=no),
                    template.format(option="yes", description=yes)]
        return [str(no), str(yes)]
    if question.kind == "score":
        descriptions = list(question.criteria)
    else:
        descriptions = [question.criteria[o] for o in options]
    template = template or FRAME
    return [
        template.format(option=o, description=_text(d))
        for o, d in zip(options, descriptions, strict=True)
    ]


def _text(value):
    """Flatten a structured description to a sentence, keep strings as they are."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return str(value)


class ZeroShotScorer:
    """NLI entailment scores over a question's options, one forward pass per option.

    `score()` returns row-normalized probabilities of shape (rows, options). The model
    loads lazily, so constructing one is free and importing this module needs no torch.
    """

    def __init__(self, model=DEFAULT_MODEL, max_length=512, batch_size=16, device=None):
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._tok = self._model = self._entail = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'zero-shot stacking needs torch and transformers: pip install "shrewd[encoder]"'
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self._model.to(self.device).eval()
        labels = {i: name.lower() for i, name in self._model.config.id2label.items()}
        entail = [i for i, name in labels.items() if name.startswith("entail")]
        if not entail:
            raise ValueError(f"{self.model_name} has no entailment label: {labels}")
        self._entail = entail[0]

    def score(self, texts, question, instructions=None):
        import torch

        self._load()
        texts = [str(t) for t in texts]
        hyps = hypotheses(question, instructions)
        out = np.zeros((len(texts), len(hyps)))
        with torch.no_grad():
            for j, hyp in enumerate(hyps):
                for start in range(0, len(texts), self.batch_size):
                    chunk = texts[start:start + self.batch_size]
                    enc = self._tok(
                        chunk, [hyp] * len(chunk), truncation=True,
                        max_length=self.max_length, padding=True, return_tensors="pt",
                    ).to(self.device)
                    probs = torch.softmax(self._model(**enc).logits, dim=-1)[:, self._entail]
                    out[start:start + len(chunk), j] = probs.float().cpu().numpy()
        total = out.sum(axis=1, keepdims=True)
        return np.divide(out, total, out=np.full_like(out, 1.0 / len(hyps)), where=total > 0)

    def to_dict(self):
        return {"model": self.model_name, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, blob):
        return cls(model=blob.get("model", DEFAULT_MODEL), max_length=blob.get("max_length", 512))


# ------------------------------------------------------------------ the stack itself


RIDGE = 1e-3  # keeps the optimum finite when a source is (near-)perfectly separable


def fit_bias(log_zs, y_idx):
    """One additive bias per hypothesis, minimizing log loss. Centered so it is readable."""
    from scipy.optimize import minimize

    k = log_zs.shape[1]
    rows = np.arange(len(y_idx))

    def nll(b):
        z = log_zs + b
        z = z - z.max(axis=1, keepdims=True)
        return -np.mean(z[rows, y_idx] - np.log(np.exp(z).sum(axis=1))) + RIDGE * np.sum(b**2)

    b = minimize(nll, np.zeros(k), method="Nelder-Mead", options={"maxiter": 5000}).x
    return b - b.mean()


def fit_weights(log_head, log_zs, y_idx):
    """Two weights on the two log-probability sources, minimizing log loss.

    Two and not k+1: per-class weights fit the pool's teacher labels a little better and
    the gold holdout no better, and with five classes that is already enough to overfit.
    The per-hypothesis bias handles the per-class part.
    """
    from scipy.optimize import minimize

    rows = np.arange(len(y_idx))

    def nll(w):
        z = w[0] * log_head + w[1] * log_zs
        z = z - z.max(axis=1, keepdims=True)
        return -np.mean(z[rows, y_idx] - np.log(np.exp(z).sum(axis=1))) + RIDGE * np.sum(w**2)

    w = minimize(nll, [1.0, 1.0], method="Nelder-Mead").x
    return [float(w[0]), float(w[1])]


def combine(head_proba, zs_proba, bias, weights):
    """Stacked probabilities from the two sources and their fitted parameters."""
    lg = lambda p: np.log(np.clip(np.asarray(p, dtype=float), 1e-9, 1.0))  # noqa: E731
    z = weights[0] * lg(head_proba) + weights[1] * (lg(zs_proba) + np.asarray(bias))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
