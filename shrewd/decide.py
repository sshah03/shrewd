"""Typed decision questions: Choice, Score, Noul.

Several questions are asked together about one document and each is answered with a
probability distribution:

    Choice  pick one of N named options      -> probabilities over the options
    Score   rate against ordered levels      -> probabilities over levels, plus their mean
    Noul    is this statement true?          -> one number, P(yes)

How a teacher produces the distribution is a backend (verbalized JSON, token logprobs,
or an ensemble). See judge.py.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np

NO, YES = "no", "yes"


# ---------------------------------------------------------------- question types


@dataclass(frozen=True)
class Choice:
    """Pick one of N named options.

    `criteria` maps option name to a description, and both are shown to the teacher. Include
    an "other" option when an input might fit none of the real ones.
    """

    instructions: str
    criteria: dict[str, str]
    hypothesis: str | None = None  # NLI template with {option}/{description}, for stacking

    kind = "choice"

    def options(self):
        return list(self.criteria)

    def render(self):
        lines = ["  type: choice — pick exactly one of the options below", "  options:"]
        lines += [f"    - {name}: {desc}" for name, desc in self.criteria.items()]
        return "\n".join(lines)

    def answer(self, proba, options):
        top = int(np.argmax(proba))
        return {
            "type": self.kind,
            "choice": options[top],
            "confidence": _confidence(proba),
            "probabilities": {o: round(float(p), 4) for o, p in zip(options, proba, strict=True)},
        }


@dataclass(frozen=True)
class Score:
    """Rate against ordered levels, lowest first.

    The headline number is the probability-weighted mean over level indices, so a document
    split between "cosmetic" and "blocking" scores in the middle. The full distribution
    travels alongside, since 0.5/0/0.5 and 0/1/0 have the same mean.
    """

    instructions: str
    criteria: list[str]
    hypothesis: str | None = None

    kind = "score"

    def __post_init__(self):
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError(
                f"a Score needs between 2 and 10 levels, got {len(self.criteria)}; "
                "beyond that the levels stop being distinguishable and a Choice is clearer"
            )

    def options(self):
        return [str(i) for i in range(len(self.criteria))]

    def render(self):
        lines = ["  type: score — rate on this scale, lowest level first", "  levels:"]
        lines += [f"    - {i}: {desc}" for i, desc in enumerate(self.criteria)]
        return "\n".join(lines)

    def answer(self, proba, options):
        levels = np.arange(len(proba), dtype=float)
        return {
            "type": self.kind,
            "score": round(float((proba * levels).sum()), 4),
            "confidence": _confidence(proba),
            "probabilities": {o: round(float(p), 4) for o, p in zip(options, proba, strict=True)},
            "legend": dict(zip(options, self.criteria, strict=True)),
        }


@dataclass(frozen=True)
class Noul:
    """Is this statement true? Answered with a single probability that it is.

    Phrase the instruction so that a high number means yes. There is no separate
    confidence: 0.5 already says "I don't know", and a second number describing the
    spread of a two-outcome distribution would carry nothing the first does not.
    """

    instructions: str
    criteria: dict[str, str] | None = None
    hypothesis: str | None = None

    kind = "noul"

    def options(self):
        return [NO, YES]

    def render(self):
        lines = ["  type: noul — answer with a single number, the probability of yes"]
        if self.criteria:
            for key, word in (("true", "yes"), ("false", "no")):
                if self.criteria.get(key):
                    lines.append(f"    {word} means: {self.criteria[key]}")
        return "\n".join(lines)

    def answer(self, proba, options):
        return {"type": self.kind, "noul": round(float(proba[options.index(YES)]), 4)}


QUESTION_TYPES = {"choice": Choice, "score": Score, "noul": Noul}


def _confidence(proba):
    """How peaked a distribution is, on 0-1 (flat = 0). A shape statistic, not a calibrated
    probability of being right. The calibrated number is in `probabilities`.
    """
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    k = len(proba)
    if k < 2:
        return 1.0
    entropy = float(-(proba * np.log(proba)).sum())
    return round(float(1.0 - entropy / np.log(k)), 4)


def from_dict(blob):
    """Rebuild a question from its serialized form."""
    blob = dict(blob)
    kind = blob.pop("type")
    cls = QUESTION_TYPES.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown question type {kind!r} "
            f"(expected one of {sorted(QUESTION_TYPES)})"
        )
    return cls(**blob)


def to_dict(question):
    blob = {"type": question.kind, "instructions": question.instructions}
    if getattr(question, "hypothesis", None):
        blob["hypothesis"] = question.hypothesis
    if question.kind == "score":
        blob["criteria"] = list(question.criteria)
    elif question.kind == "choice" or question.criteria:
        blob["criteria"] = dict(question.criteria)
    return blob


def validate(questions):
    """Check a question map before anything expensive happens."""
    if not questions:
        raise ValueError("no questions given")
    for key, question in questions.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"question id {key!r} must be a plain identifier; ids name the columns "
                "of the answer table, and the teacher never sees them"
            )
        if question.kind not in QUESTION_TYPES:
            raise ValueError(f"question {key!r} has unknown type {question.kind!r}")
        if question.kind == "choice":
            if len(question.criteria) < 2:
                raise ValueError(f"choice question {key!r} needs at least 2 options")
            if len(question.criteria) > 255:
                raise ValueError(f"choice question {key!r} has more than 255 options")
    return questions


# ---------------------------------------------------------------- the prompt

PREAMBLE = """You answer typed questions about a document with calibrated probabilities.

Answer every question independently, judging only the document below. A question's
answer must not depend on any other question.

Report probabilities that mean what they say: across all the documents you would give
0.8 to, about 80% should turn out that way. Do not round to 0 or 1 to look decisive,
and do not retreat to 0.5 to look humble. If the document genuinely settles the
question, a confident number is the honest one.

Reply with only a JSON object holding one entry per question id, in this shape:

{"answers": {"<noul id>": <0-1>, "<choice or score id>": {"<option>": <0-1>, ...}, ...}}

A noul question takes a single number: the probability that the answer is yes. A choice
or score question takes a map covering exactly the options listed for it, summing to 1.
Use the option names exactly as they are written below.
"""


def default_header(instructions=None):
    """The free text above the rendered questions: the preamble plus the panel's instructions."""
    parts = [PREAMBLE]
    if instructions:
        parts.append(f"Context for every question:\n{instructions.strip()}\n")
    return "\n".join(parts)


def build_prompt(questions, instructions=None, header=None):
    """The system prompt for one set of questions, also used as the cache namespace.

    `header` replaces the preamble and instructions (it is what `Decisions.optimize()`
    writes). The questions are always rendered from the objects.
    """
    parts = [header.rstrip() + "\n" if header else default_header(instructions)]
    parts.append("Questions:")
    for key, question in questions.items():
        parts.append(f'- id "{key}"\n  instructions: {question.instructions}\n{question.render()}')
    return "\n".join(parts)


# ---------------------------------------------------------------- response parsing


def _coerce(obj):
    """Pull the first JSON object out of a model response."""
    if not obj:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(obj).strip()).strip()
    candidates = [text] if text.startswith("{") else []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                candidates.append(text[start : i + 1])
    for blob in candidates:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _distribution(raw, options):
    """Probability vector over `options` from whatever the model wrote: a map, a bare number
    for yes/no (true/false and 1/0 also accepted), or an option name. None when nothing
    usable is there.
    """
    lookup = {o.lower(): i for i, o in enumerate(options)}
    if options == [NO, YES]:
        lookup.update({"true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0})
    vec = np.zeros(len(options), dtype=float)

    if isinstance(raw, dict):
        probs = raw.get("probabilities", raw)
        if isinstance(probs, dict):
            found = False
            for key, value in probs.items():
                i = lookup.get(str(key).strip().lower())
                if i is not None and isinstance(value, (int, float)) and np.isfinite(value):
                    vec[i] += max(float(value), 0.0)
                    found = True
            if found and vec.sum() > 0:
                return vec / vec.sum()
        for field_name in ("noul", "probability", "p"):
            value = raw.get(field_name)
            if isinstance(value, (int, float)) and len(options) == 2:
                return _binary(float(value), options)
        for field_name in ("choice", "label", "answer", "score"):
            value = raw.get(field_name)
            i = lookup.get(str(value).strip().lower()) if value is not None else None
            if i is not None:
                vec[i] = 1.0
                return vec
    elif isinstance(raw, (int, float)) and len(options) == 2:
        return _binary(float(raw), options)
    elif isinstance(raw, str):
        i = lookup.get(raw.strip().lower())
        if i is not None:
            vec[i] = 1.0
            return vec
    return None


def _binary(p, options):
    p = float(np.clip(p, 0.0, 1.0))
    vec = np.zeros(2, dtype=float)
    vec[options.index(YES)] = p
    vec[options.index(NO)] = 1.0 - p
    return vec


def parse_answers(content, questions):
    """Response text -> {question id: probability vector}. Missing ids are simply absent."""
    parsed = _coerce(content)
    if parsed is None:
        return {}
    answers = parsed.get("answers") if isinstance(parsed.get("answers"), dict) else parsed
    if not isinstance(answers, dict):
        return {}
    by_id = {str(k).strip(): v for k, v in answers.items()}
    out = {}
    for key, question in questions.items():
        if key not in by_id:
            continue
        vec = _distribution(by_id[key], question.options())
        if vec is not None:
            out[key] = vec
    return out


def missing_ids(answers, questions):
    return [key for key in questions if key not in answers]


# ---------------------------------------------------------------- answer objects


@dataclass
class Answer:
    """One question's answer: the typed value plus the distribution behind it."""

    id: str
    kind: str
    probabilities: dict
    payload: dict = field(default_factory=dict)

    def __getattr__(self, name):
        try:
            return self.__dict__["payload"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self):
        head = {"choice": "choice", "score": "score", "noul": "noul"}[self.kind]
        return f"<Answer {self.id} {head}={self.payload.get(head)!r}>"


def make_answer(key, question, proba):
    options = question.options()
    payload = question.answer(np.asarray(proba, dtype=float), options)
    return Answer(
        id=key,
        kind=question.kind,
        probabilities=payload.get("probabilities", {}),
        payload=payload,
    )
