"""Asking a teacher many typed questions about one document in one call.

The document is the long part of the prompt, so one call per document is where the
saving is. The prompt tells the teacher to judge each question independently. How the
teacher produces a distribution is a backend: verbalized (asked for in the reply, the
only option for Claude), logprobs (OpenAI-compatible endpoints), or an ensemble.
"""

import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from shrewd import decide
from shrewd.teacher import _cache_lock, cached_complete, mark_prefix_cache, resolve_model

CORRECTION = (
    "That response was missing or unusable for these question ids: {missing}. "
    "Reply with only the JSON object, including every one of them."
)


# ---------------------------------------------------------------- backends


class VerbalizedBackend:
    """Ask the model to state the probabilities in its JSON reply. The only option for a
    provider without token logprobs. Frontier Claude models were well calibrated this way
    on balanced multi-class questions and overconfident on rare yes/no ones (BENCHMARKS.md).
    """

    name = "verbalized"

    def __init__(self, temperature=0.0):
        self.temperature = temperature

    def judge(self, state, questions, prompt, model, conn):
        content, cost = cached_complete(conn, model, prompt, state, self.temperature)
        answers = decide.parse_answers(content, questions)
        missing = decide.missing_ids(answers, questions)
        if missing:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": state},
                {"role": "assistant", "content": content or "(empty response)"},
                {"role": "user", "content": CORRECTION.format(missing=", ".join(missing))},
            ]
            retry, retry_cost = cached_complete(
                conn, model, prompt, state, self.temperature, attempt=1, messages=messages
            )
            cost += retry_cost
            answers = {**decide.parse_answers(retry, questions), **answers}
        return answers, cost


class LogprobBackend:
    """Read the distribution off the model's own token probabilities, one question per call
    with a single-token answer. Needs a provider that returns logprobs (OpenAI, vLLM,
    Together, Fireworks) and raises for Anthropic. Gives up the one-call-per-document saving.
    """

    name = "logprobs"

    def __init__(self, temperature=0.0, top_logprobs=20):
        self.temperature = temperature
        self.top_logprobs = top_logprobs

    def judge(self, state, questions, prompt, model, conn):
        total = 0.0
        answers = {}
        for key, question in questions.items():
            options = question.options()
            single = decide.build_prompt({key: question})
            single += (
                "\n\nReply with only the single option token that best fits, nothing else. "
                f"One of: {', '.join(options)}"
            )
            content, cost = _logprob_call(
                conn, model, single, state, options, self.temperature, self.top_logprobs
            )
            total += cost
            vec = decide._distribution(json.loads(content) if content else None, options)
            if vec is not None:
                answers[key] = vec
        return answers, total


class EnsembleBackend:
    """Average the distributions of several teachers. Unlike majority voting this keeps the
    disagreement (0.6 and 0.9 average to 0.75). `models` may mix providers.
    """

    name = "ensemble"

    def __init__(self, models, backend=None, weights=None):
        if len(models) < 2:
            raise ValueError("an ensemble needs at least 2 models")
        self.models = [resolve_model(m) for m in models]
        self.backend = backend or VerbalizedBackend()
        self.weights = None
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("weights must line up with models")
            total = float(sum(weights))
            self.weights = [float(w) / total for w in weights]

    def judge(self, state, questions, prompt, model, conn):
        pooled, total = {}, 0.0
        weights = self.weights or [1.0 / len(self.models)] * len(self.models)
        for member, weight in zip(self.models, weights, strict=True):
            answers, cost = self.backend.judge(state, questions, prompt, member, conn)
            total += cost
            for key, vec in answers.items():
                pooled.setdefault(key, []).append((weight, vec))
        out = {}
        for key, parts in pooled.items():
            stacked = np.average([v for _, v in parts], axis=0, weights=[w for w, _ in parts])
            out[key] = stacked / stacked.sum()
        return out, total


BACKENDS = {
    "verbalized": VerbalizedBackend,
    "logprobs": LogprobBackend,
    "ensemble": EnsembleBackend,
}


def resolve_backend(backend):
    """Accept a backend object, a name, or None (meaning the default)."""
    if backend is None:
        return VerbalizedBackend()
    if isinstance(backend, str):
        cls = BACKENDS.get(backend.strip().lower())
        if cls is None:
            raise ValueError(f"unknown backend {backend!r} (expected one of {sorted(BACKENDS)})")
        if cls is EnsembleBackend:
            raise ValueError("the ensemble backend needs models: EnsembleBackend([...])")
        return cls()
    if not hasattr(backend, "judge"):
        raise TypeError("a backend needs a .judge(state, questions, prompt, model, conn) method")
    return backend


def _logprob_call(conn, model, prompt, state, options, temperature, top_logprobs):
    """One constrained call, cached as the extracted distribution rather than as text."""
    import litellm

    from shrewd.teacher import _cache_get, _cache_put, _key

    key = _key(model, prompt, state, temperature, 0, 0)
    if conn is not None:
        hit = _cache_get(conn, key)
        if hit is not None:
            return hit, 0.0
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state},
    ]
    try:
        resp = litellm.completion(
            model=model,
            messages=mark_prefix_cache(model, messages),
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=4,
            drop_params=True,
        )
    except Exception as exc:  # noqa: BLE001 - provider errors vary too much to enumerate
        raise RuntimeError(
            f"{model} did not return logprobs ({type(exc).__name__}: {exc}). "
            "Anthropic models never do; use the verbalized backend for those, or point "
            "`teacher=` at an OpenAI-compatible endpoint (OpenAI, vLLM, Together, Fireworks)."
        ) from exc
    content = _extract_logprobs(resp, options)
    try:
        cost = litellm.completion_cost(completion_response=resp) or 0.0
    except Exception:
        cost = 0.0
    if conn is not None:
        _cache_put(conn, key, content, model)
    return content, cost


def _extract_logprobs(resp, options):
    """Renormalize the first token's top logprobs over the option tokens."""
    try:
        entries = resp.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        return json.dumps(None)
    scores = dict.fromkeys(options, -np.inf)
    for entry in entries:
        token = str(getattr(entry, "token", "")).strip().strip('"').lower()
        logprob = float(getattr(entry, "logprob", -np.inf))
        for option in options:
            # first-token match: "bil" resolves to "billing" when the tokenizer splits it
            if option.lower().startswith(token) and token:
                scores[option] = np.logaddexp(scores[option], logprob)
    values = np.array([scores[o] for o in options], dtype=float)
    if not np.isfinite(values).any():
        return json.dumps(None)
    values = np.where(np.isfinite(values), values, -np.inf)
    values = np.exp(values - values.max())
    return json.dumps({"probabilities": dict(zip(options, (values / values.sum()).tolist(),
                                                 strict=True))})


# ---------------------------------------------------------------- the labeling run


def probability_columns(questions):
    """Flat column names for the answer table: one per (question, option) pair."""
    return [f"{key}__{option}" for key, q in questions.items() for option in q.options()]


# ------------------------------------------------- the per-question answer cache

ANSWER_TABLE = (
    "CREATE TABLE IF NOT EXISTS answers "
    "(key TEXT PRIMARY KEY, probs TEXT, model TEXT, created_at TEXT)"
)


def _answer_key(model, key, question, state, header=None):
    """Cache identity of one answer: the model, the question itself, and the document (not the
    whole prompt), so adding a question to a panel re-uses every answer already paid for.
    Sound because the prompt asks for each question to be judged independently.
    """
    spec = json.dumps({"id": key, **decide.to_dict(question)}, sort_keys=True)
    # an optimized header changes what the teacher was asked, but answers under the default
    # header keep their old keys so existing caches stay valid
    parts = [model, spec, state] + ([header] if header else [])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _answer_get(conn, model, key, question, state, options, header=None):
    if conn is None:
        return None
    with _cache_lock:
        conn.execute(ANSWER_TABLE)
        row = conn.execute(
            "SELECT probs FROM answers WHERE key = ?",
            (_answer_key(model, key, question, state, header),),
        ).fetchone()
    if not row:
        return None
    try:
        values = np.array(json.loads(row[0]), dtype=float)
    except (ValueError, TypeError):
        return None
    if values.shape != (len(options),) or not np.isfinite(values).all():
        return None
    total = values.sum()
    return values / total if total > 0 else None


def _answer_put(conn, model, key, question, state, vec, header=None):
    if conn is None:
        return
    with _cache_lock:
        conn.execute(ANSWER_TABLE)
        conn.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
            (
                _answer_key(model, key, question, state, header),
                json.dumps([round(float(v), 6) for v in vec]),
                model,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()


def judge_texts(
    texts,
    questions,
    model,
    instructions=None,
    backend=None,
    conn=None,
    concurrency=8,
    desc="judging",
    header=None,
):
    """Ask every question about every text. Returns (DataFrame, total_cost_usd).

    The frame is one row per input text with a `text` column and one probability column
    per (question, option) pair, so it reads as a plain CSV and reloads without a schema.
    A question the teacher never answered for a row is left as NaN rather than guessed at.
    """
    decide.validate(questions)
    backend = resolve_backend(backend)
    model = resolve_model(model)
    unique = list(dict.fromkeys(texts))
    results, total_cost = {}, 0.0

    def work(state):
        # ask only about what is not already answered for this document, so a panel
        # that gains a question pays for that question and nothing else
        cached, missing = {}, {}
        for key, question in questions.items():
            hit = _answer_get(conn, model, key, question, state, question.options(), header)
            if hit is None:
                missing[key] = question
            else:
                cached[key] = hit
        if not missing:
            return state, cached, 0.0
        prompt = decide.build_prompt(missing, instructions, header=header)
        answers, cost = backend.judge(state, missing, prompt, model, conn)
        for key, vec in answers.items():
            _answer_put(conn, model, key, questions[key], state, vec, header)
        return state, {**cached, **answers}, cost

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, state) for state in unique]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                state, answers, cost = future.result()
                results[state] = answers
                total_cost += cost
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    columns = probability_columns(questions)
    rows = []
    for state in texts:
        answers = results.get(state, {})
        row = {"text": state}
        for key, question in questions.items():
            vec = answers.get(key)
            for i, option in enumerate(question.options()):
                row[f"{key}__{option}"] = float(vec[i]) if vec is not None else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["text", *columns])

    unanswered = frame[columns].isna().all(axis=1).sum()
    if unanswered:
        warnings.warn(
            f"{unanswered} of {len(frame)} rows came back with no usable answer and are "
            "left as NaN; they are dropped from training rather than counted as a verdict",
            stacklevel=2,
        )
    return frame, total_cost


def answers_from_row(row, questions):
    """One row of the answer table back into typed Answer objects."""
    out = {}
    for key, question in questions.items():
        options = question.options()
        vec = np.array([row.get(f"{key}__{o}", np.nan) for o in options], dtype=float)
        if np.isnan(vec).any() or vec.sum() <= 0:
            continue
        out[key] = decide.make_answer(key, question, vec / vec.sum())
    return out
