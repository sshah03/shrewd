import hashlib
import json
import random
import re
import sqlite3
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import litellm
import pandas as pd
from tqdm import tqdm

from shrewd.data import INVALID

TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
)
BACKOFF_SECONDS = (2, 8, 30)

DEFAULT_MODELS = {
    "anthropic": "anthropic/claude-opus-4-8",
    "claude": "anthropic/claude-opus-4-8",
    "openai": "openai/gpt-5.1",
}


def resolve_model(model):
    """Expand a provider alias ("anthropic", "claude", "openai") to that provider's
    default model. Anything else (any litellm model string) passes through unchanged."""
    return DEFAULT_MODELS.get(model.strip().lower(), model)


def _temperature(votes):
    # single vote asks for determinism, ballots for diversity. But frontier Claude
    # models reject temperature outright (fixed at 1.0, drop_params strips it), so
    # there this only namespaces the cache. haiku-class and OpenAI teachers honor it
    return 0.0 if votes == 1 else 0.7


_cache_lock = threading.Lock()
_cost_warned = set()
_cacheable_models = {}


def _supports_prompt_cache(model):
    """Does the provider bill cached prefix reads at a discount? litellm's price table
    knows (`cache_read_input_token_cost`), and an unknown model is treated as not."""
    if model not in _cacheable_models:
        try:
            _cacheable_models[model] = bool(
                litellm.get_model_info(model).get("cache_read_input_token_cost")
            )
        except Exception:
            _cacheable_models[model] = False
    return _cacheable_models[model]


def mark_prefix_cache(model, messages):
    """Ask the provider to cache the system prompt. Every shrewd call shares one system
    prompt across the documents of a run (the classification prompt, or the decision
    header plus questions) and only the user turn varies, so the prefix is read from
    cache at a fraction of the input price (10% on Claude) once it is longer than the
    provider's minimum (512-1,024 tokens). Providers without the flag get the messages
    unchanged. OpenAI caches long prefixes on its own."""
    if not messages or messages[0].get("role") != "system" or not _supports_prompt_cache(model):
        return messages
    system = messages[0]
    if not isinstance(system.get("content"), str):
        return messages
    block = {"type": "text", "text": system["content"], "cache_control": {"type": "ephemeral"}}
    return [{**system, "content": [block]}, *messages[1:]]


def open_cache(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS calls "
        "(key TEXT PRIMARY KEY, response TEXT, model TEXT, created_at TEXT)"
    )
    conn.commit()
    return conn


def _key(model, prompt, text, temperature, vote_index, attempt):
    raw = "|".join([model, prompt, text, f"{temperature:g}", str(vote_index), str(attempt)])
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(conn, key):
    with _cache_lock:
        row = conn.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(conn, key, response, model):
    with _cache_lock:
        conn.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?)",
            (key, response, model, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def complete(model, messages, temperature=None):
    """One litellm call with retries on transient errors. Returns (content, cost_usd)."""
    litellm.suppress_debug_info = True  # set at call time so importing shrewd mutates nothing
    kwargs = {} if temperature is None else {"temperature": temperature}
    for delay in (*BACKOFF_SECONDS, None):
        try:
            # drop_params: newer models (Sonnet 5, Opus 4.8+, Fable 5) reject sampling params
            resp = litellm.completion(
                model=model, messages=mark_prefix_cache(model, messages), drop_params=True,
                **kwargs,
            )
            break
        except TRANSIENT_ERRORS:
            if delay is None:
                raise
            time.sleep(delay + random.uniform(0, delay / 4))
    content = resp.choices[0].message.content or ""
    try:
        cost = litellm.completion_cost(completion_response=resp) or 0.0
    except Exception:
        if model not in _cost_warned:
            warnings.warn(
                f"litellm has no price data for {model}; costs will show as $0.00",
                stacklevel=2,
            )
            _cost_warned.add(model)
        cost = 0.0
    return content, cost


def cached_complete(conn, model, prompt, text, temperature, vote_index=0, attempt=0, messages=None):
    """`complete` with a sqlite cache in front. Cache hits make no API call."""
    if messages is None:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
    key = _key(model, prompt, text, temperature, vote_index, attempt)
    if conn is not None:
        hit = _cache_get(conn, key)
        if hit is not None:
            return hit, 0.0
    content, cost = complete(model, messages, temperature)
    if conn is not None:
        _cache_put(conn, key, content, model)
    return content, cost


def parse_label(content, labels):
    """Leniently pull one of `labels` out of a model response. None if it can't be done."""
    if not content:
        return None
    canon = {label.lower(): label for label in labels}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip()).strip()
    candidates = ([text] if text.startswith("{") else []) + re.findall(r"\{[^{}]*\}", text)
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("label"), str):
            hit = canon.get(obj["label"].strip().lower())
            if hit:
                return hit
    return canon.get(text.strip("\"'` .").lower())


def classify(text, prompt, model, labels, temperature=0.0, conn=None, vote_index=0):
    """Label one text with the teacher. Returns (label_or_None, cost_usd).

    On an unparseable response, retries once with a correction message appended,
    then gives up (None).
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    content, cost = cached_complete(conn, model, prompt, text, temperature, vote_index, 0, messages)
    label = parse_label(content, labels)
    if label is not None:
        return label, cost
    correction = (
        'Reply with only JSON {"label": "<label>"} where <label> is exactly one of: '
        + ", ".join(labels)
    )
    messages = messages + [
        {"role": "assistant", "content": content or "(empty response)"},
        {"role": "user", "content": correction},
    ]
    content, retry_cost = cached_complete(
        conn, model, prompt, text, temperature, vote_index, 1, messages
    )
    return parse_label(content, labels), cost + retry_cost


def label_texts(texts, prompt, model, labels, votes=1, conn=None, concurrency=8, desc="labeling"):
    """Label many texts concurrently, aggregating votes.

    Returns (DataFrame[text, label, confidence], total_cost_usd). Unresolvable items
    get the INVALID label and confidence 0.0.
    """
    temperature = _temperature(votes)
    unique = list(dict.fromkeys(texts))  # duplicates get one set of calls and one verdict
    ballots = [[None] * votes for _ in unique]
    total_cost = 0.0

    def work(i, v):
        label, cost = classify(unique[i], prompt, model, labels, temperature, conn, vote_index=v)
        return i, v, label, cost

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, i, v) for i in range(len(unique)) for v in range(votes)]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                i, v, label, cost = future.result()
                ballots[i][v] = label
                total_cost += cost
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    verdicts = {}
    for text, votes_cast in zip(unique, ballots, strict=True):
        valid = [label for label in votes_cast if label is not None]
        if not valid:
            verdicts[text] = (INVALID, 0.0)
        else:
            winner, n = Counter(valid).most_common(1)[0]
            # two decimals, so 2-of-3 agreement (0.67) survives a min_confidence=0.67 cutoff
            verdicts[text] = (winner, 1.0 if votes == 1 else round(n / votes, 2))
    rows = [(text, *verdicts[text]) for text in texts]
    return pd.DataFrame(rows, columns=["text", "label", "confidence"]), total_cost


def estimate_calls(texts, prompt, model, votes, conn):
    """Count not-yet-cached calls for a labeling run and roughly price them.

    Returns (n_uncached_calls, cost_usd_or_None). Cost is None when litellm has
    no price data for the model. Ignores correction retries since this is an estimate.
    """
    temperature = _temperature(votes)
    n = sum(
        1
        for text in dict.fromkeys(texts)
        for v in range(votes)
        if conn is None or _cache_get(conn, _key(model, prompt, text, temperature, v, 0)) is None
    )
    if n == 0:
        return 0, 0.0
    avg_text_chars = sum(len(t) for t in texts) / len(texts)
    prompt_tokens = int(n * (len(prompt) + avg_text_chars) / 4)
    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=12 * n
        )
        return n, input_cost + output_cost
    except Exception:
        return n, None
