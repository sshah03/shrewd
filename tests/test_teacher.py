import json
import warnings

import litellm
import pytest
from conftest import LABELS

from shrewd import teacher

CLASSES = list(LABELS)


class TestParseLabel:
    def test_plain_json(self):
        assert teacher.parse_label('{"label": "bug"}', CLASSES) == "bug"

    def test_code_fences(self):
        assert teacher.parse_label('```json\n{"label": "bug"}\n```', CLASSES) == "bug"

    def test_case_insensitive(self):
        assert teacher.parse_label('{"label": "BILLING"}', CLASSES) == "billing"

    def test_extra_keys_and_prose(self):
        content = 'Sure! Here you go: {"label": "cancellation", "why": "asked to cancel"}'
        assert teacher.parse_label(content, CLASSES) == "cancellation"

    def test_bare_label(self):
        assert teacher.parse_label("bug", CLASSES) == "bug"
        assert teacher.parse_label('"Billing".', CLASSES) == "billing"

    def test_garbage(self):
        assert teacher.parse_label("I am not sure about this one", CLASSES) is None
        assert teacher.parse_label('{"label": "unknown"}', CLASSES) is None
        assert teacher.parse_label("", CLASSES) is None
        assert teacher.parse_label("{broken json", CLASSES) is None


def classify(text, **kwargs):
    defaults = {"prompt": "p", "model": "test/fake-model", "labels": CLASSES}
    return teacher.classify(text, **{**defaults, **kwargs})


def test_classify_happy_path(fake_teacher):
    label, cost = classify("I was charged twice")
    assert label == "billing"
    assert fake_teacher.calls == 1
    assert cost == pytest.approx(0.001)


def test_classify_correction_retry(fake_teacher):
    fake_teacher.respond = (
        lambda messages: "no idea" if len(messages) == 2 else json.dumps({"label": "bug"})
    )
    label, _ = classify("whatever")
    assert label == "bug"
    assert fake_teacher.calls == 2


def test_classify_gives_up(fake_teacher):
    fake_teacher.respond = lambda messages: "still no idea"
    label, _ = classify("whatever")
    assert label is None
    assert fake_teacher.calls == 2


def test_transient_errors_retried(fake_teacher):
    original = fake_teacher.respond

    def flaky(messages):
        if fake_teacher.calls <= 2:
            raise litellm.RateLimitError("slow down", llm_provider="test", model="m")
        return original(messages)

    fake_teacher.respond = flaky
    label, _ = classify("need a refund")
    assert label == "billing"
    assert fake_teacher.calls == 3


def test_transient_errors_exhausted(fake_teacher):
    def always_fails(messages):
        raise litellm.ServiceUnavailableError("down", llm_provider="test", model="m")

    fake_teacher.respond = always_fails
    with pytest.raises(litellm.ServiceUnavailableError):
        classify("anything")
    assert fake_teacher.calls == 4  # first try + 3 retries


def test_cache_prevents_repeat_calls(fake_teacher, tmp_path):
    conn = teacher.open_cache(tmp_path / "cache.db")
    texts = ["charge me less", "app crashed", "cancel it"]
    first, _ = teacher.label_texts(texts, "p", "m", CLASSES, conn=conn)
    assert fake_teacher.calls == 3
    second, _ = teacher.label_texts(texts, "p", "m", CLASSES, conn=conn)
    assert fake_teacher.calls == 3
    assert first.equals(second)


def test_cache_hit_costs_nothing(fake_teacher, tmp_path):
    conn = teacher.open_cache(tmp_path / "cache.db")
    _, first_cost = teacher.label_texts(["hello"], "p", "m", CLASSES, conn=conn)
    _, second_cost = teacher.label_texts(["hello"], "p", "m", CLASSES, conn=conn)
    assert first_cost > 0
    assert second_cost == 0


def test_votes_confidence(fake_teacher):
    answers = iter(["billing", "billing", "bug"])
    fake_teacher.respond = lambda messages: json.dumps({"label": next(answers)})
    labeled, _ = teacher.label_texts(["x"], "p", "m", CLASSES, votes=3, concurrency=1)
    assert labeled.iloc[0]["label"] == "billing"
    assert labeled.iloc[0]["confidence"] == round(2 / 3, 2)
    # 2-of-3 agreement must survive the documented min_confidence=0.67 cutoff
    assert labeled.iloc[0]["confidence"] >= 0.67


def test_single_vote_confidence_is_one(fake_teacher):
    labeled, _ = teacher.label_texts(["the app crashes"], "p", "m", CLASSES)
    assert labeled.iloc[0]["label"] == "bug"
    assert labeled.iloc[0]["confidence"] == 1.0


def test_invalid_vote_counts_against_agreement(fake_teacher):
    # vote 0 parses; votes 1 and 2 stay invalid through their correction retries
    answers = iter(["billing", "nonsense", "nonsense", "nonsense", "nonsense"])
    fake_teacher.respond = lambda messages: json.dumps({"label": next(answers)})
    labeled, _ = teacher.label_texts(["x"], "p", "m", CLASSES, votes=3, concurrency=1)
    assert labeled.iloc[0]["label"] == "billing"
    assert labeled.iloc[0]["confidence"] == round(1 / 3, 2)


def test_all_votes_invalid(fake_teacher):
    fake_teacher.respond = lambda messages: "garbage"
    labeled, _ = teacher.label_texts(["x"], "p", "m", CLASSES)
    assert labeled.iloc[0]["label"] == "__invalid__"
    assert labeled.iloc[0]["confidence"] == 0.0


def test_duplicate_texts_get_one_call_and_one_verdict(fake_teacher):
    labeled, _ = teacher.label_texts(
        ["same text", "other text", "same text"], "p", "m", CLASSES
    )
    assert fake_teacher.calls == 2
    assert len(labeled) == 3
    assert labeled.iloc[0].equals(labeled.iloc[2])


def test_estimate_counts_only_uncached(fake_teacher, tmp_path):
    conn = teacher.open_cache(tmp_path / "cache.db")
    texts = ["one text", "another text"]
    n, _ = teacher.estimate_calls(texts, "p", "m", 3, conn)
    assert n == 6
    teacher.label_texts(texts, "p", "m", CLASSES, votes=3, conn=conn)
    n, cost = teacher.estimate_calls(texts, "p", "m", 3, conn)
    assert n == 0
    assert cost == 0.0
    assert fake_teacher.calls == 6


def test_drop_params_sent_per_call(fake_teacher):
    # models that reject sampling params (Sonnet 5+) must not break classify(),
    # and importing shrewd must not mutate litellm globals for the host app
    assert litellm.drop_params is not True
    fake_teacher.respond = lambda messages: "ok"
    teacher.complete("m", [{"role": "user", "content": "x"}], temperature=0.0)
    assert fake_teacher.last_kwargs.get("drop_params") is True


def test_cost_unknown_warns_once(fake_teacher, monkeypatch):
    def boom(completion_response):
        raise Exception("no price")

    monkeypatch.setattr("shrewd.teacher.litellm.completion_cost", boom)
    monkeypatch.setattr(teacher, "_cost_warned", set())
    fake_teacher.respond = lambda messages: "ok"
    with pytest.warns(UserWarning, match="no price data"):
        _, cost = teacher.complete("m", [{"role": "user", "content": "x"}])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a repeat of the same model must stay quiet
        _, cost2 = teacher.complete("m", [{"role": "user", "content": "x"}])
    with pytest.warns(UserWarning, match="no price data"):
        teacher.complete("m2", [{"role": "user", "content": "x"}])  # a new model warns
    assert cost == 0.0 and cost2 == 0.0


def test_empty_response_content(fake_teacher):
    fake_teacher.respond = lambda messages: None  # provider returned no content
    label, _ = classify("x")
    assert label is None
    assert fake_teacher.calls == 2
