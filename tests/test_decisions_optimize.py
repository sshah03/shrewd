import json
import types

import numpy as np
import pytest
from conftest import QUESTIONS, fake_answer, make_docs

from shrewd import decide
from shrewd.decisions import Decisions
from shrewd.optimize import decisions_evaluator


def _fake_gepa(monkeypatch, best_header="OPTIMIZED HEADER: be honest about probabilities."):
    """Stand in for gepa.optimize: return a result whose best candidate is our header."""
    calls = {}

    def optimize(**kwargs):
        calls.update(kwargs)
        # exercise the adapter once so the task/evaluator plumbing is actually run
        batch = kwargs["trainset"][:3]
        kwargs["adapter"].evaluate(batch, {"system_prompt": best_header})
        return types.SimpleNamespace(
            best_candidate={"system_prompt": best_header},
            val_aggregate_scores=[0.61, 0.74],
            best_idx=1,
            total_metric_calls=len(batch),
            candidates=[{"system_prompt": kwargs["seed_candidate"]["system_prompt"]},
                        {"system_prompt": best_header}],
        )

    monkeypatch.setattr("shrewd.optimize.gepa.optimize", optimize)
    return calls


# ---------------------------------------------------------------- the objective


def test_evaluator_scores_a_proper_rule_not_accuracy():
    ev = decisions_evaluator({"angry": QUESTIONS["angry"]})
    gold = {"answer": {"angry": 1}}
    confident_right = ev(gold, json.dumps({"answers": {"angry": 0.95}}))
    hedged_right = ev(gold, json.dumps({"answers": {"angry": 0.6}}))
    confident_wrong = ev(gold, json.dumps({"answers": {"angry": 0.05}}))
    assert confident_right.score > hedged_right.score > confident_wrong.score
    assert confident_right.score == pytest.approx(1 - (0.05**2 + 0.05**2) / 2)
    assert "gold `yes`" in confident_wrong.feedback


def test_evaluator_penalizes_a_missing_question_and_explains():
    ev = decisions_evaluator(QUESTIONS)
    gold = {"answer": {"department": 0, "angry": 1, "severity": 2}}
    out = ev(gold, json.dumps({"answers": {"angry": 0.9}}))
    assert out.score == pytest.approx(np.mean([0.0, 1 - (0.1**2 + 0.1**2) / 2, 0.0]))
    assert "`department`: no usable answer" in out.feedback
    assert "`severity`: no usable answer" in out.feedback


def test_evaluator_skips_questions_without_gold():
    ev = decisions_evaluator(QUESTIONS)
    out = ev({"answer": {"department": -1, "angry": -1, "severity": -1}}, "{}")
    assert out.score == 0.0 and "no gold" in out.feedback


def test_evaluator_feedback_carries_the_option_description():
    ev = decisions_evaluator({"department": QUESTIONS["department"]})
    out = ev({"answer": {"department": 0}},
             json.dumps({"answers": {"department": {"bug": 0.9, "billing": 0.1}}}))
    assert "billing means: charges, invoices, refunds" in out.feedback


# ---------------------------------------------------------------- header plumbing


def test_build_prompt_header_replaces_preamble_but_keeps_questions():
    default = decide.build_prompt(QUESTIONS, instructions="tickets")
    custom = decide.build_prompt(QUESTIONS, instructions="tickets", header="MY HEADER")
    assert default.startswith(decide.PREAMBLE.splitlines()[0])
    assert custom.startswith("MY HEADER")
    assert "Context for every question" not in custom
    for key in QUESTIONS:
        assert f'id "{key}"' in custom  # questions are structural, never dropped


def test_optimize_writes_a_header_that_judge_then_uses(tmp_path, decision_teacher, monkeypatch):
    docs = make_docs(120)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model",
                  instructions="Customer tickets.")
    d.add_seed(docs.iloc[:60], test_frac=0.4)
    with pytest.raises(FileNotFoundError, match="add_seed"):
        Decisions(tmp_path / "q", questions=QUESTIONS, teacher="test/fake-model").optimize()

    seen_headers = []
    original = decision_teacher.respond
    decision_teacher.respond = lambda messages: (
        seen_headers.append(messages[0]["content"]), original(messages))[1]

    calls = _fake_gepa(monkeypatch)
    log = d.optimize(budget=50)
    assert (d.dir / "prompt_header.txt").read_text().startswith("OPTIMIZED HEADER")
    assert log["best_score"] == 0.74
    assert calls["seed_candidate"]["system_prompt"] == decide.default_header("Customer tickets.")
    # the adapter's task call composed the full prompt: header + rendered questions
    assert any(h.startswith("OPTIMIZED HEADER") and 'id "angry"' in h for h in seen_headers)

    seen_headers.clear()
    d.judge(docs.iloc[60:][["text"]])
    assert seen_headers and all(h.startswith("OPTIMIZED HEADER") for h in seen_headers)


def test_optimize_skips_when_a_header_exists(tmp_path, decision_teacher, capsys):
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    (d.dir / "prompt_header.txt").write_text("hand-written header")
    assert d.optimize() is None
    assert "skipping" in capsys.readouterr().out
    assert (d.dir / "prompt_header.txt").read_text() == "hand-written header"


def test_a_new_header_does_not_reuse_answers_from_the_old_prompt(tmp_path, decision_teacher):
    """The per-question cache is keyed on the header: an optimized prompt asks again."""
    docs = make_docs(80)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:40], test_frac=0.4)
    d.judge(docs.iloc[40:][["text"]])
    first = decision_teacher.calls
    d.judge(docs.iloc[40:][["text"]], overwrite=True)
    assert decision_teacher.calls == first                      # same header: all cached
    (d.dir / "prompt_header.txt").write_text("a different framing of the task")
    d.judge(docs.iloc[40:][["text"]], overwrite=True)
    assert decision_teacher.calls == first + 40                 # new header: asked again


def test_optimize_has_no_path_to_the_test_set():
    from pathlib import Path

    from shrewd import optimize

    assert "seed_test" not in Path(optimize.__file__).read_text()


def test_fake_teacher_answers_survive_the_evaluator():
    """Sanity: the conftest teacher scores well under the proper-rule objective."""
    ev = decisions_evaluator(QUESTIONS)
    text = "ticket 3: the refund problem again this is unacceptable"
    gold = {"answer": {"department": 0, "angry": 1, "severity": -1}}
    out = ev(gold, fake_answer(text, QUESTIONS))
    assert out.score > 0.95


def test_reflection_is_told_the_word_budget(tmp_path, decision_teacher, monkeypatch):
    """The header rides along on every teacher call for the life of the project, so its
    length is a recurring cost; the reflection model is asked to respect a budget."""
    docs = make_docs(120)
    d = Decisions(tmp_path / "p", questions=QUESTIONS, teacher="test/fake-model")
    d.add_seed(docs.iloc[:60], test_frac=0.4)
    seen = []
    original = decision_teacher.respond
    def spy(messages):
        if len(messages) == 1:                       # the reflection call: one user turn
            seen.append(messages[0]["content"])
            return "short header"
        return original(messages)

    decision_teacher.respond = spy

    def optimize(**kwargs):
        kwargs["reflection_lm"]("Please improve this prompt.")          # exercise the wrapper
        return types.SimpleNamespace(
            best_candidate={"system_prompt": "short header"}, val_aggregate_scores=[0.5, 0.6],
            best_idx=1, total_metric_calls=1,
            candidates=[{"system_prompt": kwargs["seed_candidate"]["system_prompt"]},
                        {"system_prompt": "short header"}])

    monkeypatch.setattr("shrewd.optimize.gepa.optimize", optimize)
    log = d.optimize(budget=10, max_header_words=120)
    assert any("at most 120 words" in m for m in seen)
    assert log["max_header_words"] == 120
    assert log["candidates"][1]["words"] == 2

