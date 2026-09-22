from concurrent.futures import ThreadPoolExecutor

import gepa
import numpy as np
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, EvaluationResult
from gepa.core.adapter import EvaluationBatch
from gepa.utils.stop_condition import ScoreThresholdStopper

from shrewd import teacher


class _ParallelAdapter(DefaultAdapter):
    """gepa's default adapter, with the task model called concurrently.

    Given a callable, the stock adapter runs one metric call at a time, which makes a
    400-call optimize() a coffee break. The reflective dataset it builds is unchanged.
    """

    def __init__(self, model_fn, evaluator, concurrency):
        super().__init__(model=model_fn, evaluator=evaluator)
        self.concurrency = concurrency

    def evaluate(self, batch, candidate, capture_traces=False):
        system = next(iter(candidate.values()))
        requests = [
            [{"role": "system", "content": system}, {"role": "user", "content": d["input"]}]
            for d in batch
        ]
        with ThreadPoolExecutor(self.concurrency) as pool:
            responses = list(pool.map(self.model, requests))
        outputs, scores = [], []
        trajectories = [] if capture_traces else None
        for d, response in zip(batch, responses, strict=True):
            result = self.evaluator(d, response)
            outputs.append({"full_assistant_response": response})
            scores.append(result.score)
            if trajectories is not None:
                trajectories.append(
                    {"data": d, "full_assistant_response": response, "feedback": result.feedback}
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)


def build_seed_prompt(instructions, labels):
    lines = [instructions, "", "Classify the input into exactly one of these labels:"]
    for name, description in labels.items():
        lines.append(f"- {name}: {description}" if description else f"- {name}")
    lines += ["", 'Reply with JSON: {"label": "<label>"}']
    return "\n".join(lines)


def run_gepa(dev_df, seed_prompt, model, labels, budget, target, reflection_model, conn, seed,
             concurrency=8, minibatch=10):
    """Optimize the system prompt with GEPA. Returns (best_prompt, log_dict, cost_usd).

    Task-model calls go through the sqlite cache. Reflection calls are uncached but their
    cost is tracked. `minibatch` is how many dev rows each reflection step sees. gepa's
    default of 3 is usually all-correct with a strong teacher, so the step is skipped.
    """
    costs = []

    def task_lm(messages):
        content, cost = teacher.cached_complete(
            conn, model, prompt=messages[0]["content"], text=messages[1]["content"],
            temperature=0.0,
        )
        costs.append(cost)
        return content

    def reflection_lm(prompt):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        content, cost = teacher.complete(reflection_model, messages)
        costs.append(cost)
        return content

    def evaluator(data, response):
        predicted = teacher.parse_label(response, labels)
        if predicted is None:
            return EvaluationResult(
                0.0, 'output was not one of the allowed labels; reply with JSON {"label": ...}'
            )
        gold = data["answer"]
        if predicted == gold:
            return EvaluationResult(1.0, "correct")
        feedback = f"predicted `{predicted}`, gold is `{gold}`"
        if labels[gold]:
            feedback += f". {gold} means: {labels[gold]}"
        return EvaluationResult(0.0, feedback)

    def to_instances(df):
        return [
            {"input": text, "answer": label, "additional_context": {}}
            for text, label in zip(df["text"], df["label"], strict=True)
        ]

    result = gepa.optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=to_instances(dev_df),
        valset=to_instances(dev_df),
        adapter=_ParallelAdapter(task_lm, evaluator, concurrency),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=minibatch,
        max_metric_calls=budget,
        stop_callbacks=[ScoreThresholdStopper(target)] if target is not None else None,
        track_best_outputs=False,
        seed=seed,
    )

    log = {
        "baseline_score": result.val_aggregate_scores[0],
        "best_score": result.val_aggregate_scores[result.best_idx],
        "total_metric_calls": result.total_metric_calls,
        "candidates": [
            {"val_score": score, "prompt": candidate["system_prompt"]}
            for candidate, score in zip(
                result.candidates, result.val_aggregate_scores, strict=True
            )
        ],
    }
    return result.best_candidate["system_prompt"], log, sum(costs)


# ---------------------------------------------------------------- decision panels


def decisions_evaluator(questions):
    """Score one teacher response to a panel against gold, with feedback for reflection. The
    score is 1 - Brier/2 per answered question, a proper scoring rule, so the prompt is
    pushed toward accurate probabilities rather than rounding to 0 and 1.
    """
    from shrewd import decide

    def evaluator(data, response):
        answers = decide.parse_answers(response, questions)
        gold = data["answer"]
        scores, notes = [], []
        for key, question in questions.items():
            g = gold.get(key, -1)
            if g < 0:
                continue
            options = question.options()
            vec = answers.get(key)
            if vec is None:
                scores.append(0.0)
                notes.append(f"`{key}`: no usable answer")
                continue
            onehot = np.zeros(len(options))
            onehot[g] = 1.0
            brier = float(((vec - onehot) ** 2).sum())
            scores.append(1.0 - brier / 2.0)
            if brier > 0.5:
                given = options[int(vec.argmax())]
                note = (f"`{key}`: said `{given}` p={vec.max():.2f}, "
                        f"gold `{options[g]}` p={vec[g]:.2f}")
                if question.kind == "choice" and question.criteria.get(options[g]):
                    note += f" ({options[g]} means: {question.criteria[options[g]]})"
                notes.append(note)
        if not scores:
            return EvaluationResult(0.0, "no gold for this row")
        feedback = "good" if not notes else "; ".join(notes[:3])
        return EvaluationResult(float(np.mean(scores)), feedback)

    return evaluator


def run_gepa_decisions(dev_df, questions, seed_header, model, budget, target, reflection_model,
                       conn, seed, concurrency=8, minibatch=10, max_header_words=None):
    """GEPA over the prompt header of a decision panel. Returns (header, log, cost_usd).

    Only the free text above the questions is optimized. The questions are rendered from the
    objects on every call. `max_header_words` asks the reflection model to keep the header
    under a length, since it is sent with every document afterwards.
    """
    from shrewd import decide
    from shrewd.decisions import gold_frame

    costs = []
    gold = gold_frame(dev_df, questions)
    keys = list(questions)

    def task_lm(messages):
        full = decide.build_prompt(questions, header=messages[0]["content"])
        content, cost = teacher.cached_complete(
            conn, model, prompt=full, text=messages[1]["content"], temperature=0.0
        )
        costs.append(cost)
        return content

    constraint = (
        f"\n\nHard constraint on your rewrite: the new prompt text must be at most "
        f"{max_header_words} words. A few sharp rules that fix the observed mistakes beat "
        f"an exhaustive list; do not enumerate every label."
    ) if max_header_words else ""

    def reflection_lm(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt + constraint}]
        else:
            messages = [dict(m) for m in prompt]
            messages[-1]["content"] = str(messages[-1].get("content", "")) + constraint
        content, cost = teacher.complete(reflection_model, messages)
        costs.append(cost)
        return content

    def to_instances(df):
        return [
            {"input": str(text), "answer": {k: int(gold[i, j]) for j, k in enumerate(keys)},
             "additional_context": {}}
            for i, text in enumerate(df["text"])
        ]

    result = gepa.optimize(
        seed_candidate={"system_prompt": seed_header},
        trainset=to_instances(dev_df),
        valset=to_instances(dev_df),
        adapter=_ParallelAdapter(task_lm, decisions_evaluator(questions), concurrency),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=minibatch,
        max_metric_calls=budget,
        stop_callbacks=[ScoreThresholdStopper(target)] if target is not None else None,
        track_best_outputs=False,
        seed=seed,
    )
    log = {
        "objective": "mean over questions of 1 - Brier/2, on the dev split",
        "baseline_score": result.val_aggregate_scores[0],
        "best_score": result.val_aggregate_scores[result.best_idx],
        "total_metric_calls": result.total_metric_calls,
        "max_header_words": max_header_words,
        "candidates": [
            {"val_score": score, "words": len(candidate["system_prompt"].split()),
             "header": candidate["system_prompt"]}
            for candidate, score in zip(result.candidates, result.val_aggregate_scores, strict=True)
        ],
    }
    return result.best_candidate["system_prompt"], log, sum(costs)
