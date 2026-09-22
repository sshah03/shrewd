import json
import threading
import types

import pandas as pd
import pytest

from shrewd.decide import Choice, Noul, Score

LABELS = {
    "billing": "charges, invoices, refunds, payment problems",
    "bug": "something in the product is broken or misbehaving",
    "cancellation": "wants to cancel, pause, or downgrade",
    "other": "anything that fits none of the above",
}

KEYWORDS = [
    ("charge", "billing"),
    ("refund", "billing"),
    ("invoice", "billing"),
    ("crash", "bug"),
    ("error", "bug"),
    ("broken", "bug"),
    ("cancel", "cancellation"),
    ("downgrade", "cancellation"),
]

SNIPPETS = {
    "billing": [
        "I was charged twice for order",
        "need a refund for invoice",
        "the charge on my receipt looks wrong",
    ],
    "bug": [
        "the app crashes when I open settings",
        "seeing an error on the dashboard page",
        "export has been broken since the update",
    ],
    "cancellation": [
        "please cancel my subscription",
        "I want to downgrade my plan",
        "cancel my account today please",
    ],
    "other": [
        "how do I change my avatar",
        "what are your office hours",
        "loving the product so far just saying",
    ],
}


def keyword_label(text):
    lowered = text.lower()
    for keyword, label in KEYWORDS:
        if keyword in lowered:
            return label
    return "other"


def make_seed(n_per_class=30):
    rows = [
        (f"{SNIPPETS[label][i % 3]} #{i}", label)
        for label in LABELS
        for i in range(n_per_class)
    ]
    return pd.DataFrame(rows, columns=["text", "label"])


def make_pool(n_per_class=10):
    rows = [
        f"{SNIPPETS[label][i % 3]} pool item {i}"
        for label in LABELS
        for i in range(n_per_class)
    ]
    return pd.DataFrame({"text": rows})


def response(content):
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class FakeTeacher:
    """Stands in for litellm.completion. Set `respond` to change behavior."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()
        self.respond = lambda messages: json.dumps(
            {"label": keyword_label(messages[1]["content"])}
        )

    def __call__(self, model, messages, **kwargs):
        with self._lock:
            self.calls += 1
            self.last_kwargs = kwargs
            return response(self.respond(messages))


@pytest.fixture
def fake_teacher(monkeypatch):
    fake = FakeTeacher()
    monkeypatch.setattr("shrewd.teacher.litellm.completion", fake)
    monkeypatch.setattr(
        "shrewd.teacher.litellm.completion_cost", lambda completion_response: 0.001
    )
    monkeypatch.setattr("shrewd.teacher.time.sleep", lambda seconds: None)
    return fake


@pytest.fixture
def project(tmp_path):
    from shrewd import Project

    return Project(
        tmp_path / "proj",
        instructions="Classify customer support tickets by the customer's primary intent.",
        labels=LABELS,
        teacher="test/fake-model",
    )


# ---------------------------------------------------------------- a decision panel

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this?",
        criteria={
            "billing": "charges, invoices, refunds",
            "bug": "something is broken",
            "cancellation": "wants to cancel or downgrade",
            "other": "none of the above",
        },
    ),
    "angry": Noul(instructions="Does the customer sound angry?"),
    "severity": Score(
        instructions="How severe is this?",
        criteria=["minor", "normal", "blocking"],
    ),
}


def make_docs(n=160):
    rows = []
    for i in range(n):
        keyword, label = KEYWORDS[i % len(KEYWORDS)]
        angry = "yes" if i % 3 == 0 else "no"
        shout = " this is unacceptable" if angry == "yes" else ""
        rows.append(
            {
                "text": f"ticket {i}: the {keyword} problem again{shout}",
                "department": label,
                "angry": angry,
                "severity": str(i % 3),
            }
        )
    return pd.DataFrame(rows)


def fake_answer(state, questions):
    """A teacher that is right most of the time and states plausible probabilities."""
    answers = {}
    for key, question in questions.items():
        options = question.options()
        if key == "department":
            truth = keyword_label(state)
        elif key == "angry":
            truth = "yes" if "unacceptable" in state else "no"
        else:
            truth = str(sum(ord(c) for c in state) % len(options))
        probs = {o: 0.1 / (len(options) - 1) for o in options}
        probs[truth] = 0.9
        answers[key] = {"probabilities": probs}
    return json.dumps({"answers": answers})


@pytest.fixture
def decision_teacher(monkeypatch):
    fake = FakeTeacher()
    fake.respond = lambda messages: fake_answer(messages[1]["content"], QUESTIONS)
    monkeypatch.setattr("shrewd.teacher.litellm.completion", fake)
    monkeypatch.setattr(
        "shrewd.teacher.litellm.completion_cost", lambda completion_response: 0.001
    )
    monkeypatch.setattr("shrewd.teacher.time.sleep", lambda seconds: None)
    return fake
