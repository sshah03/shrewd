"""Generate a synthetic seed + pool CSV pair so the full pipeline can be demoed
cheaply with a small teacher model.

Usage: python examples/make_demo_data.py  (writes demo_seed.csv and demo_pool.csv here)
"""

import random
from pathlib import Path

import pandas as pd

TEMPLATES = {
    "billing": [
        "I was charged {n} times for my {plan} plan this month",
        "there is an unexpected ${amount} charge on my latest invoice",
        "still waiting on the refund for order {order}",
        "why did my bill jump to ${amount} this cycle?",
        "the receipt shows {n} charges but I only bought one seat",
    ],
    "bug": [
        "the {page} page crashes every time I open it",
        "getting error {code} when I export to {fmt}",
        "search has been broken since the {release} update",
        "the {page} screen freezes on {browser}",
        "file uploads fail with error {code} in {browser}",
    ],
    "cancellation": [
        "please cancel my {plan} subscription effective today",
        "I want to downgrade from {plan} to the free tier",
        "how do I pause my account for {n} months?",
        "we are shutting down, terminate the {plan} plan",
        "switching to a competitor next week, please cancel",
    ],
    "other": [
        "do you offer discounts for {org} organizations?",
        "how do I add {n} more teammates to my workspace?",
        "is there an API for the {page} dashboard?",
        "what are your support hours in {region}?",
        "loving the new {page} design, great work team",
    ],
}
FILLS = {
    "n": ["two", "three", "2", "3", "4", "5"],
    "amount": ["19", "49", "99", "120", "500"],
    "plan": ["starter", "pro", "team", "enterprise"],
    "order": ["#1023", "#88412", "#55T9", "#20250"],
    "page": ["settings", "billing", "analytics", "reports", "admin"],
    "code": ["500", "403", "ERR_TIMEOUT", "0x80070057"],
    "fmt": ["csv", "pdf", "xlsx"],
    "release": ["Tuesday", "March", "v2.3"],
    "browser": ["Safari", "Firefox", "Chrome"],
    "org": ["nonprofit", "education", "government"],
    "region": ["Europe", "the US", "Asia"],
}
PREFIXES = ["", "hi, ", "hello team, ", "urgent: ", "quick question - ", "hey - "]
SUFFIXES = ["", " thanks", " please advise", " this is the second time I'm asking", " any update?"]


def make_rows(n, rng):
    rows = set()
    while len(rows) < n:
        label = rng.choice(list(TEMPLATES))
        template = rng.choice(TEMPLATES[label])
        text = template.format(**{k: rng.choice(v) for k, v in FILLS.items()})
        rows.add((rng.choice(PREFIXES) + text + rng.choice(SUFFIXES), label))
    return sorted(rows)


if __name__ == "__main__":
    rng = random.Random(7)
    seed = make_rows(300, rng)
    pool = make_rows(1200, rng)
    out = Path.cwd()
    pd.DataFrame(seed, columns=["text", "label"]).to_csv(out / "demo_seed.csv", index=False)
    pd.DataFrame([t for t, _ in pool], columns=["text"]).to_csv(out / "demo_pool.csv", index=False)
    print(f"wrote {out}/demo_seed.csv (300 labeled) and {out}/demo_pool.csv (1200 unlabeled)")
