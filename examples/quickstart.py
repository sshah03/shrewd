"""The README quickstart, runnable end to end on the generated demo data.

    python examples/make_demo_data.py
    export ANTHROPIC_API_KEY=...   # or any provider litellm supports
    python examples/quickstart.py

No API key? It still runs: every teacher call for this demo is bundled in
examples/demo_cache.db (a real haiku run), so the whole pipeline replays with no API calls.
"""

import os
import shutil
from pathlib import Path

import pandas as pd

from shrewd import Project, load

if not Path("demo_seed.csv").exists():
    raise SystemExit("run `python examples/make_demo_data.py` first")

if not os.environ.get("ANTHROPIC_API_KEY") and not Path("runs/demo/cache.db").exists():
    bundled = Path(__file__).parent / "demo_cache.db"
    if bundled.exists():
        Path("runs/demo").mkdir(parents=True, exist_ok=True)
        shutil.copy(bundled, "runs/demo/cache.db")
        print("no ANTHROPIC_API_KEY; replaying the bundled demo cache (zero API calls)")

proj = Project(
    "runs/demo",
    instructions="Classify customer support tickets by the customer's primary intent.",
    labels={
        "billing": "charges, invoices, refunds, payment problems",
        "bug": "something in the product is broken or misbehaving",
        "cancellation": "wants to cancel, pause, or downgrade",
        "other": "anything that fits none of the above",
    },
    # A cheap explicit model keeps this demo inexpensive. For real tasks, teacher="anthropic"
    # or teacher="openai" picks that provider's default. Any litellm model string works.
    teacher="anthropic/claude-haiku-4-5",
)

proj.add_seed(pd.read_csv("demo_seed.csv"))
proj.optimize(budget=150, target=0.95)
proj.label(pd.read_csv("demo_pool.csv"))
result = proj.distill(student="tfidf")
print(result.report())

clf = load("runs/demo")
print(clf.predict(["I was double charged last month"]))  # -> ['billing']
