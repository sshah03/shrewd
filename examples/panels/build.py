"""Compile one panel end to end and score it on its holdout.

    python examples/panels/build.py messages [--teacher ...] [--from-judged] [--features encoder]

`--from-judged` reuses the teacher answers committed under judged/, so no API calls.
Stages: add_seed -> judge -> distill -> holdout scores for the questions with public
labels. Writes runs/panel-<name>/holdout.json and keeps each featurization's student
side by side for pick.py.
"""

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panels import GOLD, PANELS  # noqa: E402

from shrewd import Decisions, load  # noqa: E402
from shrewd.decisions import gold_index, question_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("panel", choices=sorted(PANELS))
ap.add_argument("--teacher", default="anthropic/claude-fable-5-1")
ap.add_argument("--features", default="auto")
ap.add_argument("--no-stack", action="store_true")
ap.add_argument("--concurrency", type=int, default=12)
ap.add_argument("--from-judged", action="store_true",
                help="reuse the teacher's committed answers from judged/; makes no API calls")
ap.add_argument("--run", default=None, help="project directory (default runs/panel-<panel>)")
args = ap.parse_args()

spec = PANELS[args.panel]
data = ROOT / "data" / args.panel
run = Path(args.run) if args.run else ROOT / "runs" / f"panel-{args.panel}"
d = Decisions(run, questions=spec["questions"], instructions=spec["instructions"],
              teacher=args.teacher)


def with_gold(df):
    """Seed/holdout frames carry gold for some questions and the rest get an empty column."""
    df = df.copy()
    for key, mapping in GOLD[args.panel].items():
        if mapping:
            df[key] = df[key].map(mapping)
    for key in spec["questions"]:
        if key not in df:
            df[key] = None
    return df


if args.from_judged and not (run / "cache.db").exists():
    # the teacher's answers for the pool and the seed splits, so judge() below hits the
    # cache for every document and costs nothing
    judged = Path(__file__).resolve().parent / "judged"
    with gzip.open(judged / f"{args.panel}.cache.db.gz", "rb") as src, \
            open(run / "cache.db", "wb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copy(judged / f"{args.panel}.pool_judged.csv", run / "pool_judged.csv")
if not (run / "seed_dev.csv").exists():
    d.add_seed(with_gold(pd.read_csv(data / "seed.csv")), test_frac=0.35)
t0 = time.time()
d.judge(pd.read_csv(data / "pool.csv"), concurrency=args.concurrency)
print(f"judge stage took {time.time() - t0:.0f}s", flush=True)
stack = not args.no_stack and args.features != "encoder"
res = d.distill(features=args.features, zero_shot=stack)
print(res.report(), flush=True)

hold = with_gold(pd.read_csv(data / "holdout.csv"))
dec = load(run)
t0 = time.time()
P = dec.predict_proba(hold["text"].astype(str).tolist())
ms = (time.time() - t0) / len(hold) * 1000
out = {"panel": args.panel, "teacher": args.teacher, "n_holdout": len(hold),
       "ms_per_doc": round(ms, 3), "questions": {}}
print(f"\nholdout ({len(hold)} rows never seen by teacher, optimizer or student; {ms:.2f} ms/doc)")
for key, q in spec["questions"].items():
    if key not in GOLD[args.panel]:
        out["questions"][key] = {"gold": False}
        print(f"  {key:16} (no public labels; teacher-labeled only)")
        continue
    gold = np.array([gold_index(q, v) if gold_index(q, v) is not None else -1 for v in hold[key]])
    m = question_metrics(q, P[key], gold)
    out["questions"][key] = {"gold": True, **m}
    numbers = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    print(f"  {key:16} " + "  ".join(f"{k} {v:.3f}" for k, v in numbers.items()))
manifest = json.loads((run / "manifest.json").read_text())
out["cost_usd"] = round(sum(s.get("cost_usd", 0) for s in manifest["stages"]), 2)
out["features"] = args.features
(run / "holdout.json").write_text(json.dumps(out, indent=2))
# keep every featurization's student and reports side by side for pick.py
tag = args.features + ("+stack" if stack else "")
shutil.copy(run / "holdout.json", run / f"holdout-{tag}.json")
shutil.copy(run / "report.json", run / f"report-{tag}.json")
if (run / f"student-{tag}").exists():
    shutil.rmtree(run / f"student-{tag}")
shutil.copytree(run / "student", run / f"student-{tag}")
print(f"total teacher cost ${out['cost_usd']:.2f}; wrote {run / 'holdout.json'} and student-{tag}/")
print(f"PANEL DONE {args.panel}", flush=True)
