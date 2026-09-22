"""Pick each panel's shipped student among the featurizations build.py tried.

    python examples/panels/pick.py [panel ...]

Each featurization is scored per gold question on the seed test split (report-<tag>.json):
AUROC for yes/no, accuracy for a choice, negative MAE for a score. The featurization with
the best mean wins and its student-<tag>/ is copied to student/, which
is what `shrewd.load()` serves. Ties (within 0.005) go to the faster one.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panels import GOLD, PANELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SPEED = {"tfidf": 0, "embed": 1, "auto": 1, "encoder": 2}     # rough latency order


def score(report, panel):
    vals = []
    for key in GOLD[panel]:
        m = report["questions"][key]["student"]
        if "auroc" in m:
            vals.append(m["auroc"])
        elif "mae" in m:
            vals.append(-m["mae"])
        else:
            vals.append(m["accuracy"])
    return sum(vals) / len(vals)


for panel in (sys.argv[1:] or sorted(PANELS)):
    run = ROOT / "runs" / f"panel-{panel}"
    cands = []
    for path in sorted(run.glob("report-*.json")):
        tag = path.stem[len("report-"):]
        speed = -SPEED.get(tag.split("+")[0], 9)
        cands.append((score(json.loads(path.read_text()), panel), speed, tag))
    if not cands:
        print(f"{panel}: nothing to pick from")
        continue
    cands.sort(reverse=True)
    best = cands[0]
    for c in cands[1:]:                       # a slower model must win by more than noise
        if best[0] - c[0] < 0.005 and c[1] > best[1]:
            best = c
    _, _, tag = best
    if (run / "student").exists():
        shutil.rmtree(run / "student")
    shutil.copytree(run / f"student-{tag}", run / "student")
    shutil.copy(run / f"holdout-{tag}.json", run / "holdout.json")
    table = "  ".join(f"{t} {s:.3f}" for s, _, t in sorted(cands, key=lambda c: c[2]))
    print(f"{panel}: {tag}  {table}")
