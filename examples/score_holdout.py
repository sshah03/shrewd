"""Score a project's trained student on gold-labeled rows the pipeline never saw.

    python examples/score_holdout.py runs/dogfood-k8s runs/dogfood-k8s-data.csv

The CSV needs text,label columns. Rows whose text appears in the project's seed
splits or labeled pool are excluded. Whatever remains is a held-out set with
labels, so the student can be scored on data that played no part in training,
prompt optimization, or the locked test set. This is how the README's kubernetes
holdout numbers are produced.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from shrewd import load
from shrewd.evaluate import compute_metrics
from shrewd.students import load_student

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
project_dir, csv_path = Path(sys.argv[1]), sys.argv[2]

gold = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
if not {"text", "label"} <= set(gold.columns):
    raise SystemExit("the labeled CSV needs text,label columns")

used = set()
for name in ("seed_dev.csv", "seed_test.csv", "pool_labeled.csv"):
    path = project_dir / name
    if path.exists():
        used |= set(pd.read_csv(path, dtype=str, keep_default_na=False)["text"])

holdout = gold[~gold["text"].isin(used)]
print(f"{len(gold)} labeled rows − {len(used)} seen by the pipeline → {len(holdout)} holdout")

clf = load(project_dir)
classes = list(clf.labels)
unknown = holdout[~holdout["label"].isin(classes)]
if len(unknown):
    print(f"dropping {len(unknown)} rows with labels outside the project's label set")
    holdout = holdout[holdout["label"].isin(classes)]

metrics = compute_metrics(holdout["label"], clf.predict(holdout["text"].tolist()), classes)
print(f"student on {len(holdout)} holdout rows: "
      f"macro-F1 {metrics['macro_f1']:.3f}, accuracy {metrics['accuracy']:.3f}")
for cls in classes:
    per = metrics["per_class"][cls]
    print(f"  {cls:15} F1 {per['f1']:.3f}  (n={per['support']})")

# compare() candidates, if any, get scored on the same rows
for cand in sorted((project_dir / "candidates").glob("*/meta.json")):
    student = load_student(cand.parent)
    m = compute_metrics(holdout["label"], student.predict(holdout["text"].tolist()), classes)
    print(f"candidate {cand.parent.name}: macro-F1 {m['macro_f1']:.3f}, "
          f"accuracy {m['accuracy']:.3f}")

print(json.dumps({"macro_f1": metrics["macro_f1"], "n_holdout": len(holdout)}))
