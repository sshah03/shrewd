"""Budget-capped controller over the Project pipeline - cheapest teacher first, escalating
only while the dev-split score falls short of `target` and the next tier fits the
remaining budget. The locked test set is evaluated once, on the winner.
"""

import json
from pathlib import Path

from shrewd import data, evaluate
from shrewd.optimize import build_seed_prompt
from shrewd.project import Project
from shrewd.students import STUDENTS, check_students
from shrewd.teacher import estimate_calls, resolve_model

# multiplier on the estimated labeling cost of a tier, to leave room for
# optimize() and the final test labeling before committing to the tier
ESTIMATE_MARGIN = 1.5


def _spent(projects):
    return sum(s["cost_usd"] for p in projects for s in p._manifest["stages"])


def _dev_probe(kept, dev, student_name, classes, seed):
    """Macro-F1 on gold dev of a student trained on pool rows alone (no dev leak)."""
    probe = STUDENTS[student_name](seed=seed)
    probe.fit(kept["text"].tolist(), kept["label"].tolist())
    return evaluate.compute_metrics(
        dev["label"], probe.predict(dev["text"].tolist()), classes
    )["macro_f1"]


def autotune(dir, instructions, labels, seed_df, pool_df, budget_usd, target=None,
             teachers=("anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"),
             students=("tfidf",), votes=1, optimize_budget=600, seed=42):
    """Run the escalation ladder under a spending cap and return the winning pipeline.

    One sub-project per teacher tier under `dir` (same seed data, so identical splits). Per
    tier: optimize -> label -> a free sweep of `students` x min_confidence on the dev split.
    Escalates while the best dev macro-F1 is below `target` (always if None) and the
    estimated cost fits.

    Returns {"teacher", "student", "min_confidence", "dir", "dev_macro_f1", "spent_usd",
    "result", "trail"}. The trail is also written to `dir`/autotune_trail.json. Re-running
    with the same `dir` resumes from caches.
    """
    check_students(students)  # fail on a missing optional extra before any spend
    parent = Path(dir)
    parent.mkdir(parents=True, exist_ok=True)
    confidence_rungs = (0.0,) if votes == 1 else (0.0, 0.67, 1.0)
    pool_texts = pool_df["text"].astype(str).tolist()
    trail, projects = [], []
    best = None

    for model in (resolve_model(m) for m in teachers):
        tier_dir = parent / model.split("/")[-1]
        project = Project(
            tier_dir, instructions=instructions, labels=labels, teacher=model, seed=seed
        )
        if not (project.dir / "seed_test.csv").exists():
            project.add_seed(seed_df)

        # budget guard - price the tier's labeling (seed prompt as proxy) before spending
        _, estimate = estimate_calls(
            pool_texts, build_seed_prompt(instructions, labels), model, votes,
            project._cache(),
        )
        remaining = round(budget_usd - _spent(projects), 2)
        if estimate is None:
            trail.append({"rung": f"teacher:{model}", "decision":
                          "no litellm price data, budget guard waived for this tier"})
        else:
            planned = round(estimate * ESTIMATE_MARGIN, 2)
            if planned > remaining:
                if best is None:
                    raise ValueError(
                        f"budget_usd={budget_usd} cannot cover the first tier "
                        f"({model}: ~${planned}); nothing was spent"
                    )
                trail.append({"rung": f"teacher:{model}", "decision":
                              f"skipped: needs ~${planned}, ${remaining} left"})
                break

        project.optimize(budget=optimize_budget, target=target)
        project.label(pool_df, votes=votes)
        projects.append(project)

        labeled = data.read_csv(project.dir / "pool_labeled.csv")
        dev = data.read_csv(project.dir / "seed_dev.csv")
        test = data.read_csv(project.dir / "seed_test.csv")
        valid, _ = data.usable_pool(labeled, dev, test)
        classes = list(project.labels)
        for min_confidence in confidence_rungs:
            kept = valid[valid["confidence"] >= min_confidence]
            if kept["label"].nunique() < 2:
                trail.append({
                    "rung": f"teacher:{model}", "min_confidence": min_confidence,
                    "decision": "probe skipped: under two labels left after filtering",
                })
                continue
            for student in students:
                dev_f1 = _dev_probe(kept, dev, student, classes, seed)
                trail.append({
                    "rung": f"teacher:{model}", "student": student,
                    "min_confidence": min_confidence, "dev_macro_f1": round(dev_f1, 4),
                    "spent_usd": round(_spent(projects), 2),
                })
                if best is None or dev_f1 > best["dev_f1"]:
                    best = {"project": project, "student": student,
                            "min_confidence": min_confidence, "dev_f1": dev_f1}
        if target is not None and best is not None and best["dev_f1"] >= target:
            trail.append({"rung": f"teacher:{model}",
                          "decision": f"dev target {target} reached, stopping early"})
            break

    if best is None:
        (parent / "autotune_trail.json").write_text(json.dumps(trail, indent=2))
        raise RuntimeError(
            "no configuration could be probed: every rung left fewer than two distinct "
            "labels (the teacher may be collapsing to one class). See "
            f"{parent / 'autotune_trail.json'} and the per-tier pool_labeled.csv files."
        )
    # the single test-set evaluation, on the winning configuration only
    result = best["project"].distill(
        student=best["student"], min_confidence=best["min_confidence"]
    )
    summary = {
        "teacher": best["project"].teacher,
        "student": best["student"],
        "min_confidence": best["min_confidence"],
        "dir": str(best["project"].dir),
        "dev_macro_f1": round(best["dev_f1"], 4),
        "spent_usd": round(_spent(projects), 2),
    }
    trail.append({"rung": "final", **summary,
                  "test_macro_f1": round(result.metrics["student"]["macro_f1"], 4)})
    (parent / "autotune_trail.json").write_text(json.dumps(trail, indent=2))
    return {**summary, "result": result, "trail": trail}
