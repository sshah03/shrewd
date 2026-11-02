import hashlib
import json
import os
import shutil
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shrewd import __version__, data, evaluate
from shrewd.optimize import build_seed_prompt, run_gepa
from shrewd.students import STUDENTS
from shrewd.teacher import (
    INVALID,
    cast_ballots,
    estimate_calls,
    label_texts,
    open_cache,
    resolve_model,
)

REVIEW_FLOOR = 0.6
SPLIT_SETTLED = 0.9


class Project:
    """One classification task: a directory of artifacts plus the pipeline stages.

    Point it at a fresh directory with `instructions`, `labels` and `teacher` to start, or at
    an existing project directory to resume (passed args must match the manifest). `teacher`
    is any litellm model string or a provider alias ("anthropic", "openai").
    """

    def __init__(self, dir, instructions=None, labels=None, teacher=None, seed=42):
        self.dir = Path(dir)
        if isinstance(labels, list):
            labels = {name: "" for name in labels}
        if teacher is not None:
            teacher = ([resolve_model(t) for t in teacher] if isinstance(teacher, (list, tuple))
                       else resolve_model(teacher))
        manifest_path = self.dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for name, passed in (
                ("instructions", instructions), ("labels", labels), ("teacher", teacher),
            ):
                if passed is not None and passed != manifest[name]:
                    raise ValueError(
                        f"{name} does not match the existing project in {self.dir}; "
                        f"use a new directory for a new task (manifest has: {manifest[name]!r})"
                    )
            self.instructions = manifest["instructions"]
            self.labels = manifest["labels"]
            self.teacher = manifest["teacher"]
            self.seed = manifest["seed"]
            self._manifest = manifest
        else:
            missing = [
                name for name, value in
                (("instructions", instructions), ("labels", labels), ("teacher", teacher))
                if not value
            ]
            if missing:
                raise ValueError(f"a new project needs: {', '.join(missing)}")
            self.instructions, self.labels, self.teacher, self.seed = (
                instructions, labels, teacher, seed,
            )
            self._manifest = {
                "version": 1,
                "shrewd": __version__,
                "instructions": instructions,
                "labels": labels,
                "teacher": teacher,
                "seed": seed,
                "stages": [],
            }
            self.dir.mkdir(parents=True, exist_ok=True)  # only after arguments validate
            self._save_manifest()
        self._cache_conn = None

    def _save_manifest(self):
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(self._manifest, f, indent=2)
        os.replace(tmp, self.dir / "manifest.json")

    def _record_stage(self, name, cost, **params):
        self._manifest["stages"].append(
            {
                "name": name,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "cost_usd": round(cost, 4),
                "params": params,
            }
        )
        self._save_manifest()

    def _cache(self):
        if self._cache_conn is None:
            self._cache_conn = open_cache(self.dir / "cache.db")
        return self._cache_conn

    def add_seed(self, df, test_frac=0.35, overwrite=False):
        """Validate hand-labeled data and split it into dev and a locked test set."""
        test_path = self.dir / "seed_test.csv"
        if test_path.exists():
            dev, test = data.prepare_seed(df, list(self.labels), test_frac, self.seed)
            splits = self._manifest.get("splits", {})
            if not overwrite and (_rows_hash(dev), _rows_hash(test)) == (
                splits.get("dev_sha256"), splits.get("test_sha256")
            ):
                print(f"seed data unchanged: {len(dev)} dev / {len(test)} test rows (kept)")
                return  # same data, same split: re-running a script is not a re-split
            if not overwrite:
                raise ValueError(
                    f"{test_path} already exists and the test set is locked. Regenerating "
                    "it silently would invalidate every score so far. Pass overwrite=True "
                    "if you really mean to."
                )
            # a re-split invalidates everything derived from the old one, including
            # models trained on it. The API cache survives, so regenerating the prompt
            # and pool labels is mostly free
            stale = [
                name for name in
                ("prompt.txt", "optimize_log.json", "pool_labeled.csv",
                 "needs_review.csv", "report.json", "compare.json")
                if (self.dir / name).exists()
            ]
            for name in stale:
                (self.dir / name).unlink()
            for name in ("student", "candidates"):
                if (self.dir / name).exists():
                    shutil.rmtree(self.dir / name)
                    stale.append(name + "/")
            if stale:
                print(f"re-split invalidated and removed: {', '.join(stale)}")
        else:
            dev, test = data.prepare_seed(df, list(self.labels), test_frac, self.seed)
        dev.to_csv(self.dir / "seed_dev.csv", index=False)
        test.to_csv(test_path, index=False)
        self._manifest["splits"] = {
            "n_dev": len(dev),
            "n_test": len(test),
            "dev_sha256": _rows_hash(dev),
            "test_sha256": _rows_hash(test),
        }
        self._record_stage("add_seed", 0.0, test_frac=test_frac)
        print(f"seed data: {len(dev)} dev / {len(test)} test rows across {len(self.labels)} labels")

    def add_examples(self, df, test_frac=None):
        """Add hand-labeled rows to the existing split. Each one goes to the test set with
        probability `test_frac`, decided by its text. Returns the (dev, test) counts added.
        """
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"examples are missing columns: {sorted(missing)}")
        dev, test = self._seed_frames()
        frame = df[["text", "label"]].astype(str)
        frame = frame[frame["text"].str.strip() != ""].drop_duplicates("text")
        unknown = sorted(set(frame["label"]) - set(self.labels))
        if unknown:
            raise ValueError(f"labels not in the project label set: {unknown}")
        frame = frame[~frame["text"].isin(set(dev["text"]) | set(test["text"]))]
        frac = test_frac if test_frac is not None else _last_test_frac(self._manifest)
        to_test = np.array([_hash_unit(t, self.seed) < frac for t in frame["text"]], dtype=bool)
        dev = pd.concat([dev, frame[~to_test]], ignore_index=True)
        test = pd.concat([test, frame[to_test]], ignore_index=True)
        self._save_seed(dev, test)
        added = (int((~to_test).sum()), int(to_test.sum()))
        self._record_stage("add_examples", 0.0, dev=added[0], test=added[1])
        print(f"added {added[0]} tuning and {added[1]} test examples "
              f"({len(dev)} dev / {len(test)} test now)")
        return added

    def fix_example(self, text, label):
        """Change the hand label of one example, in whichever split it's in."""
        if label not in self.labels:
            raise ValueError(f"{label!r} is not one of {list(self.labels)}")
        dev, test = self._seed_frames()
        for frame, split in ((dev, "dev"), (test, "test")):
            hit = frame["text"] == text
            if hit.any():
                old = frame.loc[hit, "label"].iloc[0]
                frame.loc[hit, "label"] = label
                self._save_seed(dev, test)
                self._record_stage("fix_example", 0.0, split=split, old=old, new=label)
                return split
        raise ValueError(f"{text[:40]!r} is not one of the seed examples")

    def remove_examples(self, texts):
        """Remove documents from your examples, in either split. Returns the (dev, test) counts
        removed.
        """
        texts = {texts} if isinstance(texts, str) else {str(t) for t in texts}
        dev, test = self._seed_frames()
        gone = (int(dev["text"].isin(texts).sum()), int(test["text"].isin(texts).sum()))
        if not any(gone):
            return gone
        self._save_seed(dev[~dev["text"].isin(texts)], test[~test["text"].isin(texts)])
        self._record_stage("remove_examples", 0.0, dev=gone[0], test=gone[1])
        if gone[1]:
            warnings.warn(f"{gone[1]} test example(s) removed; test scores from before aren't "
                          "comparable with the next distill()'s", stacklevel=2)
        return gone

    def set_labels(self, descriptions):
        """Change label descriptions. Label the pool again afterwards."""
        unknown = sorted(set(descriptions) - set(self.labels))
        if unknown:
            raise ValueError(f"not labels of this project: {unknown}; names can't change")
        labels = {name: descriptions.get(name, desc) for name, desc in self.labels.items()}
        if labels == self.labels:
            return False
        self.labels = labels
        self._manifest["labels"] = labels
        prompt_path = self.dir / "prompt.txt"
        tuned = any(s["name"] == "optimize" for s in self._manifest["stages"])
        if tuned:
            warnings.warn(
                "prompt.txt was tuned from the old descriptions and is unchanged; re-run "
                "optimize(overwrite=True), or edit prompt.txt by hand", stacklevel=2)
        elif prompt_path.exists():
            prompt_path.write_text(build_seed_prompt(self.instructions, labels))
        self._record_stage("set_labels", 0.0, labels=sorted(descriptions))
        return True

    def add_label(self, name, description=""):
        """Add a label. Label the pool again for the teachers to use it."""
        name = str(name).strip()
        if not name:
            raise ValueError("a label needs a name")
        if name.lower() in {k.lower() for k in self.labels}:
            raise ValueError(f"{name!r} is already a label of this project")
        self._set_label_dict({**self.labels, name: description})
        if (self.dir / "pool_labeled.csv").exists():
            warnings.warn(f"the pool was labeled before {name!r} existed; label() it again for "
                          "the teachers to use it", stacklevel=2)
        self._record_stage("add_label", 0.0, label=name)
        return True

    def merge_labels(self, mapping):
        """Merge labels into others, e.g. `{"mixed": "neutral"}`, in your examples, the labeled
        pool and the review files.
        """
        mapping = {str(source): str(target) for source, target in mapping.items()}
        for source, target in mapping.items():
            if source not in self.labels:
                raise ValueError(f"{source!r} is not a label of this project")
            if target not in self.labels or target in mapping:
                raise ValueError(f"{source!r} must merge into a label that stays, not {target!r}")
        if not mapping:
            return {}
        changed = self._rewrite_labels(lambda label: mapping.get(label, label), mapping)
        self._set_label_dict({k: v for k, v in self.labels.items() if k not in mapping})
        self._pool_matches_prompt()
        self._warn_stale("merged")
        self._record_stage("merge_labels", 0.0, mapping=mapping, changed=changed)
        return changed

    def remove_label(self, name, into=None):
        """Remove a label. With `into=`, its rows are merged into that label; otherwise they're
        dropped.
        """
        if into is not None:
            return self.merge_labels({name: into})
        if name not in self.labels:
            raise ValueError(f"{name!r} is not a label of this project")
        if len(self.labels) <= 2:
            raise ValueError("a classifier needs at least two labels")
        changed = self._rewrite_labels(lambda label: None if label == name else label, {})
        self._set_label_dict({k: v for k, v in self.labels.items() if k != name})
        self._pool_matches_prompt()
        lost = changed.get("seed_test.csv", 0)
        if lost:
            warnings.warn(f"{lost} test examples had {name!r} and were removed from the locked "
                          "test set; scores from before aren't comparable", stacklevel=2)
        self._warn_stale("removed")
        self._record_stage("remove_label", 0.0, label=name, changed=changed)
        return changed

    def _set_label_dict(self, labels):
        self.labels = labels
        self._manifest["labels"] = labels
        prompt_path = self.dir / "prompt.txt"
        if any(s["name"] == "optimize" for s in self._manifest["stages"]):
            warnings.warn("prompt.txt was tuned for the old labels and is unchanged; re-run "
                          "optimize(overwrite=True), or edit prompt.txt by hand", stacklevel=3)
        elif prompt_path.exists():
            prompt_path.write_text(build_seed_prompt(self.instructions, labels))
        self._save_manifest()

    def _rewrite_labels(self, rename, merged):
        changed = {}

        def relabel(frame, column):
            values = frame[column].astype(str)
            mapped = values.map(lambda v: rename(v) if v.strip() else v)
            keep = mapped.notna()
            n = int((mapped[keep] != values[keep]).sum() + (~keep).sum())
            frame = frame[keep].copy()
            frame[column] = mapped[keep]
            return frame, n

        if (self.dir / "seed_test.csv").exists():
            dev, test = self._seed_frames()
            dev, n_dev = relabel(dev, "label")
            test, n_test = relabel(test, "label")
            self._save_seed(dev.reset_index(drop=True), test.reset_index(drop=True))
            changed["seed_dev.csv"], changed["seed_test.csv"] = n_dev, n_test
        for name, columns in (("pool_labeled.csv", ["label"]), ("reviewed.csv", ["label"]),
                              ("needs_review.csv", ["label", "human_label"])):
            path = self.dir / name
            if not path.exists():
                continue
            frame = pd.read_csv(path, dtype={"text": str})
            n = 0
            for column in columns:
                if column not in frame.columns:
                    continue
                if column == "human_label":
                    filled = frame[column].fillna("").astype(str)
                    mapped = filled.map(lambda v: (rename(v) or "") if v.strip() else v)
                    n += int((mapped != filled).sum())
                    frame[column] = mapped
                else:
                    frame, k = relabel(frame, column)
                    n += k
            frame.to_csv(path, index=False)
            changed[name] = n
        votes = self.dir / "votes_pool.csv"
        if votes.exists():
            frame = pd.read_csv(votes)
            prefix = "label__"
            for column in [c for c in frame.columns if c.startswith(prefix)]:
                label = column[len(prefix):]
                target = merged.get(label)
                if target is not None:
                    kept = prefix + target
                    frame[kept] = frame[kept].fillna(0) + frame[column].fillna(0)
                if rename(label) is None or target is not None:
                    frame = frame.drop(columns=column)
            share = [c for c in frame.columns if c.startswith(prefix)]
            total = frame[share].sum(axis=1).replace(0, np.nan)
            frame[share] = frame[share].div(total, axis=0)
            frame.to_csv(votes, index=False)
        return changed

    def _pool_matches_prompt(self):
        prompt_path = self.dir / "prompt.txt"
        if "label_prompt_sha256" in self._manifest and prompt_path.exists():
            self._manifest["label_prompt_sha256"] = hashlib.sha256(
                prompt_path.read_text().encode()).hexdigest()
            self._save_manifest()

    def _warn_stale(self, verb):
        if (self.dir / "student").exists() or (self.dir / "candidates").exists():
            warnings.warn(f"trained models still predict the labels you {verb}; compare() or "
                          "distill() again to retrain them", stacklevel=3)

    def _seed_frames(self):
        if not (self.dir / "seed_test.csv").exists():
            raise FileNotFoundError("no seed data; call add_seed() with your examples first")
        return (data.read_csv(self.dir / "seed_dev.csv"),
                data.read_csv(self.dir / "seed_test.csv"))

    def _save_seed(self, dev, test):
        dev.to_csv(self.dir / "seed_dev.csv", index=False)
        test.to_csv(self.dir / "seed_test.csv", index=False)
        self._manifest["splits"] = {
            **self._manifest.get("splits", {}),
            "n_dev": len(dev), "n_test": len(test),
            "dev_sha256": _rows_hash(dev), "test_sha256": _rows_hash(test),
        }
        self._save_manifest()

    def optimize(self, budget=800, target=None, reflection_model=None, overwrite=False):
        """GEPA-optimize the teacher prompt against the dev split. Never sees the test set.

        The whole dev split is both the reflection pool and the validation set. `budget` counts
        teacher calls, and each accepted candidate costs one pass over dev. `reflection_model`
        writes the new prompts and is called a few dozen times, so a stronger model than the
        teacher costs little and did better in my tests.
        """
        prompt_path = self.dir / "prompt.txt"
        if prompt_path.exists() and not overwrite:
            print(f"{prompt_path} exists, skipping optimization (pass overwrite=True to redo)")
            return
        dev_path = self.dir / "seed_dev.csv"
        if not dev_path.exists():
            raise FileNotFoundError(
                "no seed data; call add_seed() with your hand-labeled examples first"
            )
        dev = data.read_csv(dev_path)
        print(
            f"optimizing teacher prompt with GEPA: budget {budget} metric calls over "
            f"{len(dev)} dev rows"
        )
        primary = self._teachers()[0]
        if len(self._teachers()) > 1:
            print(f"several teachers configured; optimizing the prompt against {primary}")
        reflection = resolve_model(reflection_model) if reflection_model else primary
        best_prompt, log, cost = run_gepa(
            dev, build_seed_prompt(self.instructions, self.labels), primary,
            self.labels, budget, target, reflection, self._cache(), self.seed,
        )
        prompt_path.write_text(best_prompt)
        (self.dir / "optimize_log.json").write_text(json.dumps(log, indent=2))
        self._record_stage("optimize", cost, budget=budget, target=target)
        print(
            f"dev-val score: {log['baseline_score']:.3f} (seed prompt) → "
            f"{log['best_score']:.3f} best of {len(log['candidates'])} candidates; "
            f"cost ${cost:.2f}"
        )
        print(f"wrote {prompt_path}")
        return log

    def label(self, df, votes=1, dry_run=False, concurrency=8, budget_usd=None, n=None,
              batch=None, combine=None):
        """Teacher-label a pool of unlabeled texts. Resumable, so pass the full pool each time.

        With neither `budget_usd` nor `n`, every row is labeled. With either, rows are acquired
        in rounds: a tfidf probe trained on what is labeled so far picks the rows it is least
        sure about, spread across clusters, and its dev accuracy is logged to label_log.json.
        Stops at the budget, at `n` rows, or when the pool is exhausted.

        With several teachers, `combine="weighted"` weights each teacher's votes by how well it
        matched your dev labels, instead of a majority vote.
        """
        if "text" not in df.columns:
            raise ValueError("pool data needs a 'text' column")
        if combine not in (None, "majority", "weighted"):
            raise ValueError(f'combine must be "majority" or "weighted", not {combine!r}')
        if combine is not None and combine != self._combine():
            if combine == "weighted" and len(self._teachers()) < 2:
                raise ValueError('combine="weighted" needs several teachers')
            self._manifest["combine"] = combine
            self._save_manifest()
        prompt_path = self.dir / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"no {prompt_path}; run optimize() first, or write that file yourself "
                "to skip optimization"
            )
        prompt = prompt_path.read_text()
        self.apply_review()  # ledger pending human labels before the queue is regenerated
        texts = df["text"].astype(str).tolist()
        teachers = self._teachers()
        if budget_usd is not None or n is not None:
            if dry_run:
                raise ValueError("dry_run prices a full labeling run; drop budget_usd/n for it")
            if len(teachers) > 1:
                raise ValueError(
                    "active acquisition (budget_usd / n) with several teachers is not "
                    "supported yet; label the whole pool, or use one teacher"
                )
            return self._label_actively(
                texts, prompt, votes, concurrency, budget_usd, n, batch
            )
        if dry_run:
            n_calls, cost = 0, 0.0
            to_price = texts
            if self._combine() == "weighted":
                to_price = texts + data.read_csv(self.dir / "seed_dev.csv")["text"].tolist()
            for model in teachers:
                calls, price = estimate_calls(to_price, prompt, model, votes, self._cache())
                n_calls += calls
                cost = None if cost is None or price is None else cost + price
            print(
                f"dry run: {len(texts)} items × {votes} vote(s) × {len(teachers)} teacher(s) = "
                f"{len(texts) * votes * len(teachers)} calls, {n_calls} not yet cached"
            )
            price = f"~${cost:.2f}" if cost is not None else f"unknown for {self.teacher}"
            print(f"estimated cost of uncached calls: {price}")
            return {"uncached_calls": n_calls, "estimated_usd": cost}
        labeled, cost = self._label_texts(texts, prompt, votes, concurrency)
        labeled = self._reapply_reviewed(labeled)
        labeled.to_csv(self.dir / "pool_labeled.csv", index=False)
        if self._combine() == "weighted":
            self._write_weighted_outputs(labeled, texts)
        else:
            self._write_review_queue(labeled)
        valid = labeled[labeled["label"] != INVALID]
        # remembered so training can warn if prompt.txt changes after labeling
        self._manifest["label_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        self._record_stage("label", cost, items=len(labeled), votes=votes)
        mean_conf = f"{valid['confidence'].mean():.2f}" if len(valid) else "n/a"
        print(
            f"labeled {len(labeled)} items ({len(labeled) - len(valid)} invalid), "
            f"mean confidence {mean_conf}, cost ${cost:.2f}"
        )
        print(f"wrote {self.dir / 'pool_labeled.csv'}")
        return labeled

    def _label_actively(self, texts, prompt, votes, concurrency, budget_usd, n, batch):
        from shrewd import acquire

        dev = data.read_csv(self.dir / "seed_dev.csv")
        test = data.read_csv(self.dir / "seed_test.csv")
        seen = set(dev["text"]) | set(test["text"])
        pool = [t for t in dict.fromkeys(texts) if t not in seen]  # seed rows need no teacher
        labeled_path = self.dir / "pool_labeled.csv"
        log_path = self.dir / "label_log.json"
        if labeled_path.exists():
            labeled = data.read_csv(labeled_path)
            if "round" not in labeled.columns:
                labeled["round"] = 0
            log = json.loads(log_path.read_text()) if log_path.exists() else []
        else:
            labeled = pd.DataFrame(columns=["text", "label", "confidence", "round"])
            log = []
        done = set(labeled["text"])
        remaining = [t for t in pool if t not in done]
        batch = batch or acquire.default_batch(len(pool))
        spent = sum(r["cost_usd"] for r in log)
        last_round_cost = log[-1]["cost_usd"] if log else None
        classes = list(self.labels)
        print(
            f"active labeling: {len(remaining)} unlabeled pool rows, batches of {batch}"
            + (f", budget ${budget_usd:.2f}" if budget_usd is not None else "")
            + (f", up to {n} rows" if n is not None else "")
        )
        while remaining:
            if n is not None and len(labeled) >= n:
                print(f"stopping: {len(labeled)} rows labeled, n={n}")
                break
            take = min(batch, len(remaining), (n - len(labeled)) if n is not None else batch)
            if budget_usd is not None:
                _, estimate = estimate_calls(remaining[:take], prompt, self.teacher, votes,
                                             self._cache())
                if estimate is None and last_round_cost is not None:
                    estimate = last_round_cost * take / batch
                if estimate is not None and spent + estimate > budget_usd:
                    print(
                        f"stopping: ${spent:.2f} spent, next batch would cost about "
                        f"${estimate:.2f} against a ${budget_usd:.2f} budget"
                    )
                    break
            valid = labeled[labeled["label"] != INVALID]
            probe = acquire.fit_probe(
                dev["text"].tolist() + valid["text"].tolist(),
                dev["label"].tolist() + valid["label"].tolist(), self.seed,
            )
            picked = acquire.select(probe, remaining, take, self.seed)
            chosen = [remaining[i] for i in picked]
            rows, cost = label_texts(
                chosen, prompt, self.teacher, classes, votes=votes, conn=self._cache(),
                concurrency=concurrency, desc=f"round {len(log) + 1}",
            )
            rows["round"] = len(log) + 1
            labeled = pd.concat([labeled, rows], ignore_index=True)
            labeled.to_csv(labeled_path, index=False)
            remaining = [t for t in remaining if t not in set(chosen)]
            spent += cost
            last_round_cost = cost
            # the learning-curve point: a probe trained on pool labels alone, scored on gold dev
            valid = labeled[labeled["label"] != INVALID]
            dev_acc = None
            if valid["label"].nunique() >= 2:
                check = acquire.fit_probe(
                    valid["text"].tolist(), valid["label"].tolist(), self.seed
                )
                dev_preds = np.array(check.predict(dev["text"].tolist()))
                dev_acc = float(np.mean(dev_preds == dev["label"].values))
            log.append({
                "round": len(log) + 1, "n_labeled": int(len(labeled)), "cost_usd": round(cost, 4),
                "spent_usd": round(spent, 4), "dev_accuracy": dev_acc,
            })
            log_path.write_text(json.dumps(log, indent=2))
            gain = ""
            if dev_acc is not None and len(log) > 1 and log[-2]["dev_accuracy"] is not None:
                gain = f" ({dev_acc - log[-2]['dev_accuracy']:+.3f})"
            print(
                f"round {len(log)}: {len(labeled)} labeled, dev accuracy "
                f"{dev_acc if dev_acc is None else f'{dev_acc:.3f}'}{gain}, spent ${spent:.2f}"
            )
            recent = [r["dev_accuracy"] for r in log[-4:] if r["dev_accuracy"] is not None]
            if len(recent) == 4 and max(recent[1:]) - recent[0] < 0.005:
                print("note: dev accuracy has been flat for three rounds; more labels are "
                      "unlikely to move this student much")
        self._manifest["label_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        self._record_stage(
            "label", sum(r["cost_usd"] for r in log[len(self._rounds_recorded()):]),
            items=int(len(labeled)), votes=votes, mode="active", budget_usd=budget_usd, n=n,
            rounds=len(log),
        )
        print(f"wrote {labeled_path} and {log_path}")
        return labeled

    def _rounds_recorded(self):
        return [
            r for s in self._manifest["stages"] if s["name"] == "label"
            and s["params"].get("mode") == "active" for r in range(s["params"]["rounds"])
        ]

    # -- teachers and human review ----------------------------------------------

    def check_teachers(self, votes=1, dry_run=False):
        """Have each teacher label the dev split and report how often it matches your labels."""
        prompt_path = self.dir / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"no {prompt_path}; run optimize() first, or write that file yourself"
            )
        prompt = prompt_path.read_text()
        if dry_run:
            texts = data.read_csv(self.dir / "seed_dev.csv")["text"].tolist()
            n_calls, cost = 0, 0.0
            for model in self._teachers():
                calls, price = estimate_calls(texts, prompt, model, votes, self._cache())
                n_calls += calls
                cost = None if cost is None or price is None else cost + price
            return {"uncached_calls": n_calls, "estimated_usd": cost}
        from shrewd.decisions import _write_teacher_check

        agreement, cost = self._weighted_agreement(prompt, votes, 8)
        return _write_teacher_check(self, agreement, cost)

    def set_teacher(self, teacher):
        """Change the teacher, or pass a list of several. Cached answers are kept."""
        teacher = ([resolve_model(t) for t in teacher] if isinstance(teacher, (list, tuple))
                   else resolve_model(teacher))
        if isinstance(teacher, list) and (not teacher or len(set(teacher)) != len(teacher)):
            raise ValueError("teacher needs distinct models")
        if not isinstance(teacher, list) or len(teacher) < 2:
            self._manifest.pop("combine", None)
        self.teacher = teacher
        self._manifest["teacher"] = teacher
        self._record_stage("set_teacher", 0.0, teacher=teacher)

    def _teachers(self):
        return list(self.teacher) if isinstance(self.teacher, (list, tuple)) else [self.teacher]

    def _combine(self):
        return self._manifest.get("combine", "majority")

    def _label_texts(self, texts, prompt, votes, concurrency, desc="labeling"):
        """Label with every configured teacher and combine: one teacher is `label_texts`, several
        label independently, the majority wins, and confidence is the share that agreed.
        """
        teachers = self._teachers()
        if self._combine() == "weighted" and len(teachers) > 1:
            return self._label_weighted(texts, prompt, votes, concurrency, desc)
        if len(teachers) == 1:
            return label_texts(
                texts, prompt, teachers[0], list(self.labels), votes=votes,
                conn=self._cache(), concurrency=concurrency, desc=desc,
            )
        frames, total = [], 0.0
        for model in teachers:
            frame, cost = label_texts(
                texts, prompt, model, list(self.labels), votes=votes, conn=self._cache(),
                concurrency=concurrency, desc=f"{desc} ({model})",
            )
            frames.append(frame)
            total += cost
        return _combine_teachers(frames, texts), total

    def _as_question(self):
        from shrewd.decide import Choice

        return {"label": Choice(instructions=self.instructions,
                                criteria={k: v or k for k, v in self.labels.items()})}

    def _vote_frames(self, texts, prompt, votes, concurrency, desc):
        classes = list(self.labels)
        columns = [f"label__{c}" for c in classes]
        frames, total = [], 0.0
        for model in self._teachers():
            unique, ballots, cost = cast_ballots(
                texts, prompt, model, classes, votes=votes, conn=self._cache(),
                concurrency=concurrency, desc=f"{desc} ({model})",
            )
            total += cost
            shares = np.full((len(unique), len(classes)), np.nan)
            for i, cast in enumerate(ballots):
                valid = [classes.index(v) for v in cast if v is not None]
                if valid:
                    shares[i] = np.bincount(valid, minlength=len(classes)) / len(valid)
            frame = pd.DataFrame(shares, columns=columns)
            frame.insert(0, "text", unique)
            frames.append(frame)
        return frames, total

    def _weighted_agreement(self, prompt, votes, concurrency):
        from shrewd.agree import Agreement
        from shrewd.decisions import gold_frame

        dev = data.read_csv(self.dir / "seed_dev.csv")
        frames, cost = self._vote_frames(dev["text"].tolist(), prompt, votes, concurrency,
                                         "teachers on dev")
        question = self._as_question()
        agreement = Agreement(question, self._teachers(), seed=self.seed).fit(
            frames, dev["text"].tolist(), gold_frame(dev, question)
        )
        return agreement, cost

    def _label_weighted(self, texts, prompt, votes, concurrency, desc):
        agreement, dev_cost = self._weighted_agreement(prompt, votes, concurrency)
        frames, cost = self._vote_frames(texts, prompt, votes, concurrency, desc)
        combined = agreement.combine(frames, list(dict.fromkeys(texts)))
        classes = list(self.labels)
        values = combined[[f"label__{c}" for c in classes]].to_numpy(dtype=float)
        verdicts = {}
        for text, row in zip(combined["text"], values, strict=True):
            if np.isnan(row).any():
                verdicts[text] = (INVALID, 0.0)
            else:
                verdicts[text] = (classes[int(row.argmax())], round(float(row.max()), 4))
        self._last_weighted = (agreement, frames)
        rows = [(text, *verdicts[text]) for text in texts]
        return pd.DataFrame(rows, columns=["text", "label", "confidence"]), cost + dev_cost

    def _write_weighted_outputs(self, labeled, texts):
        from shrewd.agree import pool_status, write_votes

        agreement, frames = self._last_weighted
        unique = list(dict.fromkeys(texts))
        write_votes(self.dir / "votes_pool.csv", self._teachers(), frames)
        summary = agreement.summary(frames, unique, floor=REVIEW_FLOOR)
        (self.dir / "agreement.json").write_text(json.dumps(summary, indent=2))
        cells = pool_status(agreement, frames, unique, floor=REVIEW_FLOOR)["cells"]
        classes = list(self.labels)
        said = {}
        for model, frame in zip(self._teachers(), frames, strict=True):
            values = frame[[f"label__{c}" for c in classes]].to_numpy(dtype=float)
            for text, row in zip(frame["text"], values, strict=True):
                said.setdefault(text, []).append(
                    f"{model}: no answer" if np.isnan(row).any()
                    else f"{model}: {classes[int(row.argmax())]} {row.max():.2f}")
        reviewed = set()
        if (self.dir / "reviewed.csv").exists():
            reviewed = set(pd.read_csv(self.dir / "reviewed.csv", dtype=str)["text"])
        queue = labeled.drop_duplicates("text").copy()
        queue["why"] = [cells.get((t, "label"), "agree") for t in queue["text"]]
        settled_split = (queue["why"] == "split") & (queue["confidence"] >= SPLIT_SETTLED)
        unsettled = ((queue["why"] != "agree") & ~settled_split) | (queue["label"] == INVALID)
        queue = queue[unsettled & ~queue["text"].isin(reviewed)].copy()
        queue["why"] = queue["why"].map({"split": "teachers split", "missing": "no answer "
                                         "from a teacher", "unsure": "unsure"}).fillna("unsure")
        queue["teachers"] = ["; ".join(said.get(t, [])) for t in queue["text"]]
        path = self.dir / "needs_review.csv"
        if queue.empty:
            path.unlink(missing_ok=True)
            return 0
        queue["human_label"] = ""
        queue.sort_values("confidence").to_csv(path, index=False)
        print(f"{len(queue)} rows the teachers split on or the combination is unsure of are in "
              f"needs_review.csv; fill in human_label to overrule them")
        return len(queue)

    def _write_review_queue(self, labeled):
        """Rows the teachers did not settle, with a column for a human to settle them."""
        unsettled = labeled[(labeled["confidence"] < 1.0) | (labeled["label"] == INVALID)]
        path = self.dir / "needs_review.csv"
        if unsettled.empty:
            path.unlink(missing_ok=True)
            return 0
        queue = unsettled.copy()
        queue["human_label"] = ""
        queue.sort_values("confidence").to_csv(path, index=False)
        print(f"{len(queue)} rows the teachers disagreed on or could not answer are in "
              f"needs_review.csv; fill in human_label to overrule them")
        return len(queue)

    def apply_review(self):
        """Overrule the teacher wherever a human filled in `human_label` in needs_review.csv.

        Each fix becomes that row's label at confidence 1.0 in pool_labeled.csv and is
        written to `reviewed.csv`, a ledger that label() re-applies after any relabeling.
        Runs at the start of distill() and compare(). Returns the number applied.
        """
        path = self.dir / "needs_review.csv"
        pool_path = self.dir / "pool_labeled.csv"
        if not path.exists() or not pool_path.exists():
            return 0
        review = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "human_label" not in review.columns:
            return 0
        fixes = review[review["human_label"].str.strip() != ""]
        if fixes.empty:
            return 0
        return self._apply_labels([
            {"text": row["text"], "label": row["human_label"]} for _, row in fixes.iterrows()
        ])

    def review(self, labels):
        """Record your labels on pool rows, as filling in needs_review.csv and calling
        apply_review() would.
        """
        if not (self.dir / "pool_labeled.csv").exists():
            raise FileNotFoundError("no pool_labeled.csv; run label() on your pool first")
        records = labels.to_dict("records") if isinstance(labels, pd.DataFrame) else list(labels)
        applied = self._apply_labels(records)
        path, ledger_path = self.dir / "needs_review.csv", self.dir / "reviewed.csv"
        if applied and path.exists() and ledger_path.exists():
            queue = pd.read_csv(path, dtype=str, keep_default_na=False)
            done = set(pd.read_csv(ledger_path, dtype=str, keep_default_na=False)["text"])
            queue[~queue["text"].isin(done)].to_csv(path, index=False)
        return applied

    def _apply_labels(self, records):
        pool_path = self.dir / "pool_labeled.csv"
        pool = data.read_csv(pool_path)
        index = {t: i for i, t in enumerate(pool["text"])}
        canon = {name.lower(): name for name in self.labels}
        ledgered = set()
        if (self.dir / "reviewed.csv").exists():
            ledgered = set(pd.read_csv(self.dir / "reviewed.csv", dtype=str,
                                       keep_default_na=False)["text"])
        before = data.read_before(self.dir)
        applied, rejected = [], []
        for row in records:
            text, wanted = str(row["text"]), str(row["label"])
            label = canon.get(wanted.strip().lower())
            if label is None:
                rejected.append(f"{wanted!r} is not one of {list(self.labels)}")
                continue
            if text not in index:
                rejected.append(f"{text[:40]!r} is not in the labeled pool")
                continue
            if text not in ledgered and text not in before:
                before[text] = [str(pool.at[index[text], "label"]),
                                float(pool.at[index[text], "confidence"])]
            pool.loc[index[text], ["label", "confidence"]] = [label, 1.0]
            applied.append({"text": text, "label": label})
        if applied:
            pool.to_csv(pool_path, index=False)
            data.write_before(self.dir, before)
            ledger_path = self.dir / "reviewed.csv"
            ledger = pd.DataFrame(applied)
            if ledger_path.exists():
                old = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
                ledger = pd.concat([old, ledger]).drop_duplicates("text", keep="last")
            ledger.to_csv(ledger_path, index=False)
            self._record_stage("review", 0.0, applied=len(applied), rejected=len(rejected))
            print(f"applied {len(applied)} human labels over the teacher's; ledger in reviewed.csv")
        if rejected:
            warnings.warn(
                f"{len(rejected)} human labels could not be applied: " + "; ".join(rejected[:5]),
                stacklevel=2,
            )
        return len(applied)

    def unreview(self, texts):
        """Undo review() on pool rows. The teachers' labels come back."""
        texts = [texts] if isinstance(texts, str) else [str(t) for t in texts]
        ledger_path, pool_path = self.dir / "reviewed.csv", self.dir / "pool_labeled.csv"
        if not ledger_path.exists() or not pool_path.exists():
            return 0
        ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
        back = [t for t in dict.fromkeys(texts) if t in set(ledger["text"])]
        if not back:
            return 0
        before = data.read_before(self.dir)
        valid = set(self.labels) | {INVALID}
        answers = {t: before[t] for t in back if t in before and before[t][0] in valid}
        missing = [t for t in back if t not in answers]
        cost = 0.0
        if missing:
            prompt = (self.dir / "prompt.txt").read_text()
            votes = next((s["params"].get("votes", 1) for s in reversed(self._manifest["stages"])
                          if s["name"] == "label"), 1)
            asked, cost = self._label_texts(missing, prompt, votes, 8, desc="teachers' answers")
            for text, label, confidence in asked[["text", "label", "confidence"]] \
                    .itertuples(index=False):
                answers[text] = [str(label), float(confidence)]
        pool = data.read_csv(pool_path)
        hit = pool["text"].isin(answers)
        pool.loc[hit, "label"] = pool.loc[hit, "text"].map(lambda t: answers[t][0])
        pool.loc[hit, "confidence"] = pool.loc[hit, "text"].map(lambda t: answers[t][1])
        pool.to_csv(pool_path, index=False)
        rest = ledger[~ledger["text"].isin(back)]
        if rest.empty:
            ledger_path.unlink()
        else:
            rest.to_csv(ledger_path, index=False)
        data.write_before(self.dir, {t: v for t, v in before.items() if t not in back})
        self._record_stage("unreview", cost, items=len(back))
        return len(back)

    def _reapply_reviewed(self, labeled):
        ledger_path = self.dir / "reviewed.csv"
        if not ledger_path.exists():
            return labeled
        ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
        fixes = dict(zip(ledger["text"], ledger["label"], strict=True))
        labeled = labeled.copy()
        hit = labeled["text"].isin(fixes)
        if hit.any():
            before = data.read_before(self.dir)
            for text, label, confidence in labeled.loc[hit, ["text", "label", "confidence"]] \
                    .itertuples(index=False):
                before[text] = [str(label), float(confidence)]
            data.write_before(self.dir, before)
            labeled.loc[hit, "label"] = labeled.loc[hit, "text"].map(fixes)
            labeled.loc[hit, "confidence"] = 1.0
            print(f"re-applied {int(hit.sum())} human labels from reviewed.csv")
        return labeled

    def _training_data(self, min_confidence):
        pool_path = self.dir / "pool_labeled.csv"
        if not pool_path.exists():
            raise FileNotFoundError("no pool_labeled.csv; run label() on your pool first")
        self.apply_review()
        pool = data.read_csv(pool_path)
        dev = data.read_csv(self.dir / "seed_dev.csv")
        test = data.read_csv(self.dir / "seed_test.csv")
        splits = self._manifest.get("splits")
        if splits and _rows_hash(test) != splits["test_sha256"]:
            raise ValueError(
                "seed_test.csv does not match the locked split recorded in manifest.json "
                "; restore the file, or re-split explicitly with add_seed(overwrite=True)"
            )

        recorded = self._manifest.get("label_prompt_sha256")
        prompt_path = self.dir / "prompt.txt"
        if recorded and prompt_path.exists():
            current = hashlib.sha256(prompt_path.read_text().encode()).hexdigest()
            if current != recorded:
                warnings.warn(
                    "prompt.txt has changed since label() ran; pool_labeled.csv still "
                    "holds the old prompt's labels. Re-run label() to refresh them.",
                    stacklevel=3,
                )

        valid, n_seed_dupes = data.usable_pool(pool, dev, test)
        if n_seed_dupes:
            print(
                f"{n_seed_dupes} pool rows duplicate seed texts and were left out of training "
                "(the locked test set stays unseen)"
            )
        kept = valid[valid["confidence"] >= min_confidence]
        held_out = len(valid) - len(kept)
        if held_out:
            print(f"{held_out} rows below min_confidence={min_confidence} left out of training")
        return pool, dev, test, valid, kept

    def distill(self, student="tfidf", min_confidence=0.0, route_budget=None, route_delta=0.05,
                route_frac=0.3, **student_kwargs):
        """Train a student on the teacher-labeled pool, then evaluate both on the test set.

        `route_budget=0.02` holds part of the pool out of training to set a routing threshold,
        so that at most 2% of the answers given locally disagree with the teacher, with
        probability 1 - `route_delta`. Off by default.
        """
        prior = sum(1 for s in self._manifest["stages"] if s["name"] == "distill")
        if prior:
            warnings.warn(
                f"this will be evaluation #{prior + 1} of the locked test set. Repeated "
                "evaluations erode what it can certify. compare() ranks candidates on "
                "dev for free; distill() is meant for the final configuration.",
                stacklevel=2,
            )
        pool, dev, test, valid, kept = self._training_data(min_confidence)
        held = None
        if route_budget is not None:
            order = np.random.default_rng(self.seed).permutation(len(kept))
            cut = int(round(route_frac * len(kept)))
            held = kept.iloc[np.sort(order[:cut])]
            kept = kept.iloc[np.sort(order[cut:])]
            print(f"holding {len(held)} labeled pool rows out of training to set the routing "
                  f"threshold (budget {route_budget:.1%}, {1 - route_delta:.0%} guarantee)")
        train_texts = kept["text"].tolist() + dev["text"].tolist()
        train_labels = kept["label"].tolist() + dev["label"].tolist()
        if isinstance(student, str):
            if student not in STUDENTS:
                raise ValueError(f"unknown student {student!r}, pick from {sorted(STUDENTS)}")
            kind = student
            student = STUDENTS[kind](seed=self.seed, **student_kwargs)
        else:
            if student_kwargs:
                raise TypeError(
                    "student kwargs only apply to registry names; configure your "
                    "instance directly instead"
                )
            kind = type(student).__name__
        print(
            f"training {kind} student on {len(train_texts)} rows "
            f"({len(kept)} pool + {len(dev)} seed dev)"
        )
        student.fit(train_texts, train_labels)
        # a previous student of a different type may have left files save() won't overwrite
        shutil.rmtree(self.dir / "student", ignore_errors=True)
        student.save(self.dir / "student")
        if held is not None:
            from shrewd.route import HELD_FILE, pick_threshold
            from shrewd.students import ROUTING_FILE

            proba = student.predict_proba(held["text"].tolist())
            said = np.array([student.classes_[i] for i in proba.argmax(axis=1)])
            scores, disagree = proba.max(axis=1), said != held["label"].to_numpy()
            _, stats = pick_threshold(scores, disagree, route_budget, route_delta)
            student.routing = stats
            (self.dir / "student" / ROUTING_FILE).write_text(json.dumps(stats, indent=2))
            (self.dir / "student" / HELD_FILE).write_text(json.dumps(
                {"scores": [round(float(s), 6) for s in scores],
                 "disagree": [bool(d) for d in disagree]}))

        prompt = (self.dir / "prompt.txt").read_text()
        teacher_labeled, cost = self._label_texts(
            test["text"].tolist(), prompt, 1, 8, desc="teacher on test"
        )
        teacher_preds = teacher_labeled["label"].tolist()
        student_proba = student.predict_proba(test["text"].tolist())
        student_preds = [student.classes_[i] for i in student_proba.argmax(axis=1)]

        classes = list(self.labels)
        metrics = {
            "classes": classes,
            "student_type": kind,
            "n_train": len(train_texts),
            "n_test": len(test),
            "teacher_invalid_on_test": teacher_preds.count(INVALID),
            "pool_rows_duplicating_seed": int(
                pool["text"].isin(set(dev["text"]) | set(test["text"])).sum()
            ),
            "teacher": evaluate.compute_metrics(test["label"], teacher_preds, classes),
            "student": {
                **evaluate.compute_metrics(test["label"], student_preds, classes),
                "selective": evaluate.selective_curve(
                    test["label"], student_proba, list(student.classes_)
                ),
            },
            "teacher_minus_student_ci": evaluate.paired_gap_interval(
                test["label"], teacher_preds, student_preds, classes
            ),
        }
        if held is not None:
            local = student._local_from(student_proba)
            gold = test["label"].to_numpy()
            n_local = int(local.sum())
            metrics["routing"] = {
                **student.routing,
                "test": {
                    "coverage": round(float(local.mean()), 4) if len(local) else 0.0,
                    "n_local": n_local,
                    "student_error": round(float((np.array(student_preds)[local]
                                                  != gold[local]).mean()), 4) if n_local else None,
                    "teacher_error": round(float((np.array(teacher_preds)[local]
                                                  != gold[local]).mean()), 4) if n_local else None,
                },
            }
        last_label = next(
            (s["params"] for s in reversed(self._manifest["stages"]) if s["name"] == "label"),
            {},
        )
        votes = last_label.get("votes", 1)
        ctx = {
            "teacher": metrics["teacher"],
            "student": metrics["student"],
            "student_type": kind,
            "gap_ci": metrics["teacher_minus_student_ci"],
            "student_selective": metrics["student"]["selective"],
            "classes": classes,
            "labels": self.labels,
            "target": next(
                (s["params"]["target"] for s in reversed(self._manifest["stages"])
                 if s["name"] == "optimize"),
                None,
            ),
            "train_counts": dict(Counter(train_labels)),
            "pool_confidence": valid["confidence"].tolist(),
            "n_pool": len(pool),
            "n_invalid": int((pool["label"] == INVALID).sum()),
            "votes": votes,
            "active": last_label.get("mode") == "active",
            "seed_counts": dict(Counter(dev["label"]) + Counter(test["label"])),
            "pool_counts": dict(Counter(valid["label"])),
            "test_examples": [
                {"text": text, "gold": gold, "teacher": pred}
                for text, gold, pred in zip(
                    test["text"], test["label"], teacher_preds, strict=True
                )
            ],
        }
        findings = evaluate.run_rules(ctx)
        result = evaluate.DistillResult(metrics=metrics, findings=findings)
        (self.dir / "report.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "findings": [asdict(f) for f in findings],
                    "params": {"student": kind, "min_confidence": min_confidence,
                               **({"route_budget": route_budget}
                                  if route_budget is not None else {})},
                },
                indent=2,
            )
        )
        self._record_stage("distill", cost, student=kind, min_confidence=min_confidence,
                           **({"route_budget": route_budget} if route_budget is not None else {}))
        print(
            f"teacher test macro-F1 {metrics['teacher']['macro_f1']:.3f} | "
            f"student {metrics['student']['macro_f1']:.3f} | "
            f"{len(findings)} finding(s); print(result.report()) for details"
        )
        return result

    def ask(self, texts):
        """The teacher's labels for `texts`, cached like label()'s. Use it as
        `load(dir).predict(texts, fallback=proj.ask)`.
        """
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        prompt = (self.dir / "prompt.txt").read_text()
        labeled, _ = self._label_texts(texts, prompt, 1, 8, desc="teacher fallback")
        labels = [None if label == INVALID else label for label in labeled["label"]]
        return labels[0] if single else labels

    def compare(self, students, min_confidence=0.0):
        """Train candidate students on the labeled pool alone and rank them on the dev split, so
        picking a winner never touches the locked test set. Each candidate is saved under
        candidates/<name>/. promote() ships one as-is, distill(student=...) retrains it on
        pool + dev and runs the one test-set evaluation.
        """
        pool, dev, test, valid, kept = self._training_data(min_confidence)
        train_texts, train_labels = kept["text"].tolist(), kept["label"].tolist()
        dev_texts = dev["text"].tolist()
        classes = list(self.labels)

        rows = []
        for entry in students:
            if isinstance(entry, str):
                if entry not in STUDENTS:
                    raise ValueError(f"unknown student {entry!r}, pick from {sorted(STUDENTS)}")
                student, name = STUDENTS[entry](seed=self.seed), entry
            else:
                student = entry
                name = type(entry).__name__.lower().removesuffix("student")
            base, n = name, 2
            while any(r["student"] == name for r in rows):
                name = f"{base}-{n}"
                n += 1
            print(f"training candidate {name} on {len(train_texts)} pool rows")
            student.fit(train_texts, train_labels)
            candidate_dir = self.dir / "candidates" / name
            shutil.rmtree(candidate_dir, ignore_errors=True)
            student.save(candidate_dir)
            student.predict(dev_texts[:1])  # warm up before timing (tokenizer, device)
            started = time.perf_counter()
            proba = student.predict_proba(dev_texts)
            ms_per_item = (time.perf_counter() - started) * 1000 / len(dev_texts)
            preds = [student.classes_[i] for i in proba.argmax(axis=1)]
            metrics = evaluate.compute_metrics(dev["label"], preds, classes)
            at_80 = evaluate.selective_curve(
                dev["label"], proba, list(student.classes_), coverages=(0.8,)
            )[0]
            size = sum(f.stat().st_size for f in candidate_dir.rglob("*") if f.is_file())
            rows.append(
                {
                    "student": name,
                    "dev_macro_f1": round(metrics["macro_f1"], 3),
                    "dev_accuracy": round(metrics["accuracy"], 3),
                    "dev_accuracy_at_80pct": round(at_80["accuracy"], 3),
                    "min_confidence_for_80pct": at_80["threshold"],
                    "ms_per_item": round(ms_per_item, 2),
                    "size_mb": round(size / 1e6, 1),
                }
            )
        rows.sort(key=lambda r: r["dev_macro_f1"], reverse=True)

        width = max(len(r["student"]) for r in rows) + 2
        print(f"\n{'candidate':{width}}  dev macro-F1  acc@80% cov  ms/item  size")
        for r in rows:
            print(
                f"{r['student']:{width}}  {r['dev_macro_f1']:>12.3f}"
                f"  {r['dev_accuracy_at_80pct']:>11.3f}"
                f"  {r['ms_per_item']:>7.2f}  {r['size_mb']:>5.1f}MB"
            )
        print(
            f"\npromote one with promote({rows[0]['student']!r}), or get its test-set "
            f"number with distill(student={rows[0]['student']!r})"
        )
        (self.dir / "compare.json").write_text(json.dumps({"candidates": rows}, indent=2))
        self._record_stage(
            "compare", 0.0, students=[r["student"] for r in rows],
            min_confidence=min_confidence,
        )
        return rows

    def promote(self, name):
        """Copy candidates/<name>/ to student/, the model load() serves."""
        source = self.dir / "candidates" / name
        if not (source / "meta.json").exists():
            have = sorted(p.name for p in (self.dir / "candidates").glob("*") if p.is_dir())
            raise FileNotFoundError(f"no candidate {name!r}; run compare() first (have: {have})")
        target = self.dir / "student"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        self._record_stage("promote", 0.0, student=name)
        print(f"promoted {name} → {target}")


def _hash_unit(text, seed):
    digest = hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()
    return int(digest[:12], 16) / 16 ** 12


def _last_test_frac(manifest):
    return next((s["params"].get("test_frac", 0.35) for s in reversed(manifest["stages"])
                 if s["name"] == "add_seed"), 0.35)


def _rows_hash(df):
    joined = "\n".join(sorted(df["text"] + "\t" + df["label"]))
    return hashlib.sha256(joined.encode()).hexdigest()


def _combine_teachers(frames, texts):
    """Majority label across teachers, confidence is the share that agreed.

    A tie goes to the teacher listed first, so order the list by trust. A teacher's
    invalid answer is not a vote, and a row every teacher failed on stays INVALID.
    """
    by_text = [dict(zip(f["text"], f["label"], strict=True)) for f in frames]
    rows = []
    for text in texts:
        votes = [m.get(text) for m in by_text]
        valid = [v for v in votes if v is not None and v != INVALID]
        if not valid:
            rows.append((text, INVALID, 0.0))
            continue
        counts = Counter(valid)
        top = max(counts.values())
        winner = next(v for v in valid if counts[v] == top)
        rows.append((text, winner, round(top / len(frames), 2)))
    return pd.DataFrame(rows, columns=["text", "label", "confidence"])
