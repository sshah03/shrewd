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
from shrewd.teacher import INVALID, estimate_calls, label_texts, open_cache, resolve_model


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
              batch=None):
        """Teacher-label a pool of unlabeled texts. Resumable, so pass the full pool each time.

        With neither `budget_usd` nor `n`, every row is labeled. With either, rows are acquired
        in rounds: a tfidf probe trained on what is labeled so far picks the rows it is least
        sure about, spread across clusters, and its dev accuracy is logged to label_log.json.
        Stops at the budget, at `n` rows, or when the pool is exhausted.
        """
        if "text" not in df.columns:
            raise ValueError("pool data needs a 'text' column")
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
            for model in teachers:
                calls, price = estimate_calls(texts, prompt, model, votes, self._cache())
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

    def _teachers(self):
        return list(self.teacher) if isinstance(self.teacher, (list, tuple)) else [self.teacher]

    def _label_texts(self, texts, prompt, votes, concurrency, desc="labeling"):
        """Label with every configured teacher and combine: one teacher is `label_texts`, several
        label independently, the majority wins, and confidence is the share that agreed.
        """
        teachers = self._teachers()
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
        pool = data.read_csv(pool_path)
        index = {t: i for i, t in enumerate(pool["text"])}
        canon = {name.lower(): name for name in self.labels}
        applied, rejected = [], []
        for _, row in fixes.iterrows():
            label = canon.get(row["human_label"].strip().lower())
            if label is None:
                rejected.append(f"{row['human_label']!r} is not one of {list(self.labels)}")
                continue
            if row["text"] not in index:
                rejected.append(f"{row['text'][:40]!r} is not in the labeled pool")
                continue
            pool.loc[index[row["text"]], ["label", "confidence"]] = [label, 1.0]
            applied.append({"text": row["text"], "label": label})
        if applied:
            pool.to_csv(pool_path, index=False)
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

    def _reapply_reviewed(self, labeled):
        ledger_path = self.dir / "reviewed.csv"
        if not ledger_path.exists():
            return labeled
        ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
        fixes = dict(zip(ledger["text"], ledger["label"], strict=True))
        labeled = labeled.copy()
        hit = labeled["text"].isin(fixes)
        if hit.any():
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

    def distill(self, student="tfidf", min_confidence=0.0, **student_kwargs):
        """Train a student on the teacher-labeled pool, then evaluate both on the test set."""
        prior = sum(1 for s in self._manifest["stages"] if s["name"] == "distill")
        if prior:
            warnings.warn(
                f"this will be evaluation #{prior + 1} of the locked test set. Repeated "
                "evaluations erode what it can certify. compare() ranks candidates on "
                "dev for free; distill() is meant for the final configuration.",
                stacklevel=2,
            )
        pool, dev, test, valid, kept = self._training_data(min_confidence)
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
                    "params": {"student": kind, "min_confidence": min_confidence},
                },
                indent=2,
            )
        )
        self._record_stage("distill", cost, student=kind, min_confidence=min_confidence)
        print(
            f"teacher test macro-F1 {metrics['teacher']['macro_f1']:.3f} | "
            f"student {metrics['student']['macro_f1']:.3f} | "
            f"{len(findings)} finding(s); print(result.report()) for details"
        )
        return result

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
