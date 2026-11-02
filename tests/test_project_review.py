import pandas as pd
import pytest
from conftest import LABELS, keyword_label, make_pool, make_seed, response


class TwoTeachers:
    """Two fake models that disagree on refund tickets: `a` says billing, `b` says bug."""

    def __init__(self):
        self.calls = 0
        self.by_model = {}

    def __call__(self, model, messages, **kwargs):
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1
        text = messages[1]["content"]
        label = keyword_label(text)
        if model == "test/b" and "refund" in text.lower():
            label = "bug"
        return response(f'{{"label": "{label}"}}')


@pytest.fixture
def two_teachers(monkeypatch):
    fake = TwoTeachers()
    monkeypatch.setattr("shrewd.teacher.litellm.completion", fake)
    monkeypatch.setattr("shrewd.teacher.litellm.completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr("shrewd.teacher.time.sleep", lambda seconds: None)
    return fake


def _project(tmp_path, teacher):
    from shrewd import Project

    p = Project(tmp_path / "proj", instructions="Classify tickets.", labels=LABELS, teacher=teacher)
    p.add_seed(make_seed())
    (p.dir / "prompt.txt").write_text("classify tickets")
    return p


def test_two_teachers_vote_and_disagreements_fill_the_review_queue(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    assert p.teacher == ["test/a", "test/b"]
    pool = make_pool()
    labeled = p.label(pool)
    assert two_teachers.by_model == {"test/a": len(pool), "test/b": len(pool)}
    refund = labeled[labeled["text"].str.contains("refund")]
    assert (refund["confidence"] == 0.5).all()
    assert (refund["label"] == "billing").all()            # tie -> the first teacher listed
    settled = labeled[~labeled["text"].str.contains("refund")]
    assert (settled["confidence"] == 1.0).all()
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    assert set(review["text"]) == set(refund["text"])
    assert (review["human_label"] == "").all()


def test_a_human_label_overrules_the_teachers_and_survives_relabeling(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    pool = make_pool()
    p.label(pool)
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_label"] = "cancellation"
    fixed_text = review.loc[0, "text"]
    review.to_csv(p.dir / "needs_review.csv", index=False)

    assert p.apply_review() == 1
    labeled = pd.read_csv(p.dir / "pool_labeled.csv")
    row = labeled[labeled["text"] == fixed_text].iloc[0]
    assert row["label"] == "cancellation" and row["confidence"] == 1.0
    assert (p.dir / "reviewed.csv").exists()

    p.label(pool)                                            # relabel: the fix comes back
    labeled = pd.read_csv(p.dir / "pool_labeled.csv")
    assert labeled[labeled["text"] == fixed_text].iloc[0]["label"] == "cancellation"

    result = p.distill(student="tfidf")                      # distill applies the queue too
    assert result.metrics["n_train"] > 0


def test_a_bad_human_label_is_reported_not_applied(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    p.label(make_pool())
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_label"] = "not-a-label"
    review.to_csv(p.dir / "needs_review.csv", index=False)
    with pytest.warns(UserWarning, match="not one of"):
        assert p.apply_review() == 0


def test_one_teacher_has_an_empty_queue_and_is_unchanged(tmp_path, fake_teacher):
    p = _project(tmp_path, "test/fake-model")
    labeled = p.label(make_pool())
    assert (labeled["confidence"] == 1.0).all()
    assert not (p.dir / "needs_review.csv").exists()


def test_active_acquisition_refuses_several_teachers(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    with pytest.raises(ValueError, match="several teachers"):
        p.label(make_pool(), n=10)


def test_dry_run_counts_every_teacher(tmp_path, two_teachers, capsys):
    p = _project(tmp_path, ["test/a", "test/b"])
    p.label(make_pool(), dry_run=True)
    assert "× 2 teacher(s) = 80 calls" in capsys.readouterr().out
    assert two_teachers.calls == 0


def test_ensemble_teacher_round_trips_through_the_manifest(tmp_path, two_teachers):
    from shrewd import Project

    _project(tmp_path, ["test/a", "test/b"])
    assert Project(tmp_path / "proj").teacher == ["test/a", "test/b"]


def test_pending_human_labels_survive_a_relabel(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    pool = make_pool()
    p.label(pool)
    review = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    review.loc[0, "human_label"] = "other"
    text = review.loc[0, "text"]
    review.to_csv(p.dir / "needs_review.csv", index=False)
    p.label(pool)                                            # no apply_review() in between
    labeled = pd.read_csv(p.dir / "pool_labeled.csv")
    assert labeled[labeled["text"] == text].iloc[0]["label"] == "other"



def test_weighted_combination_learns_which_teacher_to_trust(tmp_path, two_teachers):
    """On dev, `b` calls every refund ticket a bug while you labeled them billing, so where
    the two split on a refund the combination sides with `a`, and is sure of it."""
    import json

    p = _project(tmp_path, ["test/a", "test/b"])
    pool = make_pool()
    labeled = p.label(pool, combine="weighted")
    assert json.loads((p.dir / "manifest.json").read_text())["combine"] == "weighted"
    refund = labeled[labeled["text"].str.contains("refund")]
    assert (refund["label"] == "billing").all()
    assert (refund["confidence"] > 0.9).all()           # majority vote would say 0.5
    summary = json.loads((p.dir / "agreement.json").read_text())
    assert summary["questions"]["label"]["method"] == "dawid-skene"
    accuracy = summary["questions"]["label"]["dev_accuracy"]
    assert accuracy["test/a"] > accuracy["test/b"]
    # they split, but the combination is sure, so nobody needs to look
    assert not (p.dir / "needs_review.csv").exists()
    votes = pd.read_csv(p.dir / "votes_pool.csv")
    assert set(votes["teacher"]) == {"test/a", "test/b"}

    result = p.distill()                                 # remembered: the test set is combined too
    assert result.metrics["teacher"]["macro_f1"] == pytest.approx(1.0)


def test_weighted_combination_needs_several_teachers(tmp_path, fake_teacher):
    p = _project(tmp_path, "test/fake-model")
    with pytest.raises(ValueError, match="several teachers"):
        p.label(make_pool(), combine="weighted")
    with pytest.raises(ValueError, match="majority"):
        p.label(make_pool(), combine="vote")


def test_review_records_labels_without_editing_a_csv(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    p.label(make_pool())
    queue = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    text = queue.iloc[0]["text"]
    assert p.review([{"text": text, "label": "Bug"}]) == 1
    pool = pd.read_csv(p.dir / "pool_labeled.csv")
    assert pool.set_index("text").loc[text, "label"] == "bug"
    after = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    assert text not in set(after["text"])


def test_a_classifier_can_route_what_it_is_unsure_of(tmp_path, two_teachers):
    import json

    from shrewd import load

    p = _project(tmp_path, "test/a")
    pool = make_pool(n_per_class=120)
    p.label(pool)
    result = p.distill(route_budget=0.05)
    routing = result.metrics["routing"]
    assert routing["n"] > 100 and "test" in routing
    assert "routing:" in result.report()
    saved = json.loads((p.dir / "student" / "routing.json").read_text())
    assert saved["budget"] == 0.05
    meta = json.loads((p.dir / "student" / "meta.json").read_text())
    assert "routing" not in meta                      # older versions read meta.json unchanged
    clf = load(p.dir)
    texts = ["I was charged twice for order", "zzz qqq unknown words"]
    local = clf.local(texts)
    routed = clf.predict(texts, route=True)
    assert [label is not None for label in routed] == list(local)
    filled = clf.predict(texts, fallback=lambda rest: ["other"] * len(rest))
    assert None not in filled
    assert p.ask("I was charged twice for order") == "billing"


def test_a_classifier_without_routing_says_how_to_add_it(tmp_path, two_teachers):
    from shrewd import load

    p = _project(tmp_path, "test/a")
    p.label(make_pool())
    p.distill()
    assert not (p.dir / "student" / "routing.json").exists()
    with pytest.raises(ValueError, match="route_budget"):
        load(p.dir).predict(["x"], route=True)


def test_checking_teachers_on_dev_finds_where_they_part_ways_with_you(tmp_path, two_teachers):
    import json

    p = _project(tmp_path, ["test/a", "test/b"])
    assert p.check_teachers(dry_run=True)["uncached_calls"] > 0
    summary = p.check_teachers()
    entry = summary["questions"]["label"]
    assert entry["dev_accuracy"]["test/a"] > entry["dev_accuracy"]["test/b"]
    gap = summary["disagreements"]["label"]
    # b calls refund tickets bugs; the combination (out of fold) mostly overrules it, so
    # whatever is left is billing vs bug
    for pair in gap["pairs"]:
        assert {pair["yours"], pair["theirs"]} == {"billing", "bug"}
        assert pair["examples"]
    assert json.loads((p.dir / "teacher_check.json").read_text())["n_dev"] == summary["n_dev"]
    calls = two_teachers.calls
    p.label(make_pool(), combine="weighted")
    # the dev answers were already paid for by the check
    assert two_teachers.calls - calls == 2 * len(make_pool())


def test_a_classifiers_limit_moves_without_retraining(tmp_path, two_teachers):
    from shrewd import load
    from shrewd.route import curve, rebudget

    p = _project(tmp_path, "test/a")
    p.label(make_pool(n_per_class=120))
    p.distill(route_budget=0.02)
    assert len(curve(p.dir)) == 4
    rebudget(p.dir, 0.1)
    assert load(p.dir).routing["budget"] == 0.1


def test_examples_can_be_added_and_fixed_without_breaking_the_lock(tmp_path, two_teachers):
    import json

    p = _project(tmp_path, "test/a")
    splits = json.loads((p.dir / "manifest.json").read_text())["splits"]
    extra = pd.DataFrame({"text": [f"refund please for order {i}" for i in range(40)],
                          "label": "billing"})
    dev_n, test_n = p.add_examples(extra)
    assert dev_n + test_n == 40 and 0 < test_n < 40
    assert p.add_examples(extra) == (0, 0)                     # already there: skipped
    test = pd.read_csv(p.dir / "seed_test.csv")
    text = test["text"].iloc[0]
    assert p.fix_example(text, "other") == "test"
    after = json.loads((p.dir / "manifest.json").read_text())
    assert after["splits"]["n_test"] == splits["n_test"] + test_n
    p.label(make_pool())
    p.distill()                                                # the lock check still passes
    with pytest.raises(ValueError, match="not one of"):
        p.fix_example(text, "nonsense")


def test_label_descriptions_can_change_and_rewrite_an_untuned_prompt(tmp_path, two_teachers):
    from shrewd import Project

    p = Project(tmp_path / "q", instructions="Classify tickets.", labels=LABELS, teacher="test/a")
    p.add_seed(make_seed())
    from shrewd.optimize import build_seed_prompt

    (p.dir / "prompt.txt").write_text(build_seed_prompt(p.instructions, p.labels))
    assert p.set_labels({"other": "greetings, praise and questions about hours"})
    assert "praise" in (p.dir / "prompt.txt").read_text()
    assert Project(tmp_path / "q").labels["other"].startswith("greetings")
    with pytest.raises(ValueError, match="names can't change"):
        p.set_labels({"refunds": "x"})


def test_labels_merge_into_others_everywhere_without_asking_again(tmp_path, two_teachers):
    import json
    import warnings

    p = _project(tmp_path, ["test/a", "test/b"])
    pool = make_pool()
    p.label(pool, combine="weighted")
    p.label(pool)  # back to majority, so the review queue exists too
    calls = two_teachers.calls
    merged, into = list(LABELS)[0], list(LABELS)[1]
    changed = p.merge_labels({merged: into})
    assert two_teachers.calls == calls                    # nothing asked again
    assert merged not in p.labels and into in p.labels
    manifest = json.loads((p.dir / "manifest.json").read_text())
    assert merged not in manifest["labels"]
    for name in ("seed_dev.csv", "seed_test.csv", "pool_labeled.csv"):
        assert merged not in set(pd.read_csv(p.dir / name)["label"])
    assert changed["seed_dev.csv"] + changed["seed_test.csv"] > 0
    votes = pd.read_csv(p.dir / "votes_pool.csv")
    assert f"label__{merged}" not in votes.columns
    share = votes[[c for c in votes.columns if c.startswith("label__")]].sum(axis=1)
    assert share.round(6).eq(1).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)       # the pool matches the new prompt
        warnings.filterwarnings("ignore", message="only .* dev rows")
        warnings.filterwarnings("ignore", message="question|label .* rarest|test scores")
        p.distill()                                       # and the lock still holds


def test_merging_needs_a_label_that_stays(tmp_path, two_teachers):
    p = _project(tmp_path, "test/a")
    a, b = list(LABELS)[:2]
    with pytest.raises(ValueError, match="not a label"):
        p.merge_labels({"nope": a})
    with pytest.raises(ValueError, match="stays"):
        p.merge_labels({a: b, b: a})
    assert p.merge_labels({}) == {}


def test_removing_a_label_drops_its_rows_and_keeps_the_lock_consistent(tmp_path, two_teachers):
    import json

    p = _project(tmp_path, "test/a")
    p.label(make_pool())
    gone = list(LABELS)[-1]
    before = pd.read_csv(p.dir / "seed_test.csv")
    with pytest.warns(UserWarning, match="test examples"):
        changed = p.remove_label(gone)
    after = pd.read_csv(p.dir / "seed_test.csv")
    assert len(after) == len(before) - changed["seed_test.csv"]
    assert gone not in set(after["label"]) and gone not in p.labels
    assert gone not in set(pd.read_csv(p.dir / "pool_labeled.csv")["label"])
    splits = json.loads((p.dir / "manifest.json").read_text())["splits"]
    assert splits["n_test"] == len(after)
    p.distill()                                           # the updated hash matches the file


def test_a_label_can_be_removed_into_another(tmp_path, two_teachers):
    p = _project(tmp_path, "test/a")
    p.label(make_pool())
    a, b = list(LABELS)[:2]
    p.remove_label(a, into=b)
    assert a not in p.labels
    assert a not in set(pd.read_csv(p.dir / "seed_dev.csv")["label"])


def test_a_classifier_keeps_at_least_two_labels(tmp_path, two_teachers):
    from shrewd import Project

    p = Project(tmp_path / "two", instructions="x", labels={"a": "", "b": ""}, teacher="test/a")
    with pytest.raises(ValueError, match="two labels"):
        p.remove_label("a")


def test_a_label_can_be_added_and_the_prompt_follows(tmp_path, two_teachers):
    p = _project(tmp_path, "test/a")
    p.label(make_pool())
    with pytest.warns(UserWarning, match="label\\(\\) it again"):
        p.add_label("shipping", "delivery and tracking")
    assert p.labels["shipping"] == "delivery and tracking"
    assert "shipping" in (p.dir / "prompt.txt").read_text()
    with pytest.raises(ValueError, match="already"):
        p.add_label("Shipping")


def test_projects_that_never_merge_are_unchanged(tmp_path, two_teachers):
    """The new methods are opt-in: a project that doesn't call them writes the same files."""
    import json

    p = _project(tmp_path, "test/a")
    p.label(make_pool())
    stages = [s["name"] for s in json.loads((p.dir / "manifest.json").read_text())["stages"]]
    assert not {"merge_labels", "remove_label", "add_label"} & set(stages)


def test_prune_keeps_what_a_shipped_classifier_needs(tmp_path, two_teachers):
    from shrewd import Project, load, prune

    p = _project(tmp_path, ["test/a", "test/b"])
    p.label(make_pool())
    p.distill()
    would = prune(p.dir, dry_run=True)
    assert {path.name for path in would} >= {"seed_test.csv", "pool_labeled.csv", "cache.db"}
    assert all(path.exists() for path in would)
    removed = prune(p.dir, keep=("notes.txt",))
    assert removed == would and not any(path.exists() for path in removed)
    # several teachers: the dev split stays so the fallback can reweigh them
    assert {path.name for path in p.dir.iterdir()} <= {
        "manifest.json", "student", "prompt.txt", "report.json", "agreement.json", "seed_dev.csv"}
    clf = load(p.dir)
    assert clf.predict(["I was charged twice for order"]) == ["billing"]
    assert Project(p.dir).ask("I was charged twice for order") == "billing"


def test_prune_needs_a_shipped_project(tmp_path, two_teachers):
    from shrewd import prune

    p = _project(tmp_path, "test/a")
    with pytest.raises(FileNotFoundError, match="distill"):
        prune(p.dir)
    assert (p.dir / "seed_test.csv").exists()


def test_unreview_puts_the_teachers_answer_back_without_asking(tmp_path, two_teachers):
    p = _project(tmp_path, ["test/a", "test/b"])
    labeled = p.label(make_pool())
    refund = labeled[labeled["text"].str.contains("refund")].iloc[0]
    other = labeled[~labeled["text"].str.contains("refund")].iloc[0]
    p.review([{"text": refund["text"], "label": "other"}, {"text": other["text"], "label": "bug"}])
    calls = two_teachers.calls
    assert p.unreview(refund["text"]) == 1
    pool = pd.read_csv(p.dir / "pool_labeled.csv").set_index("text")
    assert pool.at[refund["text"], "label"] == refund["label"]
    assert pool.at[refund["text"], "confidence"] == refund["confidence"]
    assert pool.at[other["text"], "label"] == "bug"           # the other answer stands
    ledger = pd.read_csv(p.dir / "reviewed.csv")
    assert list(ledger["text"]) == [other["text"]]
    queue = pd.read_csv(p.dir / "needs_review.csv", dtype=str, keep_default_na=False)
    assert refund["text"] not in set(queue["text"])           # you've seen it: not asked again
    assert two_teachers.calls == calls
    assert p.unreview(refund["text"]) == 0                    # nothing left to take back

    # after relabeling, the answer put back is the fresh one; and without the saved answers
    # (reviews from older versions) they come from the cache
    p.label(make_pool())
    (p.dir / "review_before.json").unlink()
    calls = two_teachers.calls
    assert p.unreview([other["text"]]) == 1
    assert not (p.dir / "reviewed.csv").exists()
    pool = pd.read_csv(p.dir / "pool_labeled.csv").set_index("text")
    assert pool.at[other["text"], "label"] == other["label"]
    assert two_teachers.calls == calls


def test_examples_can_be_removed_and_the_test_hash_follows(tmp_path, two_teachers):
    import json as _json

    p = _project(tmp_path, "test/a")
    dev, test = pd.read_csv(p.dir / "seed_dev.csv"), pd.read_csv(p.dir / "seed_test.csv")
    with pytest.warns(UserWarning, match="comparable"):
        assert p.remove_examples([dev["text"].iloc[0], test["text"].iloc[0], "nope"]) == (1, 1)
    splits = _json.loads((p.dir / "manifest.json").read_text())["splits"]
    assert (splits["n_dev"], splits["n_test"]) == (len(dev) - 1, len(test) - 1)
    p.label(make_pool())
    p.distill()                  # the updated hash means no tampering complaint
    assert p.remove_examples("nope") == (0, 0)
