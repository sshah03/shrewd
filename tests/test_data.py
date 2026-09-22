import pandas as pd
import pytest
from conftest import LABELS, make_seed

from shrewd import data

CLASSES = list(LABELS)


def prepare(df, **kwargs):
    defaults = {"labels": CLASSES, "test_frac": 0.35, "seed": 42}
    return data.prepare_seed(df, **{**defaults, **kwargs})


def test_missing_columns():
    with pytest.raises(ValueError, match="missing columns.*label"):
        prepare(pd.DataFrame({"text": ["hi"]}))


def test_unknown_labels():
    df = make_seed()
    df.loc[0, "label"] = "spam"
    with pytest.raises(ValueError, match="spam"):
        prepare(df)


def test_null_rows():
    df = make_seed()
    df.loc[3, "text"] = None
    with pytest.raises(ValueError, match="empty text or label"):
        prepare(df)


def test_class_too_small():
    df = make_seed(30)
    df = df[~((df["label"] == "bug") & (df.index % 5 != 0))]  # keep 6 bug rows
    with pytest.raises(ValueError, match=r"bug \(6\)"):
        prepare(df)


def test_duplicates_dropped_with_warning():
    df = pd.concat([make_seed(), make_seed().head(5)], ignore_index=True)
    with pytest.warns(UserWarning, match="5 duplicate"):
        dev, test = prepare(df)
    assert len(dev) + len(test) == len(make_seed())


def test_small_test_class_warning():
    with pytest.warns(UserWarning, match="fewer than 25 test examples"):
        prepare(make_seed(30))


def test_small_dev_warning():
    with pytest.warns(UserWarning, match="works best with 100"):
        prepare(make_seed(20))


def test_no_size_warnings_when_big():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare(make_seed(80))


def test_split_deterministic_and_disjoint():
    dev1, test1 = prepare(make_seed())
    dev2, test2 = prepare(make_seed())
    pd.testing.assert_frame_equal(dev1, dev2)
    pd.testing.assert_frame_equal(test1, test2)
    assert not set(dev1["text"]) & set(test1["text"])


def test_split_stratified():
    dev, test = prepare(make_seed(40))
    for frame in (dev, test):
        counts = frame["label"].value_counts()
        assert counts.max() - counts.min() <= 1


def test_read_csv_preserves_text(tmp_path):
    df = pd.DataFrame({"text": ["NA", "null", "line\nbreak", "123"], "label": ["other"] * 4})
    df.to_csv(tmp_path / "x.csv", index=False)
    back = data.read_csv(tmp_path / "x.csv")
    assert back["text"].tolist() == ["NA", "null", "line\nbreak", "123"]
