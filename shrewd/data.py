import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

INVALID = "__invalid__"  # the label given to texts that never produced a parseable response


def read_csv(path):
    return pd.read_csv(path, dtype={"text": str, "label": str}, keep_default_na=False)


def prepare_seed(df, labels, test_frac, seed):
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"seed data is missing columns: {sorted(missing)}")
    df = df[["text", "label"]]
    blank = df.isna().any(axis=1)
    df = df.astype(str)
    blank |= df["text"].str.strip().eq("") | df["label"].str.strip().eq("")
    if blank.any():
        raise ValueError(
            f"seed data has {int(blank.sum())} rows with empty text or label; fix or drop them"
        )

    unknown = sorted(set(df["label"]) - set(labels))
    if unknown:
        raise ValueError(f"seed labels not in the project label set: {unknown}")

    before = len(df)
    df = df.drop_duplicates(subset="text")
    if len(df) < before:
        warnings.warn(f"dropped {before - len(df)} duplicate texts from seed data", stacklevel=2)

    counts = df["label"].value_counts().reindex(labels, fill_value=0)
    tiny = counts[counts < 8]
    if len(tiny):
        raise ValueError(
            "each label needs at least 8 seed examples; too few: "
            + ", ".join(f"{name} ({n})" for name, n in tiny.items())
        )

    dev, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )

    small = test["label"].value_counts().reindex(labels, fill_value=0)
    small = small[small < 25]
    if len(small):
        warnings.warn(
            "fewer than 25 test examples for: "
            + ", ".join(f"{name} ({n})" for name, n in small.items())
            + "; test metrics for these labels will be noisy",
            stacklevel=2,
        )
    if len(dev) < 100:
        warnings.warn(
            f"only {len(dev)} dev rows; GEPA prompt optimization works best with 100+",
            stacklevel=2,
        )
    return dev.reset_index(drop=True), test.reset_index(drop=True)


def usable_pool(pool, dev, test):
    """Drop invalid rows and any pool text that also appears in the seed.

    The pool is usually the user's whole export, hand-labeled rows included. A test
    text that slips into training makes the locked split worthless, and a dev text
    would be counted twice with two possibly different labels. Returns
    (valid_rows, n_dropped_as_seed_duplicates).
    """
    valid = pool[pool["label"] != INVALID]
    seen = set(dev["text"]) | set(test["text"])
    overlap = valid["text"].isin(seen)
    return valid[~overlap], int(overlap.sum())
