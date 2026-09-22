"""Fetch GoEmotions to turn into a "decisions" panel: eight yes/no questions per comment,
with the crowd rater fraction kept alongside the majority answer as `<name>__rate`.

    python examples/fetch_goemotions.py     # writes seed.csv, pool.csv, holdout.csv here

Source: https://github.com/google-research/goemotions (Apache-2.0, Demszky et al. 2020).
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/"
PARTS = ("goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv")
META = (
    "text", "id", "author", "subreddit", "link_id", "parent_id",
    "created_utc", "rater_id", "example_very_unclear",
)

# eight emotions spanning the base-rate range, from a 25% question down to a 2.5% one,
# and including the two most contested ones in the corpus (approval, annoyance) on
# purpose since a panel where every question is easy would not test calibration at all
QUESTIONS = (
    "neutral", "admiration", "gratitude", "approval",
    "amusement", "curiosity", "annoyance", "anger",
)
MIN_RATERS = 3
SPLITS = {"seed": (0, 500), "pool": (500, 2000), "holdout": (2000, 6000)}


def download(out_dir):
    frames = []
    for part in PARTS:
        path = out_dir / part
        if not path.exists():
            print(f"downloading {part} ...")
            urllib.request.urlretrieve(BASE + part, path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main():
    out_dir = Path(__file__).parent
    raw = download(out_dir)
    print(f"{len(raw):,} rater annotations")

    # one row per annotation -> one row per comment, carrying the fraction of raters
    # who chose each emotion
    raw = raw[raw["example_very_unclear"] == 0]
    emotions = [c for c in raw.columns if c not in META]
    grouped = raw.groupby("id")
    agg = grouped[emotions].mean()
    agg["text"] = grouped["text"].first()
    agg["subreddit"] = grouped["subreddit"].first()
    agg["n_raters"] = grouped.size()
    agg = agg[agg["n_raters"] >= MIN_RATERS].reset_index()
    agg = agg[agg["text"].str.len().between(15, 600)].reset_index(drop=True)
    print(f"{len(agg):,} comments with {MIN_RATERS}+ raters")

    rng = np.random.default_rng(42)
    agg = agg.iloc[rng.permutation(len(agg))].reset_index(drop=True)

    for name, (start, stop) in SPLITS.items():
        chunk = agg.iloc[start:stop]
        out = pd.DataFrame({"text": chunk["text"].values,
                            "n_raters": chunk["n_raters"].values,
                            "subreddit": chunk["subreddit"].values})
        for emotion in QUESTIONS:
            out[emotion] = (chunk[emotion].values >= 0.5).astype(int)
            out[f"{emotion}__rate"] = chunk[emotion].values.round(4)
        out.to_csv(out_dir / f"{name}.csv", index=False)
        rates = ", ".join(f"{e} {out[e].mean():.1%}" for e in QUESTIONS)
        print(f"{name:8} {len(out):5,} rows | {rates}")


if __name__ == "__main__":
    main()
