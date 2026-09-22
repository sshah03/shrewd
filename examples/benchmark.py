"""Benchmark the whole pipeline on a public labeled dataset.

    export ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY + TEACHER below
    python examples/benchmark.py           # AG News, 4 topics
    python examples/benchmark.py banking77 # Banking77

A few hundred labeled rows become the "hand-labeled" seed. The labels of a larger
sample are hidden and it plays the unlabeled pool. Because the pool's gold labels
are secretly known, the script can score the teacher's bulk labels directly, a
check the real workflow can't do. Interrupted runs resume from cache.
"""

import os
import sys

import pandas as pd

from shrewd import Project

SEED_PER_CLASS = 60
POOL_PER_CLASS = int(os.environ.get("SHREWD_POOL_PER_CLASS", "250"))
BUDGET = int(os.environ.get("SHREWD_BUDGET", "300"))
# cheap default, override without editing: SHREWD_TEACHER="openai/..."
TEACHER = os.environ.get("SHREWD_TEACHER", "anthropic/claude-haiku-4-5")


def agnews():
    url = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
        "master/data/ag_news_csv/train.csv"
    )
    df = pd.read_csv(url, header=None, names=["class", "title", "description"])
    df["text"] = (df["title"] + ". " + df["description"]).str.replace("\\", " ", regex=False)
    df["label"] = df["class"].map({1: "world", 2: "sports", 3: "business", 4: "sci/tech"})
    labels = {
        "world": "international news: politics, conflicts, diplomacy, world events",
        "sports": "sports: games, athletes, teams, competitions",
        "business": "business and finance: markets, companies, deals, the economy",
        "sci/tech": "science and technology: research, software, gadgets, space",
    }
    return df, labels, "Classify the news snippet by its topic."


def banking77():
    url = (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/train.csv"
    )
    df = pd.read_csv(url).rename(columns={"category": "label"})
    keep = sorted(df["label"].unique())[:6]
    df = df[df["label"].isin(keep)]
    labels = {name: name.replace("_", " ") for name in keep}
    return df, labels, "Classify the customer's banking query by its intent."


def sms():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
    labels = {
        "ham": "a normal personal or transactional message",
        "spam": "unsolicited promotional, phishing, or scam content",
    }
    return df, labels, "Classify the SMS message."


def sst2():
    url = (
        "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/"
        "master/data/SST2/train.tsv"
    )
    df = pd.read_csv(url, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].map({0: "negative", 1: "positive"})
    labels = {
        "negative": "the reviewer's overall sentiment is unfavorable",
        "positive": "the reviewer's overall sentiment is favorable",
    }
    return df, labels, "Classify the sentiment of the movie-review sentence."


def newsgroups():
    from sklearn.datasets import fetch_20newsgroups

    categories = [
        "rec.autos", "rec.motorcycles",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "sci.electronics",
    ]
    data = fetch_20newsgroups(
        subset="train", categories=categories, remove=("headers", "footers", "quotes")
    )
    df = pd.DataFrame(
        {"text": data.data, "label": [data.target_names[i] for i in data.target]}
    )
    df["text"] = df["text"].str.strip().str.slice(0, 1200)
    df = df[df["text"].str.len() > 30]
    labels = {
        "rec.autos": "cars: buying, maintaining, discussing automobiles",
        "rec.motorcycles": "motorcycles and riding",
        "comp.sys.ibm.pc.hardware": "PC/IBM-compatible hardware: cards, drives, upgrades",
        "comp.sys.mac.hardware": "Apple Macintosh hardware",
        "sci.electronics": "general electronics and circuit design",
    }
    return df, labels, "Classify which discussion group the post belongs to."


DATASETS = {
    "agnews": agnews,
    "banking77": banking77,
    "sms": sms,
    "sst2": sst2,
    "newsgroups": newsgroups,
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "agnews"
    df, labels, instructions = DATASETS[name]()
    df = df[["text", "label"]].drop_duplicates("text").sample(frac=1, random_state=0)

    seed = df.groupby("label").head(SEED_PER_CLASS)
    pool = df.drop(seed.index).groupby("label").head(POOL_PER_CLASS)
    print(f"{name}: {len(seed)} seed rows, {len(pool)} pool rows (gold labels hidden)")

    run_dir = f"runs/{name}-{TEACHER.split('/')[-1]}"  # one project dir per teacher model
    proj = Project(run_dir, instructions=instructions, labels=labels, teacher=TEACHER)
    if not (proj.dir / "seed_test.csv").exists():  # re-runs keep the locked split and cache
        proj.add_seed(seed)
    proj.optimize(budget=BUDGET, target=0.93)
    proj.label(pool[["text"]], dry_run=True)
    proj.label(pool[["text"]])
    result = proj.distill(student="tfidf")
    print()
    print(result.report())

    # compare the teacher's pool labels against the hidden gold labels
    labeled = pd.read_csv(f"{run_dir}/pool_labeled.csv")
    merged = labeled.merge(pool, on="text", suffixes=("_teacher", "_gold"))
    agreement = (merged["label_teacher"] == merged["label_gold"]).mean()
    print(f"\nteacher pool labels vs hidden gold: {agreement:.1%} agreement on {len(merged)} rows")


if __name__ == "__main__":
    main()
