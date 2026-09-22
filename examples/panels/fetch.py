"""Fetch the public data behind the pre-built panels into data/<panel>/ as seed.csv
(labeled, dev/test split), pool.csv (text only) and holdout.csv (labeled, never used).

    python examples/panels/fetch.py [panel ...]     # default: all four

    messages     SMS Spam Collection (ucirvine/sms_spam)
    email        phishing-email-dataset (zefang-liu), its "phishing" class is spam broadly
    guardrail    lmsys/toxic-chat, xTRam1/safe-guard-prompt-injection,
                 jackhhao/jailbreak-classification
    pii          ai4privacy/pii-masking-200k (synthetic sentences with span labels)

Only some questions per panel have public labels. Each guardrail row carries labels only
for the questions its source annotated.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent.parent / "data"
SEED_ROWS, POOL_ROWS, HOLDOUT_ROWS = 500, 1500, 4000
MAX_CHARS = 3000                      # long emails are cut, the student sees the same cut


def _dl(repo, path):
    return hf_hub_download(repo, path, repo_type="dataset")


def _split(df, name, pool_rows=POOL_ROWS, rng=None, shuffle_seed=True):
    """Shuffle once so seed, pool and holdout never overlap. The pool keeps no labels."""
    rng = rng or np.random.default_rng(0)
    if shuffle_seed:
        df = df.iloc[rng.permutation(len(df))]
    df = df.reset_index(drop=True)
    seed, pool = df.iloc[:SEED_ROWS], df.iloc[SEED_ROWS:SEED_ROWS + pool_rows]
    hold = df.iloc[SEED_ROWS + pool_rows:SEED_ROWS + pool_rows + HOLDOUT_ROWS]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    seed.to_csv(out / "seed.csv", index=False)
    pool[["text"]].to_csv(out / "pool.csv", index=False)
    hold.to_csv(out / "holdout.csv", index=False)
    print(f"{name}: seed {len(seed)}, pool {len(pool)}, holdout {len(hold)} -> {out}")


def messages():
    df = pd.read_parquet(_dl("ucirvine/sms_spam", "plain_text/train-00000-of-00001.parquet"))
    df = pd.DataFrame({"text": df["sms"].str.strip(),
                       "unsolicited": df["label"].map({1: "yes", 0: "no"})})
    _split(df.drop_duplicates("text"), "messages")


def email():
    df = pd.read_csv(_dl("zefang-liu/phishing-email-dataset", "Phishing_Email.csv"))
    # the dataset's "Phishing Email" class is spam in the broad sense (casino bonuses,
    # snoring cures, address lists, bounce notices), not only credential or payment lures
    df = pd.DataFrame({
        "text": df["Email Text"].astype(str).str.strip().str.slice(0, MAX_CHARS),
        "spam": df["Email Type"].map({"Phishing Email": "yes", "Safe Email": "no"}),
    })
    df = df[df["text"].str.len() > 20].drop_duplicates("text")
    _split(df, "email")


PII_GROUPS = {
    "contact": {"EMAIL", "PHONENUMBER", "STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "ZIPCODE"},
    "identity": {"FIRSTNAME", "LASTNAME", "MIDDLENAME", "DOB", "SSN"},
    "financial": {"CREDITCARDNUMBER", "CREDITCARDCVV", "IBAN", "BIC", "ACCOUNTNUMBER",
                  "ACCOUNTNAME", "BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS"},
    "credentials": {"PASSWORD", "PIN"},
    "device": {"IPV4", "IPV6", "IP", "MAC", "PHONEIMEI", "USERAGENT", "NEARBYGPSCOORDINATE",
               "VEHICLEVIN", "VEHICLEVRM"},
}


def guardrail():
    """Two sources, one text column. toxic-chat rows carry gold for `harmful` and
    `jailbreak`, safe-guard rows carry gold for `injection`, and the other cells stay empty."""
    tc = pd.read_csv(_dl("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_all.csv"))
    a = pd.DataFrame({
        "text": tc["user_input"].astype(str).str.strip().str.slice(0, MAX_CHARS),
        "harmful": tc["toxicity"].map({1: "yes", 0: "no"}),
        "jailbreak": tc["jailbreaking"].map({1: "yes", 0: "no"}),
    })
    sg = pd.concat([
        pd.read_parquet(_dl("xTRam1/safe-guard-prompt-injection",
                            f"data/{p}-00000-of-00001.parquet"))
        for p in ("train", "test")
    ])
    b = pd.DataFrame({
        "text": sg["text"].astype(str).str.strip().str.slice(0, MAX_CHARS),
        "injection": sg["label"].map({1: "yes", 0: "no"}),
    })
    # toxic-chat's 204 jailbreaks are mostly the same few templates pasted repeatedly (~70
    # distinct texts), so a third source supplies the jailbreak labels
    jb = pd.read_csv(_dl("jackhhao/jailbreak-classification",
                         "balanced/jailbreak_dataset_full_balanced.csv"))
    c = pd.DataFrame({
        "text": jb["prompt"].astype(str).str.strip().str.slice(0, MAX_CHARS),
        "jailbreak": jb["type"].map({"jailbreak": "yes", "benign": "no"}),
    })
    rng = np.random.default_rng(0)
    b = b.iloc[rng.permutation(len(b))].head(len(a) * 2 // 3)   # 60/40 real prompts / injection set
    df = pd.concat([a, b, c], ignore_index=True)
    df = df[df["text"].str.len() > 3].drop_duplicates("text")
    # jailbreaks are 2% of real prompts: a random 500-row seed would hold ~5, too few to
    # gate a stack or read an AUROC on. The seed takes every positive it can up to a cap;
    # pool and holdout stay at the natural rate, so calibration and the holdout numbers do.
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    want = pd.concat([df[df["jailbreak"] == "yes"].head(80), df[df["harmful"] == "yes"].head(100),
                      df[df["injection"] == "yes"].head(80)]).drop_duplicates("text")
    rest = df.drop(want.index)
    seed = pd.concat([want, rest.head(SEED_ROWS - len(want))])
    _split(pd.concat([seed, rest.iloc[SEED_ROWS - len(want):]]), "guardrail", shuffle_seed=False)


def pii():
    with open(_dl("ai4privacy/pii-masking-200k", "english_pii_43k.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    out = pd.DataFrame({"text": [r["source_text"].strip()[:MAX_CHARS] for r in rows]})
    for group, labels in PII_GROUPS.items():
        out[group] = ["yes" if any(m["label"] in labels for m in r["privacy_mask"]) else "no"
                      for r in rows]
    _split(out.drop_duplicates("text"), "pii")


if __name__ == "__main__":
    names = sys.argv[1:] or ["messages", "email", "guardrail", "pii"]
    for n in names:
        globals()[n]()
