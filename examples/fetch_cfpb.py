"""Fetch CFPB consumer-complaint narratives: a base window plus two later "drift"
windows, six financial products, consumer-selected product as the gold label.

    python examples/fetch_cfpb.py     # writes base.csv, drift6.csv, drift12.csv here

This is the data behind the complaint-routing case study in the main readme: run the
quickstart on base.csv (seed+pool), then score the trained student on each
window's rows with examples/score_holdout.py to measure temporal drift.
The public API rate-limits aggressively so the fetch takes a few minutes.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PRODUCTS = [
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Credit card",
    "Checking or savings account",
    "Mortgage",
    "Money transfer, virtual currency, or money service",
]
WINDOWS = {
    "base": ("2025-01-01", "2025-06-30", 700),
    "drift6": ("2025-07-01", "2025-12-31", 300),
    "drift12": ("2026-01-01", "2026-06-30", 300),
}
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
out = Path.cwd()

for window, (lo, hi, per_product) in WINDOWS.items():
    frames = []
    for product in PRODUCTS:
        params = urllib.parse.urlencode({
            "date_received_min": lo, "date_received_max": hi,
            "has_narrative": "true", "product": product,
            "size": per_product, "no_aggs": "true",
        }, quote_via=urllib.parse.quote)
        for attempt in range(8):  # the API rate-limits aggressively (429s)
            try:
                with urllib.request.urlopen(f"{API}?{params}", timeout=180) as resp:
                    hits = json.load(resp)["hits"]["hits"]
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 400, 503) or attempt == 7:
                    raise
                wait = 20 * (attempt + 1)
                print(f"  HTTP {e.code}, retrying in {wait}s", flush=True)
                time.sleep(wait)
        df = pd.DataFrame([{
            "text": h["_source"].get("complaint_what_happened") or "",
            "label": h["_source"].get("product"),
            "Date received": h["_source"].get("date_received"),
        } for h in hits])
        frames.append(df)
        print(f"{window} / {product}: {len(df)} rows", flush=True)
        time.sleep(8)
    all_df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    all_df["text"] = all_df["text"].astype(str).str.strip().str.slice(0, 1500)
    all_df = all_df[all_df["text"].str.len() > 100].drop_duplicates("text")
    all_df.to_csv(out / f"{window}.csv", index=False)
    print(f"== {window}: {len(all_df)} rows, labels: "
          f"{all_df['label'].value_counts().to_dict()}", flush=True)
