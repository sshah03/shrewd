"""Fetch labeled GitHub issues as a shrewd-ready CSV (text,label).

Usage:
    python examples/fetch_github_issues.py kubernetes/kubernetes \\
        kind/bug=bug kind/feature=feature kind/support=support -o issues.csv

Needs the `gh` CLI, authenticated. Uses the search API (issues only, no PRs;
GitHub caps each label's results at 1,000). Issues carrying several of the mapped
labels are dropped as ambiguous. Bodies are cleaned of things that would leak
the label into the text: bot/prow commands (/kind ...), HTML comments, and common
issue-template headings.
"""
import csv
import json
import re
import subprocess
import sys
import time

command_line = re.compile(r"^\s*/(kind|sig|area|priority|triage|assign|cc|wg|committee|"
                          r"label|retitle|milestone).*$", re.MULTILINE | re.IGNORECASE)
html_comment = re.compile(r"<!--.*?-->", re.DOTALL)
template_noise = re.compile(
    r"^#+\s*(what happened|what you expected to happen|how to reproduce.*|anything else.*|"
    r"environment|.*version.*|cloud provider|install tools|container runtime.*|"
    r"expected behavior|actual behavior|steps to reproduce.*)[:?]?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def main():
    args = [a for a in sys.argv[1:] if a != "-o"]
    out_path = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "issues.csv"
    if "-o" in sys.argv:
        args.remove(out_path)
    repo, mappings = args[0], args[1:]
    label_map = dict(m.split("=", 1) for m in mappings)
    if not label_map:
        sys.exit("pass at least one github-label=class mapping")

    rows, seen = [], set()
    for gh_label, klass in label_map.items():
        fetched = 0
        for page in range(1, 11):  # search API cap: 1,000 results per query
            out = subprocess.run(
                ["gh", "api", "-X", "GET", "search/issues",
                 "-f", f"q=repo:{repo} label:{gh_label} is:issue",
                 "-f", "per_page=100", "-f", f"page={page}",
                 "-f", "sort=created", "-f", "order=desc"],
                capture_output=True, text=True, timeout=60,
            )
            if out.returncode != 0:
                print(f"{gh_label} page {page}: {out.stderr.strip()[:100]}", file=sys.stderr)
                break
            items = json.loads(out.stdout).get("items", [])
            if not items:
                break
            for issue in items:
                mapped = [n["name"] for n in issue["labels"] if n["name"] in label_map]
                if issue["number"] in seen or len(mapped) != 1:
                    continue
                seen.add(issue["number"])
                body = issue.get("body") or ""
                body = html_comment.sub(" ", body)
                body = command_line.sub(" ", body)
                body = template_noise.sub(" ", body)
                text = re.sub(r"\s+", " ", f"{issue['title']}. {body}").strip()[:1200]
                if len(text) > 40:
                    rows.append({"text": text, "label": klass})
                    fetched += 1
            time.sleep(2.5)  # search API allows ~30 requests/minute
        print(f"{gh_label}: {fetched} issues")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
