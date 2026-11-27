"""Summarize the runs in data/results.jsonl next to the CUDA/CPU runs.

Usage:
    .venv/bin/python scripts/summarize.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA_OUT, REPO  # noqa: E402

CUDA_RESULTS = os.path.join(REPO, "speech_vote_agreement", "embedding_stance_cuda",
                            "data", "results.jsonl")
OLD_RESULTS = os.path.join(REPO, "speech_vote_agreement", "embedding_stance",
                           "data", "embedding_results.txt")


def load_jsonl(p):
    rows = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def print_table(title, rows):
    print(f"== {title} ==")
    hdr = f"{'model':<14} {'axis':<8} {'scope':<13} {'n':>6} {'agreement':>9} {'auc':>6} {'wall_s':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["model"], r["axis"], r["scope"])):
        print(f"{r['model']:<14} {r['axis']:<8} {r['scope']:<13} {r['n']:>6} "
              f"{r['agreement']:>9.3f} {r['auc']:>6.3f} {r.get('seconds', 0):>7.1f}")
    print()


def main():
    print_table("embedding-stance results (this project, data/results.jsonl)",
                load_jsonl(os.path.join(DATA_OUT, "results.jsonl")))
    print_table("earlier CUDA embedding-stance results (for comparison)",
                load_jsonl(CUDA_RESULTS))

    if os.path.exists(OLD_RESULTS):
        print("== old CPU baseline (../embedding_stance/data/embedding_results.txt) ==")
        with open(OLD_RESULTS, encoding="utf-8") as f:
            print(f.read().strip())
        print()

    print("reference points: lexicon agreement 0.592 (primary), "
          "supervised TF-IDF AUC 0.63; chance = 0.500")
    print("validation: this GPU run (minilm) should match the earlier cuda/minilm "
          "(voters case-wide n=5525 agreement=0.511 auc=0.498)")


if __name__ == "__main__":
    sys.exit(main())
