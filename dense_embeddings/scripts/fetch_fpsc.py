"""Rebuild data/speeches.json from the public FPSC dataset (davidilag/FPSC).

Prefers the local parquet cache of the HF metadata
(<repo>/data/fpsc/fpsc_train_metadata.parquet, same rows as
data/train/metadata.jsonl); falls back to downloading metadata.jsonl
(~212 MB) from HuggingFace when the cache is absent.

Writes data/speeches.json (in this project) in the same {"speeches": [...]}
shape the original scripts used, keeping only utterances with a non-empty
transcript (48,028 rows).

Usage:
    .venv/bin/python scripts/fetch_fpsc.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA_OUT, FPSC_PARQUET, SPEECHES_JSON  # noqa: E402

RAW_DIR = os.path.join(DATA_OUT, "raw")
FPSC_FILE = "data/train/metadata.jsonl"


def rows_from_parquet(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = set(t.column_names)
    def col(row, *names):
        for n in names:
            if n in cols:
                return row[n]
        return None
    for row in t.to_pylist():
        yield {
            "id": row.get("id"),
            "speaker_id": col(row, "speaker_id", "mp_id"),
            "meeting_id": row.get("meeting_id"),
            "topic": row.get("topic"),
            "winner_text": col(row, "winner_text", "text"),
            "contribution_type": row.get("contribution_type"),
        }


def rows_from_hf():
    from huggingface_hub import hf_hub_download
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"downloading davidilag/FPSC:{FPSC_FILE} ...", flush=True)
    path = hf_hub_download("davidilag/FPSC", FPSC_FILE, repo_type="dataset",
                           local_dir=RAW_DIR)
    print(f"downloaded to {path}", flush=True)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    os.makedirs(DATA_OUT, exist_ok=True)
    if os.path.exists(SPEECHES_JSON):
        with open(SPEECHES_JSON, encoding="utf-8") as f:
            n = len(json.load(f)["speeches"])
        print(f"{SPEECHES_JSON} already exists ({n} speeches) — nothing to do")
        return

    if os.path.exists(FPSC_PARQUET):
        print(f"using local cache {FPSC_PARQUET}")
        rows = rows_from_parquet(FPSC_PARQUET)
    else:
        rows = rows_from_hf()

    speeches = []
    n_raw = 0
    for r in rows:
        n_raw += 1
        text = (r.get("winner_text") or r.get("text") or "").strip()
        if not text:
            continue
        speeches.append({
            "id": r["id"],
            "speaker_id": r.get("speaker_id") or r.get("mp_id"),
            "meeting_id": r.get("meeting_id"),
            "topic": r.get("topic"),
            "text": text,
            "contribution_type": r.get("contribution_type"),
        })

    with open(SPEECHES_JSON, "w", encoding="utf-8") as f:
        json.dump({"speeches": speeches}, f, ensure_ascii=False)
    n_case = sum(1 for s in speeches if s.get("topic"))
    print(f"read {n_raw} rows -> wrote {len(speeches)} speeches "
          f"({n_case} with a topic) to {SPEECHES_JSON}")


if __name__ == "__main__":
    sys.exit(main())
