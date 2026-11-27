"""Shared linkage + model registry for the dense-embedding stance pipeline.

Adapted from models/embedding_stance_rocm/scripts/common.py so the whole
pipeline is self-contained in this UV project (scripts_dense_embeddings/).

Linkage (identical to the speech-vote agreement analysis): case-linked FPSC
speeches (topic -> case + stage), non-chairman, stage <= vote stage,
contested votes only (both Fyri and Ímóti).

Paths:
  ROOT      = this project folder (scripts_dense_embeddings/)
  DATA_OUT  = ROOT/data/        speeches.json, speech_embeddings_<slug>.npz,
                                results.jsonl, vote_level_scores_*.csv, ...
  DATA_IN   = <repo>/data/      votes/ballots CSVs + fpsc parquet cache
                                (override with env DENSE_EMB_DATA_IN)
  LINK_DIR  = <repo>/speech_vote_agreement/   speech_scores.csv, mp_agreement.csv
                                (override with env DENSE_EMB_SVA_DIR)

The model registry contains the 5 models already run on CUDA (for validation:
results must reproduce speech_vote_agreement/embedding_stance_cuda/data/results.jsonl),
the "to try next" candidates that did not fit the 8 GB RTX 4070 but fit the
24 GB RX 7900 XTX (gfx1100), and the follow-up queue: bigger multilingual
encoders + the domain-adapted FoBERT produced by pretrain_fobert_mlm.py.
"""

import csv
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                # scripts_dense_embeddings/
REPO = os.path.dirname(ROOT)                                # repository root

DATA_IN = os.environ.get("DENSE_EMB_DATA_IN") or os.path.join(REPO, "data")
LINK_DIR = os.environ.get("DENSE_EMB_SVA_DIR") or os.path.join(REPO, "speech_vote_agreement")
DATA_OUT = os.path.join(ROOT, "data")
SPEECHES_JSON = os.path.join(DATA_OUT, "speeches.json")
# local cache of the FPSC metadata (davidilag/FPSC data/train/metadata.jsonl)
FPSC_PARQUET = os.path.join(DATA_IN, "fpsc", "fpsc_train_metadata.parquet")
# domain-adapted FoBERT written by scripts/pretrain_fobert_mlm.py
FOVERT_FPSC_DIR = os.path.join(ROOT, "models", "fobert-fpsc-mlm")

TOPIC_RE = re.compile(r"(\d+)\.\s*viðgerð\s+(LM-\d+/\d{4})", re.IGNORECASE)
VOTE_STAGE_RE = re.compile(r"^\s*(?:beinleiðis\s+)?(\d+)\s*\.", re.IGNORECASE)
V2Y = {"Fyri": 1, "Ímóti": 0}

# model name -> HF id (or local path). Plain HF encoders (no sentence-transformers
# modules.json) and local paths are wrapped with mean pooling + L2 norm in
# embed_rocm.py; sentence-transformers models load directly.
MODELS = {
    # --- already run on CUDA (RTX 4070); re-run here to validate the port ---
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "scandibert": "vesteinn/ScandiBERT",
    "fobert": "vesteinn/FoBERT",
    "nbbert": "NbAiLab/nb-bert-base",
    # --- "to try next" (needed >8 GB, fits 24 GB XTX) ---
    "xlmr-enis": "vesteinn/XLMR-ENIS",            # XLM-R EN+IS+NO+SV (plain encoder)
    "e5large": "intfloat/multilingual-e5-large",  # 560M contrastive sentence encoder (ST)
    "icebert-large": "mideind/IceBERT-large",     # Icelandic BERT large, 406M (plain encoder)
    "bge-m3": "BAAI/bge-m3",                      # multilingual dense retriever (ST)
    "danskbert": "vesteinn/DanskBERT",            # Danish BERT (plain encoder)
    # --- follow-up queue (scripts/run_queue.sh) ---
    "arctic-l-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",  # 568M multilingual (ST)
    "labse": "sentence-transformers/LaBSE",                # 471M, 109 langs, similarity (ST)
    "nbbert-large": "NbAiLab/nb-bert-large",               # 356M Norwegian large (plain)
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",                 # 4.0B multilingual (ST, bf16)
    "fobert-fpsc": FOVERT_FPSC_DIR,                        # domain-adapted FoBERT (local path)
}
# models that are NOT sentence-transformers repos and need the manual wrapper
PLAIN_ENCODERS = {
    "vesteinn/ScandiBERT", "vesteinn/FoBERT", "NbAiLab/nb-bert-base",
    "vesteinn/XLMR-ENIS", "mideind/IceBERT-large", "vesteinn/DanskBERT",
    "NbAiLab/nb-bert-large",
}
# models that need a text prefix on every input (e5 uses query:/passage:)
PREFIXES = {"intfloat/multilingual-e5-large": "passage: "}
# load these in bf16 to fit VRAM (big models)
BF16_MODELS = {"Qwen/Qwen3-Embedding-4B"}
TASKS = {}  # sentence-transformers `task` adapters, if a model needs one
# default batch size per slug (large encoders get a smaller batch)
DEFAULT_BATCH = {
    "minilm": 256, "mpnet": 128, "scandibert": 128, "fobert": 128,
    "nbbert": 128, "xlmr-enis": 128, "danskbert": 128,
    "icebert-large": 64, "e5large": 64, "bge-m3": 64,
    "arctic-l-v2": 32, "labse": 64, "nbbert-large": 64,
    "qwen3-4b": 8, "fobert-fpsc": 128,
}


def norm_meeting_id(mid):
    return str(mid or "").replace(".0", "")


def vote_stage_n(stage):
    m = VOTE_STAGE_RE.match(stage or "")
    return int(m.group(1)) if m else 3


def emb_cache_path(slug):
    return os.path.join(DATA_OUT, f"speech_embeddings_{slug}.npz")


def load_speeches_by_case():
    """case -> [speech dicts] for case-linked, non-chairman speeches."""
    with open(SPEECHES_JSON, encoding="utf-8") as f:
        speeches = json.load(f)["speeches"]
    by_case = defaultdict(list)
    for x in speeches:
        m = TOPIC_RE.search(x.get("topic") or "")
        if not m or x.get("contribution_type") == "chairman":
            continue
        by_case[m.group(2).upper()].append({
            "id": x["id"], "mp": x["speaker_id"], "stage": int(m.group(1)),
            "meeting": norm_meeting_id(x.get("meeting_id")),
            "text": (x.get("text") or "")[:4000],
        })
    return by_case, len(speeches)


def load_votes_ballots():
    votes = {}
    with open(os.path.join(DATA_IN, "votes_2020-2023.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["stage_n"] = vote_stage_n(r["stage"])
            r["contested"] = float(r["n_for"] or 0) > 0 and float(r["n_against"] or 0) > 0
            r["meeting_id"] = norm_meeting_id(r["meeting_id"])
            votes[r["vote_id"]] = r
    with open(os.path.join(DATA_IN, "ballots_2020-2023.csv"), encoding="utf-8") as f:
        ballots = list(csv.DictReader(f))
    return votes, ballots


def needed_speech_ids(by_case, votes, ballots):
    """Speech ids needed for contested-vote ballots (same set as the CPU/CUDA runs)."""
    needed = {}
    for b in ballots:
        v = votes.get(b["vote_id"])
        if not v or not v["contested"] or b["vote"] not in V2Y:
            continue
        for sp in by_case.get(v["case_number"].upper(), []):
            if sp["mp"] == b["mp_id"] and sp["stage"] <= v["stage_n"]:
                needed[sp["id"]] = sp["text"]
    return needed


def contested_ballots_by_vote(votes, ballots):
    out = defaultdict(list)
    for b in ballots:
        v = votes.get(b["vote_id"])
        if v and v["contested"] and b["vote"] in V2Y:
            out[b["vote_id"]].append(b)
    return out
