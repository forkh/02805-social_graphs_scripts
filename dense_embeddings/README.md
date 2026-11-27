# scripts_dense_embeddings — dense-embedding stance pipeline (UV project)

Self-contained copy of the dense-embedding ("embedding stance") pipeline that
originally lived in `models/embedding_stance_rocm/scripts/` (ROCm port of
`speech_vote_agreement/embedding_stance_cuda/scripts/`). Everything runs from
this folder; Python and package versions are pinned by `uv.lock`.

**Question:** for each contested Løgtingið vote (2020–2023), does a dense
embedding axis built from the speakers' own speeches predict the ballot?
Two axes are tested per encoder, in two scopes (case-wide / same-meeting):

- **voters**: `mean(emb of Fyri voters' speeches) − mean(emb of Ímóti voters' speeches)`,
  rebuilt **leave-one-out** per ballot so an MP never anchors their own axis;
- **lexicon**: same idea, but anchored on the lexicon ±1 scores from the
  speech–vote agreement pipeline (`../speech_vote_agreement/data/speech_scores.csv`).

An MP's mean speech embedding is projected on the unit axis; sign > 0 ⇒ predict
Fyri. Metrics: agreement and rank-AUC vs. the actual ballot (chance = 0.5).

## Layout

```
scripts_dense_embeddings/
├── pyproject.toml / uv.lock / .python-version   # UV project (Python 3.13, pinned deps)
├── scripts/
│   ├── common.py            # linkage + model registry + paths (single source of truth)
│   ├── fetch_fpsc.py        # 1. rebuild data/speeches.json from the FPSC parquet cache (or HF)
│   ├── embed_rocm.py        # 2. embed the needed speeches on the GPU (cached per model)
│   ├── analyze_rocm.py      # 3. leave-one-out pro/con axis analysis -> data/results.jsonl
│   ├── pretrain_fobert_mlm.py  # optional: MLM-continue FoBERT on FPSC -> models/fobert-fpsc-mlm
│   ├── summarize.py         # 4. print the results table
│   ├── make_models.py       # 5. models.html (registry + results)
│   ├── make_findings.py     # 6. findings.html (one-page summary + bar chart)
│   ├── setup.sh             # uv sync --extra rocm|cuda + GPU check
│   ├── run_all.sh           # embed+analyze the 10 base models
│   └── run_queue.sh         # follow-up encoders + FoBERT pretraining
└── data/                    # created at runtime: speeches.json, *.npz, results.jsonl, *.csv
```

## Setup

```bash
cd scripts_dense_embeddings
bash scripts/setup.sh rocm     # AMD GPU: uv sync + torch 2.14.0+rocm10.1.0 (TheRock wheels)
# or:  bash scripts/setup.sh cuda     # NVIDIA GPU: uv sync + torch 2.15.0+cu132
```

`uv sync` creates `.venv` from `uv.lock` (Python 3.13 from `.python-version`).
`setup.sh` then installs the pinned backend-specific **torch** build — torch is
deliberately not in `uv.lock`, because its wheel depends on the GPU backend:

| backend | pinned torch | index |
|---|---|---|
| `rocm` (AMD) | `2.14.0+rocm10.1.0` | https://stable.repo.amd.com/rocm/whl-next/ |
| `cuda` (NVIDIA) | `2.15.0+cu132` | https://download.pytorch.org/whl/cu132 |

On AMD, pass the GPU through to the container: `podman run --device /dev/kfd --device /dev/dri ...`.
After setup, run scripts with `.venv/bin/python` (not `uv run`, which would
re-sync the venv from the lock and drop the backend-specific torch).

## Pipeline

```bash
.venv/bin/python scripts/fetch_fpsc.py                  # data/speeches.json (48,028 rows)
.venv/bin/python scripts/embed_rocm.py --model minilm    # cached per model in data/*.npz
.venv/bin/python scripts/analyze_rocm.py --model minilm --axis voters
.venv/bin/python scripts/analyze_rocm.py --model minilm --axis lexicon
bash scripts/run_all.sh                                  # all 10 base models, both axes
bash scripts/run_queue.sh                                # follow-ups + FoBERT MLM (fobert-fpsc)
.venv/bin/python scripts/summarize.py                    # table vs. CUDA/CPU runs
.venv/bin/python scripts/make_models.py                  # models.html
.venv/bin/python scripts/make_findings.py                # findings.html
```

## Inputs (read-only, outside this folder)

| path | env override | content |
|---|---|---|
| `../data/votes_2020-2023.csv`, `../data/ballots_2020-2023.csv` | `DENSE_EMB_DATA_IN` | scraped Løgtingið votes/ballots (from `../scripts/scrape_votes.py`) |
| `../data/fpsc/fpsc_train_metadata.parquet` | (inside `DENSE_EMB_DATA_IN`) | FPSC metadata cache used by `fetch_fpsc.py` |
| `../speech_vote_agreement/data/speech_scores.csv` | `DENSE_EMB_SVA_DIR` | lexicon scores for the `--axis lexicon` mode |
| `../speech_vote_agreement/data/mp_agreement.csv` | `DENSE_EMB_SVA_DIR` | joined into `mp_embedding_agreement_<slug>.csv` |

The linkage (case from the speech `topic`, stage ≤ vote stage, contested votes
only) is implemented in `scripts/common.py` and is identical to
`../speech_vote_agreement/scripts/build_ballot_stance.py`.

## Outputs (in `data/`)

- `speech_embeddings_<slug>.npz` — per-model L2-normalized embedding matrix (cache)
- `results.jsonl` — one line per (model, axis, scope) with agreement/AUC
- `vote_level_scores_<slug>_<axis>_<scope>.csv` — per-ballot scores
- `mp_embedding_agreement_<slug>.csv` — per-MP agreement (voters axis)
- `embedding_results.txt` — human-readable log
