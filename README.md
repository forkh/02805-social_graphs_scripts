# scripts_to_publish

Scripts that accompany the report, grouped by workstream. **Only code lives here —
no data files.** Data is either scraped/fetched by these scripts or comes from the
external FPSC corpus.

## Contents

- `data/` — data acquisition and validation:
  - `scrape_votes.py` — scrapes votes, ballots, cases and members (2020–2023) from
    logting.fo via its public Umbraco Surface endpoints; resumable via a raw-HTML
    cache. Run: `python3 scrape_votes.py --years 2020 2021 2022 2023`
  - `validate_data.py` — consistency checks (tallies vs. ballots, joins) and
    verification of the join keys with the FPSC speech dataset
  - `fetch_fpsc.py` — downloads/rebuilds the FPSC speech corpus
    (`davidilag/FPSC`, CC-BY-4.0) into `speeches.json`
  - `common.py` — shared path constants and loader helpers (**required** by
    `fetch_fpsc.py`; keep it in the same folder)
- `network_analysis/` — self-contained reproduction of the voting/speaking-order
  network analyses from the scraped data only: `run_all.py` runs the steps
  (`voting_network.py`, `temporal_network.py`, `speaking_order.py`,
  `formal_roles.py`, `alternation.py`, `fig1.py`) plus the helper modules
  (`config.py`, `roles.py`, `netlib.py`, `speaking_lib.py`, `auditlib.py`,
  `within_bloc.py`, `figstyle.py`). Run `python network_analysis/run_all.py`
  from the folder that contains `data/` and `network_analysis/`; see
  `network_analysis/README.md` for the run order, the expected output
  checksums and where the reported numbers come from
- `dense_embeddings/` — dense-vector / ROCm embedding pipeline: `scripts/`
  (`analyze_rocm.py`, `embed_rocm.py`, `common.py`, `fetch_fpsc.py`,
  `pretrain_fobert_mlm.py`, `make_models.py`, `make_findings.py`,
  `make_model_fig.py`, `summarize.py`, `run_all.sh`, `run_queue.sh`, `setup.sh`),
  plus `README.md`, `pyproject.toml`, `.python-version`
- `text_analysis/` — document-level text analysis: `text_comparison.ipynb`
  (TF-IDF/PPMI speech-vs-vote tests behind `fig:text`)

## Caveats

1. **Output paths are relative to the run location.** `scrape_votes.py` writes to
   `--out` (default `../data`) and `common.py` derives its output paths from its own
   location. If you run the scripts from inside `data/`, generated files (CSVs,
   parquet, `scrape_log.json`, raw HTML cache, `speeches.json`) land in that same
   folder — keep the published folder code-only and let outputs go elsewhere.
2. **No data is bundled here.** Voting data is scraped from logting.fo at run time;
   speech data comes from the external FPSC dataset (CC-BY-4.0, `davidilag/FPSC`).

## dense_embeddings caveats

1. **GPU required.** The embedding/analysis scripts (`embed_rocm.py`,
   `analyze_rocm.py`) need an AMD (ROCm) or NVIDIA (CUDA) GPU. `torch` is
   intentionally **not** pinned in `pyproject.toml` — `scripts/setup.sh` installs
   the backend-specific wheel afterwards (ROCm: `torch==2.14.0+rocm10.1.0`,
   CUDA: `torch==2.15.0+cu132`). Python >=3.12,<3.14 (see `.python-version`).
2. **Lexicon-axis mode needs extra data.** `analyze_rocm.py --axis lexicon` reads
   `speech_scores.csv` from the say–do analysis (`speech_vote_agreement/data/`),
   which is not published here. The paper uses the **voters axis** only
   (`--axis voters`, the default), which needs just the FPSC speeches + scraped
   votes.

## text_analysis caveats

1. **Depends on the `data/` outputs.** The notebook reads
   `votes/ballots/members_2020-2023.parquet` and `fpsc/fpsc_train_metadata.parquet`
   via `D = '../data/'`. With the notebook placed directly in `text_analysis/`,
   `../data/` resolves to `scripts_to_publish/data/` — so run the `data/` scripts
   there first to produce the parquet files (at run time; keep the published
   folder code-only). Keep the notebook one level below the data folder, or adjust
   the `D` cell.
2. **Dependencies** are in the repo-root `requirements.txt` (`pandas`, `numpy`,
   `scikit-learn`, `scipy`, `matplotlib`) plus `jupyterlab`/`ipykernel` to run it.

## network_analysis caveats

1. **Data is read-only.** The scripts read `data/*.parquet`, `data/*.csv` and
   `data/fpsc/fpsc_train_metadata.parquet`; all outputs go to
   `reproduce_networks_outputs/` next to `data/` (already in `.gitignore`).
   `run_all.py` stops if any file outside the output folder is created, changed
   or deleted, so the inputs cannot be modified accidentally.
2. **Dependencies:** Python 3.12.2 with numpy, pandas, pyarrow, scipy, networkx,
   matplotlib and scikit-learn (exact versions in `network_analysis/README.md`).
   Runtime is about 9–10 minutes on a laptop; fixed seeds are used throughout,
   and `network_analysis/README.md` lists the expected SHA-256 of every key
   output file, so a successful run can be checked byte-for-byte.

