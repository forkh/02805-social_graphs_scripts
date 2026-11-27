#!/usr/bin/env bash
# Follow-up queue for the GPU: new encoders (cheap -> slow), then
# FoBERT MLM domain pretraining on FPSC, then the adapted encoder, then a
# summary. Logs to stdout; run in the background:
#   nohup bash scripts/run_queue.sh > run_queue.log 2>&1 &
#
# Run scripts/setup.sh first (installs .venv + the GPU torch build).
# Every embed step is cached (data/speech_embeddings_<slug>.npz), so re-running
# the queue only recomputes what is missing. Set SKIP_PRETRAIN=1 to skip the
# FoBERT pretraining block, or ENCODERS="qwen3-4b" to run a subset.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "run scripts/setup.sh [rocm|cuda] first" >&2
    exit 1
fi
PY=.venv/bin/python

if [ ! -f data/speeches.json ]; then
    $PY scripts/fetch_fpsc.py
fi

# Set ENCODERS= (empty) to skip the encoder loop — embeds are cached anyway,
# and re-running analyze would duplicate rows in results.jsonl.
ENCODERS="${ENCODERS-arctic-l-v2 labse nbbert-large qwen3-4b}"
for m in $ENCODERS; do
    echo "=================== embed+analyze $m ==================="
    $PY scripts/embed_rocm.py --model "$m"
    $PY scripts/analyze_rocm.py --model "$m" --axis voters
    $PY scripts/analyze_rocm.py --model "$m" --axis lexicon
done

if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
    echo "=================== FoBERT MLM on FPSC ==================="
    $PY scripts/pretrain_fobert_mlm.py
    echo "=================== embed+analyze fobert-fpsc ==================="
    $PY scripts/embed_rocm.py --model fobert-fpsc
    $PY scripts/analyze_rocm.py --model fobert-fpsc --axis voters
    $PY scripts/analyze_rocm.py --model fobert-fpsc --axis lexicon
fi

$PY scripts/summarize.py
$PY scripts/make_findings.py
echo "queue done — see run_queue.log, data/results.jsonl and findings.html"
