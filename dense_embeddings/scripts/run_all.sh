#!/usr/bin/env bash
# Embed + analyze every registered model on the GPU.
# Usage: bash scripts/run_all.sh [slug ...]   (default: the 10 base models)
# Run scripts/setup.sh first (installs .venv + the GPU torch build).
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

MODELS="${@:-minilm mpnet scandibert fobert nbbert xlmr-enis e5large icebert-large bge-m3 danskbert}"
for m in $MODELS; do
    echo "=================== $m ==================="
    $PY scripts/embed_rocm.py --model "$m"
    $PY scripts/analyze_rocm.py --model "$m" --axis voters
    $PY scripts/analyze_rocm.py --model "$m" --axis lexicon
done

$PY scripts/summarize.py
