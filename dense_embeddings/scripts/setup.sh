#!/usr/bin/env bash
# Create/sync the UV environment (uv.lock), install the pinned GPU torch build,
# and verify the GPU is visible.
#
# Usage:
#   bash scripts/setup.sh rocm    # AMD GPU  -> torch 2.14.0+rocm10.1.0 (TheRock wheels)
#   bash scripts/setup.sh cuda    # NVIDIA GPU -> torch 2.15.0+cu132
#
# On AMD, the container must have the GPU devices passed through, e.g.
#   podman run --device /dev/kfd --device /dev/dri ...
# PyTorch's ROCm build exposes HIP via the torch.cuda API, so the scripts use
# device="cuda" unchanged on both backends.
#
# After this script, run the pipeline with .venv/bin/python (NOT `uv run`,
# which would re-sync and drop the backend-specific torch build).
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${1:-rocm}"
case "$BACKEND" in
    rocm)
        TORCH_SPEC="torch==2.14.0+rocm10.1.0"
        TORCH_INDEX="https://stable.repo.amd.com/rocm/whl-next/"
        ;;
    cuda)
        TORCH_SPEC="torch==2.15.0+cu132"
        TORCH_INDEX="https://download.pytorch.org/whl/cu132"
        ;;
    *)
        echo "usage: bash scripts/setup.sh [rocm|cuda]" >&2; exit 1 ;;
esac

uv sync                       # base deps from uv.lock (Python from .python-version)
uv pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| hip:", getattr(torch.version, "hip", None),
      "| cuda attr:", torch.version.cuda)
print("cuda(=HIP) available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
else:
    raise SystemExit("no GPU visible - check the torch build and /dev/kfd passthrough")
EOF
