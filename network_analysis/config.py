"""Paths of reproduce_networks/ (derived from this file's location; the only folder read outside it is data/).

REPO_ROOT  parent of reproduce_networks/
DATA_DIR   REPO_ROOT/data (original datasets, read only)
OUT_DIR    REPO_ROOT/reproduce_networks_outputs (all outputs)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "reproduce_networks_outputs"

VOTING = OUT_DIR / "voting"                # voting_network.py
TEMPORAL = OUT_DIR / "temporal"            # temporal_network.py
SPEAKING = OUT_DIR / "speaking_order"      # speaking_order.py
FORMAL = OUT_DIR / "formal_roles"          # formal_roles.py, alternation.py
FIGURES = OUT_DIR / "figures"              # fig1.py
