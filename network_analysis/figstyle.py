"""Figure style constants used by fig1.py (the needed part of report_networks/figure_code/figure_config.py, values unchanged)."""
from __future__ import annotations

import matplotlib as mpl

PARTY_ORDER = ["Javnaðarflokkurin", "Tjóðveldi", "Sambandsflokkurin", "Fólkaflokkurin", "Framsókn", "Miðflokkurin", "Sjálvstýri"]
PARTY_MARKERS = dict(zip(PARTY_ORDER, ["o", "s", "^", "D", "v", "P", "X"]))
INK = "#1a1a19"
FONT_SIZE, TITLE_SIZE, TICK_SIZE, SMALL_SIZE = 7.0, 7.5, 6.5, 6.0
DPI_PNG = 300
SEED_VOTING_LAYOUT = 2805


def apply_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_SIZE, "axes.titlesize": TITLE_SIZE, "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE, "legend.fontsize": SMALL_SIZE, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "lines.linewidth": 1.1, "figure.dpi": 120, "savefig.dpi": DPI_PNG,
        "savefig.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False,
    })
