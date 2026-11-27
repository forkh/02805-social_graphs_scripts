"""PNAS figures: voters-axis AUC per encoder, case-wide vs same-meeting.

Two variants, written in one run, for the 6-page report (the 15-row model table
in README.md goes to the supplement):

  figures/fig_model_auc.png/.pdf            dashed chance line + dotted supervised-TF-IDF lines
  figures/fig_model_auc_no_tfidf.png/.pdf   dashed chance line only

Reads data/results.jsonl (axis="voters") and writes PNG (300 dpi) + PDF (vector,
for Overleaf) plus a figure_values*.json with the printed numbers — same audit
pattern as outputs/networks/publication_figures/. Style comes from
src/plotting.py (PNAS widths, DejaVu Sans 7-7.5 pt, INK/MUTED). Nothing here
recomputes the analysis; the results file is only read.

Usage (from embedding_stance_rocm/):
    ../../../.venv/bin/python scripts/make_model_fig.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # embedding_stance_rocm/
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src"))  # repo src/
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plotting as P  # noqa: E402

RESULTS = os.path.join(ROOT, "data", "results.jsonl")
FIG_DIR = os.path.join(ROOT, "figures")

AXIS = "voters"
SCOPES = ["case-wide", "same-meeting"]
CHANCE = 0.500
# supervised TF-IDF (text-only) + logistic regression, GroupKFold by case,
# from ../vote_prediction/data/vote_prediction_results.txt (tfidf-only auc;
# its sample is n=5,600 / 2,526, close to but not identical to the embedding
# runs' 5,525 / 2,464). The 0.670 case-wide number there is controls-only
# *accuracy*, a different metric, so it is deliberately not used here.
TFIDF_AUC = {"case-wide": 0.604, "same-meeting": 0.627}
TFIDF_SRC = "../vote_prediction/data/vote_prediction_results.txt (tfidf-only auc)"
# models highlighted (the domain-adaptation story): base FoBERT and FoBERT+MLM
HIGHLIGHT = {"fobert": "#7ea6dd", "fobert-fpsc": "#155799"}
BAR_FILL = "#c8c7c1"
XMIN, XMAX = 0.44, 0.66

VARIANTS = [("fig_model_auc", True), ("fig_model_auc_no_tfidf", False)]


def load_voters_results():
    rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["axis"] == AXIS]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["scope"]] = r
    # order as in the README table: by same-meeting AUC, best first
    order = sorted(by_model, key=lambda m: -by_model[m]["same-meeting"]["auc"])
    return [(m, by_model[m]) for m in order]


def draw_panel(ax, data, scope, label, include_tfidf):
    ys = list(range(len(data)))
    aucs = [d[scope]["auc"] for _, d in data]
    fills = [HIGHLIGHT.get(m, BAR_FILL) for m, _ in data]
    ax.barh(ys, aucs, height=0.62, color=fills, edgecolor="none", zorder=3)
    for y, a in zip(ys, aucs):
        ax.text(a + 0.0015, y, f"{a:.3f}", va="center", ha="left", fontsize=6.5, color=P.INK, zorder=4)
    ax.axvline(CHANCE, color=P.MUTED, lw=0.8, ls="--", zorder=2)
    if include_tfidf:
        v = TFIDF_AUC[scope]
        ax.axvline(v, color=P.MUTED, lw=0.8, ls=":", zorder=2)
        ax.text(v - 0.002, 0.97, f"TF-IDF {v:.3f}", transform=ax.get_xaxis_transform(),
                fontsize=6.5, color=P.MUTED, ha="right", va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([m for m, _ in data])
    ax.invert_yaxis()
    ax.set_xlim(XMIN, XMAX)
    n = data[0][1][scope]["n"]
    ax.set_title(f"{scope} (n = {n:,})", loc="left")
    ax.set_xlabel("voters-axis AUC")  # "leave-one-out" detail lives in the LaTeX caption
    ax.tick_params(length=0)
    P.panel_label(ax, label)


def main():
    P.set_style()
    data = load_voters_results()
    os.makedirs(FIG_DIR, exist_ok=True)

    for stem, include_tfidf in VARIANTS:
        fig, axes = plt.subplots(1, 2, figsize=(P.COL_TWO, 3.4), sharey=True)
        for ax, scope, label in zip(axes, SCOPES, ("a", "b")):
            draw_panel(ax, data, scope, label, include_tfidf)
        axes[1].tick_params(labelleft=False)
        handles = [Patch(facecolor=HIGHLIGHT["fobert"], label="fobert (base FoBERT)"),
                   Patch(facecolor=HIGHLIGHT["fobert-fpsc"], label="fobert-fpsc (FoBERT + FPSC MLM)"),
                   Line2D([0], [0], color=P.MUTED, lw=0.8, ls="--", label=f"chance = {CHANCE:.3f}")]
        if include_tfidf:
            handles.append(Line2D([0], [0], color=P.MUTED, lw=0.8, ls=":",
                                  label="supervised TF-IDF (text-only) AUC"))
        fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.06))
        fig.align_ylabels(axes)

        for ext in ("png", "pdf"):
            p = os.path.join(FIG_DIR, f"{stem}.{ext}")
            fig.savefig(p)
            print("wrote", p)
        plt.close(fig)

        refs = {"chance": CHANCE}
        if include_tfidf:
            refs["supervised_tfidf_auc_text_only"] = dict(TFIDF_AUC)
            refs["tfidf_source"] = TFIDF_SRC
        values = {"variant": stem, "axis": AXIS, "source": os.path.relpath(RESULTS, ROOT),
                  "references": refs,
                  "models": {m: {s: {"auc": d[s]["auc"], "agreement": d[s]["agreement"],
                                     "n": d[s]["n"]} for s in SCOPES} for m, d in data}}
        vpath = os.path.join(FIG_DIR, f"figure_values{'' if include_tfidf else '_no_tfidf'}.json")
        with open(vpath, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2)
        print("wrote", vpath)


if __name__ == "__main__":
    main()
