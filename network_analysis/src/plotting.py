"""Plotting helpers for the publication figures (DTU 02805 voting-network project).

Nothing here recomputes the analysis: every function takes numbers that were produced by the existing
notebooks (read from ``network_analysis/outputs``) and only draws them. The one exception is the layout/backbone
helper for the network figure, which derives a *drawing* subgraph from the existing pairwise table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from networks import PARTY_COLORS, PARTY_MARKERS, PARTY_ORDER, find_repo_root

INK, MUTED, GRID = "#1c1c1b", "#6b6a66", "#d9d8d3"
COL_ONE, COL_TWO = 7.1, 3.45          # PNAS-like figure widths in inches (two columns / one column)
PRE_FILL, POST_FILL = "#4a4a48", "#ffffff"   # pre/post encoded by dark vs open bars + hatch, not by party colours


def set_style() -> None:
    """One consistent style for every publication figure."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.2,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False,
    })


def output_dir(root: Path | None = None) -> Path:
    base = root or find_repo_root()
    d = (base / "network_analysis" if (base / "network_analysis").is_dir() else base / "networks") / "outputs" / "figures" / "publication"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig, name: str, root: Path | None = None) -> dict[str, Path]:
    """PNG (300 dpi, for sharing) and PDF (vector, for Overleaf) in network_analysis/outputs/figures/publication/."""
    d = output_dir(root)
    paths = {}
    for ext in ("png", "pdf"):
        p = d / f"{name}.{ext}"
        fig.savefig(p)
        paths[ext] = p
    return paths


def panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def luminance(hex_colour: str) -> float:
    """Relative luminance (sRGB), used to check that colours also differ in grey scale."""
    r, g, b = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in to_rgb(hex_colour)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def party_legend_handles(parties=PARTY_ORDER, filled=True, size=5.5):
    return [Line2D([0], [0], marker=PARTY_MARKERS[p], ls="", markersize=size, markeredgecolor="white" if filled else INK,
                   markerfacecolor=PARTY_COLORS[p], label=p) for p in parties]


# ----------------------------------------------------------------------------
# Figure A -- party similarity heatmap
# ----------------------------------------------------------------------------
def party_similarity_table(pairs: pd.DataFrame, min_overlap: int):
    """Mean pairwise voting similarity and number of MP pairs for every party pair (symmetric matrices)."""
    k = pairs[pairs.n_shared_divisions >= min_overlap]
    order = {p: i for i, p in enumerate(PARTY_ORDER)}
    swap = k.party_1.map(order) > k.party_2.map(order)
    k = k.assign(pa=np.where(swap, k.party_2, k.party_1), pb=np.where(swap, k.party_1, k.party_2))
    g = k.groupby(["pa", "pb"]).similarity.agg(["mean", "size"])
    mean = pd.DataFrame(np.nan, index=PARTY_ORDER, columns=PARTY_ORDER)
    count = pd.DataFrame(0, index=PARTY_ORDER, columns=PARTY_ORDER, dtype=int)
    for (x, y), r in g.iterrows():
        mean.loc[x, y] = mean.loc[y, x] = r["mean"]
        count.loc[x, y] = count.loc[y, x] = int(r["size"])
    return mean, count


def plot_party_heatmap(ax, mean: pd.DataFrame, count: pd.DataFrame, low_n: int = 5, cbar=True):
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#efeeea")
    data = np.ma.masked_invalid(mean.to_numpy())
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    n = len(mean)
    for i in range(n):
        for j in range(n):
            v, c = mean.iloc[i, j], count.iloc[i, j]
            if np.isnan(v):
                ax.text(j, i, "no pair", ha="center", va="center", fontsize=5.5, color=MUTED)
                continue
            dark = v > 0.55
            col = "white" if dark else INK
            if c < low_n:
                ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=7, color=col, style="italic")
                ax.text(j, i + 0.27, f"n={c}", ha="center", va="center", fontsize=5.5, color=col)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor=col, lw=0, alpha=0.35))
            else:
                ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=col, fontweight="bold")
                ax.text(j, i + 0.27, f"n={c}", ha="center", va="center", fontsize=5.5, color=col)
    ax.set_xticks(range(n)); ax.set_xticklabels(mean.columns, rotation=40, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(mean.index)
    for t, p in zip(ax.get_xticklabels(), mean.columns):
        t.set_color(INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Mean pairwise voting similarity")
        cb.outline.set_visible(False)
    return im


# ----------------------------------------------------------------------------
# Figure B -- similarity network
# ----------------------------------------------------------------------------
def threshold_backbone(pairs: pd.DataFrame, min_overlap: int, tau: float) -> nx.Graph:
    """Drawing subgraph: MP pairs with >= min_overlap shared divisions AND similarity >= tau (weight = similarity)."""
    k = pairs[(pairs.n_shared_divisions >= min_overlap) & (pairs.similarity >= tau)]
    G = nx.Graph()
    G.add_weighted_edges_from(zip(k.mp_id_1, k.mp_id_2, k.similarity))
    return G


def component_layout(G: nx.Graph, seed: int, pad: float = 0.55, target_aspect: float = 0.95) -> dict:
    """Force-directed layout (NetworkX default, course Week 1) of the backbone, computed per connected
    component (``k`` enlarged so that near-complete groups are spread out), each component scaled so its
    area grows with its number of nodes, then packed in rows; the row width that gives a bounding box
    closest to ``target_aspect`` (width / height) is used. Distances between components are arbitrary."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    n_max = len(comps[0])
    items = []
    for c in comps:
        sub = G.subgraph(c)
        n = len(sub)
        if n == 1:
            p = {next(iter(c)): np.zeros(2)}
        else:
            p = nx.spring_layout(sub, weight=None, seed=seed, k=2.0 / np.sqrt(n), iterations=500)
        pts = np.array(list(p.values()))
        pts = (pts - pts.mean(axis=0)) * np.sqrt(n / n_max)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        items.append((list(p.keys()), pts - lo, (hi - lo) + pad))

    def pack(row_limit):
        placed, x, y, row_h, width = [], 0.0, 0.0, 0.0, 0.0
        for keys, pts, (w, h) in items:
            if x > 0 and x + w > row_limit + 1e-9:
                x, y, row_h = 0.0, y + row_h, 0.0
            placed.append((keys, pts, x, y))
            x += w
            width = max(width, x)
            row_h = max(row_h, h)
        return placed, width, y + row_h

    widths = np.cumsum([w for _, _, (w, _) in items])
    candidates = sorted(set([max(w for _, _, (w, _) in items)] + list(widths)))
    best = min(candidates, key=lambda rl: abs(np.log((lambda r: r[1] / r[2])(pack(rl)) / target_aspect)))
    placed, _, _ = pack(best)
    pos = {}
    for keys, pts, x, y in placed:
        for k, pt in zip(keys, pts):
            pos[k] = np.array([x + pt[0] + pad / 2, -(y + pt[1] + pad / 2)])
    return pos


def outward_offsets(G: nx.Graph, pos: dict, nodes, length: float = 14.0) -> dict:
    """Label offsets (points) pointing away from the centre of the node's own component, so that labels of
    the densest groups spread outwards instead of piling up."""
    out = {}
    for c in nx.connected_components(G):
        centre = np.mean([pos[n] for n in c], axis=0)
        for n in c:
            if n in nodes:
                v = pos[n] - centre
                nrm = np.linalg.norm(v)
                v = v / nrm if nrm > 1e-9 else np.array([1.0, 1.0])
                out[n] = (length * v[0], length * v[1])
    return out


def draw_similarity_network(ax, B: nx.Graph, pos: dict, node_party: dict, strength: dict, tau: float,
                            node_fill: dict | None = None, labels: dict | None = None, label_offsets: dict | None = None,
                            size_range=(10, 62), width_range=(0.25, 1.6)):
    """Draw the backbone. Edge width encodes similarity (tau..1 -> width_range); node size encodes weighted
    strength; marker = party; colour = party (or ``node_fill`` if given)."""
    for u, v, d in B.edges(data=True):
        w = width_range[0] + (width_range[1] - width_range[0]) * (d["weight"] - tau) / max(1 - tau, 1e-9)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#8d8c87", lw=w, alpha=0.38, zorder=1, solid_capstyle="round")
    smax = max(strength.values())
    for p in PARTY_ORDER:
        ns = [n for n in B if node_party[n] == p]
        if not ns:
            continue
        s = [size_range[0] + (size_range[1] - size_range[0]) * strength[n] / smax for n in ns]
        if node_fill is None:
            fc, ec, lw = [PARTY_COLORS[p]] * len(ns), "white", 0.5
        else:
            fc, ec, lw = [node_fill[n] for n in ns], INK, 0.8
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns], s=s, c=fc, marker=PARTY_MARKERS[p], edgecolors=ec, linewidths=lw, zorder=3)
    for n, text in (labels or {}).items():
        dx, dy = (label_offsets or {}).get(n, (4, 4))
        ax.annotate(text, pos[n], xytext=(dx, dy), textcoords="offset points", fontsize=5.8, color=INK, zorder=5,
                    ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top",
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": MUTED, "shrinkA": 0, "shrinkB": 2},
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.margins(0.2)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_axis_off()


# ----------------------------------------------------------------------------
# Figure C helpers
# ----------------------------------------------------------------------------
def paired_bars(ax, values: dict, ylabel: str, title: str, ylim=None, fmt="{:.2f}"):
    """Two bars (pre dark, post open + hatch) with their values printed; no error bars (none were calculated)."""
    xs = [0, 1]
    ax.bar(xs[0], values["pre"], width=0.62, color=PRE_FILL, edgecolor=INK, lw=0.7)
    ax.bar(xs[1], values["post"], width=0.62, color=POST_FILL, edgecolor=INK, lw=0.7, hatch="////")
    top = max(values["pre"], values["post"])
    for x, k in zip(xs, ("pre", "post")):
        ax.text(x, values[k] + (ylim[1] if ylim else top) * 0.02, fmt.format(values[k]), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(["before", "after"])
    ax.set_xlim(-0.6, 1.6)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
