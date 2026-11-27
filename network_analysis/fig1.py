"""Fig. 1 of the paper (= report_networks/figures_formal_roles/fig_svt_final_v4.*), PNAS double column, panels a-c.

    python reproduce_networks/fig1.py            (writes OUT_DIR/figures/fig_svt_final_v4.{pdf,png} and fig_svt_final_v4_provenance.json)

Copied from networks/formal_roles/src/make_fig_svt_final_v4.py (panels b, c, layout, checks) and make_fig_svt_final.py (panel a); the code of
every panel is unchanged. Changes: paths (OUT_DIR), MP names are looked up at run time from DATA_DIR/members_2020-2023.csv (the ringed MP is
given by its MP ID), no caption draft is written. Style constants: figstyle.py.
a  agreement by pair type, pooled per contested division (speech window), 95% bootstrap intervals  [OUT_DIR/formal_roles]
b  pre-election voting network (P1, pairs with >= 40 shared contested divisions, edges >= 0.5); one spring layout per connected component
c  P1 agreement with government MPs vs other opposition MPs for the ringed MP and the opposition parties JAV, TJO, FRA
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as CFG  # noqa: E402
import figstyle as C  # noqa: E402   (style constants of report_networks/figure_code/figure_config.py)
import roles as R  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# constants of make_figures_formal_roles.py / make_fig_svt_final.py / make_fig_svt_final_v4.py
SHORT = dict(zip(C.PARTY_ORDER, ["JAV", "TJÓ", "SAM", "FÓL", "FRA", "MIÐ", "SJÁ"]))
MIN_FONT, SEED_JITTER, WIDTH_IN, EDGE_MIN = 6.0, 2805, 17.8 / 2.54, 0.5
MINUS = lambda s: re.sub(r"(?<![\w])-(?=\d)", "−", s)  # noqa: E731
TYPES = [("same_party", "same party", "#1f4e8c"), ("other_party_same_side", "other party, same bloc", "#6fa3dc"),
         ("opposite_side", "opposite blocs", "#eb6834")]
COL = {k: c for k, _, c in TYPES}
FOCUS, SECOND = "MPID000035", "MPID000036"          # the ringed Sjálvstýri MP; the second Sjálvstýri MP (only checked in panel c)
RING = "#c62828"
STEM = "fig_svt_final_v4"
PRE = "pairwise_similarity_pre_election.csv"            # OUT_DIR/temporal, written by temporal_network.py
GOV_PARTIES, OPP_PARTIES = R.TABLE["P1"]["government"], R.TABLE["P1"]["opposition"]
GOV_FILL, OPP_FILL = "#ffffff", "#9a9a9a"
THR_B = 40
GAP = 0.4
FOCUS_LABEL = None                                  # "<initial>. <surname>" of FOCUS, set in main() from the members table


def check(ok, what):
    if not ok:
        raise AssertionError("CHECK FAILED: " + what)


def panel_a(ax, T, res, V):
    D = pd.read_csv(T / "a_pair_types_per_division_window.csv")
    S = res["a_pair_types_per_division"]["window_pooled"]; ref = S["all"]["agreement"]
    check(abs(D.match_all.sum() / D.pairs_all.sum() - ref) < 1e-12, "all-pairs agreement = results.json")
    rng = np.random.default_rng(SEED_JITTER); out = {}
    for i, (k, lab, c) in enumerate(TYPES):
        p, m = D[f"pairs_{k}"], D[f"match_{k}"]
        y = float(m.sum() / p.sum()); lo, hi = S[k]["ci95_divisions"]
        check(abs(y - S[k]["agreement"]) < 1e-12 and int(p.sum()) == S[k]["pair_votes"] and int((p > 0).sum()) == S[k]["divisions"], f"{k} = results.json")
        dots = (m[p > 0] / p[p > 0]).to_numpy()
        ax.bar(i, y, 0.62, color=c, zorder=1)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(dots)), dots, s=1.2, color="#8d8c87", alpha=0.3, linewidths=0, zorder=2)
        ax.hlines(y, i - 0.31, i + 0.31, color=c, lw=1.2, zorder=4)
        ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], color=C.INK, lw=0.8, capsize=2, zorder=5)
        yt = max(y, dots.max()) + 0.03
        if yt < ref + 0.03 and yt + 0.13 > ref - 0.01:
            yt = ref + 0.03
        ax.text(i, yt, f"{y:.2f}", ha="center", va="bottom", fontsize=MIN_FONT)
        out[k] = {"agreement": y, "ci95": [lo, hi], "pair_votes": int(p.sum()), "divisions_with_pairs": int((p > 0).sum())}
    ax.axhline(ref, color=C.INK, lw=0.8, ls="--", zorder=3, label=f"all pairs ({ref:.2f})")
    ax.legend(loc="upper right", fontsize=MIN_FONT, handlelength=1.8, borderaxespad=0.1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{l_}\n({out[k]['pair_votes'] / 1000:.1f}k)" for (k, _, _), l_ in
                        zip(TYPES, ["same\nparty", "other party,\nsame bloc", "opposite\nblocs"])], fontsize=MIN_FONT)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("pair type (pair-votes)")
    ax.set_ylabel("agreement on contested divisions\n(share of matching pair-votes)")
    V["a"] = {"pair_types": out, "all_pairs": ref, "divisions": int(len(D)), "jitter_seed": SEED_JITTER}


def p1_frame(res):
    P = pd.read_csv(CFG.TEMPORAL / PRE)
    E = P[P.meets_min_overlap_20].copy()
    check(bool((E.n_shared_divisions >= 20).all()), "meets_min_overlap_20 = n_shared >= 20")
    side = {p: "government" for p in GOV_PARTIES} | {p: "opposition" for p in OPP_PARTIES}
    check(set(E.party_1) | set(E.party_2) <= set(side), "every party in the P1 role table")
    E["same_bloc"] = E.party_1.map(side) == E.party_2.map(side)
    E["ptype"] = np.where(E.party_1 == E.party_2, "same_party", np.where(E.same_bloc, "other_party_same_side", "opposite_side"))
    row = next(x for x in res["a_pair_level"] if x["network"] == "P1" and x["min_overlap"] == 20)
    check(len(E) == row["pairs"] == 815 and row["pairs_side_changed"] == 0, "815 P1 pairs; no pair changes bloc within P1")
    for k in ("same_party", "other_party_same_side", "opposite_side"):
        y = E[E.ptype == k].similarity
        check(len(y) == row[f"n_{k}"] and abs(y.mean() - row[f"mean_{k}"]) < 1e-12, f"P1 {k}: n and mean = results.json")
    return E, side


def network_stats(E, side, names, thr):
    """Counts of the P1 network at a shared-division threshold (edges >= EDGE_MIN); no layout."""
    Et = E[E.n_shared_divisions >= thr]
    party = {**dict(zip(Et.mp_id_1, Et.party_1)), **dict(zip(Et.mp_id_2, Et.party_2))}
    nodes = sorted(party)
    K = Et[Et.similarity >= EDGE_MIN]
    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(zip(K.mp_id_1, K.mp_id_2))
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: (-len(c), c[0]))
    opp = K[K.ptype == "opposite_side"]; per = {}
    for a, p1, b, p2 in zip(opp.mp_id_1, opp.party_1, opp.mp_id_2, opp.party_2):
        o_ = a if side[p1] == "opposition" else b
        per[o_] = per.get(o_, 0) + 1
    return {"min_shared_divisions": thr, "pairs": int(len(Et)), "mps": len(nodes), "edges": int(len(K)),
            "edges_by_type": {k: int((K.ptype == k).sum()) for k in COL}, "opposite_bloc_edges": int(len(opp)),
            "opposite_bloc_edges_by_opposition_mp": {m: {"name": names[m], "party": party[m], "edges": n_} for m, n_ in sorted(per.items())},
            "component_sizes_ge2": [len(c) for c in comps if len(c) >= 2],
            "isolated_mps": [{"mp_id": c[0], "name": names[c[0]], "party": party[c[0]]} for c in comps if len(c) == 1]}


def component_report(E, side, names, thr):
    """Composition of the components at the 0.5 cut, the opposite-bloc edges, and component stability over edge cuts (no layout)."""
    Et = E[E.n_shared_divisions >= thr]
    party = {**dict(zip(Et.mp_id_1, Et.party_1)), **dict(zip(Et.mp_id_2, Et.party_2))}
    nodes = sorted(party)

    def comps_at(cut):
        K = Et[Et.similarity >= cut]
        G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(zip(K.mp_id_1, K.mp_id_2))
        return sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: (-len(c), c[0]))
    base = [c for c in comps_at(EDGE_MIN) if len(c) >= 2]
    comp = []
    for c in base:
        d = pd.DataFrame({"mp": c, "party": [party[m] for m in c], "bloc": [side[party[m]] for m in c]})
        comp.append({"size": len(c), "by_bloc": d.bloc.value_counts().sort_index().to_dict(),
                     "by_party": d.groupby(["bloc", "party"]).size().rename("n").reset_index().to_dict(orient="records"),
                     "opposition_mps_in_component_if_government_majority": [names[m] for m in c if side[party[m]] == "opposition"]
                     if (d.bloc == "government").mean() > 0.5 else None})
    tot = {b: sum(side[party[m]] == b for m in nodes) for b in ("government", "opposition")}
    K5 = Et[Et.similarity >= EDGE_MIN]; opp = K5[K5.ptype == "opposite_side"]
    stab = []
    for cut in np.round(np.arange(0.50, 0.701, 0.05), 2):
        cs = [c for c in comps_at(cut) if len(c) >= 2]
        stab.append({"cut": float(cut), "components_ge2": [len(c) for c in cs], "same_membership_as_0.5": cs == base,
                     "isolated": [names[c[0]] for c in comps_at(cut) if len(c) == 1]})
    merge = None
    for cut in np.round(np.arange(0.50, -0.001, -0.01), 2):          # highest cut below 0.5 at which the two components are joined
        if len([c for c in comps_at(cut) if len(c) >= 2]) == 1:
            merge = float(cut); break
    return {"min_shared_divisions": thr, "mps_by_bloc": tot, "components": comp,
            "opposite_bloc_edges": {"n": int(len(opp)), "all_involve": sorted({names[m] for a, b in zip(opp.mp_id_1, opp.mp_id_2) for m in (a, b)
                                                                              if side[party[m]] == "opposition"}),
                                    "similarity_range": [float(opp.similarity.min()), float(opp.similarity.max())] if len(opp) else None},
            "stability_over_edge_cuts": stab, "highest_cut_below_0.5_where_components_merge": merge}


def panel_b(ax, E, side, names, V):
    Et = E[E.n_shared_divisions >= THR_B]
    P = pd.read_csv(CFG.TEMPORAL / PRE)
    check(set(zip(Et.mp_id_1, Et.mp_id_2)) == set(zip(*P[P.meets_min_overlap_40][["mp_id_1", "mp_id_2"]].to_numpy().T)),
          "P1 pairs >= 40 = meets_min_overlap_40 in the P1 table")
    party = {**dict(zip(Et.mp_id_1, Et.party_1)), **dict(zip(Et.mp_id_2, Et.party_2))}
    nodes = sorted(party)
    check(FOCUS in nodes and party[FOCUS] == "Sjálvstýri", "ringed MP in the P1 network at >= 40")
    K = Et[Et.similarity >= EDGE_MIN]
    G = nx.Graph(); G.add_nodes_from(range(len(nodes))); ix = {n: i for i, n in enumerate(nodes)}
    G.add_weighted_edges_from((ix[a], ix[b], float(w)) for a, b, w in zip(K.mp_id_1, K.mp_id_2, K.similarity))
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: (-len(c), c[0]))
    iso = [c[0] for c in comps if len(c) == 1]
    conn = sorted(i for c in comps if len(c) >= 2 for i in c)
    # one spring layout PER CONNECTED COMPONENT (weight = similarity, k = 2/sqrt(n), 500 iterations, seed 2805), components side by side,
    # larger left, gap GAP; components come from the drawn network only (no bloc, party or Louvain information). A single layout of the
    # disconnected network pushes the components 30-50x their spread apart (decision of the project author, see provenance).
    # Integer node labels in sorted mp_id order keep it independent of PYTHONHASHSEED.
    big = [c for c in comps if len(c) >= 2]
    pos, x0, n_max = {}, 0.0, len(big[0])
    for c in big:
        pc = nx.spring_layout(G.subgraph(c), weight="weight", seed=C.SEED_VOTING_LAYOUT, k=2.0 / np.sqrt(len(c)), iterations=500)
        pts = np.array([pc[i] for i in c]); pts = (pts - pts.mean(0)) * np.sqrt(len(c) / n_max)
        pts[:, 0] += x0 - pts[:, 0].min()
        for i, xy in zip(c, pts):
            pos[nodes[i]] = xy
        x0 = pts[:, 0].max() + GAP
    Pp = np.array(list(pos.values())); xmin, ymin = Pp.min(0); xmax, ymax = Pp.max(0); span = max(xmax - xmin, ymax - ymin)
    for k_, i in enumerate(iso):                                     # isolated MPs: bottom-right edge of the panel
        pos[nodes[i]] = np.array([xmax - 0.1 * span * k_, ymin - 0.12 * span])
    lo, hi = float(K.similarity.min()), float(K.similarity.max())
    order = {"same_party": 0, "other_party_same_side": 0, "opposite_side": 1}
    Ks = K.assign(o=K.ptype.map(order)).sort_values(["o", "similarity", "mp_id_1", "mp_id_2"])   # opposite-bloc edges last
    for r in Ks.itertuples():
        a, b = pos[r.mp_id_1], pos[r.mp_id_2]; f = (r.similarity - lo) / (hi - lo)
        if r.ptype == "opposite_side":
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COL[r.ptype], lw=0.8, alpha=0.6, zorder=2, solid_capstyle="round")
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COL[r.ptype], lw=0.3 + 0.5 * f, alpha=0.06 + 0.5 * f ** 2, zorder=1, solid_capstyle="round")
    for n in nodes:
        fill = GOV_FILL if side[party[n]] == "government" else OPP_FILL
        ax.scatter(*pos[n], s=22, marker=C.PARTY_MARKERS[party[n]], facecolor=fill, edgecolor=C.INK, linewidths=0.55, zorder=3)
    ax.scatter(*pos[FOCUS], s=85, marker="o", facecolor="none", edgecolor=RING, linewidths=0.9, zorder=4)
    # diagnostics AFTER the layout (bloc used only to describe the result); opposition reference group excludes Sjálvstýri MPs
    isoset = {nodes[i] for i in iso}
    comp_of_focus = {nodes[i] for c in comps if ix[FOCUS] in c for i in c}
    gov = np.array([pos[n] for n in nodes if side[party[n]] == "government" and n not in isoset])
    oppm = np.array([pos[n] for n in nodes if side[party[n]] == "opposition" and party[n] != "Sjálvstýri" and n not in isoset])
    cg, co = gov.mean(0), oppm.mean(0)
    spread = float(np.mean([np.linalg.norm(gov - cg, axis=1).mean(), np.linalg.norm(oppm - co, axis=1).mean()]))
    sep = float(np.linalg.norm(cg - co)); two_lobes = sep > 3 * spread
    dg = float(np.linalg.norm(gov - pos[FOCUS], axis=1).mean()); do = float(np.linalg.norm(oppm - pos[FOCUS], axis=1).mean())
    p = pos[FOCUS]
    lobe = np.array([pos[n] for n in nodes if n in comp_of_focus])
    label_xy = (lobe[:, 0].min(), lobe[:, 1].min() - 0.10 * span)          # below the lobe, at its left edge (outside the lobe)
    ax.annotate(FOCUS_LABEL, xy=tuple(p), xytext=label_xy, ha="left", va="top", fontsize=MIN_FONT, color=C.INK, zorder=5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C.INK, shrinkA=1, shrinkB=4.5))
    if two_lobes:                                                    # small lobe labels above each lobe
        for c_, ext, txt in ((cg, gov, "government"), (co, oppm, "opposition")):
            ax.text(c_[0], ext[:, 1].max() + 0.08 * span, txt, ha="center", va="bottom", fontsize=MIN_FONT, color=C.INK, style="italic")
    ax.set_aspect("equal", adjustable="datalim"); ax.set_axis_off(); ax.margins(0.03)
    V["_label_bbox_check"] = (ax, [pos[n] for n in nodes])
    present = [p_ for p_ in C.PARTY_ORDER if p_ in set(party.values())]
    st = network_stats(E, side, names, THR_B)
    check(st["mps"] == len(nodes) and st["edges"] == len(K) and [nodes[i] for i in iso] == [x["mp_id"] for x in st["isolated_mps"]], "stats = drawn network")
    V["b"] = {"network": f"P1 (pre-election), pairs with >= {THR_B} shared contested divisions, edges >= {EDGE_MIN}", "edge_cut": EDGE_MIN,
              "at_40": st, "at_20_for_comparison": network_stats(E, side, names, 20), "components_at_40": component_report(E, side, names, THR_B),
              "two_lobes": {"answer": bool(two_lobes), "rule": "distance between the layout centroids of government MPs and of non-SJÁ opposition MPs "
                            "> 3 x their mean within-group spread", "centroid_distance": sep, "mean_within_spread": spread},
              "ringed_mp": {"mp_id": FOCUS, "mean_layout_distance_to_government_MPs": dg, "mean_layout_distance_to_opposition_MPs_excl_SJA": do,
                                    "nearest_lobe": "government" if dg < do else "opposition"},
              "node_fill": {"government MP": GOV_FILL, "opposition MP": OPP_FILL},
              "edge_style": "same-bloc edges: opacity 0.06 + 0.5 f^2, width 0.3 + 0.5 f; opposite-bloc edges drawn last: opacity 0.6, width 0.8 pt",
              "layout": "networkx spring_layout per connected component of the drawn network (edges >= 0.5, weight = similarity), k = 2/sqrt(n), "
                        f"500 iterations, seed 2805; each component centred, scaled by sqrt(n / n_largest), placed side by side (larger left, gap {GAP} "
                        "layout units; distance arbitrary); no bloc, party or Louvain information; isolated MPs (if any) at the bottom-right edge",
              "layout_decision": "per-component layout chosen by the project author: at >= 40 shared divisions the network has two components, and a single "
                                 "converged spring layout puts them 30-50x their internal spread apart",
              "layout_coordinates": {n: [float(v) for v in pos[n]] for n in nodes},
              "layout_seed": C.SEED_VOTING_LAYOUT, "party_abbreviations": {SHORT[p_]: p_ for p_ in present}}
    return present


def legend_b(ax_a, ax, present):
    h = [Line2D([0], [0], marker=C.PARTY_MARKERS[p], ls="", markersize=4.5, markerfacecolor="none", markeredgecolor=C.INK,
                markeredgewidth=0.6, label=SHORT[p]) for p in present]
    h += [Line2D([0], [0], marker="o", ls="", markersize=4.5, markerfacecolor=GOV_FILL, markeredgecolor=C.INK, markeredgewidth=0.6, label="government MP"),
          Line2D([0], [0], marker="o", ls="", markersize=4.5, markerfacecolor=OPP_FILL, markeredgecolor=C.INK, markeredgewidth=0.6, label="opposition MP"),
          Line2D([0], [0], marker="o", ls="", markersize=6.5, markerfacecolor="none", markeredgecolor=RING, markeredgewidth=0.9, label="Sjálvstýri MP (ringed)")]
    xc = (ax_a.get_position().x0 + ax.get_position().x1) / 2          # 2-row legend centred under panels (a)-(b), below the x-axis labels
    ax.figure.legend(handles=h, loc="upper center", bbox_to_anchor=(xc, -0.17), ncol=5, fontsize=MIN_FONT, handletextpad=0.1, columnspacing=0.9,
                     borderaxespad=0, frameon=False)


def panel_c(ax, res, V):
    e = res["e_sjalvstyri"]["P1"]; ref = res["e_sjalvstyri"]["reference_opposition_parties_P1"]
    check(e[FOCUS]["divisions"] == 83 and e[SECOND]["divisions"] == 24, "division counts 83 and 24")
    rows = [(f"{FOCUS_LABEL}\n(SJÁ, {e[FOCUS]['divisions']} div.)", e[FOCUS])] + \
           [(f"{SHORT[p]}\n({ref[p]['divisions']} div.)", ref[p]) for p in ["Javnaðarflokkurin", "Tjóðveldi", "Framsókn"]]
    cg, co = COL["opposite_side"], COL["other_party_same_side"]
    check(cg == "#eb6834" and co == "#6fa3dc", "colours as requested")
    out = []
    for i, (lab, r) in enumerate(rows):
        for key, ci, c, dy in (("agreement_with_government", "ci95_government", cg, -0.14), ("agreement_with_other_opposition", "ci95_other_opposition", co, 0.14)):
            y = r[key]; lo, hi = r[ci]
            check(lo <= y + 1e-12 <= hi + 2e-12, f"{lab} {key} inside its interval")
            ax.errorbar(y, i + dy, xerr=[[y - lo], [hi - y]], fmt="o", ms=3, color=c, lw=0.8, capsize=1.5)
        out.append({"label": lab.replace("\n", " "), "divisions": r["divisions"], "with_government": r["agreement_with_government"],
                    "ci_government": r["ci95_government"], "with_other_opposition": r["agreement_with_other_opposition"],
                    "ci_other_opposition": r["ci95_other_opposition"], "sign_flip_p": r["sign_flip_null"]["p_two_sided"]})
    ax.axhline(0.5, color="#b8b8b4", lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=MIN_FONT, linespacing=1.0); ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.03); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("agreement before the election (P1)"); ax.set_ylabel("opposition MP or party", labelpad=2)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3, color=cg, label="with the opposite bloc (government)"),
                       Line2D([0], [0], marker="o", ls="", ms=3, color=co, label="with other opposition parties")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=MIN_FONT, borderaxespad=0, handletextpad=0.2)
    V["c"] = {"rows": out, "colours": {"with government MPs": cg, "with other opposition MPs": co}, "dropped": ["MPID000036 (24 divisions)", "MPID000038 (1 division)", "pooled Sjálvstýri row"]}


def main() -> int:
    global FOCUS_LABEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CFG.FIGURES))
    a = ap.parse_args()
    out = Path(a.out)
    paths = {k: out / f"{STEM}{s}" for k, s in (("pdf", ".pdf"), ("png", ".png"), ("prov", "_provenance.json"))}
    if any(p.exists() for p in paths.values()):
        print("refusing to overwrite:", ", ".join(p.name for p in paths.values() if p.exists())); return 2
    T = CFG.FORMAL
    res = json.loads((T / "results.json").read_text(encoding="utf-8"))
    members = pd.read_csv(CFG.DATA_DIR / "members_2020-2023.csv").set_index("mp_id")
    check(members.loc[FOCUS, "parties"] == "Sjálvstýri", "the ringed MP is a Sjálvstýri MP (members table)")
    nm = members.loc[FOCUS, "name"].split()
    FOCUS_LABEL = f"{nm[0][0]}. {nm[-1]}"
    C.apply_style()
    plt.rcParams.update({"font.size": 7.0, "axes.labelsize": 7.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.grid": False})
    E1, side = p1_frame(res); V = {}
    names = members.name.to_dict()
    fig = plt.figure(figsize=(WIDTH_IN, 2.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.55, 0.85], wspace=0.42)
    inner = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.12)
    ax_a, ax_b, ax_c = fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1]), fig.add_subplot(outer[0, 1])
    panel_a(ax_a, T, res, V)
    pv = V["a"]["pair_types"]
    ticks = [f"same\nparty\n({pv['same_party']['pair_votes'] / 1000:.1f}k)", f"same bloc,\nother party\n({pv['other_party_same_side']['pair_votes'] / 1000:.1f}k)",
             f"opposite\nblocs\n({pv['opposite_side']['pair_votes'] / 1000:.1f}k)"]
    check(ticks == ["same\nparty\n(20.5k)", "same bloc,\nother party\n(37.6k)", "opposite\nblocs\n(61.1k)"], f"panel (a) tick labels {ticks}")
    ax_a.set_xticklabels(ticks, fontsize=MIN_FONT, linespacing=1.05)
    present = panel_b(ax_b, E1, side, names, V); legend_b(ax_a, ax_b, present); panel_c(ax_c, res, V)
    for ax, lab, x in ((ax_a, "a", -0.32), (ax_b, "b", 0.0), (ax_c, "c", -0.62)):
        ax.text(x, 1.02, lab, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tb = [t.get_window_extent(rend) for t in ax_a.get_xticklabels()]
    check(all(tb[k].x1 < tb[k + 1].x0 for k in range(len(tb) - 1)), "panel (a) tick labels do not overlap")
    axb, pts = V.pop("_label_bbox_check")
    ann = [t for t in axb.texts if t.get_text() == FOCUS_LABEL][0]
    lab = matplotlib.text.Text.get_window_extent(ann, rend).expanded(1.05, 1.1)          # text only (the leader line ends at the ringed node)
    xy = axb.transData.transform(np.array(pts))
    check(not ((xy[:, 0] >= lab.x0 - 4) & (xy[:, 0] <= lab.x1 + 4) & (xy[:, 1] >= lab.y0 - 4) & (xy[:, 1] <= lab.y1 + 4)).any(),
          "ringed-MP label overlaps no node")
    bb = axb.get_window_extent(rend)
    check(bb.x0 <= lab.x0 and lab.x1 <= bb.x1 and bb.y0 <= lab.y0, "ringed-MP label inside panel (b)")
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    check(not [t.get_text() for t in texts if t.get_fontsize() < MIN_FONT - 1e-9], f"text >= {MIN_FONT} pt")
    check(not any("side" in t.get_text().lower() for t in texts), "no 'side' in the figure")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["pdf"], bbox_inches="tight", metadata={"CreationDate": None}); fig.savefig(paths["png"], bbox_inches="tight", dpi=300); plt.close(fig)
    check(b"/Type3" not in paths["pdf"].read_bytes(), "Type 3 fonts")
    prov = {"figure": [paths["pdf"].name, paths["png"].name], "script": "reproduce_networks/fig1.py (copy of networks/formal_roles/src/make_fig_svt_final_v4.py)",
            "inputs": ["OUT_DIR/formal_roles/results.json", "OUT_DIR/formal_roles/a_pair_types_per_division_window.csv", f"OUT_DIR/temporal/{PRE}",
                       "DATA_DIR/members_2020-2023.csv"], "values": V,
            "pair_type_colours": {lab: c for _, lab, c in TYPES}, "ring_colour": RING, "seeds": {"layout": C.SEED_VOTING_LAYOUT, "jitter": SEED_JITTER},
            "width_cm": 17.8, "png_dpi": 300, "min_font_pt": MIN_FONT,
            "environment": {"python": sys.version.split()[0], "platform": platform.system(), "matplotlib": matplotlib.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__, "networkx": nx.__version__}}
    paths["prov"].write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("written:", ", ".join(p.name for p in paths.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
