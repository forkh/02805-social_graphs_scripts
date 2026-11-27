"""Helpers for the temporal (pre/post 2022 Løgting election) network analysis.

Builds on ``networks.py`` (which is left unchanged). Randomness only comes from
``numpy.random.Generator`` objects passed in by the caller.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

import networks as N

# Faroese general election to the Løgting, and the day the new government took office
# (government.fo; see network_analysis/reports/election_timeline.md for sources).
ELECTION = pd.Timestamp("2022-12-08")
NEW_GOVERNMENT = pd.Timestamp("2022-12-22")


def assign_period(dates: pd.Series) -> pd.Series:
    """pre (< election), transition (election .. new government, inclusive), post (> new government)."""
    out = pd.Series("undated", index=dates.index, dtype=object)
    ok = dates.notna()
    out[ok & (dates < ELECTION)] = "pre"
    out[ok & (dates >= ELECTION) & (dates <= NEW_GOVERNMENT)] = "transition"
    out[ok & (dates > NEW_GOVERNMENT)] = "post"
    return out


def period_matrix(M_contested: pd.DataFrame, keep_cols) -> pd.DataFrame:
    """Columns (divisions) of a contested MP x division matrix; MPs with no vote in them dropped."""
    X = M_contested.loc[:, list(keep_cols)]
    return X.loc[X.notna().any(axis=1)]


def period_parties(ballots: pd.DataFrame, vote_ids) -> pd.Series:
    """Party label for each MP *within one period*: the party they voted under most often there."""
    return N.modal_party(ballots[ballots.vote_id.isin(set(vote_ids))])


def party_assortativity(G: nx.Graph, weight: str | None = None) -> float:
    """Newman's assortativity of the party attribute on the non-isolated part of G.

    Without ``weight`` every edge counts 1 (identical to networkx's attribute assortativity). The
    thresholded similarity graphs here are almost complete, so the unweighted version is close to
    zero by construction; pass ``weight="weight"`` to let each edge count by its similarity.
    """
    H = G.subgraph([n for n, d in G.degree() if d > 0])
    if H.number_of_edges() == 0:
        return float("nan")
    cats = sorted({H.nodes[n]["party"] for n in H})
    ix = {c: i for i, c in enumerate(cats)}
    E = np.zeros((len(cats), len(cats)))
    for u, v, d in H.edges(data=True):
        w = d.get(weight, 1.0) if weight else 1.0
        i, j = ix[H.nodes[u]["party"]], ix[H.nodes[v]["party"]]
        E[i, j] += w
        E[j, i] += w
    E /= E.sum()
    a2 = float((E.sum(axis=1) ** 2).sum())
    return float((np.trace(E) - a2) / (1 - a2)) if a2 < 1 else float("nan")


def scenario_stats(Mx: pd.DataFrame, parties: pd.Series, names: pd.Series, thr: int, seeds,
                   label: str = "", include_blank: bool = False, return_objects: bool = False):
    """Pairwise table -> thresholded graph -> the headline statistics of one (period, threshold)."""
    px = parties.loc[Mx.index]
    pr = N.pairwise_table(Mx, px, include_blank=include_blank, thresholds=(thr,))
    Gx = N.similarity_graph(pr, thr, px, names, Mx.index)
    gs = N.graph_summary(Gx)
    kept = pr[pr.n_shared_divisions >= thr]
    wb = N.within_between(pr, thr) if len(kept) else {}
    out = {"scenario": label, "min_overlap": thr, "n_divisions": int(Mx.shape[1]), "n_mps": int(Mx.shape[0]),
           "n_mps_with_edge": int(Mx.shape[0] - gs["n_isolates"]),
           "share_mps_with_edge": float((Mx.shape[0] - gs["n_isolates"]) / Mx.shape[0]),
           "n_pairs_total": len(pr), "n_pairs_eligible": gs["n_edges"], "density": gs["density"],
           "n_components": gs["n_components"], "largest_component": gs["largest_component"],
           "n_within_pairs": wb.get("n_within", 0), "n_between_pairs": wb.get("n_between", 0),
           "mean_within": wb.get("mean_within", np.nan), "mean_between": wb.get("mean_between", np.nan),
           "gap_within_minus_between": wb.get("gap_within_minus_between", np.nan),
           "party_assortativity": party_assortativity(Gx),
           "party_assortativity_weighted": party_assortativity(Gx, "weight")}
    part = None
    if gs["n_edges"] > 0:
        part, mods, stab = N.louvain_runs(Gx, seeds)
        out.update({"louvain_n_communities": sum(1 for c in part if any(Gx.degree(n) > 0 for n in c)),
                    "louvain_modularity": float(max(mods)), "louvain_modularity_min": float(min(mods)),
                    "louvain_stability_NMI": stab, "party_NMI": N.party_nmi(Gx, part),
                    "party_partition_modularity": nx.community.modularity(Gx, N.party_partition(Gx), weight="weight")})
    return (out, pr, Gx, part) if return_objects else out


def pair_matrix(pairs: pd.DataFrame, ids, thr: int) -> np.ndarray:
    """Square similarity matrix over ``ids`` (NaN where the pair has < thr shared divisions)."""
    idx = {m: i for i, m in enumerate(ids)}
    k = pairs[(pairs.n_shared_divisions >= thr) & pairs.mp_id_1.isin(idx) & pairs.mp_id_2.isin(idx)]
    S = np.full((len(ids), len(ids)), np.nan)
    i, j = k.mp_id_1.map(idx).to_numpy(), k.mp_id_2.map(idx).to_numpy()
    S[i, j] = S[j, i] = k.similarity.to_numpy()
    return S


def _spearman(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok])[0])


def mantel(S1: np.ndarray, S2: np.ndarray, n_perm: int, rng) -> tuple[float, np.ndarray]:
    """Mantel-type Spearman correlation of two similarity matrices over the same MPs.

    Null: relabel the MPs of S2 (rows and columns together), which keeps the dependence between
    dyads that share an MP. Returns (observed, null array).
    """
    iu = np.triu_indices(S1.shape[0], 1)
    a = S1[iu]
    obs = _spearman(a, S2[iu])
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(S1.shape[0])
        null[k] = _spearman(a, S2[np.ix_(p, p)][iu])
    return obs, null


def change_statistics(A: np.ndarray, is_post: np.ndarray, labels: np.ndarray, thr: int, pair_mask=None) -> dict:
    """Pre->post change of pair similarity for the same MPs, over pairs eligible in both subsets.

    A pair is eligible when it has >= thr shared divisions in *both* subsets. ``pair_mask`` (a boolean
    vector over ``np.triu_indices`` pairs) restricts the statistic to a fixed set of pairs, so that
    observed and permuted values are computed over the same pairs.
    """
    sh0, ma0 = N.agreement_counts(A[:, ~is_post])
    sh1, ma1 = N.agreement_counts(A[:, is_post])
    i, j = np.triu_indices(A.shape[0], 1)
    ok = (sh0[i, j] >= thr) & (sh1[i, j] >= thr)
    if pair_mask is not None:
        ok &= pair_mask
    if ok.sum() == 0:
        return {"n_pairs": 0}
    s0, s1 = ma0[i, j][ok] / sh0[i, j][ok], ma1[i, j][ok] / sh1[i, j][ok]
    same = (labels[i] == labels[j])[ok]
    d = s1 - s0
    return {"n_pairs": int(ok.sum()), "mean_abs_change": float(np.abs(d).mean()),
            "mean_change_within": float(d[same].mean()) if same.any() else float("nan"),
            "mean_change_between": float(d[~same].mean()) if (~same).any() else float("nan"),
            "spearman_pre_post": _spearman(s0, s1),
            "median_shared_pre": float(np.median(sh0[i, j][ok])), "median_shared_post": float(np.median(sh1[i, j][ok]))}


def eligible_pair_mask(A: np.ndarray, is_post: np.ndarray, thr: int) -> np.ndarray:
    """Boolean vector over ``np.triu_indices`` pairs: >= thr shared divisions in both subsets."""
    sh0, _ = N.agreement_counts(A[:, ~is_post])
    sh1, _ = N.agreement_counts(A[:, is_post])
    i, j = np.triu_indices(A.shape[0], 1)
    return (sh0[i, j] >= thr) & (sh1[i, j] >= thr)


def block_permutation(blocks: np.ndarray, n_post: int, rng) -> np.ndarray:
    """Random 'post' indicator over divisions that keeps whole blocks (e.g. cases) together.

    Blocks are put in random order and assigned to 'post' until the number of post divisions is as
    close as possible to ``n_post`` (within one block's size).
    """
    ids = pd.unique(blocks)
    order = rng.permutation(ids)
    sizes = pd.Series(blocks).value_counts().loc[order].to_numpy()
    cum = np.cumsum(sizes)
    k = int(np.searchsorted(cum, n_post))
    if k > 0 and abs(cum[k - 1] - n_post) <= abs(cum[min(k, len(cum) - 1)] - n_post):
        k -= 1
    return np.isin(blocks, order[: k + 1])


def period_label_permutation(A: np.ndarray, is_post: np.ndarray, labels: np.ndarray, thr: int,
                             n_perm: int, rng, blocks=None, pair_mask=None) -> pd.DataFrame:
    """Null for 'the same MPs behave differently in the two sets of divisions'.

    The pre/post label is re-assigned at random and the change statistics recomputed. With
    ``blocks=None`` individual divisions are exchangeable; with ``blocks`` (case ids) whole cases move
    together, which preserves the clustering of divisions within a case. The null captures sampling
    variation of the split only: it is blind to everything that varies with time (agenda, calendar,
    roles) and does not keep each MP's period-specific attendance.
    """
    n_post = int(is_post.sum())
    rows = []
    for _ in range(n_perm):
        ip = rng.permutation(is_post) if blocks is None else block_permutation(blocks, n_post, rng)
        rows.append(change_statistics(A, ip, labels, thr, pair_mask))
    return pd.DataFrame(rows)


def split_half_benchmark(A: np.ndarray, labels: np.ndarray, thr: int, n_perm: int, rng, blocks=None) -> pd.DataFrame:
    """Within one period: split its divisions into two random halves and compare pair similarity.

    A 'no change' reference at the period's real attendance patterns. Halves have half the divisions,
    so ``thr`` should be lowered accordingly; the result is then conservative (noisier than the real
    pre-vs-post comparison).
    """
    L = A.shape[1]
    rows = []
    for _ in range(n_perm):
        if blocks is None:
            half = np.zeros(L, bool)
            half[rng.choice(L, L // 2, replace=False)] = True
        else:
            half = block_permutation(blocks, L // 2, rng)
        rows.append(change_statistics(A, half, labels, thr))
    return pd.DataFrame(rows)


def partition_agreement(part_a, part_b, nodes) -> float:
    """NMI between two partitions restricted to ``nodes`` (all must appear in both)."""
    return float(normalized_mutual_info_score(N.communities_to_labels(part_a, nodes),
                                              N.communities_to_labels(part_b, nodes)))


def make_graph_stats(labels: np.ndarray, min_overlap: int, seed: int):
    """Return f(shared, match) -> Louvain modularity and party NMI of the thresholded graph.

    Used as ``graph_fn`` of ``networks.division_null`` so that observed and null networks are
    summarised by exactly the same function (single Louvain seed).
    """
    def graph_stats(shared, match):
        a, b, s = N.upper_arrays(shared, match, min_overlap)
        g = nx.Graph()
        g.add_nodes_from(range(shared.shape[0]))
        g.add_weighted_edges_from(zip(a, b, s))
        part = nx.community.louvain_communities(g, weight="weight", seed=seed)
        nodes = [n for n, d in g.degree() if d > 0]
        lab = {n: k for k, c in enumerate(part) for n in c}
        return {"louvain_Q": nx.community.modularity(g, part, weight="weight"),
                "party_NMI": float(normalized_mutual_info_score(labels[nodes], [lab[n] for n in nodes]))}
    return graph_stats


def community_alignment(part_a, part_b, nodes) -> tuple[pd.DataFrame, dict]:
    """Contingency of two partitions on ``nodes`` and a best one-to-one matching of their communities.

    Returns (crosstab, {node: True if the node's community is not the one matched to its
    community in part_a}) -- a simple 'changed community' flag; meaningful only when the
    partitions have comparable communities.
    """
    from scipy.optimize import linear_sum_assignment
    la, lb = N.communities_to_labels(part_a, nodes), N.communities_to_labels(part_b, nodes)
    ct = pd.crosstab(pd.Series(la, index=nodes, name="a"), pd.Series(lb, index=nodes, name="b"))
    r, c = linear_sum_assignment(-ct.to_numpy())
    match = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
    changed = {n: match.get(a) != b for n, a, b in zip(nodes, la, lb)}
    return ct, changed
