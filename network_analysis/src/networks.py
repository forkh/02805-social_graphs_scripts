"""Network-analysis helpers for the Løgting voting data (DTU 02805).

Everything here is deterministic given its inputs (randomness comes from an
explicit ``numpy.random.Generator`` passed by the caller) and nothing writes to
``data/``: the only writer, :func:`output_path`, refuses any path outside
``network_analysis/outputs/`` (and ``network_analysis/reports/`` for the temporal notebook's ``.md`` files).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

FOR, AGAINST, BLANK = "Fyri", "Ímóti", "Blankt"
VOTE_CODE = {FOR: 1, AGAINST: -1, BLANK: 0}  # absent (empty label) -> NaN
STEM = "2020-2023"
# speech-dataset (FPSC) coverage, used as the window of the main comparison
WINDOW_START, WINDOW_END = pd.Timestamp("2020-07-29"), pd.Timestamp("2023-05-04")

# Fixed party order, colours (validated categorical palette, slots 1-7) and
# marker shapes. With 7 parties colour alone is not CVD-safe for all pairs, so
# figures must also use the marker shape / a legend.
PARTY_ORDER = ["Javnaðarflokkurin", "Tjóðveldi", "Sambandsflokkurin", "Fólkaflokkurin",
               "Framsókn", "Miðflokkurin", "Sjálvstýri"]
PARTY_COLORS = dict(zip(PARTY_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                      "#e87ba4", "#008300", "#6250d6"]))
PARTY_MARKERS = dict(zip(PARTY_ORDER, ["o", "s", "^", "D", "v", "P", "X"]))


# ----------------------------------------------------------------------------
# paths and loading
# ----------------------------------------------------------------------------
def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the folder holding data/ and network_analysis/src/."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data").is_dir() and ((cand / "network_analysis" / "src").is_dir() or (cand / "networks" / "src").is_dir()):
            return cand
    raise FileNotFoundError("repository root (with data/ and network_analysis/src/) not found")


def output_path(root: Path, *parts: str) -> Path:
    """Path under network_analysis/outputs/ (created if needed); refuses anything else.

    Layout: ``tables/``, ``figures/``, ``results/``. The temporal notebook passes ``("temporal", ...)``
    and its files are routed by kind: figures -> ``figures/temporal/``, ``.md`` -> ``network_analysis/reports/``,
    ``.json`` -> ``results/temporal/``, everything else (csv) -> ``tables/temporal/``.
    """
    net = ((Path(root) / "network_analysis") if (Path(root) / "network_analysis").is_dir() else (Path(root) / "networks")).resolve()
    base = net / "outputs"
    if parts and parts[0] == "temporal":
        rest = Path(*parts[1:])
        if rest.parts[0] == "figures":
            p = base / "figures" / "temporal" / Path(*rest.parts[1:])
        elif rest.suffix == ".md":
            p = net / "reports" / rest
        elif rest.suffix == ".json":
            p = base / "results" / "temporal" / rest
        else:
            p = base / "tables" / "temporal" / rest
    else:
        p = base / Path(*parts)
    p = p.resolve()
    allowed = (base, net / "reports")
    if not any(a == p or a in p.parents for a in allowed):
        raise ValueError(f"refusing to write outside {base} (or {net / 'reports'}): {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_data(root: Path) -> dict[str, pd.DataFrame]:
    """Read the four source tables (read-only) from data/*_2020-2023.parquet."""
    d = Path(root) / "data"
    out = {n: pd.read_parquet(d / f"{n}_{STEM}.parquet")
           for n in ("ballots", "votes", "cases", "members")}
    out["cases"]["case_id"] = out["cases"]["case_id"].astype("int64")  # str in the file
    out["votes"]["meeting_date"] = pd.to_datetime(out["votes"]["meeting_date"])
    return out


def restrict_window(votes: pd.DataFrame, ballots: pd.DataFrame,
                    start: pd.Timestamp = WINDOW_START, end: pd.Timestamp = WINDOW_END):
    """Votes whose plenary meeting date lies in [start, end] (undated votes drop out)."""
    v = votes[votes.meeting_date.between(start, end)]
    return v, ballots[ballots.vote_id.isin(v.vote_id)]


def modal_party(ballots: pd.DataFrame) -> pd.Series:
    """Party under which each MP cast most ballots (party at vote time, mode)."""
    return ballots.groupby("mp_id").party.agg(lambda s: s.value_counts().idxmax())


# ----------------------------------------------------------------------------
# votes -> matrices
# ----------------------------------------------------------------------------
def vote_matrix(ballots: pd.DataFrame) -> pd.DataFrame:
    """MP x division matrix: +1 For, -1 Against, 0 Blank, NaN absent / not listed."""
    cast = ballots[ballots.vote.isin(VOTE_CODE)]
    return (cast.assign(code=cast.vote.map(VOTE_CODE).astype(float))
                .pivot(index="mp_id", columns="vote_id", values="code")
                .sort_index())


def contested_mask(M: pd.DataFrame, min_minority: int = 1) -> pd.Series:
    """Divisions with >= min_minority For AND >= min_minority Against votes."""
    n_for, n_against = (M == 1).sum(axis=0), (M == -1).sum(axis=0)
    return (n_for >= min_minority) & (n_against >= min_minority)


def agreement_counts(A: np.ndarray, include_blank: bool = False):
    """Pairwise (n_shared, n_match) from a +1/-1/0/NaN matrix.

    A division is jointly observed for a pair when both cast a *valid* vote
    (For/Against; also Blank when include_blank). They match when equal.
    """
    cats = [1, -1] + ([0] if include_blank else [])
    ind = {c: (A == c).astype(float) for c in cats}
    valid = sum(ind.values())
    shared = valid @ valid.T
    match = sum(i @ i.T for i in ind.values())
    return shared, match


def pairwise_table(M: pd.DataFrame, parties: pd.Series, include_blank: bool = False,
                   thresholds=(20, 40, 60)) -> pd.DataFrame:
    """Long table, one row per unordered MP pair (mp_id_1 < mp_id_2), all pairs kept."""
    ids = list(M.index)
    shared, match = agreement_counts(M.to_numpy(), include_blank)
    i, j = np.triu_indices(len(ids), 1)
    sh, ma = shared[i, j], match[i, j]
    df = pd.DataFrame({
        "mp_id_1": np.array(ids)[i], "mp_id_2": np.array(ids)[j],
        "similarity": np.where(sh > 0, ma / np.where(sh > 0, sh, 1), np.nan),
        "n_shared_divisions": sh.astype(int), "n_matching": ma.astype(int)})
    for t in thresholds:
        df[f"meets_min_overlap_{t}"] = df.n_shared_divisions >= t
    df["party_1"] = df.mp_id_1.map(parties).to_numpy()
    df["party_2"] = df.mp_id_2.map(parties).to_numpy()
    df["same_party"] = df.party_1 == df.party_2
    return df


# ----------------------------------------------------------------------------
# graphs
# ----------------------------------------------------------------------------
def build_bipartite(ballots: pd.DataFrame, votes: pd.DataFrame, parties: pd.Series,
                    names: pd.Series) -> nx.Graph:
    """MP-division bipartite graph; an edge = the MP cast For/Against/Blank (not absent)."""
    cast = ballots[ballots.vote.isin(VOTE_CODE)]
    B = nx.Graph()
    for mp in sorted(cast.mp_id.unique()):
        B.add_node(mp, bipartite=0, party=parties.get(mp), name=names.get(mp))
    vinfo = votes.set_index("vote_id")
    for vid in sorted(cast.vote_id.unique()):
        r = vinfo.loc[vid]
        B.add_node(f"vote_{vid}", bipartite=1, case_id=int(r.case_id),
                   meeting_date=str(r.meeting_date.date()) if pd.notna(r.meeting_date) else None)
    B.add_edges_from((r.mp_id, f"vote_{r.vote_id}", {"vote": r.vote})
                     for r in cast.itertuples(index=False))
    return B


def similarity_graph(pairs: pd.DataFrame, min_overlap: int, parties: pd.Series,
                     names: pd.Series, nodes) -> nx.Graph:
    """Weighted undirected MP-MP graph; keeps pairs with >= min_overlap shared divisions.

    Edge attributes: weight (= agreement rate), n_shared, and length = 1 / weight
    (a distance for shortest-path measures; weights of 0 are floored at 0.01).
    """
    G = nx.Graph()
    for mp in nodes:
        G.add_node(mp, party=parties.get(mp), name=names.get(mp))
    keep = pairs[pairs.n_shared_divisions >= min_overlap]
    for r in keep.itertuples(index=False):
        G.add_edge(r.mp_id_1, r.mp_id_2, weight=float(r.similarity),
                   n_shared=int(r.n_shared_divisions),
                   length=1.0 / max(float(r.similarity), 0.01))
    return G


def disparity_filter(G: nx.Graph, alpha: float = 0.05) -> nx.Graph:
    """Serrano et al. disparity-filter backbone (course Week 4), for visualisation only.

    An edge survives if it is significant for at least one endpoint: with
    p = w / strength(node) and degree k, alpha_ij = (1 - p)^(k - 1) < alpha.
    """
    keep = set()
    for u in G:
        k = G.degree(u)
        if k < 2:
            continue
        s = G.degree(u, weight="weight")
        for v in G[u]:
            if (1 - G[u][v]["weight"] / s) ** (k - 1) < alpha:
                keep.add(frozenset((u, v)))
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from((u, v, G[u][v]) for u, v in map(tuple, keep))
    return H


def graph_summary(G: nx.Graph) -> dict:
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    w = [d["weight"] for *_, d in G.edges(data=True)]
    return {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "density": nx.density(G), "n_components": len(comps),
            "largest_component": comps[0] if comps else 0,
            "n_isolates": sum(1 for _, d in G.degree() if d == 0),
            "mean_degree": float(np.mean([d for _, d in G.degree()])),
            "mean_edge_weight": float(np.mean(w)) if w else float("nan")}


# ----------------------------------------------------------------------------
# communities
# ----------------------------------------------------------------------------
def louvain_runs(G: nx.Graph, seeds, resolution: float = 1.0, weight="weight"):
    """Louvain for several seeds. Returns (best partition, all modularities, mean pairwise NMI).

    "Best" = highest modularity. NMI between runs (non-isolated nodes only) shows how
    stable the partition is; isolated nodes are singleton communities and are left out.
    """
    parts = [nx.community.louvain_communities(G, weight=weight, resolution=resolution, seed=s)
             for s in seeds]
    mods = [nx.community.modularity(G, p, weight=weight) for p in parts]
    nodes = [n for n, d in G.degree() if d > 0]
    labs = [communities_to_labels(p, nodes) for p in parts]
    nmis = [normalized_mutual_info_score(a, b) for a, b in combinations(labs, 2)]
    return parts[int(np.argmax(mods))], mods, float(np.mean(nmis)) if nmis else float("nan")


def communities_to_labels(partition, nodes) -> list[int]:
    lab = {n: k for k, c in enumerate(partition) for n in c}
    return [lab[n] for n in nodes]


def party_nmi(G: nx.Graph, partition) -> float:
    """NMI between a node partition and party labels, on non-isolated nodes."""
    nodes = [n for n, d in G.degree() if d > 0]
    return float(normalized_mutual_info_score(
        [G.nodes[n]["party"] for n in nodes], communities_to_labels(partition, nodes)))


def party_partition(G: nx.Graph) -> list[set]:
    groups: dict = {}
    for n, d in G.nodes(data=True):
        groups.setdefault(d["party"], set()).add(n)
    return list(groups.values())


# ----------------------------------------------------------------------------
# within / between party and null models
# ----------------------------------------------------------------------------
def within_between(pairs: pd.DataFrame, min_overlap: int) -> dict:
    p = pairs[pairs.n_shared_divisions >= min_overlap]
    w, b = p[p.same_party].similarity, p[~p.same_party].similarity
    return {"n_pairs": len(p), "n_within": len(w), "n_between": len(b),
            "mean_within": float(w.mean()), "mean_between": float(b.mean()),
            "gap_within_minus_between": float(w.mean() - b.mean())}


def upper_arrays(shared: np.ndarray, match: np.ndarray, min_overlap: int):
    """Upper-triangle (i, j, similarity) of the pairs meeting the overlap threshold."""
    i, j = np.triu_indices(shared.shape[0], 1)
    ok = shared[i, j] >= min_overlap
    return i[ok], j[ok], match[i, j][ok] / shared[i, j][ok]


def gap(sim: np.ndarray, same: np.ndarray) -> float:
    return float(sim[same].mean() - sim[~same].mean())


def permute_party_gap(i, j, sim, labels: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """Null of the within-minus-between gap: shuffle party labels across MPs."""
    out = np.empty(n_perm)
    for k in range(n_perm):
        lab = rng.permutation(labels)
        out[k] = gap(sim, lab[i] == lab[j])
    return out


def shuffle_within_divisions(A: np.ndarray, rng) -> np.ndarray:
    """Randomise For/Against votes among the MPs who cast one in each division.

    Preserves, per division, who participated, how many For/Against there were and
    where the Blanks/absences are; destroys everything that is MP-specific.
    """
    out = A.copy()
    for c in range(out.shape[1]):
        idx = np.flatnonzero(np.isin(out[:, c], (1, -1)))
        out[idx, c] = rng.permutation(out[idx, c])
    return out


def division_null(A: np.ndarray, labels: np.ndarray, min_overlap: int, n_perm: int, rng,
                  graph_fn=None) -> pd.DataFrame:
    """Null distributions under within-division vote shuffling.

    Per permutation: mean/sd of pairwise similarity, within-minus-between party gap
    and, if graph_fn(shared, match) is given, whatever scalar dict it returns.
    """
    rows = []
    for _ in range(n_perm):
        sh, ma = agreement_counts(shuffle_within_divisions(A, rng))
        i, j, sim = upper_arrays(sh, ma, min_overlap)
        row = {"mean_sim": sim.mean(), "sd_sim": sim.std(ddof=1),
               "gap": gap(sim, labels[i] == labels[j])}
        if graph_fn is not None:
            row.update(graph_fn(sh, ma))
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_p(obs: float, null: np.ndarray, two_sided: bool = False) -> float:
    """Permutation p-value with the +1 correction."""
    null = np.asarray(null)
    if two_sided:
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


# ----------------------------------------------------------------------------
# dissent (exploratory; the dissent workstream may define this differently)
# ----------------------------------------------------------------------------
def dissent_table(ballots: pd.DataFrame, min_party_voters: int = 3) -> pd.DataFrame:
    """Per-MP dissent from the party's strict For/Against plurality in each division.

    A party-division unit counts when >= min_party_voters members cast For/Against and
    the split is not tied. Blank and absent ballots are neither dissent nor support.
    """
    cast = ballots[ballots.vote.isin([FOR, AGAINST])].copy()
    cast["f"] = (cast.vote == FOR).astype(int)
    cast["a"] = (cast.vote == AGAINST).astype(int)
    unit = cast.groupby(["vote_id", "party"])[["f", "a"]].sum()
    unit = unit[(unit.f + unit.a >= min_party_voters) & (unit.f != unit.a)]
    unit["majority"] = np.where(unit.f > unit.a, FOR, AGAINST)
    m = cast.merge(unit[["majority"]].reset_index(), on=["vote_id", "party"])
    m["dissent"] = m.vote != m.majority
    return (m.groupby("mp_id").agg(n_party_divisions=("dissent", "size"),
                                   n_dissent=("dissent", "sum")).assign(
        dissent_rate=lambda d: d.n_dissent / d.n_party_divisions))
