"""Voting-network helpers: the functions of networks/src/networks.py and networks/src/temporal.py that the chain uses.
Code copied unchanged (paths aside); randomness only from the seeds passed by the caller.
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
WINDOW_START, WINDOW_END = pd.Timestamp("2020-07-29"), pd.Timestamp("2023-05-04")
ELECTION = pd.Timestamp("2022-12-08")
NEW_GOVERNMENT = pd.Timestamp("2022-12-22")


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the four source tables (read-only) from data/*_2020-2023.parquet."""
    d = Path(data_dir)
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
    """Pairwise (n_shared, n_match) from a +1/-1/0/NaN matrix (a division is shared when both cast a valid vote)."""
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


def similarity_graph(pairs: pd.DataFrame, min_overlap: int, parties: pd.Series,
                     names: pd.Series, nodes) -> nx.Graph:
    """Weighted undirected MP-MP graph; keeps pairs with >= min_overlap shared divisions."""
    G = nx.Graph()
    for mp in nodes:
        G.add_node(mp, party=parties.get(mp), name=names.get(mp))
    keep = pairs[pairs.n_shared_divisions >= min_overlap]
    for r in keep.itertuples(index=False):
        G.add_edge(r.mp_id_1, r.mp_id_2, weight=float(r.similarity),
                   n_shared=int(r.n_shared_divisions),
                   length=1.0 / max(float(r.similarity), 0.01))
    return G


def communities_to_labels(partition, nodes) -> list[int]:
    lab = {n: k for k, c in enumerate(partition) for n in c}
    return [lab[n] for n in nodes]


def louvain_runs(G: nx.Graph, seeds, resolution: float = 1.0, weight="weight"):
    """Louvain for several seeds. Returns (best partition, all modularities, mean pairwise NMI); best = highest modularity."""
    parts = [nx.community.louvain_communities(G, weight=weight, resolution=resolution, seed=s)
             for s in seeds]
    mods = [nx.community.modularity(G, p, weight=weight) for p in parts]
    nodes = [n for n, d in G.degree() if d > 0]
    labs = [communities_to_labels(p, nodes) for p in parts]
    nmis = [normalized_mutual_info_score(a, b) for a, b in combinations(labs, 2)]
    return parts[int(np.argmax(mods))], mods, float(np.mean(nmis)) if nmis else float("nan")


def dissent_table(ballots: pd.DataFrame, min_party_voters: int = 3) -> pd.DataFrame:
    """Per-MP dissent from the party's strict For/Against plurality in each division (>= min_party_voters, no tie)."""
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


# ---------------------------------------------------------------- temporal (networks/src/temporal.py)
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
    """Party label for each MP within one period: the party they voted under most often there."""
    return modal_party(ballots[ballots.vote_id.isin(set(vote_ids))])


# ---------------------------------------------------------------- null models and validation (networks/src/networks.py, temporal.py)
PARTY_ORDER = ["Javnaðarflokkurin", "Tjóðveldi", "Sambandsflokkurin", "Fólkaflokkurin",
               "Framsókn", "Miðflokkurin", "Sjálvstýri"]


def upper_arrays(shared: np.ndarray, match: np.ndarray, min_overlap: int):
    """Upper-triangle (i, j, similarity) of the pairs meeting the overlap threshold."""
    i, j = np.triu_indices(shared.shape[0], 1)
    ok = shared[i, j] >= min_overlap
    return i[ok], j[ok], match[i, j][ok] / shared[i, j][ok]


def gap(sim: np.ndarray, same: np.ndarray) -> float:
    return float(sim[same].mean() - sim[~same].mean())


def shuffle_within_divisions(A: np.ndarray, rng) -> np.ndarray:
    """Randomise For/Against votes among the MPs who cast one in each division (division totals and attendance kept)."""
    out = A.copy()
    for c in range(out.shape[1]):
        idx = np.flatnonzero(np.isin(out[:, c], (1, -1)))
        out[idx, c] = rng.permutation(out[idx, c])
    return out


def division_null(A: np.ndarray, labels: np.ndarray, min_overlap: int, n_perm: int, rng,
                  graph_fn=None) -> pd.DataFrame:
    """Null distributions under within-division vote shuffling (mean/sd similarity, party gap, graph_fn statistics)."""
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


def _spearman(a, b) -> float:
    from scipy import stats
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok])[0])


def change_statistics(A: np.ndarray, is_post: np.ndarray, labels: np.ndarray, thr: int, pair_mask=None) -> dict:
    """Pre->post change of pair similarity for the same MPs, over pairs with >= thr shared divisions in both subsets."""
    sh0, ma0 = agreement_counts(A[:, ~is_post])
    sh1, ma1 = agreement_counts(A[:, is_post])
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
    """Boolean vector over np.triu_indices pairs: >= thr shared divisions in both subsets."""
    sh0, _ = agreement_counts(A[:, ~is_post])
    sh1, _ = agreement_counts(A[:, is_post])
    i, j = np.triu_indices(A.shape[0], 1)
    return (sh0[i, j] >= thr) & (sh1[i, j] >= thr)


def block_permutation(blocks: np.ndarray, n_post: int, rng) -> np.ndarray:
    """Random 'post' indicator over divisions that keeps whole blocks (cases) together."""
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
    """Null for 'the same MPs behave differently in the two sets of divisions' (pre/post label re-assigned at random)."""
    n_post = int(is_post.sum())
    rows = []
    for _ in range(n_perm):
        ip = rng.permutation(is_post) if blocks is None else block_permutation(blocks, n_post, rng)
        rows.append(change_statistics(A, ip, labels, thr, pair_mask))
    return pd.DataFrame(rows)
