"""Speaking-order helpers: the functions of networks/interaction_analysis/src/interaction_networks.py that the chain uses (code unchanged
apart from paths). Successions are consecutive rows of the same sitting; self-successions are dropped.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

SEED = 20260121
MIN_CHAIR_ROWS = 50   # an MP with at least this many chairman rows counts as a "chair MP"


def load_rows(data_dir: Path) -> pd.DataFrame:
    """Transcribed FPSC rows (read-only), sorted in meeting order by the sequence number in `id`."""
    cols = ["id", "meeting_id", "mp_id", "contribution_type", "date", "name", "political_party_affiliation"]
    d = pq.read_table(Path(data_dir) / "fpsc/fpsc_train_metadata.parquet", columns=cols).to_pandas()
    d["seq"] = d["id"].str.extract(r"_S(\d+)$")[0].astype(int)
    d["date"] = pd.to_datetime(d["date"])
    mdate = d.groupby("meeting_id")["date"].min().rename("meeting_date")   # a meeting can pass midnight
    d = d.join(mdate, on="meeting_id").sort_values(["meeting_id", "seq"]).reset_index(drop=True)
    return d


def load_parties(pairs_csv: Path) -> pd.Series:
    """Primary party label per MP: modal party of the MP's ballots, from the voting pair table."""
    p = pd.read_csv(pairs_csv, usecols=["mp_id_1", "mp_id_2", "party_1", "party_2"])
    s = pd.concat([p.set_index("mp_id_1").party_1, p.set_index("mp_id_2").party_2])
    return s[~s.index.duplicated()].sort_index().rename("party")


def mp_names(rows: pd.DataFrame) -> pd.Series:
    return rows.groupby("mp_id")["name"].agg(lambda s: s.value_counts().idxmax())


def directed_transitions(rows: pd.DataFrame, exclude_chairs: bool):
    """Directed transition counts between consecutive rows of the same meeting (chairman rows dropped first when exclude_chairs).
    Returns (Counter{(source, target): n}, stats). Self-loops are dropped and counted."""
    D = Counter()
    st = Counter()
    for _, g in rows.groupby("meeting_id", sort=True):
        ch = (g.contribution_type == "chairman").to_numpy()
        keep = ~ch if exclude_chairs else np.ones(len(g), bool)
        mp = g.mp_id.to_numpy()[keep]
        for i in range(len(mp) - 1):
            st["adjacent_row_pairs"] += 1
            if mp[i] == mp[i + 1]:
                st["self_loops_dropped"] += 1
                continue
            D[(mp[i], mp[i + 1])] += 1
            st["transitions_kept"] += 1
    return D, dict(st)


def directed_frame(D: Counter, definition: str) -> pd.DataFrame:
    df = pd.DataFrame([(s, t, w) for (s, t), w in D.items()], columns=["source", "target", "weight"])
    df = df.sort_values(["weight", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    df["network_definition"] = definition
    return df


def undirected_frame(D: Counter, definition: str) -> pd.DataFrame:
    """Reciprocal directed weights are SUMMED: w(a,b) = n(a->b) + n(b->a)."""
    U = Counter()
    for (s, t), w in D.items():
        U[(s, t) if s < t else (t, s)] += w
    df = pd.DataFrame([(a, b, w) for (a, b), w in U.items()],
                      columns=["mp_id_1", "mp_id_2", "interaction_weight"])
    df = df.sort_values(["interaction_weight", "mp_id_1", "mp_id_2"], ascending=[False, True, True]).reset_index(drop=True)
    df["network_definition"] = definition
    return df


def vote_pair_frame(pairs_csv: Path, U: pd.DataFrame, chair_mps: set) -> pd.DataFrame:
    v = pd.read_csv(pairs_csv)
    w = U.set_index(["mp_id_1", "mp_id_2"]).interaction_weight
    v["interaction_weight"] = [float(w.get((a, b), 0.0)) for a, b in zip(v.mp_id_1, v.mp_id_2)]
    v["involves_chair_mp"] = v.mp_id_1.isin(chair_mps) | v.mp_id_2.isin(chair_mps)
    return v


def _rank_corr(x, y):
    return stats.spearmanr(x, y).statistic


def weight_matrix(U: pd.DataFrame, universe: list[str]):
    """Full symmetric interaction matrix over all MPs of the universe (zero = no transition)."""
    idx = {n: i for i, n in enumerate(universe)}
    W = np.zeros((len(universe), len(universe)))
    for a, b, w in zip(U.mp_id_1, U.mp_id_2, U.interaction_weight):
        W[idx[a], idx[b]] = W[idx[b], idx[a]] = w
    return W, idx


def mantel_spearman(v: pd.DataFrame, W: np.ndarray, idx: dict, perm_nodes: list[str], groups: dict | None,
                    n_perm: int, seed: int, positive_only: bool):
    """Spearman(interaction, voting similarity) over the pairs in `v`, with a node-label permutation null.

    The labels of `perm_nodes` are permuted (all of them, or only within `groups`) and the permuted pair weights are looked
    up in the FULL interaction matrix W. MPs outside `perm_nodes` stay fixed.
    """
    ia = np.array([idx[a] for a in v.mp_id_1]); ib = np.array([idx[b] for b in v.mp_id_2])
    sim = v.similarity.to_numpy()
    obs_w = W[ia, ib]
    sel = obs_w > 0 if positive_only else np.ones(len(v), bool)
    if sel.sum() < 10:
        return {"n_pairs": int(sel.sum()), "rho": np.nan}
    rho = _rank_corr(obs_w[sel], sim[sel])
    rng = np.random.default_rng(seed)
    pn = np.array([idx[n] for n in perm_nodes])
    if groups is None:
        blocks = [pn]
    else:
        g = {}
        for n in perm_nodes:
            g.setdefault(groups.get(n, "(none)"), []).append(idx[n])
        blocks = [np.array(b) for b in g.values()]
    base = np.arange(W.shape[0])
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = base.copy()
        for b in blocks:
            p[b] = rng.permutation(b)
        w = W[p[ia], p[ib]]
        s2 = (w > 0) if positive_only else sel
        null[k] = _rank_corr(w[s2], sim[s2]) if s2.sum() > 2 else np.nan
    null = null[~np.isnan(null)]
    return {"n_pairs": int(sel.sum()), "rho": float(rho), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_2.5%": float(np.quantile(null, .025)), "null_97.5%": float(np.quantile(null, .975)),
            "p_two_sided": float((1 + (np.abs(null - null.mean()) >= abs(rho - null.mean())).sum()) / (len(null) + 1)),
            "n_permutations": int(len(null)), "seed": seed}


def party_mixing(D: Counter, parties: pd.Series, n_perm: int, seed: int) -> dict:
    """Weighted party mixing of directed transitions (source party x target party); summary only.

    Assortativity: Newman's discrete coefficient on the normalised weighted mixing matrix. Null: party labels permuted among
    the MPs that have a party.
    """
    nodes = sorted({n for e in D for n in e})
    has = [n for n in nodes if n in parties.index]
    plist = sorted(parties.loc[has].unique())
    pid = {p: i for i, p in enumerate(plist)}
    idx = {n: i for i, n in enumerate(has)}
    S, T, W = [], [], []
    miss_w = 0
    for (s, t), w in D.items():
        if s in idx and t in idx:
            S.append(idx[s]); T.append(idx[t]); W.append(w)
        else:
            miss_w += w
    S, T, W = np.array(S), np.array(T), np.array(W, float)
    lab = np.array([pid[parties[n]] for n in has])
    k = len(plist)

    def stat(lbl):
        e = np.zeros((k, k))
        np.add.at(e, (lbl[S], lbl[T]), W)
        e /= e.sum()
        a, b = e.sum(1), e.sum(0)
        within = np.trace(e)
        exp = float(a @ b)
        return within, exp, (within - exp) / (1 - exp), e
    within, exp, r, e = stat(lab)
    rng = np.random.default_rng(seed)
    null_r = np.empty(n_perm)
    null_w = np.empty(n_perm)
    for i in range(n_perm):
        w_, _, r_, _ = stat(rng.permutation(lab))
        null_r[i], null_w[i] = r_, w_
    return {"parties": plist, "mps_with_party": len(has), "mps_in_network": len(nodes), "mps_without_party": len(nodes) - len(has),
            "directed_weight_with_both_ends_assigned_share": float(W.sum() / (W.sum() + miss_w)),
            "within_party_weight_share": float(within), "expected_within_share_independent_mixing": exp,
            "assortativity_weighted": float(r),
            "perm_null_assortativity_mean": float(null_r.mean()), "perm_null_assortativity_sd": float(null_r.std(ddof=1)),
            "perm_null_assortativity_p95": float(np.quantile(null_r, .95)),
            "perm_p_one_sided_ge": float((1 + (null_r >= r).sum()) / (n_perm + 1)),
            "n_permutations": n_perm, "seed": seed,
            "within_party_share_null_mean": float(null_w.mean()), "within_party_share_null_sd": float(null_w.std(ddof=1))}
