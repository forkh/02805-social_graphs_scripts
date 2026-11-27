"""Independent helpers of networks/audit/scripts/common.py used by formal_roles.py (code unchanged apart from paths).
Raw data = data/*.csv only. BLOC is the party-level two-way partition equal to the Louvain partition of the voting network; it is used
only to compute the Louvain reference values reported next to the formal-bloc results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as CFG

FOR, AGAINST, BLANK = "Fyri", "Ímóti", "Blankt"
W0, W1 = pd.Timestamp("2020-07-29"), pd.Timestamp("2023-05-04")
ELEC, GOV = pd.Timestamp("2022-12-08"), pd.Timestamp("2022-12-22")
BLOC = {"Sambandsflokkurin": 0, "Fólkaflokkurin": 0, "Miðflokkurin": 0, "Sjálvstýri": 0, "Javnaðarflokkurin": 1, "Tjóðveldi": 1, "Framsókn": 1}


def load_raw():
    b = pd.read_csv(CFG.DATA_DIR / "ballots_2020-2023.csv", keep_default_na=False)
    v = pd.read_csv(CFG.DATA_DIR / "votes_2020-2023.csv", parse_dates=["meeting_date"])
    m = pd.read_csv(CFG.DATA_DIR / "members_2020-2023.csv")
    return b, v, m


def contested_ids(b, min_minority=1):
    c = b[b.vote.isin([FOR, AGAINST])]
    k = c.groupby("vote_id").vote.agg(f=lambda s: (s == FOR).sum(), a=lambda s: (s == AGAINST).sum())
    return k[(k.f >= min_minority) & (k.a >= min_minority)].index


def modal_party(b):
    return b.groupby("mp_id").party.agg(lambda s: s.value_counts().idxmax())


def vote_matrix(b, ids, mps, include_blank=False):
    """MP x division array: +1 For, -1 Against, (0 Blank if include_blank), NaN otherwise."""
    code = {FOR: 1, AGAINST: -1}
    if include_blank:
        code[BLANK] = 0
    c = b[b.vote_id.isin(set(ids)) & b.vote.isin(code)]
    ix = {m: k for k, m in enumerate(mps)}; col = {x: k for k, x in enumerate(ids)}
    A = np.full((len(mps), len(ids)), np.nan)
    for r in c.itertuples(index=False):
        if r.mp_id in ix:
            A[ix[r.mp_id], col[r.vote_id]] = code[r.vote]
    return A


def pair_counts(A, include_blank=False):
    """(shared, match) matrices; a division is shared when both have a valid vote (For/Against, plus Blank if include_blank)."""
    cats = [1, -1] + ([0] if include_blank else [])
    ind = [(A == c).astype(float) for c in cats]
    valid = sum(ind)
    return valid @ valid.T, sum(i @ i.T for i in ind)


def pair_table(mps, A, party, include_blank=False):
    sh, ma = pair_counts(A, include_blank)
    i, j = np.triu_indices(len(mps), 1)
    df = pd.DataFrame({"mp_id_1": np.array(mps)[i], "mp_id_2": np.array(mps)[j], "n_shared": sh[i, j].astype(int), "n_match": ma[i, j].astype(int)})
    df["sim"] = np.where(df.n_shared > 0, df.n_match / df.n_shared.where(df.n_shared > 0, 1), np.nan)
    df["p1"] = df.mp_id_1.map(party); df["p2"] = df.mp_id_2.map(party)
    return df


def window(b, v, start=W0, end=W1):
    vw = v[v.meeting_date.between(start, end)]
    return vw, b[b.vote_id.isin(vw.vote_id)]
