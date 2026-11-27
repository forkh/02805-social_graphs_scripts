"""Formal government/opposition roles: analyses (a)-(e) and (g) (port of networks/formal_roles/src/compute_formal_roles.py; the letters
are those of the original analysis plan).

    python reproduce_networks/formal_roles.py            -> OUT_DIR/formal_roles/

Inputs: data/ballots_2020-2023.csv, data/votes_2020-2023.csv and the outputs of voting_network.py, temporal_network.py and
speaking_order.py in OUT_DIR. Louvain communities / the party-level blocs are used only for the Louvain reference values reported next to
the formal-bloc results (bloc_three_way.csv, within_bloc_vote_comparison.csv) and as the outcome compared with formal blocs in (d).
Every grouping of the analyses uses formal roles (roles.py). Writes only into OUT_DIR/formal_roles (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from itertools import combinations
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as CFG  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

import roles as R  # noqa: E402
import auditlib as A  # noqa: E402   (audit helpers; independent of netlib)
import within_bloc as WB  # noqa: E402
import speaking_lib as I  # noqa: E402

SEED = 2805
OFF = {"a_boot": 107, "b_null": 101, "b_exact": 0, "d_null": 102, "e_boot": 103, "e_null": 104, "g_null": 105, "g_boot": 106}
N_BOOT, N_PERM, N_PERM_G = 10_000, 10_000, 1_000
INPUTS = {  # key -> (folder, file name)
    "ballots": (CFG.DATA_DIR, "ballots_2020-2023.csv"), "votes": (CFG.DATA_DIR, "votes_2020-2023.csv"),
    "int_vs_vote": (CFG.SPEAKING, "pairwise_interaction_vs_vote_similarity_min40.csv"),
    "mp_map": (CFG.SPEAKING, "mp_id_name_party_map.csv"),
    "communities": (CFG.VOTING, "network_communities.csv"),
    "temporal_nodes": (CFG.TEMPORAL, "temporal_node_metrics.csv"),
    "dissent": (CFG.VOTING, "dissent_network_summary.csv"),
    "network_results": (CFG.VOTING, "network_results.json"),
}
CHECKS: list[dict] = []


def check(ok, what, **kw):
    CHECKS.append({"check": what, "pass": bool(ok), **{k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in kw.items()}})
    if not ok:
        raise AssertionError(f"CHECK FAILED: {what} {kw}")


def _js(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def inp(key):
    d, f = INPUTS[key]
    return d / f


def rel(key):
    d, f = INPUTS[key]
    return f"{'data' if d == CFG.DATA_DIR else 'OUT_DIR/' + d.name}/{f}"


def sha(key):
    return hashlib.sha256(inp(key).read_bytes()).hexdigest()


def boot_ratio(num, den, rng, n=N_BOOT):
    """Percentile 95% interval of sum(num)/sum(den) with divisions resampled."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    k = len(num)
    idx = rng.integers(0, k, size=(n, k))
    r = num[idx].sum(1) / np.where(den[idx].sum(1) > 0, den[idx].sum(1), np.nan)
    return [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]


# ----------------------------------------------------------------------------------------------- data and coverage
def load():
    b, v, _ = A.load_raw()
    x = R.annotate(b, v)
    return b, v, x


def coverage(b, v, x):
    d = v.meeting_date
    per = pd.Series([R.period(t) for t in d], index=v.index)
    inwin = d.between(A.W0, A.W1)
    out = {"divisions_total": int(len(v)), "divisions_undated": int(d.isna().sum()),
           "undated_vote_ids": sorted(int(i) for i in v.vote_id[d.isna()]),
           "divisions_per_period_all": {k: int((per == k).sum()) for k in ("P1", "P2", "P3")},
           "divisions_per_period_window": {k: int(((per == k) & inwin).sum()) for k in ("P1", "P2", "P3")},
           "first_last_dated": [str(d.min().date()), str(d.max().date())],
           "parties_in_ballots": sorted(b.party.unique().tolist()),
           "rules": R.RULES}
    call = set(A.contested_ids(b))
    out["contested_per_period_all"] = {k: int(sum(i in call for i in v.vote_id[per == k])) for k in ("P1", "P2", "P3")}
    out["contested_per_period_window"] = {k: int(sum(i in call for i in v.vote_id[(per == k) & inwin])) for k in ("P1", "P2", "P3")}
    check(out["divisions_total"] == 1125 and out["divisions_undated"] == 7, "1,125 divisions, 7 undated (inputs)")
    check(out["divisions_per_period_all"] == {"P1": 672, "P2": 3, "P3": 443}, "period counts as in the plan")
    check(set(out["parties_in_ballots"]) == set(R.PARTIES), "every ballot party is in the role table")
    check(not ((x.party == "Sjálvstýri") & (x.period == "P3")).any(), "Sjálvstýri has no P3 ballots")
    check(len(x) == int(b.vote_id.isin(v.vote_id[d.notna()]).sum()), "every dated ballot has a side")
    return out, call


# ----------------------------------------------------------------------------------------------- MP-level side group
def mp_side_groups(x):
    """Majority side group (X/Y) over each MP's dated ballots in the speech window (any recorded vote)."""
    w = x[x.meeting_date.between(A.W0, A.W1) & (x.vote != "")]
    c = w.groupby(["mp_id", "side_group"]).size().unstack(fill_value=0)
    for g in ("X", "Y"):
        if g not in c:
            c[g] = 0
    grp = pd.Series(np.where(c.X >= c.Y, "X", "Y"), index=c.index)
    check(not (c.X == c.Y).any(), "no MP has a tie between side groups")
    mixed = c[(c.X > 0) & (c.Y > 0)]
    return grp.to_dict(), [{"mp_id": m, "ballots_X": int(r.X), "ballots_Y": int(r.Y), "assigned": grp[m]} for m, r in mixed.iterrows()]


# ----------------------------------------------------------------------------------------------- reproduction checks
def repro_bloc_three_way(b, v):
    """s06_bloc_everywhere.py with the party-level blocs (Louvain reference values); written to OUT_DIR/formal_roles/bloc_three_way.csv."""
    dated = v.dropna(subset=["meeting_date"]); has_b = set(b.vote_id); call = set(A.contested_ids(b)); rows = []

    def three(name, pt, thr):
        e = pt[pt.n_shared >= thr]; sp = e.p1 == e.p2; sb = e.p1.map(A.BLOC) == e.p2.map(A.BLOC)
        r = {"network": name, "min_overlap": thr, "pairs": len(e)}
        for lab, m in (("same_party", sp), ("diff_party_same_bloc", ~sp & sb), ("diff_bloc", ~sb)):
            r[f"n_{lab}"] = int(m.sum()); r[f"mean_{lab}"] = float(e[m].sim.mean()) if m.any() else np.nan
        r["gap_party"] = r["mean_same_party"] - float(e[~sp].sim.mean())
        r["gap_bloc"] = float(e[sb].sim.mean() - e[~sb].sim.mean())
        r["party_beyond_bloc"] = r["mean_same_party"] - r["mean_diff_party_same_bloc"]
        rows.append(r)

    vw, bw = A.window(b, v)
    cids = list(A.contested_ids(bw)); mps = sorted(bw.mp_id.unique())
    P = A.pair_table(mps, A.vote_matrix(bw, cids, mps), A.modal_party(bw))
    for t in (20, 40, 60):
        three("baseline window (all MPs)", P, t)
    SEL = {"pre": dated[dated.meeting_date < A.ELEC], "post_full": dated[dated.meeting_date > A.GOV],
           "post_window": dated[(dated.meeting_date > A.GOV) & (dated.meeting_date <= A.W1)]}
    PER = {}
    for k, xs in SEL.items():
        xs = xs[xs.vote_id.isin(has_b)]; ids = [i for i in xs.vote_id if i in call]; bb = b[b.vote_id.isin(xs.vote_id)]; par = A.modal_party(bb)
        allm = sorted(bb.mp_id.unique()); pres = A.vote_matrix(b, ids, allm, include_blank=True); mm = [m for m, r in zip(allm, pres) if (~np.isnan(r)).any()]
        PER[k] = (mm, A.vote_matrix(b, ids, mm), par, A.pair_table(mm, A.vote_matrix(b, ids, mm), par))
        three(k, PER[k][3], 20)
    pre_m = {m for m, r in zip(PER["pre"][0], PER["pre"][1]) if (~np.isnan(r)).any()}
    post_m = {m for m, r in zip(PER["post_full"][0], PER["post_full"][1]) if (~np.isnan(r)).any()}
    cont = sorted(pre_m & post_m)
    sub = lambda k: PER[k][1][[PER[k][0].index(m) for m in cont]]  # noqa: E731
    Pp, Pq = A.pair_table(cont, sub("pre"), PER["pre"][2]), A.pair_table(cont, sub("post_full"), PER["post_full"][2])
    el = (Pp.n_shared >= 20) & (Pq.n_shared >= 20)
    three("continuing 113 pairs: before", Pp[el], 20); three("continuing 113 pairs: after", Pq[el], 20)
    new = pd.DataFrame(rows)
    check(len(new) == 8 and int(new.pairs.iloc[1]) == 712, "bloc_three_way: 8 rows; 712 pairs at >= 40 shared divisions")
    return new


def repro_within_bloc():
    """within_bloc_test.run() with the Louvain blocs (Louvain reference values); written to OUT_DIR/formal_roles/within_bloc_vote_comparison.csv."""
    new, _ = WB.run()
    check(len(new) == 10, "within_bloc_vote_comparison: 10 rows")
    return new


# ----------------------------------------------------------------------------------------------- (a) pair types per division
def pair_type_counts(x, ids):
    """Per division: pairs and matching pairs for same party / other party same side / opposite side (formal side on that date)."""
    c = x[x.vote_id.isin(set(ids)) & x.vote.isin([A.FOR, A.AGAINST])]
    rows = []
    for vid, g in c.groupby("vote_id", sort=True):
        f = (g.vote == A.FOR).to_numpy(); n = len(g)
        tot, mat = comb(n, 2), comb(int(f.sum()), 2) + comb(int((~f).sum()), 2)
        sp = sm = ss = ssm = 0
        for _, h in g.groupby("party"):
            k = len(h); kf = int((h.vote == A.FOR).sum()); sp += comb(k, 2); sm += comb(kf, 2) + comb(k - kf, 2)
        for _, h in g.groupby("side"):
            k = len(h); kf = int((h.vote == A.FOR).sum()); ss += comb(k, 2); ssm += comb(kf, 2) + comb(k - kf, 2)
        rows.append({"vote_id": vid, "period": g.period.iloc[0], "date": str(g.meeting_date.iloc[0].date()), "n_valid": n,
                     "pairs_same_party": sp, "match_same_party": sm,
                     "pairs_other_party_same_side": ss - sp, "match_other_party_same_side": ssm - sm,
                     "pairs_opposite_side": tot - ss, "match_opposite_side": mat - ssm, "pairs_all": tot, "match_all": mat})
    return pd.DataFrame(rows)


TYPES = ["same_party", "other_party_same_side", "opposite_side"]


def summarise_types(D, rng):
    out = {}
    for t in TYPES + ["all"]:
        p, m = D[f"pairs_{t}"].to_numpy(), D[f"match_{t}"].to_numpy()
        out[t] = {"agreement": float(m.sum() / p.sum()), "pair_votes": int(p.sum()), "divisions": int((p > 0).sum()),
                  "ci95_divisions": boot_ratio(m, p, rng)}
    out["party_beyond_side"] = out["same_party"]["agreement"] - out["other_party_same_side"]["agreement"]
    out["side_gap"] = out["other_party_same_side"]["agreement"] - out["opposite_side"]["agreement"]
    return out


# ----------------------------------------------------------------------------------------------- pair tables with formal relation
def formal_pair_table(b, x, ids, mps, party):
    """s06-style pair table (+ formal relation): share of shared contested divisions on which the two MPs were on the same side."""
    Am = A.vote_matrix(b, ids, mps)
    P = A.pair_table(mps, Am, party)
    g = x[x.vote_id.isin(set(ids)) & x.vote.isin([A.FOR, A.AGAINST])]
    ix = {m: k for k, m in enumerate(mps)}; col = {i: k for k, i in enumerate(ids)}
    GX = np.zeros(Am.shape); GY = np.zeros(Am.shape)
    for r in g.itertuples(index=False):
        if r.mp_id in ix:
            (GX if r.side_group == "X" else GY)[ix[r.mp_id], col[r.vote_id]] = 1
    check(np.array_equal((GX + GY) > 0, ~np.isnan(Am)), "side-group matrix covers exactly the valid votes")
    same = GX @ GX.T + GY @ GY.T
    i, j = np.triu_indices(len(mps), 1)
    P["n_same_side"] = same[i, j].astype(int)
    P["share_same_side"] = np.where(P.n_shared > 0, P.n_same_side / P.n_shared.where(P.n_shared > 0, 1), np.nan)
    P["same_side"] = P.share_same_side > 0.5
    P["side_changed"] = (P.n_same_side > 0) & (P.n_same_side < P.n_shared)
    return P


def three_formal(name, e):
    sp = e.p1 == e.p2; ss = e.same_side
    r = {"network": name, "pairs": len(e), "pairs_side_changed": int(e.side_changed.sum())}
    for lab, m in (("same_party", sp), ("other_party_same_side", ~sp & ss), ("opposite_side", ~ss)):
        r[f"n_{lab}"] = int(m.sum()); r[f"mean_{lab}"] = float(e[m].sim.mean()) if m.any() else np.nan
    r["party_beyond_side"] = r["mean_same_party"] - r["mean_other_party_same_side"]
    r["side_gap"] = r["mean_other_party_same_side"] - r["mean_opposite_side"]
    return r


def period_pairs(b, x, v, sel, call):
    xs = v[sel & v.vote_id.isin(set(b.vote_id))]
    ids = [i for i in xs.vote_id if i in call]; bb = b[b.vote_id.isin(xs.vote_id)]; par = A.modal_party(bb)
    allm = sorted(bb.mp_id.unique()); pres = A.vote_matrix(b, ids, allm, include_blank=True)
    mm = [m for m, r in zip(allm, pres) if (~np.isnan(r)).any()]
    return mm, par, formal_pair_table(b, x, ids, mm, par), ids


# ----------------------------------------------------------------------------------------------- (b) party beyond side
def party_beyond_side_tests(P, thr, period_key, rng):
    e = P[P.n_shared >= thr].reset_index(drop=True)
    nodes = sorted(set(e.mp_id_1) | set(e.mp_id_2)); ix = {m: k for k, m in enumerate(nodes)}
    i1 = e.mp_id_1.map(ix).to_numpy(); i2 = e.mp_id_2.map(ix).to_numpy(); sim = e.sim.to_numpy()
    par = {**dict(zip(e.mp_id_1, e.p1)), **dict(zip(e.mp_id_2, e.p2))}
    plab = np.array([par[m] for m in nodes], dtype=object)
    side_of_party = {p: s for s in (R.GOV, R.OPP) for p in R.TABLE[period_key][s]}
    slab = np.array([side_of_party[p] for p in plab], dtype=object)
    ss = slab[i1] == slab[i2]
    check(bool((ss == e.same_side.to_numpy()).all()), f"{period_key}: per-pair formal relation constant within the period")

    def stat(pl):
        sp = pl[i1] == pl[i2]
        return sim[sp].mean() - sim[~sp & ss].mean()
    obs = float(stat(plab))
    null = np.empty(N_PERM); blocks = [np.where(slab == s)[0] for s in (R.GOV, R.OPP)]
    for k in range(N_PERM):
        pl = plab.copy()
        for bl in blocks:
            pl[bl] = plab[rng.permutation(bl)]
        null[k] = stat(pl)
    # exact party-level enumeration: side gap among different-party pairs for every split of the parties into the formal group sizes
    dp = plab[i1] != plab[i2]
    parties = sorted(set(plab)); gov = sorted(p for p in parties if side_of_party[p] == R.GOV); k_ = len(gov)
    splits = []
    for c in combinations(parties, k_):
        if 2 * k_ == len(parties) and parties[0] not in c:
            continue                                                  # unordered splits only
        grp = np.array([p in c for p in plab]); same = grp[i1] == grp[i2]
        splits.append((c, float(sim[dp & same].mean() - sim[dp & ~same].mean())))
    obs_gap = float(sim[dp & ss].mean() - sim[dp & ~ss].mean())
    n_ge = sum(g >= obs_gap - 1e-12 for _, g in splits)
    return {"threshold": thr, "pairs": int(len(e)), "mps": len(nodes), "party_beyond_side": obs,
            "null_within_side": {"n_permutations": N_PERM, "permuted": "party labels of MPs within formal side (party sizes per side kept)",
                                 "mean": float(null.mean()), "sd": float(null.std(ddof=1)), "p_upper": float((1 + (null >= obs - 1e-12).sum()) / (N_PERM + 1))},
            "exact_party_split": {"statistic": "mean similarity of different-party pairs on the same side minus on opposite sides",
                                  "observed_side_gap": obs_gap, "n_parties": len(parties), "group_sizes": [k_, len(parties) - k_],
                                  "n_distinct_splits": len(splits), "n_splits_ge_observed": int(n_ge), "p_exact": n_ge / len(splits),
                                  "min_possible_p": 1 / len(splits), "max_gap_over_splits": max(g for _, g in splits)}}


# ----------------------------------------------------------------------------------------------- (c) successions within side
def successions_within_side(groups_side):
    bloc, universe, party, S, U = WB.load()
    groups = {n: groups_side.get(n, "(none)") for n in universe}
    rows = []
    for d in ("chair_inclusive", "chair_excluded"):
        W, idx = I.weight_matrix(U[d], universe)
        v = S[S.network_definition == d].reset_index(drop=True)
        v = v.assign(same_side=[groups[a] == groups[b] for a, b in zip(v.mp_id_1, v.mp_id_2)])
        specs = [("pooled", v, False, "tested (same design as the within-bloc test; formal sides)", WB.N_PERM),
                 ("pooled", v, True, "exploratory", WB.N_PERM_EXPL),
                 ("same_side", v[v.same_side], False, "exploratory", WB.N_PERM_EXPL),
                 ("opposite_side", v[~v.same_side], False, "exploratory", WB.N_PERM_EXPL)]
        for stratum, vv, pos, cls, npm in specs:
            m = I.mantel_spearman(vv, W, idx, list(universe), groups, npm, WB.SEED, pos)
            rows.append({"network_definition": d, "null": "within_formal_side", "stratum": stratum, "pairs_with_interaction_only": pos,
                         "evidence_class": cls, "n_pairs_in_sample": len(vv), "n_pairs_interacting": int((vv.interaction_weight > 0).sum()), **m})
        # sensitivity: within formal side, but MPs outside the 712-pair sample are permuted only among themselves (within side);
        # separates the effect of the side labels from the effect of mixing sample and non-sample MPs (no Louvain information used)
        ins = set(v.mp_id_1) | set(v.mp_id_2)
        g2 = {n: f"{groups[n]}|{'in' if n in ins else 'out'}" for n in universe}
        m = I.mantel_spearman(v, W, idx, list(universe), g2, WB.N_PERM, WB.SEED, False)
        rows.append({"network_definition": d, "null": "within_formal_side_x_in_sample", "stratum": "pooled", "pairs_with_interaction_only": False,
                     "evidence_class": "sensitivity (exploratory)", "n_pairs_in_sample": len(v), "n_pairs_interacting": int((v.interaction_weight > 0).sum()), **m})
    gs = pd.Series(groups)
    return pd.DataFrame(rows), {"mps_in_universe": len(universe), "side_group_sizes": {k: int(v) for k, v in gs.value_counts().sort_index().items()}}


# ----------------------------------------------------------------------------------------------- (d) communities vs sides
def communities_vs_sides(groups_side, rng):
    T = pd.read_csv(inp("temporal_nodes"))
    out = {}
    cases = [("pre", "P1"), ("post_full", "P3"), ("post_window", "P3")]
    for per, key in cases:
        d = T[(T.period == per) & T.community.notna() & (T.community != "isolated")].copy()     # as nb04: MPs with an edge only
        sop = {p: s for s in (R.GOV, R.OPP) for p in R.TABLE[key][s]}
        d["side"] = d.party.map(sop)
        check(d.side.notna().all(), f"{per}: every MP's party in the {key} role table")
        out[per] = _agreement(d.community.to_numpy(), d.side.to_numpy(), d.party.to_numpy(), sop, rng)
        out[per]["roles_used"] = key
        # the temporal notebook's coalition NMI (coalition parties only, Sjálvstýri excluded), computed as in the notebook
        cp = {"Sambandsflokkurin", "Fólkaflokkurin", "Miðflokkurin", "Javnaðarflokkurin", "Tjóðveldi", "Framsókn"}
        dd = d[d.party.isin(cp)]
        nb04 = float(normalized_mutual_info_score(dd.side == R.GOV, dd.community))
        out[per]["saved_nb04_coalition_NMI_excl_Sjalvstyri"] = nb04
    C = pd.read_csv(inp("communities")); C = C[C.community >= 0]
    sg = C.mp_id.map(groups_side)
    check(sg.notna().all(), "pooled communities: every MP has a side group")
    pooled_sop = {p: ("X" if p in R.TABLE["P1"][R.GOV] else "Y") for p in R.PARTIES}
    out["pooled_window"] = _agreement(C.community.to_numpy(), sg.to_numpy(), C.party.to_numpy(), pooled_sop, rng)
    out["pooled_window"]["roles_used"] = "MP side group (majority over window ballots; X = Sambandsflokkurin's side)"
    return out


def _agreement(comm, side, party, side_of_party, rng):
    nmi = float(normalized_mutual_info_score(side, comm)); ari = float(adjusted_rand_score(side, comm))
    nn, na = np.empty(N_PERM), np.empty(N_PERM)
    for k in range(N_PERM):
        s = rng.permutation(side); nn[k] = normalized_mutual_info_score(s, comm); na[k] = adjusted_rand_score(s, comm)
    parties = sorted(set(party)); first = sorted(p for p in parties if side_of_party[p] == side_of_party[parties[0]])
    k_ = len(first); res = []
    for c in combinations(parties, k_):
        if 2 * k_ == len(parties) and parties[0] not in c:
            continue
        lab = np.array([p in c for p in party]); res.append(normalized_mutual_info_score(lab, comm))
    obs_split_count = sum(r >= nmi - 1e-12 for r in res)
    tab = pd.crosstab(pd.Series(side, name="side"), pd.Series(comm, name="community"))
    return {"n_mps": int(len(comm)), "NMI": nmi, "ARI": ari,
            "null_mp_label_permutation": {"n_permutations": N_PERM, "NMI_mean": float(nn.mean()), "ARI_mean": float(na.mean()),
                                          "p_upper_NMI": float((1 + (nn >= nmi - 1e-12).sum()) / (N_PERM + 1)),
                                          "p_upper_ARI": float((1 + (na >= ari - 1e-12).sum()) / (N_PERM + 1))},
            "exact_party_split": {"n_distinct_splits": len(res), "n_splits_NMI_ge_observed": int(obs_split_count),
                                  "p_exact": obs_split_count / len(res), "min_possible_p": 1 / len(res)},
            "crosstab": {str(i): {str(c): int(tab.loc[i, c]) for c in tab.columns} for i in tab.index},
            "mps_off_majority": int(len(comm) - sum(tab.max(axis=0)))}


# ----------------------------------------------------------------------------------------------- (e) Sjálvstýri
def focal_agreement(x, ids, focal_party):
    """Per (division, focal MP): agreement with government MPs and with opposition MPs of other parties."""
    c = x[x.vote_id.isin(set(ids)) & x.vote.isin([A.FOR, A.AGAINST])]
    rows = []
    for vid, g in c.groupby("vote_id", sort=True):
        foc = g[g.party == focal_party]
        if foc.empty:
            continue
        gov = g[g.side == R.GOV].vote.to_numpy(); opp = g[(g.side == R.OPP) & (g.party != focal_party)].vote.to_numpy()
        for r in foc.itertuples(index=False):
            if focal_party == "Sjálvstýri":
                check(r.side == R.OPP, f"Sjálvstýri in opposition on {vid}")
            rows.append({"vote_id": vid, "period": r.period, "mp_id": r.mp_id, "n_gov": len(gov), "m_gov": int((gov == r.vote).sum()),
                         "n_opp": len(opp), "m_opp": int((opp == r.vote).sum())})
    return pd.DataFrame(rows)


def focal_summary(F, rng_b, rng_n):
    D = F.groupby("vote_id")[["n_gov", "m_gov", "n_opp", "m_opp"]].sum()
    D = D[(D.n_gov > 0) & (D.n_opp > 0)]
    k = len(D)
    idx = rng_b.integers(0, k, size=(N_BOOT, k))
    rg = D.m_gov.to_numpy()[idx].sum(1) / D.n_gov.to_numpy()[idx].sum(1)
    ro = D.m_opp.to_numpy()[idx].sum(1) / D.n_opp.to_numpy()[idx].sum(1)
    obs_g, obs_o = D.m_gov.sum() / D.n_gov.sum(), D.m_opp.sum() / D.n_opp.sum()
    dd = (D.m_gov / D.n_gov - D.m_opp / D.n_opp).to_numpy()
    flips = rng_n.choice([-1.0, 1.0], size=(N_PERM, k))
    null = (flips * dd).mean(1)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"divisions": int(k), "ballots": int(len(F[F.vote_id.isin(D.index)])),
            "agreement_with_government": float(obs_g), "ci95_government": q(rg),
            "agreement_with_other_opposition": float(obs_o), "ci95_other_opposition": q(ro),
            "difference_gov_minus_opp": float(obs_g - obs_o), "ci95_difference": q(rg - ro),
            "mean_per_division_difference": float(dd.mean()),
            "sign_flip_null": {"n_permutations": N_PERM, "statistic": "mean over divisions of (agreement with government - with other opposition)",
                               "p_two_sided": float((1 + (np.abs(null) >= abs(dd.mean()) - 1e-12).sum()) / (N_PERM + 1))}}


def sjalvstyri(x, call, seed_b, seed_n):
    ids1 = sorted(i for i in x[x.period == "P1"].vote_id.unique() if i in call)
    ids2 = sorted(i for i in x[x.period == "P2"].vote_id.unique() if i in call)
    out = {"P1": {}, "P2_descriptive": {}, "reference_opposition_parties_P1": {}}
    F = focal_agreement(x, ids1, "Sjálvstýri")
    out["P1"]["party_pooled"] = focal_summary(F, np.random.default_rng(seed_b), np.random.default_rng(seed_n))
    for k, m in enumerate(sorted(F.mp_id.unique())):
        out["P1"][m] = focal_summary(F[F.mp_id == m], np.random.default_rng(seed_b + 10 + k), np.random.default_rng(seed_n + 10 + k))
    F2 = focal_agreement(x, ids2, "Sjálvstýri")
    out["P2_descriptive"] = {"divisions": int(F2.vote_id.nunique()) if len(F2) else 0, "ballots": int(len(F2)),
                             "agreement_with_government": float(F2.m_gov.sum() / F2.n_gov.sum()) if len(F2) and F2.n_gov.sum() else None,
                             "agreement_with_other_opposition": float(F2.m_opp.sum() / F2.n_opp.sum()) if len(F2) and F2.n_opp.sum() else None}
    for k, p in enumerate(["Javnaðarflokkurin", "Tjóðveldi", "Framsókn"]):
        out["reference_opposition_parties_P1"][p] = focal_summary(focal_agreement(x, ids1, p), np.random.default_rng(seed_b + 50 + k),
                                                                  np.random.default_rng(seed_n + 50 + k))
    return out, F


# ----------------------------------------------------------------------------------------------- (g) side dissent
CODE = {A.FOR: 1, A.AGAINST: -1, A.BLANK: 0}


def _dissent_arrays(vote, side_gov):
    """vote: (..., n) codes; side_gov: (n,) bool. Returns dissent, denominator, tie, noref masks (..., n)."""
    isF = vote == 1; isA = vote == -1; valid = isF | isA
    Fg = isF[..., side_gov].sum(-1); Ag = isA[..., side_gov].sum(-1); Fo = isF[..., ~side_gov].sum(-1); Ao = isA[..., ~side_gov].sum(-1)
    Fs = np.where(side_gov, Fg[..., None], Fo[..., None]) - isF
    As = np.where(side_gov, Ag[..., None], Ao[..., None]) - isA
    noref = valid & (Fs + As == 0); tie = valid & (Fs == As) & ~noref
    den = valid & ~noref & ~tie
    plural = np.where(Fs > As, 1, -1)
    return den & (vote != plural), den, tie, noref


def side_dissent(x, call, seed_null, seed_boot):
    c = x[x.vote_id.isin(call) & x.vote.isin(list(CODE))].copy()
    c["code"] = c.vote.map(CODE)
    c["in_window"] = c.meeting_date.between(A.W0, A.W1)
    rng = np.random.default_rng(seed_null)
    recs, per_div = [], []
    for vid, g in c.groupby("vote_id", sort=True):
        g = g.sort_values("mp_id")
        vote = g.code.to_numpy(); sg = (g.side == R.GOV).to_numpy()
        d, den, tie, noref = _dissent_arrays(vote, sg)
        recs.append(pd.DataFrame({"vote_id": vid, "mp_id": g.mp_id.to_numpy(), "party": g.party.to_numpy(), "side": g.side.to_numpy(),
                                  "period": g.period.to_numpy(), "in_window": g.in_window.to_numpy(), "valid": vote != 0,
                                  "dissent": d, "denominator": den, "tie": tie, "no_reference": noref}))
        Vp = rng.permuted(np.tile(vote, (N_PERM_G, 1)), axis=1)
        dn, denn, _, _ = _dissent_arrays(Vp, sg)
        row = {"vote_id": vid, "period": g.period.iloc[0], "in_window": bool(g.in_window.iloc[0])}
        for s, m in (("government", sg), ("opposition", ~sg)):
            row[f"d_{s}"] = int(d[m].sum()); row[f"n_{s}"] = int(den[m].sum())
            row[f"null_d_{s}"] = dn[:, m].sum(1); row[f"null_n_{s}"] = denn[:, m].sum(1)
        per_div.append(row)
    B = pd.concat(recs, ignore_index=True); Dv = pd.DataFrame(per_div)
    brng = np.random.default_rng(seed_boot)
    scopes = {"window_pooled": Dv.in_window, "window_P1": Dv.in_window & (Dv.period == "P1"), "window_P3": Dv.in_window & (Dv.period == "P3"),
              "P3_full": Dv.period == "P3", "all_dated_pooled": Dv.period.notna()}
    summ = {}
    for sc, msk in scopes.items():
        S = Dv[msk]; summ[sc] = {"divisions": int(len(S))}
        for s in ("government", "opposition", "both"):
            ss = ["government", "opposition"] if s == "both" else [s]
            dd = sum(S[f"d_{q}"].to_numpy() for q in ss); nn = sum(S[f"n_{q}"].to_numpy() for q in ss)
            nd = sum(np.stack(S[f"null_d_{q}"].to_list()).sum(0) for q in ss); nden = sum(np.stack(S[f"null_n_{q}"].to_list()).sum(0) for q in ss)
            obs = dd.sum() / nn.sum(); nr = nd / nden
            summ[sc][s] = {"dissent": int(dd.sum()), "denominator": int(nn.sum()), "rate": float(obs), "ci95_divisions": boot_ratio(dd, nn, brng),
                           "null_within_division_shuffle": {"n_permutations": N_PERM_G, "mean_rate": float(nr.mean()),
                                                            "p2.5": float(np.percentile(nr, 2.5)), "p97.5": float(np.percentile(nr, 97.5)),
                                                            "p_lower": float((1 + (nr <= obs + 1e-12).sum()) / (N_PERM_G + 1))}}
    W = B[B.in_window]
    summ["window_ties_excluded"] = int(W.tie.sum()); summ["window_no_reference_excluded"] = int(W.no_reference.sum())
    summ["window_valid_ballots"] = int(W.valid.sum()); summ["window_blank_ballots"] = int((~W.valid).sum())
    summ["P2_in_window"] = {"divisions": int(Dv[Dv.period == "P2"].shape[0]), "dissent": int(B[B.period == "P2"].dissent.sum()),
                            "denominator": int(B[B.period == "P2"].denominator.sum())}
    agg = lambda G: G.agg(dissent=("dissent", "sum"), denominator=("denominator", "sum"), ties=("tie", "sum"), no_reference=("no_reference", "sum"),  # noqa: E731
                          valid=("valid", "sum")).assign(rate=lambda d: d.dissent / d.denominator.where(d.denominator > 0))
    by_party = agg(W.groupby(["party", "side"])).reset_index()
    by_party_period = agg(W[W.period != "P2"].groupby(["period", "party", "side"])).reset_index()
    by_mp = agg(W.groupby(["mp_id", "party", "side"])).reset_index()
    return summ, B, by_party, by_party_period, by_mp


def sjalvstyri_dissent(B, seed):
    rng = np.random.default_rng(seed); out = {}
    S = B[B.in_window & (B.party == "Sjálvstýri")]
    for m, g in S.groupby("mp_id"):
        per = g.groupby("vote_id")[["dissent", "denominator"]].sum()
        out[m] = {"dissent": int(per.dissent.sum()), "denominator": int(per.denominator.sum()),
                  "rate": float(per.dissent.sum() / per.denominator.sum()) if per.denominator.sum() else None,
                  "ci95_divisions": boot_ratio(per.dissent, per.denominator, rng) if per.denominator.sum() else None}
    return out


def existing_party_dissent():
    T = pd.read_csv(inp("dissent"))
    res = json.loads((inp("network_results")).read_text(encoding="utf-8"))["dissent_exploratory"]
    check(int(T.n_dissent.sum()) == int(res["n_dissenting_ballots"]) == 70, "existing party dissent: 70 dissenting ballots")
    g = T.groupby("party")[["n_party_divisions", "n_dissent"]].sum()
    return {"definition": res["definition"], "source": rel("dissent"), "dissent": int(T.n_dissent.sum()),
            "denominator": int(T.n_party_divisions.sum()), "rate": float(T.n_dissent.sum() / T.n_party_divisions.sum()),
            "by_party": {p: {"dissent": int(r.n_dissent), "denominator": int(r.n_party_divisions), "rate": float(r.n_dissent / r.n_party_divisions)}
                         for p, r in g.iterrows()},
            "per_period": "not available in existing outputs (window pooled only)"}, T[["mp_id", "party", "n_party_divisions", "n_dissent", "dissent_rate"]]


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CFG.FORMAL))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print("refusing to overwrite", out.name); return 2
    out.mkdir(parents=True, exist_ok=True)
    res = {"inputs": {k: {"path": rel(k), "sha256": sha(k)} for k in INPUTS}, "seeds": {"base": SEED, **{k: SEED + o for k, o in OFF.items()},
           "successions": WB.SEED}, "role_table": {k: {s: sorted(p) for s, p in t.items()} for k, t in R.TABLE.items()},
           "role_periods": R.PERIOD_TEXT}
    b, v, x = load()
    res["coverage"], call = coverage(b, v, x)
    print("coverage ok")
    # reproduction checks first: stop if they fail
    out.mkdir(parents=True, exist_ok=True)
    repro_bloc_three_way(b, v).to_csv(out / "bloc_three_way.csv", index=False)
    repro_within_bloc().to_csv(out / "within_bloc_vote_comparison.csv", index=False)
    saved3 = pd.read_csv(out / "bloc_three_way.csv"); saved_wb = pd.read_csv(out / "within_bloc_vote_comparison.csv")
    print("Louvain reference tables written")
    groups_side, mixed = mp_side_groups(x)
    res["mp_side_groups"] = {"rule": "majority side group over the MP's dated window ballots (X = Sambandsflokkurin's side)",
                             "mps": len(groups_side), "mixed": mixed}
    pd.Series(groups_side, name="side_group").rename_axis("mp_id").reset_index().to_csv(out / "mp_side_group.csv", index=False)
    # (a)
    win_ids = sorted(i for i in v[v.meeting_date.between(A.W0, A.W1)].vote_id if i in call)
    Dpd = pair_type_counts(x, win_ids)
    Dp3 = pair_type_counts(x, sorted(i for i in v[v.meeting_date >= R.P3_START].vote_id if i in call))
    Dpd.to_csv(out / "a_pair_types_per_division_window.csv", index=False)
    rng = np.random.default_rng(SEED + OFF["a_boot"])
    res["a_pair_types_per_division"] = {"window_pooled": summarise_types(Dpd, rng), "window_P1": summarise_types(Dpd[Dpd.period == "P1"], rng),
                                        "window_P3": summarise_types(Dpd[Dpd.period == "P3"], rng), "P3_full": summarise_types(Dp3, rng),
                                        "window_P2_divisions": int((Dpd.period == "P2").sum())}
    vw, bw = A.window(b, v)
    cids = list(A.contested_ids(bw)); mps = sorted(bw.mp_id.unique())
    Pw = formal_pair_table(b, x, cids, mps, A.modal_party(bw))
    Pw.to_csv(out / "a_pairs_window_formal_relation.csv", index=False)
    pair_rows = [three_formal("baseline window (all MPs)", Pw[Pw.n_shared >= t]) | {"min_overlap": t} for t in (20, 40, 60)]
    PP = {}
    for key, sel in (("P1", v.meeting_date.between(R.P1_START, R.P2_START - pd.Timedelta(days=1))),
                     ("P3_full", v.meeting_date >= R.P3_START), ("P3_window", v.meeting_date.between(R.P3_START, A.W1))):
        PP[key] = period_pairs(b, x, v, sel, call)
        pair_rows.append(three_formal(key, PP[key][2][PP[key][2].n_shared >= 20]) | {"min_overlap": 20})
    pd.DataFrame(pair_rows).to_csv(out / "a_pair_level_three_way_formal.csv", index=False)
    res["a_pair_level"] = pair_rows
    s40 = saved3[(saved3.network == "baseline window (all MPs)") & (saved3.min_overlap == 40)].iloc[0]
    e40 = Pw[Pw.n_shared >= 40]
    check(len(e40) == 712 and int(s40.pairs) == 712, "712 pairs at >= 40 shared divisions")
    res["louvain_saved_values"] = {"bloc_three_way_baseline_40": {k: (float(s40[k]) if k != "network" else s40[k]) for k in s40.index},
                                   "bloc_three_way_rows": saved3.to_dict(orient="records")}
    print("(a) done")
    # (b)
    rngb = np.random.default_rng(SEED + OFF["b_null"])
    res["b_party_beyond_side"] = {"P1": party_beyond_side_tests(PP["P1"][2], 20, "P1", rngb),
                                  "P3_full": party_beyond_side_tests(PP["P3_full"][2], 20, "P3", rngb),
                                  "P3_window": party_beyond_side_tests(PP["P3_window"][2], 20, "P3", rngb)}
    print("(b) done")
    # (c)
    tab_c, info_c = successions_within_side(groups_side)
    tab_c.to_csv(out / "c_successions_within_formal_side.csv", index=False)
    res["c_successions"] = {"info": info_c, "rows": tab_c.to_dict(orient="records"),
                            "louvain_saved_rows": saved_wb[saved_wb["null"] == "within_bloc"].to_dict(orient="records")}
    print("(c) done")
    # (d)
    res["d_communities_vs_sides"] = communities_vs_sides(groups_side, np.random.default_rng(SEED + OFF["d_null"]))
    print("(d) done")
    # (e)
    res["e_sjalvstyri"], Fe = sjalvstyri(x, call, SEED + OFF["e_boot"], SEED + OFF["e_null"])
    Fe.to_csv(out / "e_sjalvstyri_per_division_P1.csv", index=False)
    print("(e) done")
    # (g)
    summ, Bg, byp, bypp, bym = side_dissent(x, call, SEED + OFF["g_null"], SEED + OFF["g_boot"])
    party_dis, party_mp = existing_party_dissent()
    bym = bym.merge(party_mp.rename(columns={"party": "party_existing_table", "n_party_divisions": "party_dissent_denominator",
                                             "n_dissent": "party_dissent", "dissent_rate": "party_dissent_rate"}), on="mp_id", how="left")
    byp.to_csv(out / "g_side_dissent_by_party_window.csv", index=False); bypp.to_csv(out / "g_side_dissent_by_party_period_window.csv", index=False)
    bym.to_csv(out / "g_side_dissent_by_mp_window.csv", index=False)
    res["g_side_dissent"] = {"definition": "valid (For/Against) vote differing from the strict plurality of the OTHER members of the MP's formal side voting For/Against "
                             "in that contested division; blanks and absences are not dissent; ties and divisions with no other side member voting are excluded",
                             "summary": summ, "sjalvstyri_mps": sjalvstyri_dissent(Bg, SEED + OFF["g_boot"] + 1), "existing_party_dissent": party_dis}
    print("(g) done")
    res["checks"] = CHECKS
    res["environment"] = {"python": sys.version.split()[0], "platform": platform.system(), "numpy": np.__version__, "pandas": pd.__version__}
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=_js), encoding="utf-8", newline="\n")
    print(f"written {out.name}: {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
