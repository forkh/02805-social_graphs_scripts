"""Voting-similarity network of the speech window (port of the cells of networks/notebooks/02_network_analysis.ipynb that feed the chain).

    python reproduce_networks/voting_network.py        -> OUT_DIR/voting/

Writes
  pairwise_voting_similarity.csv  every MP pair: share of matching For/Against votes on contested divisions (cell 18)
  network_communities.csv         Louvain communities of the network of pairs with >= 40 shared contested divisions,
                                  best modularity of 20 seeds; -1 = no edge (cell 21; used only as a reference / outcome)
  dissent_network_summary.csv     per-MP dissent from the own party's strict For/Against plurality (cell 41)
  network_results.json            network size, Louvain modularity, within-division shuffle null, turnover break, blank-vote
                                  sensitivity (cells 12, 20, 21, 25-27) and the dissent summary (cell 41)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG  # noqa: E402
import netlib as N  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

SEED = 2805                      # as in the notebook
MIN_OVERLAPS = (20, 40, 60)
MAIN_OVERLAP = 40
PERM_DIVISION = 1_000            # within-division shuffle null (cell 26)


def main() -> int:
    out = CFG.VOTING
    if out.exists() and any(out.iterdir()):
        print("refusing to overwrite", out.name); return 2
    out.mkdir(parents=True, exist_ok=True)
    data = N.load_data(CFG.DATA_DIR)
    ballots, votes, members = data["ballots"], data["votes"], data["members"]
    votes_w, ballots_w = N.restrict_window(votes, ballots)
    parties = N.modal_party(ballots_w)
    names = members.set_index("mp_id")["name"]
    M_all = N.vote_matrix(ballots_w)
    Mc = M_all.loc[:, N.contested_mask(M_all)]
    pairs = N.pairwise_table(Mc, parties, thresholds=MIN_OVERLAPS)
    G = N.similarity_graph(pairs, MAIN_OVERLAP, parties, names, Mc.index)
    degree = dict(G.degree())
    best_part, mods, stab = N.louvain_runs(G, range(SEED, SEED + 20))
    nonisol_comms = sorted([c for c in best_part if any(degree[n] > 0 for n in c)], key=len, reverse=True)
    comm_of = {n: k for k, c in enumerate(nonisol_comms) for n in c}
    nm = pd.DataFrame({"mp_id": list(G.nodes)})
    nm["name"] = nm.mp_id.map(names)
    nm["party"] = nm.mp_id.map(parties)
    nm["community"] = nm.mp_id.map(comm_of).fillna(-1).astype(int)     # -1 = isolated (no edge)
    dis = N.dissent_table(ballots_w)
    dsum = nm[["mp_id", "name", "party"]].merge(dis, left_on="mp_id", right_index=True, how="left")
    pairs.to_csv(out / "pairwise_voting_similarity.csv", index=False)
    nm[["mp_id", "name", "party", "community"]].to_csv(out / "network_communities.csv", index=False)
    dsum.round(6).to_csv(out / "dissent_network_summary.csv", index=False)

    # graph size (cell 20)
    n_isol = sum(1 for _, d in G.degree() if d == 0)
    graph_main = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(), "n_isolates": n_isol,
                  "n_mps_with_edge": G.number_of_nodes() - n_isol, "min_shared_contested_divisions": MAIN_OVERLAP}
    # turnover break in the speech window (cell 12): MPs voting only after the break and their contested divisions
    vdates = votes_w.set_index("vote_id").meeting_date.loc[M_all.columns]
    cast = ballots_w[ballots_w.vote.isin(N.VOTE_CODE)].assign(meeting_date=lambda d: d.vote_id.map(vdates))
    first_vote, last_vote = cast.groupby("mp_id").meeting_date.min(), cast.groupby("mp_id").meeting_date.max()
    ud = pd.DatetimeIndex(sorted(vdates.dropna().unique()))
    cand = pd.DataFrame({d: {"only_before": int((last_vote < d).sum()), "only_from": int((first_vote >= d).sum())}
                         for d in ud[1:]}).T
    cand["one_sided_mps"] = np.where(cand.min(axis=1) >= 10, cand.only_before + cand.only_from, 0)
    split_date = cand.one_sided_mps.idxmax()
    prev_day = ud[ud < split_date].max()
    mp_periods = pd.Series(np.where(last_vote < split_date, "P1", np.where(first_vote >= split_date, "P2", "both")), index=first_vote.index)
    period = pd.Series(np.where(vdates >= split_date, "P2", "P1"), index=vdates.index)
    n_contested = Mc.notna().sum(axis=1)
    p2_only = mp_periods[mp_periods == "P2"].index
    period_break = {"split_date": str(split_date.date()), "previous_voting_day": str(prev_day.date()),
                    "days_without_division_before": int((split_date - prev_day).days),
                    "mps_p1_only": int((mp_periods == "P1").sum()), "mps_p2_only": int(len(p2_only)), "mps_both": int((mp_periods == "both").sum()),
                    "contested_p1": int((period.loc[Mc.columns] == "P1").sum()), "contested_p2": int((period.loc[Mc.columns] == "P2").sum()),
                    "max_contested_divisions_of_mps_voting_only_after_break": int(n_contested.reindex(p2_only).fillna(0).max())}
    # within-division shuffle null of Louvain modularity (cells 25-26): 1,000 permutations, generator seed SEED + 1, Louvain seed SEED
    A = Mc.to_numpy()
    labels = pd.Categorical(parties.loc[Mc.index], categories=N.PARTY_ORDER).codes
    sh, ma = N.agreement_counts(A)

    def graph_stats(shared, match, min_overlap=MAIN_OVERLAP):
        a, b, s = N.upper_arrays(shared, match, min_overlap)
        g = nx.Graph()
        g.add_nodes_from(range(shared.shape[0]))
        g.add_weighted_edges_from(zip(a, b, s))
        part = nx.community.louvain_communities(g, weight="weight", seed=SEED)
        nodes = [n for n, d in g.degree() if d > 0]
        lab = {n: k for k, c in enumerate(part) for n in c}
        return {"louvain_Q": nx.community.modularity(g, part, weight="weight"),
                "party_NMI": normalized_mutual_info_score(labels[nodes], [lab[n] for n in nodes])}
    obs_graph = graph_stats(sh, ma)
    null_div = N.division_null(A, labels, MAIN_OVERLAP, PERM_DIVISION, np.random.default_rng(SEED + 1), graph_fn=graph_stats)
    division_shuffle = {"n_perm": PERM_DIVISION, "louvain_Q_observed": float(obs_graph["louvain_Q"]),
                        "louvain_Q_null_mean": float(null_div.louvain_Q.mean()), "louvain_Q_null_sd": float(null_div.louvain_Q.std()),
                        "louvain_Q_null_max": float(null_div.louvain_Q.max())}
    # blank votes as a third category (cell 27): Spearman of pair similarities, pairs with >= 40 shared divisions in both versions
    pb = N.pairwise_table(Mc, parties, include_blank=True, thresholds=(MAIN_OVERLAP,))
    both = pairs.n_shared_divisions.ge(MAIN_OVERLAP) & pb.n_shared_divisions.ge(MAIN_OVERLAP)
    blank_rho = float(stats.spearmanr(pairs.loc[both, "similarity"], pb.loc[both, "similarity"])[0])

    res = {"graph_main": graph_main, "period_break": period_break, "within_division_shuffle": division_shuffle,
           "blank_sensitivity": {"spearman": blank_rho, "n_pairs": int(both.sum())},
           "louvain_main": {"community_sizes": [len(c) for c in nonisol_comms], "modularity_best_of_20": float(max(mods)),
                            "seeds": f"{SEED}..{SEED + 19}"},
           "dissent_exploratory": {"definition": "vote against the strict For/Against plurality of own party; units with >=3 party voters, no ties",
                                   "n_dissenting_ballots": int(dis.n_dissent.sum()), "n_mps_with_rate": int(dsum.dissent_rate.notna().sum())}}
    (out / "network_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(f"voting network: {Mc.shape[0]} MPs, {Mc.shape[1]} contested divisions, {len(pairs)} pairs, communities {res['louvain_main']['community_sizes']}; "
          f"shuffle-null Q {division_shuffle['louvain_Q_null_mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
