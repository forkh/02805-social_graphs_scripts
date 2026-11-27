"""Within-bloc permutation test of Spearman(succession weight, voting similarity) with the Louvain blocs (load() and run() of
networks/interaction_analysis/src/within_bloc_test.py, code unchanged apart from paths). Used only for the Louvain reference values
reported next to the formal-bloc test; called by formal_roles.py.
"""
from __future__ import annotations

import pandas as pd

import config as CFG
import speaking_lib as I

N_PERM, N_PERM_EXPL, SEED = 10_000, 2_000, I.SEED       # same seed and counts as the pre-specified tests
T = CFG.SPEAKING


def load():
    comm = pd.read_csv(CFG.VOTING / "network_communities.csv")
    bloc = {m: int(c) for m, c in zip(comm.mp_id, comm.community) if c >= 0}      # -1 = no edge in the voting network
    mp = pd.read_csv(T / "mp_id_name_party_map.csv")
    universe = sorted(mp.mp_id)
    party = {m: p for m, p in zip(mp.mp_id, mp.party)}
    S = pd.read_csv(T / "pairwise_interaction_vs_vote_similarity_min40.csv")
    U = {d: pd.read_csv(T / f"undirected_edges_chair_{d.split('_')[1]}.csv") for d in ("chair_inclusive", "chair_excluded")}
    return bloc, universe, party, S, U


def run(n_perm_main=N_PERM, n_perm_expl=N_PERM_EXPL):
    bloc, universe, party, S, U = load()
    groups_bloc = {n: bloc.get(n, "(none)") for n in universe}
    groups_party = {n: party.get(n, "(none)") for n in universe}
    rows = []
    for d in ("chair_inclusive", "chair_excluded"):
        W, idx = I.weight_matrix(U[d], universe)
        v = S[S.network_definition == d].reset_index(drop=True)
        v = v.assign(same_bloc=[bloc[a] == bloc[b] for a, b in zip(v.mp_id_1, v.mp_id_2)])
        specs = [("pooled", v, False, "tested (authorised after the main results; same design as the within-party test)", n_perm_main),
                 ("pooled", v, True, "exploratory", n_perm_expl),
                 ("same_bloc", v[v.same_bloc], False, "exploratory", n_perm_expl),
                 ("different_bloc", v[~v.same_bloc], False, "exploratory", n_perm_expl)]
        for stratum, vv, pos, cls, npm in specs:
            for null_name, groups in (("within_bloc", groups_bloc), ("within_party", groups_party)):
                if null_name == "within_party" and not (stratum == "pooled" and not pos):
                    continue
                m = I.mantel_spearman(vv, W, idx, list(universe), groups, npm, SEED, pos)
                rows.append({"network_definition": d, "null": null_name, "stratum": stratum, "pairs_with_interaction_only": pos,
                             "evidence_class": cls if null_name == "within_bloc" else "check (reproduces vote_comparison_tests.csv)",
                             "n_pairs_in_sample": len(vv), "n_pairs_interacting": int((vv.interaction_weight > 0).sum()), **m})
    return pd.DataFrame(rows), bloc
