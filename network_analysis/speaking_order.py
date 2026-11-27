"""Speaking-order successions (port of the cells of networks/interaction_analysis/notebooks/01_fpsc_interaction_networks.ipynb that
feed the chain).

    python reproduce_networks/speaking_order.py        -> OUT_DIR/speaking_order/   (needs OUT_DIR/voting/ from voting_network.py)

Writes
  mp_id_name_party_map.csv                       every speaker, party = modal ballot party ("(no party)" if none)  (cell 3)
  directed_edges_chair_{inclusive,excluded}.csv  directed succession counts                                         (cells 5, 7)
  undirected_edges_chair_{inclusive,excluded}.csv w(a,b) = n(a->b) + n(b->a)                                        (cells 5, 7)
  pairwise_interaction_vs_vote_similarity_min40.csv  voting pairs with >= 40 shared contested divisions and their
                                                 succession weight, both definitions                               (cell 18)
  speaking_order_results.json                    chair MPs and their weight share, overlap of the two definitions,
                                                 party mixing of the chair-excluded network                        (cells 9, 11, 14)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG  # noqa: E402
import speaking_lib as I  # noqa: E402

import pandas as pd  # noqa: E402

PRIMARY_THRESHOLD = 40
N_PERM_MIXING = 10_000


def main() -> int:
    out = CFG.SPEAKING
    if out.exists() and any(out.iterdir()):
        print("refusing to overwrite", out.name); return 2
    out.mkdir(parents=True, exist_ok=True)
    pairs_csv = CFG.VOTING / "pairwise_voting_similarity.csv"
    rows = I.load_rows(CFG.DATA_DIR)
    parties = I.load_parties(pairs_csv)
    names = I.mp_names(rows)
    universe = sorted(rows.mp_id.unique())
    chair_counts = rows[rows.contribution_type == "chairman"].mp_id.value_counts()
    chair_mps = set(chair_counts[chair_counts >= I.MIN_CHAIR_ROWS].index)
    mp_map = pd.DataFrame({"mp_id": universe, "name": [names[n] for n in universe],
                           "party": [parties.get(n, "(no party)") for n in universe],
                           "speech_rows": [int((rows.mp_id == n).sum()) for n in universe],
                           "chairman_rows": [int(chair_counts.get(n, 0)) for n in universe]})
    mp_map["is_chair_mp"] = mp_map.mp_id.isin(chair_mps)
    mp_map.to_csv(out / "mp_id_name_party_map.csv", index=False)
    U, scatter, DD = {}, [], {}
    for d, excl in (("chair_inclusive", False), ("chair_excluded", True)):
        D, st = I.directed_transitions(rows, exclude_chairs=excl)
        DD[d] = D
        I.directed_frame(D, d).to_csv(out / f"directed_edges_{d}.csv", index=False)
        U[d] = I.undirected_frame(D, d)
        U[d].to_csv(out / f"undirected_edges_{d}.csv", index=False)
        v_all = I.vote_pair_frame(pairs_csv, U[d], chair_mps)
        v = v_all[v_all[f"meets_min_overlap_{PRIMARY_THRESHOLD}"] & v_all.similarity.notna()].reset_index(drop=True)
        scatter.append(v.assign(network_definition=d))
        print(f"{d}: {st.get('transitions_kept', 0)} successions, {len(U[d])} pairs")
    pd.concat(scatter).to_csv(out / "pairwise_interaction_vs_vote_similarity_min40.csv", index=False)

    # chair MPs and their share of the chair-inclusive weight (cell 11, network_summary)
    ui = U["chair_inclusive"]
    total = float(ui.interaction_weight.sum())
    chair_share = float(ui[ui.mp_id_1.isin(chair_mps) | ui.mp_id_2.isin(chair_mps)].interaction_weight.sum() / total)
    # overlap of the two definitions (cell 9): strongest 20 pairs shared
    a = U["chair_inclusive"].set_index(["mp_id_1", "mp_id_2"]).interaction_weight
    b = U["chair_excluded"].set_index(["mp_id_1", "mp_id_2"]).interaction_weight
    top20_shared = len(set(a.index[:20]) & set(b.sort_values(ascending=False).index[:20]))
    # party mixing of the chair-excluded network (cell 14): 10,000 label permutations, seed of the interaction analysis
    mixing = I.party_mixing(DD["chair_excluded"], parties, N_PERM_MIXING, I.SEED)
    res = {"chair_mps": {"min_chairman_rows": I.MIN_CHAIR_ROWS, "n_chair_mps": len(chair_mps),
                         "fewest_chairman_rows_among_chair_mps": int(chair_counts[sorted(chair_mps)].min()),
                         "share_weight_on_edges_touching_chair_mps_chair_inclusive": chair_share},
           "definition_overlap": {"pairs_inclusive": len(a), "pairs_exclusive": len(b), "pairs_in_both": len(a.index.intersection(b.index)),
                                  "top20_pairs_shared": top20_shared},
           "party_mixing_chair_excluded": mixing}
    (out / "speaking_order_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"chair share {chair_share:.3f}; top-20 shared {top20_shared}; assortativity {mixing['assortativity_weighted']:.4f}, p {mixing['perm_p_one_sided_ge']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
