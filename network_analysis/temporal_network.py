"""Pre/post-election voting networks (port of the cells of networks/notebooks/04_temporal_network_analysis.ipynb that feed the chain).

    python reproduce_networks/temporal_network.py      -> OUT_DIR/temporal/

Periods by vote date: pre (< 8 Dec 2022 election), post (> 22 Dec 2022 new government); post_full = all post-election divisions,
post_window = post-election divisions up to the end of the speech window (4 May 2023). Party labels are period-specific.
Writes
  pairwise_similarity_pre_election.csv  pre-election pair table (cell 16; Fig. 1b)
  temporal_node_metrics.csv             Louvain community of every MP per period at the temporal threshold (cells 16, 27, 49;
                                        used only as the outcome compared with the formal blocs)
  temporal_results.json                 threshold, election timeline (cell 4), continuing-MP pair change and its case-level null (cell 26)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG  # noqa: E402
import netlib as N  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 2805
THRESHOLDS = (10, 20, 30, 40)
SEEDS = range(SEED, SEED + 20)            # Louvain seeds, as in the notebook
RULE_SHARE = 0.85                         # threshold rule of the notebook (section 4)
PERM_SPLIT = 1_000                        # case-level split null of the continuing MPs (cell 26)


def main() -> int:
    out = CFG.TEMPORAL
    if out.exists() and any(out.iterdir()):
        print("refusing to overwrite", out.name); return 2
    out.mkdir(parents=True, exist_ok=True)
    data = N.load_data(CFG.DATA_DIR)
    ballots, votes, members = data["ballots"], data["votes"], data["members"]
    names = members.set_index("mp_id")["name"]
    dated = votes.dropna(subset=["meeting_date"])
    dated_ballots = ballots[ballots.vote_id.isin(dated.vote_id)]
    M_all = N.vote_matrix(dated_ballots)
    Mc = M_all.loc[:, N.contested_mask(M_all)]
    div_date = dated.set_index("vote_id").meeting_date
    div_period = N.assign_period(div_date.loc[M_all.columns])
    ddate = div_date.loc[M_all.columns]
    select = {"pre": div_period == "pre", "post_full": div_period == "post",
              "post_window": (div_period == "post") & (ddate <= N.WINDOW_END)}
    DS = {}
    for key, sel in select.items():
        all_cols = M_all.columns[sel.to_numpy()]
        cols = Mc.columns[Mc.columns.isin(all_cols)]
        DS[key] = {"M": N.period_matrix(Mc, cols), "parties": N.period_parties(dated_ballots, all_cols), "cols": cols}
    # temporal threshold: largest threshold with >= 85% of MPs having an eligible pair in pre and post_full (the notebook's rule)
    share = {}
    for k in ("pre", "post_full"):
        Mx = DS[k]["M"]; pr = N.pairwise_table(Mx, DS[k]["parties"].loc[Mx.index], thresholds=THRESHOLDS)
        for t in THRESHOLDS:
            kept = pr[pr.n_shared_divisions >= t]
            share[(k, t)] = len(set(kept.mp_id_1) | set(kept.mp_id_2)) / Mx.shape[0]
    ok = [t for t in THRESHOLDS if share[("pre", t)] >= RULE_SHARE and share[("post_full", t)] >= RULE_SHARE]
    REC = max(ok) if ok else min(THRESHOLDS)
    assert REC == 20, f"temporal threshold {REC}, expected 20"
    rows = []
    for k in ("pre", "post_full", "post_window"):
        Mx = DS[k]["M"]; px = DS[k]["parties"].loc[Mx.index]
        pr = N.pairwise_table(Mx, px, thresholds=(REC,))
        G = N.similarity_graph(pr, REC, px, names, Mx.index)
        part, mods, stab = N.louvain_runs(G, SEEDS)
        comms = sorted([c for c in part if any(G.degree(n) > 0 for n in c)], key=len, reverse=True)
        lab = {n: f"C{ix + 1}" for ix, c in enumerate(comms) for n in c}
        for n in G.nodes:
            rows.append({"period": k, "mp_id": n, "name": names.get(n), "party": G.nodes[n]["party"],
                         "n_contested_votes": int(Mx.loc[n].notna().sum()), "degree": G.degree(n),
                         "community": lab.get(n, "isolated") if G.degree(n) > 0 else "isolated"})
    # timeline (cell 4): days without a division around the election
    vdays = pd.DatetimeIndex(sorted(dated.meeting_date.unique()))
    before, after = vdays[vdays < N.ELECTION].max(), vdays[vdays > N.ELECTION].min()
    timeline = {"last_vote_before_election": str(before.date()), "first_vote_after_election": str(after.date()),
                "days_between": int((after - before).days),
                "votes_between_election_and_new_government": int(((dated.meeting_date >= N.ELECTION) & (dated.meeting_date <= N.NEW_GOVERNMENT)).sum())}
    # continuing MPs (cells 19, 26): change of pair similarity, pairs eligible in both periods, case-level split null (seed SEED + 31)
    cont = sorted(set(DS["pre"]["M"].index) & set(DS["post_full"]["M"].index))
    pp_pre = DS["pre"]["parties"]
    cols_pre, cols_post = list(DS["pre"]["cols"]), list(DS["post_full"]["cols"])
    A_c = Mc.loc[cont, cols_pre + cols_post].to_numpy()
    is_post = np.r_[np.zeros(len(cols_pre), bool), np.ones(len(cols_post), bool)]
    lab_c = pd.Categorical(pp_pre.loc[cont]).codes
    blocks = dated.set_index("vote_id").case_id.loc[cols_pre + cols_post].to_numpy()          # one case = one block
    mask_obs = N.eligible_pair_mask(A_c, is_post, REC)
    obs_ch = N.change_statistics(A_c, is_post, lab_c, REC)
    null_block = N.period_label_permutation(A_c, is_post, lab_c, REC, PERM_SPLIT, np.random.default_rng(SEED + 31), blocks=blocks, pair_mask=mask_obs)
    continuing = {"n_continuing_mps": len(cont), "n_pairs_eligible_both": obs_ch["n_pairs"], "threshold": REC,
                  "mean_abs_change_observed": obs_ch["mean_abs_change"], "spearman_pre_post": obs_ch["spearman_pre_post"],
                  "case_level_null": {"n_perm": PERM_SPLIT, "mean_abs_change_null_mean": float(null_block.mean_abs_change.mean()),
                                      "mean_abs_change_null_sd": float(null_block.mean_abs_change.std()),
                                      "n_pairs_null_mean": float(null_block.n_pairs.mean())}}
    pre = N.pairwise_table(DS["pre"]["M"], DS["pre"]["parties"].loc[DS["pre"]["M"].index], thresholds=THRESHOLDS)
    pre.to_csv(out / "pairwise_similarity_pre_election.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "temporal_node_metrics.csv", index=False)
    (out / "temporal_results.json").write_text(json.dumps({"threshold": REC, "rule_share": RULE_SHARE, "louvain_seeds": f"{SEED}..{SEED + 19}",
                                                           "share_mps_with_eligible_pair": {f"{k}@{t}": v for (k, t), v in share.items()},
                                                           "timeline": timeline, "continuing": continuing},
                                                          indent=1), encoding="utf-8", newline="\n")
    print(f"temporal: threshold {REC}; pre pairs {len(pre)}; node rows {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
