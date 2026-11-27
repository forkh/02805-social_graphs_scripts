# reproduce_networks

Self-contained reproduction of the network analyses, from the original datasets in `data/` only.

**Reproduces:** the voting-similarity network of the speech window and its formal government/opposition blocs; party and bloc dissent;
the pre/post-election comparison; Louvain communities compared with the formal blocs; speaking-order successions with the within-bloc
null, party mixing and the alternation between blocs; Fig. 1 of the paper (`fig_svt_final_v4`).
Louvain communities are used only as reference values next to the formal-bloc results and as the outcome compared with the formal
blocs; every grouping of the analyses uses the formal roles in `roles.py`.

## Requirements

Python 3.12.2; numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scipy 1.18.1, networkx 3.7, matplotlib 3.11.2, scikit-learn 1.9.1.

## Data (read only)

`data/ballots_2020-2023.parquet`, `data/votes_2020-2023.parquet`, `data/members_2020-2023.parquet`, `data/cases_2020-2023.parquet`,
`data/ballots_2020-2023.csv`, `data/votes_2020-2023.csv`, `data/members_2020-2023.csv`, `data/fpsc/fpsc_train_metadata.parquet`.

## Run

From the folder that contains `data/` and `reproduce_networks/`:

    python reproduce_networks/run_all.py

All outputs go to `reproduce_networks_outputs/` next to `data/` (suggested `.gitignore` line: `reproduce_networks_outputs/`).
Each step runs in a fresh Python process; the run stops if any file outside the output folder is created, changed or deleted
(SHA-256, size and mtime checked). `--from <step>` resumes. Fixed seeds are used throughout; the outputs do not depend on
`PYTHONHASHSEED`. Runtime: about 9–10 minutes on a laptop (formal_roles 7 minutes, voting and alternation 1 minute each, the rest seconds).

## Run order

| step | script | produces |
|---|---|---|
| voting | `voting_network.py` | `voting/`: pairwise voting similarity (contested divisions, speech window), Louvain communities, party dissent; `network_results.json` (network size, modularity, within-division shuffle null, turnover break, blank-vote sensitivity) |
| temporal | `temporal_network.py` | `temporal/`: pre-election pair table, Louvain communities per period; `temporal_results.json` (election timeline, continuing-MP pair change and its case-level null) |
| speaking_order | `speaking_order.py` | `speaking_order/`: succession edge tables (chair inclusive/excluded), MP map, succession weight vs voting similarity per pair; `speaking_order_results.json` (chair MPs, overlap of the two definitions, party mixing) |
| formal_roles | `formal_roles.py` | `formal_roles/results.json` and CSVs: pair types by formal bloc, party beyond bloc, successions within bloc, communities vs blocs, Sjálvstýri test, bloc dissent; Louvain reference tables `bloc_three_way.csv`, `within_bloc_vote_comparison.csv` |
| alternation | `alternation.py` | `formal_roles/alternation/`: alternation between blocs in speaking order |
| fig1 | `fig1.py` | `figures/fig_svt_final_v4.{pdf,png}` and its provenance JSON |

Helper modules: `config.py` (paths), `roles.py` (formal government/opposition table), `netlib.py`, `speaking_lib.py`, `auditlib.py`,
`within_bloc.py` (ported network helpers), `figstyle.py` (figure style).

## Checking the result

Expected SHA-256 (paths relative to `reproduce_networks_outputs/`):

| file | SHA-256 |
|---|---|
| `formal_roles/results.json` | `ab7a7343ce98b2f8831cb8d852af098461d8b717b19e8c6092bd6559a5bf5f43` |
| `voting/network_results.json` | `44a14e6ef0934bbe8bce990cb6072dea2d1c45870053e16333fb80a177271054` |
| `temporal/temporal_results.json` | `edb174572f951a41098ba1eb31173fbf8205338da3e4dc340b12536d4a8aa672` |
| `speaking_order/speaking_order_results.json` | `777eb531b70c378b344a5062479361d5c7d3ef60e6b84ea14f70a5c791157ead` |
| `formal_roles/a_pair_types_per_division_window.csv` | `7ab6a85f08637685da0865b16b57191465d3e0855d5c0eda0615453129b9b59f` |
| `formal_roles/a_pair_level_three_way_formal.csv` | `f83c08bb4a1689d7cea4495624ee0f173e2b95415d7fa1e497c3c592694b70d9` |
| `formal_roles/c_successions_within_formal_side.csv` | `7869c21d8485fdffb512cd0daeb13fa09a625b183d3ea44464f20175c5693c31` |
| `formal_roles/e_sjalvstyri_per_division_P1.csv` | `5f328fee6d961d3cacc18069d702e0cdba08ef92b1efdcafe9a85b03e4c90cc4` |
| `formal_roles/g_side_dissent_by_party_window.csv` | `c8cf7a65e99b1bf0bc024177b38a7bcdb57b6fedfe3d3e1166277a6321cf1efe` |
| `formal_roles/g_side_dissent_by_mp_window.csv` | `525ea33264f20add960716e235cf1f5966b071070e57842dcef165d3fb920e06` |
| `formal_roles/bloc_three_way.csv` | `fccf2a46b00796397fb01ca77772cfae3b359c4c056cee1fe2f90edc8eb860ec` |
| `formal_roles/within_bloc_vote_comparison.csv` | `1d7f66a5e5bf224bcc8b67894eee43c8d76be5226213a4977b15ff54a907fbbf` |
| `formal_roles/alternation/alternation_results.json` | `5145b8afe685207eed284c0f9f79d3ae97380e8e418576962532f053b0488340` |
| `voting/pairwise_voting_similarity.csv` | `89ab9388aa3e133e026be16c2bbfa36d9addeb38272a7a46eb9fd8f2d5685772` |
| `voting/network_communities.csv` | `b0e3794d371ab69c884ff97471767e56c81f9f27d9ce672f0d670ddc3f7cdba6` |
| `temporal/pairwise_similarity_pre_election.csv` | `b9e287141beddd1ab5afe8764820db9a6d3f028d6435c6b7961ee18367fd6f3f` |
| `speaking_order/directed_edges_chair_excluded.csv` | `936e22772d610ebb030a2600e5ef967a4971a7e0db773212a2495e4988f8fd5c` |
| `speaking_order/pairwise_interaction_vs_vote_similarity_min40.csv` | `e24b495cb277a1181e4be2b4b3fe4ac3025855becc231abf0af05c5041845e4f` |
| `figures/fig_svt_final_v4.png` | `ec50a8e5e777c785e699d2853fd2d843e158c0ddc6e830667f5d3b230c327cdf` |
| `figures/fig_svt_final_v4.pdf` | `93fc55494eec86f3a0b043e09eb377e2238139738619ce0549ccfdbe9628ff41` |

Where the reported numbers come from (`results.json` = `formal_roles/results.json`):

| numbers | key |
|---|---|
| divisions per period, undated, contested | `coverage` |
| agreement by pair type per division, 95% intervals | `a_pair_types_per_division.window_pooled` (also `window_P1`, `P3_full`) |
| pair-level means (712 pairs) | `a_pair_level` (network "baseline window (all MPs)", min_overlap 40); Louvain reference `louvain_saved_values` |
| party beyond bloc, exact party splits | `b_party_beyond_side.{P1,P3_full,P3_window}` |
| successions vs voting similarity, within-bloc nulls | `c_successions.rows` (primary: null `within_formal_side_x_in_sample`; sensitivity: `within_formal_side`); Louvain reference `c_successions.louvain_saved_rows` |
| Louvain communities vs formal blocs (NMI, ARI) | `d_communities_vs_sides.{pre,post_full,post_window,pooled_window}` |
| Sjálvstýri test (party and per MP) | `e_sjalvstyri.P1` |
| bloc dissent and shuffle null; party dissent | `g_side_dissent.summary`; `g_side_dissent.existing_party_dissent` |
| alternation between blocs | `alternation/alternation_results.json` → `results.{pooled,P1,P3}` |
| Fig. 1b network (MPs, edges, components, merge cut) | `figures/fig_svt_final_v4_provenance.json` → `values.b.at_40`, `values.b.components_at_40` |
| `\backboneCut` (edge cut of Fig. 1b) | `figures/fig_svt_final_v4_provenance.json` → `values.b.edge_cut` |
| `\nNetMPs`, `\nPairsMain`, `\netIsolates` | `voting/network_results.json` → `graph_main.n_mps_with_edge`, `graph_main.n_edges`, `graph_main.n_isolates` |
| `\louvainQ` | `voting/network_results.json` → `louvain_main.modularity_best_of_20` |
| `\divShuffleQ` | `voting/network_results.json` → `within_division_shuffle.louvain_Q_null_mean` |
| MPs who first voted in 2023 had at most 26 contested divisions | `voting/network_results.json` → `period_break` (`split_date`, `max_contested_divisions_of_mps_voting_only_after_break`, `contested_p2`) |
| Blankt as a third category: Spearman 0.99 | `voting/network_results.json` → `blank_sensitivity.spearman` |
| 91-day pause without divisions after the election | `temporal/temporal_results.json` → `timeline.days_between` |
| `\contPairs`, `\contAbs`, `\contNullMean` | `temporal/temporal_results.json` → `continuing.n_pairs_eligible_both`, `continuing.mean_abs_change_observed`, `continuing.case_level_null.mean_abs_change_null_mean` |
| `\soChairWeightPct`, `\soNChairMPs`; chair MPs have at least 50 procedural turns | `speaking_order/speaking_order_results.json` → `chair_mps` (`share_weight_on_edges_touching_chair_mps_chair_inclusive` × 100, `n_chair_mps`, `min_chairman_rows`, `fewest_chairman_rows_among_chair_mps`) |
| `\soWithinPartyPct`, `\soWithinPartyExpPct`, `\soAssort`, `\soAssortP` | `speaking_order/speaking_order_results.json` → `party_mixing_chair_excluded` (`within_party_weight_share` × 100, `expected_within_share_independent_mixing` × 100, `assortativity_weighted`, `perm_p_one_sided_ge`) |
| heaviest edges of the chair-inclusive and chair-excluded networks do not overlap | `speaking_order/speaking_order_results.json` → `definition_overlap.top20_pairs_shared` (0) |
