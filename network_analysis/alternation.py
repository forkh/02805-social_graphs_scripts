"""Alternation between formal blocs in speaking order (chair-excluded successions, all MPs).

    python reproduce_networks/alternation.py            -> OUT_DIR/formal_roles/alternation/   (copy of networks/formal_roles/src/alternation.py)

Successions: consecutive non-chairman rows of the same sitting, self-successions dropped (the validated chair-excluded
definition; the aggregated counts are asserted equal to the directed edge table written by speaking_order.py).
Bloc of a speaker = formal side (roles.py) of the speaker's party on the sitting date. Party on the date = (1) the MP's ballot
party that day; else (2) the ballot party on the nearest ballot date in the same period; else (3) the FPSC party label, where
"A → B (Nov 2022)" is A before 1 Nov 2022, B after 30 Nov 2022 and undetermined within November 2022. Successions with a speaker
without a bloc are dropped and counted. Null: speaking order shuffled within each sitting (same rows, same speakers and
counts), self-successions dropped again, 10,000 shuffles. Louvain communities are not used. Writes only into --out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as CFG  # noqa: E402


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import roles as R  # noqa: E402
import speaking_lib as I  # noqa: E402

SEED, N_SHUF = 2805 + 201, 10_000
SAVED_EDGES = "OUT_DIR/speaking_order/directed_edges_chair_excluded.csv"     # written by speaking_order.py
NOV_START, NOV_END = pd.Timestamp("2022-11-01"), pd.Timestamp("2022-11-30")
CHECKS = []


def check(ok, what):
    CHECKS.append({"check": what, "pass": bool(ok)})
    if not ok:
        raise AssertionError("CHECK FAILED: " + what)


def party_on(label: str, date: pd.Timestamp):
    lab = re.sub(r"\s*\([^()]*\)", "", label).strip()            # drop English glosses / "(Nov 2022)"
    if "→" in lab:
        a, b = [s.strip() for s in lab.split("→")]
        check("(Nov 2022)" in label, f"switch label dated Nov 2022: {label}")
        return a if date < NOV_START else (b if date > NOV_END else None)
    return lab


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(CFG.FORMAL / "alternation")); a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print("refusing to overwrite", out.name); return 2
    out.mkdir(parents=True, exist_ok=True)
    rows = I.load_rows(CFG.DATA_DIR)                                    # reads DATA_DIR/fpsc/fpsc_train_metadata.parquet
    x = rows[rows.contribution_type != "chairman"].copy()
    x["period"] = [R.period(d) for d in x.meeting_date]
    # party on the sitting date: (1) ballot party that day, (2) ballot party on the nearest ballot date in the same period,
    # (3) FPSC label (switch rule above). The FPSC label is one fixed label per MP, so ballots are preferred.
    b = pd.read_csv(CFG.DATA_DIR / "ballots_2020-2023.csv", keep_default_na=False)
    v = pd.read_csv(CFG.DATA_DIR / "votes_2020-2023.csv", parse_dates=["meeting_date"])
    bp = b.merge(v[["vote_id", "meeting_date"]], on="vote_id").dropna(subset=["meeting_date"])
    bp = bp.groupby(["mp_id", "meeting_date"]).party.agg(lambda s: s.value_counts().idxmax()).reset_index()
    bp["period"] = [R.period(d) for d in bp.meeting_date]
    md = x[["mp_id", "meeting_date", "period", "political_party_affiliation"]].drop_duplicates(["mp_id", "meeting_date"]).reset_index(drop=True)
    same = md.merge(bp[["mp_id", "meeting_date", "party"]], on=["mp_id", "meeting_date"], how="left").party
    near = []
    for r in md.itertuples(index=False):
        c = bp[(bp.mp_id == r.mp_id) & (bp.period == r.period)]
        near.append(None if c.empty else c.iloc[int(np.argmin(np.abs((c.meeting_date - r.meeting_date).dt.days.to_numpy())))].party)
    lab = [party_on(l_, d) for l_, d in zip(md.political_party_affiliation, md.meeting_date)]
    md["party_on_date"] = [s if isinstance(s, str) else (n if n is not None else l_) for s, n, l_ in zip(same, near, lab)]
    md["party_source"] = ["ballot same day" if isinstance(s, str) else ("ballot nearest date, same period" if n is not None else "FPSC label")
                          for s, n in zip(same, near)]
    cross = {"mp_days": int(len(md)), "party_source_counts": md.party_source.value_counts().to_dict(),
             "fpsc_label_differs_from_ballot_party_mp_days": int(sum(1 for p_, l_, s_ in zip(md.party_on_date, lab, md.party_source)
                                                                     if s_ != "FPSC label" and p_ != l_)),
             "differences": [{"mp_id": r.mp_id, "date": str(r.meeting_date.date()), "ballot_party": r.party_on_date, "fpsc_label": l_}
                             for r, l_ in zip(md.itertuples(index=False), lab) if r.party_source != "FPSC label" and r.party_on_date != l_]}
    pm = {(m, d): p_ for m, d, p_ in zip(md.mp_id, md.meeting_date, md.party_on_date)}
    x["party_on_date"] = [pm[(m, d)] for m, d in zip(x.mp_id, x.meeting_date)]
    x["bloc"] = [R.side(p_, d) if isinstance(p_, str) and p_ in R.PARTIES else None for p_, d in zip(x.party_on_date, x.meeting_date)]
    # observed successions (validated definition) with sitting and blocs
    rec, D = [], Counter()
    for mid, g in x.groupby("meeting_id", sort=True):
        mp, bl, pa = g.mp_id.to_numpy(), g.bloc.to_numpy(dtype=object), g.party_on_date.to_numpy(dtype=object)
        for i in range(len(mp) - 1):
            if mp[i] != mp[i + 1]:
                D[(mp[i], mp[i + 1])] += 1
                rec.append((mid, g.meeting_date.iloc[0], g.period.iloc[0], mp[i], mp[i + 1], bl[i], bl[i + 1], pa[i], pa[i + 1]))
    S = pd.DataFrame(rec, columns=["meeting_id", "meeting_date", "period", "source", "target", "bloc_source", "bloc_target", "party_source", "party_target"])
    saved = pd.read_csv(CFG.SPEAKING / "directed_edges_chair_excluded.csv")
    check(sum(D.values()) == int(saved.weight.sum()) and {(s, t): w for s, t, w in zip(saved.source, saved.target, saved.weight)} == dict(D),
          "successions reproduce the chair-excluded directed edges of speaking_order.py exactly")
    S["has_bloc"] = S.bloc_source.notna() & S.bloc_target.notna()
    dropped = S[~S.has_bloc]
    def reason(r):
        out_ = []
        for b_, p_ in ((r.bloc_source, r.party_source), (r.bloc_target, r.party_target)):
            if pd.isna(b_):
                out_.append("undetermined party (switch label, Nov 2022)" if pd.isna(p_) else f"no formal bloc ({p_})")
        return " + ".join(out_)
    drop_info = {"successions_total": int(len(S)), "dropped_no_bloc": int(len(dropped)),
                 "dropped_by_reason": {k: int(n) for k, n in Counter(reason(r) for r in dropped.itertuples()).items()}}
    K = S[S.has_bloc]
    code = {"government": "G", "opposition": "O"}
    K = K.assign(dir=[code[s] + code[t] for s, t in zip(K.bloc_source, K.bloc_target)])
    K.to_csv(out / "successions_with_blocs.csv", index=False)
    # null: shuffle the order of the non-chairman rows within each sitting
    rng = np.random.default_rng(SEED)
    keys = ["GG", "GO", "OG", "OO"]
    obs, null = {}, {}
    for mid, g in x.groupby("meeting_id", sort=True):
        per = g.period.iloc[0]
        mp = pd.factorize(g.mp_id)[0]; bl = np.array([-1 if pd.isna(s) else (0 if s == "government" else 1) for s in g.bloc])
        n = len(mp)
        if n < 2:
            continue
        idx = np.argsort(rng.random((N_SHUF, n)), axis=1)
        M, B = mp[idx], bl[idx]
        ok = (M[:, :-1] != M[:, 1:]) & (B[:, :-1] >= 0) & (B[:, 1:] >= 0)
        cnt = np.stack([(ok & (B[:, :-1] == s) & (B[:, 1:] == t)).sum(1) for s, t in ((0, 0), (0, 1), (1, 0), (1, 1))], 1)
        ok0 = (mp[:-1] != mp[1:]) & (bl[:-1] >= 0) & (bl[1:] >= 0)
        c0 = np.array([(ok0 & (bl[:-1] == s) & (bl[1:] == t)).sum() for s, t in ((0, 0), (0, 1), (1, 0), (1, 1))])
        for sc in ("pooled", per):
            obs[sc] = obs.get(sc, 0) + c0; null[sc] = null.get(sc, 0) + cnt
    res = {}
    for sc in ["pooled", "P1", "P2", "P3"]:
        if sc not in obs:
            continue
        o = obs[sc]; nl = null[sc]; tot = o.sum()
        check(tot == int((K.period == sc).sum() if sc != "pooled" else len(K)), f"{sc}: observed counts equal the succession table")
        cr = (o[1] + o[2]) / tot; ncr = (nl[:, 1] + nl[:, 2]) / nl.sum(1)
        rows_sc = x if sc == "pooled" else x[x.period == sc]
        rb = rows_sc.bloc.dropna(); pg = float((rb == "government").mean()); po = 1 - pg
        res[sc] = {"sittings": int(rows_sc.meeting_id.nunique()), "successions": int(tot), "crossing_share": float(cr),
                   "directed_shares": {k: float(o[i] / tot) for i, k in enumerate(keys)}, "directed_counts": {k: int(o[i]) for i, k in enumerate(keys)},
                   "null_within_sitting_shuffle": {"n_shuffles": N_SHUF, "seed": SEED, "mean_crossing": float(ncr.mean()), "sd": float(ncr.std(ddof=1)),
                                                   "p2.5": float(np.percentile(ncr, 2.5)), "p97.5": float(np.percentile(ncr, 97.5)),
                                                   "p_upper_more_crossing": float((1 + (ncr >= cr - 1e-12).sum()) / (N_SHUF + 1)),
                                                   "mean_directed_shares": {k: float((nl[:, i] / nl.sum(1)).mean()) for i, k in enumerate(keys)}},
                   "independence": {"definition": "bloc shares of the bloc-assigned non-chairman rows: crossing = 2 pG pO; directed = pS pT",
                                    "share_government_rows": pg, "crossing": 2 * pg * po,
                                    "directed": {"GG": pg * pg, "GO": pg * po, "OG": po * pg, "OO": po * po}}}
    result = {"definition": __doc__.strip().splitlines()[0], "rules": {"party": "ballot party that day; else ballot party on the nearest ballot date in the same period; else FPSC label "
              "(switch labels: old party before 1 Nov 2022, new party after 30 Nov 2022, undetermined in November 2022)", "bloc": "formal side on the sitting date (roles.py)"},
              "drops": drop_info, "party_label_vs_ballot_party": cross, "results": res, "checks": CHECKS, "seed": SEED,
              "inputs": ["data/fpsc/fpsc_train_metadata.parquet", SAVED_EDGES, "data/ballots_2020-2023.csv", "data/votes_2020-2023.csv"]}
    (out / "alternation_results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"alternation written; {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
