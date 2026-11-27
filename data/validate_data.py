#!/usr/bin/env python3
"""
Validate the scraped Løgting voting data (data/*_2020-2023.parquet) and its
joinability with the FPSC speech dataset (data/fpsc/fpsc_train_metadata.parquet).

Writes a human-readable report to data/validation_report.txt and prints a summary.
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
STEM = "2020-2023"

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def main():
    votes = pd.read_parquet(DATA / f"votes_{STEM}.parquet")
    ballots = pd.read_parquet(DATA / f"ballots_{STEM}.parquet")
    cases = pd.read_parquet(DATA / f"cases_{STEM}.parquet")
    members = pd.read_parquet(DATA / f"members_{STEM}.parquet")
    scrape_log = json.loads((DATA / "scrape_log.json").read_text())

    log("=== Voting data validation: logting.fo votes "
        f"{min(votes.year)}-{max(votes.year)} ===")
    log(f"scrape_log: requests={scrape_log['requests']} errors={len(scrape_log['errors'])}")
    for e in scrape_log["errors"][:10]:
        log(f"  error: {e}")

    log(f"\ncases: {len(cases)} | votes: {len(votes)} | ballots: {len(ballots)} | members: {len(members)}")
    log("\nVotes per year/type:")
    log(votes.groupby(["year", "case_type"]).size().rename("votes").reset_index().to_string(index=False))
    log("\nBallots per year:")
    log(ballots.merge(votes[["vote_id", "year"]], on="vote_id")
        .groupby("year").size().rename("ballots").reset_index().to_string(index=False))
    log(f"\nVote labels seen: {sorted(ballots.vote.unique())}")
    log(f"Stages seen: {sorted(votes.stage.dropna().unique())}")

    # 1) tally consistency: parsed tally (n_for/n_middle/n_against) vs actual ballot labels.
    #    Members with empty vote label were absent/did not vote and are NOT part of the tally.
    v = votes.set_index("vote_id")
    fyri = ballots[ballots.vote == "Fyri"].groupby("vote_id").size()
    imoti = ballots[ballots.vote == "Ímóti"].groupby("vote_id").size()
    blankt = ballots[ballots.vote == "Blankt"].groupby("vote_id").size()
    absent = ballots[ballots.vote == ""].groupby("vote_id").size()
    chk = v[["case_number", "result", "n_for", "n_middle", "n_against"]].join(
        fyri.rename("n_fyri")).join(imoti.rename("n_imoti")).join(
        blankt.rename("n_blankt")).join(absent.rename("n_absent")).fillna(0)
    bad = chk[(chk.n_for != chk.n_fyri) | (chk.n_middle != chk.n_blankt) | (chk.n_against != chk.n_imoti)]
    no_breakdown = v[~v.index.isin(ballots.vote_id.unique())]
    log(f"\n[check] n_for==count('Fyri'), n_middle==count('Blankt'), n_against==count('Ímóti'): "
        f"{len(bad)} mismatches out of {len(chk)} votes with ballots")
    if len(bad):
        log(bad.head(15).to_string())
    log(f"[info] votes with tally but NO per-member breakdown on the site: {len(no_breakdown)}")
    if len(no_breakdown):
        log(no_breakdown[["case_number", "result"]].to_string())
    log(f"[info] ballot rows with empty label (absent / did not vote): {int(chk.n_absent.sum())}")

    # 2) uniqueness and id consistency
    dup = ballots.duplicated(subset=["vote_id", "actor_id"]).sum()
    log(f"[check] duplicate (vote_id, actor_id) rows: {dup}")
    bad_mp = ballots[ballots.mp_id != ballots.actor_id.map(lambda a: f"MPID{a:06d}")]
    log(f"[check] mp_id != MPID%06d(actor_id): {len(bad_mp)} rows")

    # 3) meeting/date attachment
    log(f"\nvotes with matched plenary meeting (meeting_id): "
        f"{votes.meeting_id.notna().sum()}/{len(votes)} "
        f"({votes.meeting_id.notna().mean():.1%})")
    log("unmatched stages:")
    log(votes[votes.meeting_id.isna()].groupby("stage").size().rename("n").reset_index().to_string(index=False))

    # 4) join with FPSC metadata parquet
    fpsc_path = DATA / "fpsc" / "fpsc_train_metadata.parquet"
    if fpsc_path.exists():
        f = pd.read_parquet(fpsc_path)
        log(f"\n=== FPSC cross-check ({len(f)} speech rows) ===")
        log(f"FPSC date range: {f.date.min()} .. {f.date.max()} | meetings: {f.meeting_id.nunique()}")
        import unicodedata
        def ascii_norm(s):
            s = " ".join(str(s).split())  # collapse whitespace (FPSC names have double spaces)
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
        f_map = f.drop_duplicates("mp_id").set_index("mp_id")["name"]
        m = members.copy()
        m["in_fpsc"] = m.mp_id.isin(f_map.index)
        m["fpsc_name"] = m.mp_id.map(f_map)
        m["name_match"] = m.in_fpsc & (m.name.map(ascii_norm) == m.fpsc_name.fillna("").map(ascii_norm))
        log(f"our members: {len(m)} | present in FPSC: {m.in_fpsc.sum()} | exact name match (ascii-normalised): {m.name_match.sum()}")
        mism = m[m.in_fpsc & ~m.name_match][["mp_id", "name", "fpsc_name"]]
        if len(mism):
            log("name mismatches (ours vs FPSC):")
            log(mism.head(20).to_string(index=False))
        mv = votes[votes.meeting_id.notna()]
        in_f = mv.meeting_id.isin(f.meeting_id.unique())
        log(f"our votes whose meeting_id exists in FPSC: {in_f.sum()}/{len(mv)}")
        # date agreement for shared meetings
        f_dates = f.drop_duplicates("meeting_id").set_index("meeting_id")["date"]
        chk = mv.assign(fpsc_date=mv.meeting_id.map(f_dates))
        agree = (chk.meeting_date == pd.to_datetime(chk.fpsc_date).dt.strftime("%Y-%m-%d")).sum()
        log(f"meeting_date agrees with FPSC date: {agree}/{len(chk)}")
        # coverage: FPSC meetings in our year range that have votes
        y0, y1 = int(votes.year.min()), int(votes.year.max())
        f_range = f.drop_duplicates("meeting_id")
        f_range = f_range[(pd.to_datetime(f_range.date) >= f"{y0}-01-01") & (pd.to_datetime(f_range.date) <= f"{y1}-12-31")]
        log(f"FPSC meetings in {y0}-{y1}: {f_range.meeting_id.nunique()} | "
            f"of these, present in our votes: {int(f_range.meeting_id.isin(mv.meeting_id.unique()).sum())}")
        log(f"our votes dated after FPSC coverage end ({str(f.date.max())[:10]}): "
            f"{int((pd.to_datetime(votes.meeting_date) > pd.to_datetime(f.date.max())).sum())}")
    else:
        log("\nFPSC parquet not found, skipping cross-check")

    report = "\n".join(lines) + "\n"
    (DATA / "validation_report.txt").write_text(report)
    log(f"\nreport written to {DATA / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
