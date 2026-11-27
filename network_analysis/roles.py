"""Formal government/opposition roles (table given by the project author; not looked up or inferred).

Sources (recorded, not fetched): government.fo announcement of 14 Sep 2019; KVF, 8 Nov 2022 (Miðflokkurin leaves the
coalition); government.fo announcement of 22 Dec 2022.

    P1  14 Sep 2019 - 7 Nov 2022   government: Sambandsflokkurin, Fólkaflokkurin, Miðflokkurin
    P2   8 Nov 2022 - 21 Dec 2022  government: Sambandsflokkurin, Fólkaflokkurin   (Miðflokkurin in opposition from 8 Nov 2022)
    P3  from 22 Dec 2022           government: Javnaðarflokkurin, Tjóðveldi, Framsókn

Each ballot gets the side of its party (party at the time of the vote, ballots.party) on the division date.
"Side group" X/Y is the stable split used for MP-level permutations: X = the side Sambandsflokkurin is on that date
(government in P1/P2, opposition in P3), Y = the other side.
"""
from __future__ import annotations

import pandas as pd

GOV, OPP = "government", "opposition"
P1_START, P2_START, P3_START = pd.Timestamp("2019-09-14"), pd.Timestamp("2022-11-08"), pd.Timestamp("2022-12-22")
PARTIES = ["Javnaðarflokkurin", "Tjóðveldi", "Sambandsflokkurin", "Fólkaflokkurin", "Framsókn", "Miðflokkurin", "Sjálvstýri"]
TABLE = {
    "P1": {"government": {"Sambandsflokkurin", "Fólkaflokkurin", "Miðflokkurin"},
           "opposition": {"Javnaðarflokkurin", "Tjóðveldi", "Framsókn", "Sjálvstýri"}},
    "P2": {"government": {"Sambandsflokkurin", "Fólkaflokkurin"},
           "opposition": {"Miðflokkurin", "Javnaðarflokkurin", "Tjóðveldi", "Framsókn", "Sjálvstýri"}},
    "P3": {"government": {"Javnaðarflokkurin", "Tjóðveldi", "Framsókn"},
           "opposition": {"Sambandsflokkurin", "Fólkaflokkurin", "Miðflokkurin"}},
}
PERIOD_TEXT = {"P1": "14 Sep 2019 - 7 Nov 2022", "P2": "8 Nov 2022 - 21 Dec 2022", "P3": "from 22 Dec 2022"}
RULES = ["Miðflokkurin in opposition from 8 Nov 2022 (P2)",
         "P2 (3 divisions) is kept in pooled pair types and pooled side dissent, and excluded from per-period tests",
         "Sjálvstýri is not listed for P3; it has no ballots in P3 (asserted)"]


def period(date) -> str | None:
    if pd.isna(date):
        return None
    d = pd.Timestamp(date)
    if d < P1_START:
        raise ValueError(f"date {d.date()} before the role table")
    return "P1" if d < P2_START else ("P2" if d < P3_START else "P3")


def side(party: str, date) -> str:
    p = period(date)
    if p is None:
        raise ValueError("undated division")
    for s in (GOV, OPP):
        if party in TABLE[p][s]:
            return s
    raise ValueError(f"party {party!r} not in the role table for {p}")


def side_group(party: str, date) -> str:
    """X = same side as Sambandsflokkurin on that date, Y = the other side."""
    return "X" if side(party, date) == side("Sambandsflokkurin", date) else "Y"


def annotate(ballots: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    """ballots + meeting_date, period, side, side_group; drops undated divisions; asserts full coverage."""
    x = ballots.merge(votes[["vote_id", "meeting_date"]], on="vote_id", how="left", validate="many_to_one")
    x = x[x.meeting_date.notna()].copy()
    x["period"] = [period(d) for d in x.meeting_date]
    key = x[["party", "period"]].drop_duplicates()
    sd = {(p, q): s for p, q in key.itertuples(index=False) for s in (GOV, OPP) if p in TABLE[q][s]}
    missing = [(p, q) for p, q in key.itertuples(index=False) if (p, q) not in sd]
    assert not missing, f"party/period not covered by the role table: {missing}"
    x["side"] = [sd[(p, q)] for p, q in zip(x.party, x.period)]
    sam = {q: next(s for s in (GOV, OPP) if "Sambandsflokkurin" in TABLE[q][s]) for q in TABLE}
    x["side_group"] = ["X" if s == sam[q] else "Y" for s, q in zip(x.side, x.period)]
    return x
