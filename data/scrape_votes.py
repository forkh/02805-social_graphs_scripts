#!/usr/bin/env python3
"""
Scrape all parliamentary votes (atkvøðugreiðslur) from logting.fo for a range
of years and store them in a tidy, FPSC-compatible format.

Endpoints used (public Umbraco Surface controllers, same ones the site's own
AJAX uses):
  GET /umbraco/Surface/MacroSurface/GetCasesByYearAndType?SelectedYear={y}&SelectedType={t}
      -> HTML table of cases: /mal/mal/?id={caseId}, number (LM-xxx/yyyy), title, committee, status
  GET /mal/mal/?id={caseId}
      -> case page: type ("Slag"), timeline events (label -> data-date)
  GET /umbraco/Surface/CaseSurface/GetCaseVotes?caseId={caseId}
      -> votes of the case: /mal/atkvodugreidsla/?id={voteId} + result tally + stage
  GET /umbraco/Surface/CaseSurface/GetCaseProtocols?caseId={caseId}
      -> protocols ("gerðabøkur") of the case: /mal/yvirlit/gerdabokur/gerdabok/?id={protocolId}, name, date
  GET /umbraco/Surface/MacroSurface/GetVoteDetail/{voteId}
      -> per-member ballots: aktorur/?id={actorId}, name, party, vote (Fyri/Ímóti/...)

Join keys with the FPSC speech dataset (davidilag/FPSC):
  actor_id  <-> FPSC mp_url ".../aktorur/?id={actor_id}"  (we also emit mp_id = "MPID%04d" like FPSC)
  meeting_id (gerðabók/protocol id) + date <-> FPSC meeting_id / url ".../gerdabok/?id={meeting_id}" / date

Output (in --out, default ../data):
  votes_2020-2023.parquet/.csv   one row per vote
  ballots_2020-2023.parquet/.csv one row per (vote, member)
  cases_2020-2023.parquet/.csv   one row per case
  members_2020-2023.csv          actor_id <-> mp_id <-> name <-> party
  scrape_log.json                run statistics and failures
  raw/                           cached raw HTML responses (reproducibility/resume)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.logting.fo"
UA = "DTU-02805-social-graphs-report (student project; contact: dtu-02805@student.dk)"

# SelectedType -> name (verified against the site's own list pages)
CASE_TYPES = {
    1: "Lógaruppskot",
    2: "Uppskot til samtyktar",
    3: "Ríkislógartilmæli",
    4: "Fíggjarlógaruppskot",
    5: "Skrivligir fyrispurningar",
    6: "Munnligir fyrispurningar",
    7: "Ófráboðaðir fyrispurningar",
    8: "Spurningar §52a",
    9: "Frágreiðingar",
    11: "Tingskipanaruppskot",
    12: "Misálit",
}
# Types that can go to a vote (questions/fyrispurningar never do)
VOTABLE_TYPES = [1, 2, 3, 4, 9, 11, 12]

RESULT_RE = re.compile(r"\((\d+)\s*-\s*(\d+)\s*-\s*(\d+)\)")


def clean(s):
    """Collapse whitespace (HTML often has double spaces inside names)."""
    return re.sub(r"\s+", " ", s or "").strip()


def parse_date_fo(s):
    """'28.05.2026' -> '2026-05-28' (None if unparseable)."""
    m = re.match(r"\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*$", s or "")
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


class Scraper:
    def __init__(self, out_dir, delay, refetch):
        self.out = Path(out_dir)
        self.raw = self.out / "raw"
        self.delay = delay
        self.refetch = refetch
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.log = {"requests": 0, "errors": [], "cases": 0, "cases_with_votes": 0,
                    "votes": 0, "ballots": 0}

    def _cache_path(self, kind, name):
        p = self.raw / kind / f"{name}.html"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get(self, url, kind, name):
        """GET with raw-HTML cache; returns text or None on failure."""
        cp = self._cache_path(kind, name)
        if cp.exists() and not self.refetch:
            return cp.read_text(encoding="utf-8", errors="replace")
        for attempt in range(4):
            try:
                r = self.s.get(url, timeout=30)
                self.log["requests"] += 1
                if r.status_code == 200:
                    text = r.text
                    cp.write_text(text, encoding="utf-8")
                    time.sleep(self.delay)
                    return text
                self.log["errors"].append(f"{url} -> HTTP {r.status_code}")
            except requests.RequestException as e:
                self.log["errors"].append(f"{url} -> {e}")
            time.sleep(self.delay * (attempt + 2))
        return None

    # ---------- parsers ----------

    @staticmethod
    def parse_case_list(html):
        """Rows of GetCasesByYearAndType -> list of dicts."""
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for tr in soup.select("tbody tr"):
            a = tr.select_one('a[href*="/mal/mal/?id="]')
            if not a:
                continue
            case_id = a["href"].split("id=")[-1]
            tds = tr.find_all("td")
            out.append({
                "case_id": case_id,
                "case_number": clean(a.get_text(" ")),
                "case_title": clean(tds[1].get_text(" ")) if len(tds) > 1 else "",
                "committee": clean(tds[2].get_text(" ")) if len(tds) > 2 else "",
                "status": clean(tds[3].get_text(" ")) if len(tds) > 3 else "",
            })
        return out

    @staticmethod
    def parse_case_votes(html):
        """GetCaseVotes -> list of votes with stage/subject/result."""
        if not html or html.lstrip().startswith('{"error"'):
            return []
        soup = BeautifulSoup(html, "html.parser")
        votes, stage = [], ""
        for row in soup.select("div.row.mb-0"):
            b = row.find("b")
            if b:
                m = re.search(r"\(([^)]+)\)", row.get_text(" "))
                if m:
                    stage = clean(m.group(1))
                continue
            a = row.select_one('a[href*="atkvodugreidsla/?id="]')
            if not a:
                continue
            vote_id = a["href"].split("id=")[-1]
            cols = row.find_all("div", class_="col")
            subject = clean(cols[0].get_text(" ")) if cols else ""
            result = clean(a.get_text(" "))
            votes.append({"vote_id": vote_id, "stage": stage, "subject": subject,
                          "result_text": result})
        return votes

    @staticmethod
    def parse_case_page(html):
        """Case page -> type label + timeline (label -> fo date)."""
        soup = BeautifulSoup(html, "html.parser")
        h5 = soup.select_one("h5.pb-5")
        head = clean(h5.get_text(" ")) if h5 else ""
        case_number, case_title = (head.split(" : ", 1) + [""])[:2] if " : " in head else (head, "")
        slag = ""
        for h6 in soup.find_all("h6"):
            if h6.get_text(strip=True) == "Slag":
                row = h6.find_parent(class_="row")
                if row:
                    cols = row.find_all("div", class_="col-8")
                    if cols:
                        slag = clean(cols[0].get_text(" "))
        timeline = {}
        for li in soup.select("li.event"):
            b = li.find("b")
            d = li.get("data-date")
            if b and d:
                timeline.setdefault(clean(b.get_text(" ")), d.strip())
        return {"case_number": case_number, "case_title": case_title,
                "case_type_label": slag, "timeline": timeline}

    @staticmethod
    def parse_protocols(html):
        """GetCaseProtocols -> list of {protocol_id, name, date_fo}."""
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for row in soup.select("div.row.mb-0"):
            a = row.select_one('a[href*="gerdabok/?id="]')
            if not a:
                continue
            pid = a["href"].split("id=")[-1]
            cols = row.find_all("div", class_="col")
            date = clean(cols[-1].get_text(" ")) if len(cols) > 1 else ""
            out.append({"protocol_id": pid, "name": clean(a.get_text(" ")), "date_fo": date})
        return out

    @staticmethod
    def parse_vote_detail(html):
        """GetVoteDetail -> {title, stage, result, ballots:[{actor_id,name,party,vote}]}."""
        soup = BeautifulSoup(html or "", "html.parser")
        title = ""
        h5 = soup.select_one("h5.pb-5")
        if h5:
            title = clean(h5.get_text(" "))
        stage, result = "", ""
        for tr in soup.select("tr"):
            th = tr.find("th")
            if not th:
                continue
            label = th.get_text(strip=True)
            td = tr.find("td")
            if not td:
                continue
            if label == "Atkvøða greidd í":
                stage = clean(td.get_text(" "))
            elif label == "Úrslit":
                result = clean(td.get_text(" "))
        ballots = []
        for row in soup.select("div.row.mb-2"):
            a = row.select_one('a[href*="aktorur/?id="]')
            if not a:
                continue
            actor_id = a["href"].split("id=")[-1]
            name = clean(a.get_text(" "))
            small = row.find("small")
            party = clean(small.get_text(" ")) if small else ""
            col = row.select_one("div.col-md-2 div")
            vote = clean(col.get_text(" ")) if col else ""
            ballots.append({"actor_id": int(actor_id), "name": name,
                            "party": party, "vote": vote})
        return {"title": title, "stage": stage, "result": result, "ballots": ballots}

    # ---------- pipeline ----------

    def run(self, years, types):
        import pandas as pd
        cases_rows, votes_rows, ballots_rows = [], [], []
        for year in years:
            for t in types:
                tname = CASE_TYPES.get(t, f"type{t}")
                url = f"{BASE}/umbraco/Surface/MacroSurface/GetCasesByYearAndType?SelectedYear={year}&SelectedType={t}"
                html = self.get(url, "case_lists", f"{year}_{t}")
                if html is None:
                    continue
                cases = self.parse_case_list(html)
                self.log["cases"] += len(cases)
                print(f"[{year}] type {t} ({tname}): {len(cases)} cases", flush=True)
                for c in cases:
                    cid = c["case_id"]
                    c.update({"case_type_id": t, "case_type": tname, "year": year,
                              "url": f"{BASE}/mal/mal/?id={cid}"})
                    cases_rows.append(c)
                    vhtml = self.get(f"{BASE}/umbraco/Surface/CaseSurface/GetCaseVotes?caseId={cid}",
                                     "case_votes", cid)
                    votes = self.parse_case_votes(vhtml)
                    if not votes:
                        continue
                    self.log["cases_with_votes"] += 1
                    # case page (timeline) + protocols (meeting ids/dates)
                    page = self.parse_case_page(self.get(f"{BASE}/mal/mal/?id={cid}", "case_pages", cid) or "")
                    if page["case_number"]:
                        c["case_number"] = page["case_number"]
                    if page["case_title"]:
                        c["case_title"] = page["case_title"]
                    if page["case_type_label"]:
                        c["case_type"] = page["case_type_label"]
                    timeline = page["timeline"]
                    protocols = self.parse_protocols(
                        self.get(f"{BASE}/umbraco/Surface/CaseSurface/GetCaseProtocols?caseId={cid}",
                                 "case_protocols", cid))
                    # protocol lookup: fo-date -> plenary protocol ("Tingfundur ...")
                    plenary = {p["date_fo"]: p for p in protocols
                               if p["name"].startswith("Tingfundur")}
                    for v in votes:
                        vid = v["vote_id"]
                        dhtml = self.get(f"{BASE}/umbraco/Surface/MacroSurface/GetVoteDetail/{vid}",
                                         "vote_details", vid)
                        detail = self.parse_vote_detail(dhtml)
                        stage = detail["stage"] or v["stage"]
                        result_text = detail["result"] or v["result_text"]
                        m = RESULT_RE.search(result_text)
                        n_for, n_mid, n_against = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
                        # match vote stage -> timeline date -> plenary protocol (meeting)
                        date_fo = timeline.get(stage)
                        proto = plenary.get(date_fo) if date_fo else None
                        meeting_id = int(proto["protocol_id"]) if proto else None
                        meeting_date = parse_date_fo(proto["date_fo"]) if proto else (
                            parse_date_fo(date_fo) if date_fo else None)
                        votes_rows.append({
                            "vote_id": int(vid), "case_id": int(cid),
                            "case_number": c["case_number"], "case_title": c["case_title"],
                            "case_type": c["case_type"], "year": year,
                            "subject": v["subject"], "vote_title": detail["title"],
                            "stage": stage, "meeting_id": meeting_id,
                            "meeting_date": meeting_date,
                            "protocol_name": proto["name"] if proto else None,
                            "result": result_text, "n_for": n_for, "n_middle": n_mid,
                            "n_against": n_against,
                            "url": f"{BASE}/mal/atkvodugreidsla/?id={vid}",
                        })
                        self.log["votes"] += 1
                        for b in detail["ballots"]:
                            ballots_rows.append({
                                "vote_id": int(vid), "case_id": int(cid),
                                "mp_id": f"MPID{b['actor_id']:06d}",
                                "actor_id": b["actor_id"], "name": b["name"],
                                "party": b["party"], "vote": b["vote"],
                            })
                        self.log["ballots"] += len(detail["ballots"])
                print(f"  running totals: cases={self.log['cases']} "
                      f"with_votes={self.log['cases_with_votes']} votes={self.log['votes']} "
                      f"ballots={self.log['ballots']}", flush=True)

        cases_df = pd.DataFrame(cases_rows)
        votes_df = pd.DataFrame(votes_rows)
        ballots_df = pd.DataFrame(ballots_rows)
        members_df = (ballots_df.groupby(["actor_id", "mp_id", "name"], as_index=False)
                      .agg(parties=("party", lambda s: " | ".join(sorted(set(s)))),
                           n_ballots=("vote_id", "count")))
        members_df = members_df.sort_values("actor_id")

        stem = f"{min(years)}-{max(years)}"
        for df, name in [(cases_df, f"cases_{stem}"), (votes_df, f"votes_{stem}"),
                         (ballots_df, f"ballots_{stem}")]:
            df.to_parquet(self.out / f"{name}.parquet", index=False)
            df.to_csv(self.out / f"{name}.csv", index=False)
        members_df.to_csv(self.out / f"members_{stem}.csv", index=False)
        members_df.to_parquet(self.out / f"members_{stem}.parquet", index=False)

        self.log["years"] = years
        self.log["types"] = types
        (self.out / "scrape_log.json").write_text(json.dumps(self.log, indent=2, default=str))
        print("\nDONE", flush=True)
        print(f"cases={len(cases_df)} votes={len(votes_df)} ballots={len(ballots_df)} "
              f"members={len(members_df)} requests={self.log['requests']} "
              f"errors={len(self.log['errors'])}")
        return cases_df, votes_df, ballots_df, members_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2024)))
    ap.add_argument("--types", type=int, nargs="+", default=VOTABLE_TYPES)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between requests")
    ap.add_argument("--refetch", action="store_true", help="ignore raw cache")
    args = ap.parse_args()
    Scraper(args.out, args.delay, args.refetch).run(args.years, args.types)


if __name__ == "__main__":
    main()
