"""Embedding-stance analysis on the GPU (leave-one-out pro/con axis).

Port of the original models/embedding_stance_rocm/scripts/analyze_rocm.py —
the torch ops (index_add_, matmul, norm) are device-agnostic and run on the GPU
through ROCm's HIP or CUDA implementation of the torch.cuda API, so
device="cuda" is kept.

For each contested vote, in embedding space:
  voters axis : axis = mean(emb of Fyri voters' speeches) - mean(emb of Ímóti
                voters' speeches), rebuilt leave-one-out per ballot.
  lexicon axis: axis = mean(emb of lexicon +1 speeches) - mean(emb of -1),
                anchored on speech_vote_agreement/data/speech_scores.csv.
An MP's mean speech embedding is projected on the unit axis; sign = predicted
stance (Fyri if > 0). Two scopes: case-wide (stage <= vote stage) and
same-meeting.

Usage (from this project folder):
    .venv/bin/python scripts/analyze_rocm.py --model minilm --axis voters
    .venv/bin/python scripts/analyze_rocm.py --model e5large --axis lexicon

Outputs (data/):
  results.jsonl                      one line per (model, axis, scope) run
  embedding_results.txt              human-readable table (appended)
  vote_level_scores_<slug>_<axis>_<scope>.csv
  mp_embedding_agreement_<slug>.csv  (voters axis, same-meeting, joined with lexicon)
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MODELS, DATA_OUT, LINK_DIR, emb_cache_path,
                    load_speeches_by_case, load_votes_ballots,
                    contested_ballots_by_vote, V2Y)


def auc(y, s):
    import scipy.stats as st
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = st.rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_emb_tensor(slug, device):
    z = np.load(emb_cache_path(slug))
    mat = z["matrix"]                      # decompress ONCE
    ids = [str(i) for i in z["ids"]]
    del z
    t = torch.from_numpy(mat).to(device)
    return {i: k for k, i in enumerate(ids)}, t


def load_lexicon_scores():
    """speech_id -> +1/-1/0 from the parent analysis' speech_scores.csv."""
    scores = {}
    p = os.path.join(LINK_DIR, "data", "speech_scores.csv")
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_chairman"] == "1":
                continue
            p_, n_ = int(r["n_pos"]), int(r["n_neg"])
            scores[r["speech_id"]] = 1 if p_ > n_ else (-1 if n_ > p_ else 0)
    return scores


def run_scope(by_case, votes, ballots_by_vote, id2row, emb_t, same_meeting,
              axis_mode, lex_scores, device):
    rows = []
    for vid, bs in ballots_by_vote.items():
        v = votes[vid]
        case = v["case_number"].upper()
        # in-scope speeches of this case, as row indices into emb_t
        rows_idx, lex = [], []
        for sp in by_case.get(case, []):
            if sp["stage"] > v["stage_n"]:
                continue
            if same_meeting and sp["meeting"] != v["meeting_id"]:
                continue
            k = id2row.get(sp["id"])
            if k is None:
                continue
            rows_idx.append(k)
            lex.append(lex_scores.get(sp["id"], 0) if axis_mode == "lexicon" else 0)
        if not rows_idx:
            continue
        E = emb_t[torch.tensor(rows_idx, device=device)]     # n_speech x d

        # per-MP mean embedding via scatter-add on mp codes
        mp_list = []
        code_of = {}
        codes = []
        for sp in by_case.get(case, []):
            if sp["stage"] > v["stage_n"]:
                continue
            if same_meeting and sp["meeting"] != v["meeting_id"]:
                continue
            if sp["id"] not in id2row:
                continue
            if sp["mp"] not in code_of:
                code_of[sp["mp"]] = len(mp_list)
                mp_list.append(sp["mp"])
            codes.append(code_of[sp["mp"]])
        n_mp = len(mp_list)
        codes_t = torch.tensor(codes, device=device)
        d = E.shape[1]
        sums = torch.zeros(n_mp, d, device=device)
        sums.index_add_(0, codes_t, E)
        counts = torch.zeros(n_mp, device=device)
        counts.index_add_(0, codes_t, torch.ones(len(codes), device=device))
        mp_emb = sums / counts.clamp(min=1)[:, None]         # n_mp x d

        if axis_mode == "lexicon":
            lex_t = torch.tensor(lex, device=device)
            pos = E[lex_t == 1]; neg = E[lex_t == -1]
            if len(pos) == 0 or len(neg) == 0:
                continue
            axis = (pos.mean(0) - neg.mean(0))
            nrm = axis.norm()
            if nrm == 0:
                continue
            axis = axis / nrm
            for b in bs:
                c = code_of.get(b["mp_id"])
                if c is None:
                    continue
                s = float(mp_emb[c] @ axis)
                rows.append({"vote_id": vid, "case_number": v["case_number"],
                             "stage": v["stage"], "meeting_date": v["meeting_date"],
                             "mp_id": b["mp_id"], "name": b["name"], "party": b["party"],
                             "vote": b["vote"], "y": V2Y[b["vote"]],
                             "score": s, "pred": 1 if s > 0 else 0})
        else:
            # voter-centroid axis, leave-one-out per ballot MP
            y_of = {}
            for b in bs:
                if b["mp_id"] in code_of:
                    y_of[code_of[b["mp_id"]]] = V2Y[b["vote"]]
            if not y_of:
                continue
            y = torch.zeros(n_mp, device=device)
            has_b = torch.zeros(n_mp, device=device)
            for c, val in y_of.items():
                y[c] = val
                has_b[c] = 1
            # only MPs who actually voted anchor the axis (as in the CPU version)
            is_for = ((y == 1) & (has_b == 1)).float()[:, None]
            is_ag = ((y == 0) & (has_b == 1)).float()[:, None]
            sum_for = (mp_emb * is_for).sum(0)
            n_for = int(is_for.sum())
            sum_ag = (mp_emb * is_ag).sum(0)
            n_ag = int(is_ag.sum())
            for b in bs:
                c = code_of.get(b["mp_id"])
                if c is None:
                    continue
                if y[c] == 1:
                    nf, na = n_for - 1, n_ag
                    mf = (sum_for - mp_emb[c]) / max(nf, 1)
                    ma = sum_ag / max(na, 1)
                else:
                    nf, na = n_for, n_ag - 1
                    mf = sum_for / max(nf, 1)
                    ma = (sum_ag - mp_emb[c]) / max(na, 1)
                if nf == 0 or na == 0:
                    continue
                axis = mf - ma
                nrm = axis.norm()
                if nrm == 0:
                    continue
                s = float(mp_emb[c] @ (axis / nrm))
                rows.append({"vote_id": vid, "case_number": v["case_number"],
                             "stage": v["stage"], "meeting_date": v["meeting_date"],
                             "mp_id": b["mp_id"], "name": b["name"], "party": b["party"],
                             "vote": b["vote"], "y": V2Y[b["vote"]],
                             "score": s, "pred": 1 if s > 0 else 0})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--axis", choices=["voters", "lexicon"], default="voters")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU visible — sync with a GPU extra first "
                 "(uv sync --extra rocm or --extra cuda) and pass the device through "
                 "(--device /dev/kfd --device /dev/dri on ROCm)")
    device = "cuda"  # ROCm torch exposes HIP via the CUDA API
    dev_name = torch.cuda.get_device_name(0)
    slug = args.model
    print(f"torch {torch.__version__} (hip {getattr(torch.version, 'hip', None)}) on "
          f"{dev_name}; model={MODELS[slug]} axis={args.axis}")

    by_case, _ = load_speeches_by_case()
    votes, ballots = load_votes_ballots()
    b2v = contested_ballots_by_vote(votes, ballots)
    id2row, emb_t = load_emb_tensor(slug, device)
    lex_scores = load_lexicon_scores() if args.axis == "lexicon" else {}
    print(f"embeddings on {device}: {tuple(emb_t.shape)}; contested votes: {len(b2v)}")

    t0 = time.time()
    results = []
    for same_meeting in (False, True):
        rows = run_scope(by_case, votes, b2v, id2row, emb_t, same_meeting,
                         args.axis, lex_scores, device)
        y = [r["y"] for r in rows]; s = [r["score"] for r in rows]
        pred = [r["pred"] for r in rows]
        acc = float(np.mean([p == t for p, t in zip(pred, y)])) if rows else float("nan")
        a = auc(y, s)
        label = "same-meeting" if same_meeting else "case-wide"
        print(f"{slug}/{args.axis}/{label}: n={len(rows)} agreement={acc:.3f} auc={a:.3f}")
        results.append({"model": slug, "hf_model": MODELS[slug], "axis": args.axis,
                        "scope": label, "n": len(rows), "agreement": acc, "auc": a})

        csv_path = os.path.join(DATA_OUT, f"vote_level_scores_{slug}_{args.axis}_{label}.csv")
        if rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    # per-MP agreement for the voters axis (joined with lexicon agreement)
    if args.axis == "voters":
        lex = {}
        p = os.path.join(LINK_DIR, "data", "mp_agreement.csv")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    lex[r["mp_id"]] = r
        per_mp = defaultdict(lambda: [0, 0])
        # recompute rows for same-meeting voters axis
        rows = run_scope(by_case, votes, b2v, id2row, emb_t, True, "voters", {}, device)
        for r in rows:
            d = per_mp[r["mp_id"]]
            d[0] += 1; d[1] += (r["pred"] == r["y"])
        mp_rows = []
        for mp_id, (n, a) in per_mp.items():
            l = lex.get(mp_id, {})
            mp_rows.append({"mp_id": mp_id, "n_scored": n, "n_agree": a,
                            "agreement_embedding": f"{a / n:.3f}",
                            "name": l.get("name", ""), "party": l.get("party", ""),
                            "agreement_lexicon_primary": l.get("agreement_primary", ""),
                            "n_scored_lexicon": l.get("n_scored_primary", "")})
        mp_rows.sort(key=lambda r: -float(r["agreement_embedding"]))
        with open(os.path.join(DATA_OUT, f"mp_embedding_agreement_{slug}.csv"),
                  "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mp_rows[0].keys()))
            w.writeheader(); w.writerows(mp_rows)

    # append to results.jsonl + human-readable txt
    os.makedirs(DATA_OUT, exist_ok=True)
    with open(os.path.join(DATA_OUT, "results.jsonl"), "a", encoding="utf-8") as f:
        for r in results:
            r["seconds"] = round(time.time() - t0, 1)
            r["backend"] = f"gpu:{dev_name} (hip {getattr(torch.version, 'hip', None)}, cuda {getattr(torch.version, 'cuda', None)})"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(DATA_OUT, "embedding_results.txt"), "a", encoding="utf-8") as f:
        f.write(f"model={MODELS[slug]} ({slug})  axis={args.axis}  "
                f"device={dev_name} (hip {getattr(torch.version, 'hip', None)}, "
                f"cuda {getattr(torch.version, 'cuda', None)})  "
                f"wall={time.time() - t0:.1f}s\n")
        for r in results:
            f.write(f"  {r['scope']:<12} n={r['n']:<6} agreement={r['agreement']:.3f}  "
                    f"auc={r['auc']:.3f}\n")
        f.write("\n")
    print(f"done in {time.time() - t0:.1f}s; peak GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB; outputs in {DATA_OUT}")


if __name__ == "__main__":
    main()
