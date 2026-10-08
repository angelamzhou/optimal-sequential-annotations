"""Freeze the session manifest for the LMArena 140k real-dynamics T=2 design.

Mirrors redial_full_design_v1/build_manifest.py. Nothing here summarises a reward
conditional on an action, a fold or a model. Per-week behaviour propensities are fit
from TRAIN only. Folds are assigned by hash of the session id and are therefore
SESSION-disjoint, not user-disjoint: the release carries no user identifier. State
this every time the folds are described.

Unit    one evaluation_session_id whose vote 1 is present in the release.
Z_1     1{a vote with evaluation_order == 2 is present}  -- ALWAYS OBSERVED (metadata),
        so it is F_0-measurable. Absorbing state when Z_1 = 0: R_2 == 0, no stage-2 label
        exists to buy.
A_t     frozen provenance partition (design.md Appendix A) via screen_arena.pair_category.
R_t     1{winner != both_bad}.
"""
import os, sys, json, hashlib
from collections import Counter, defaultdict
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena_paths import DATA  # noqa: E402
import screen_arena as S            # load(): dedup (15 rows), week, frozen pair, R

SEED = 20260914
PARQUET = os.path.join(DATA, "arena140k.parquet")
CATS = ("closed-closed", "mixed", "open-open")

def fold_of(sid):
    u = int(hashlib.sha256(f"{SEED}|{sid}".encode()).hexdigest()[:12], 16) / 16 ** 12
    return "train" if u < .6 else ("dev" if u < .8 else "test")

def main():
    df = S.load(PARQUET)
    n_votes = len(df)
    orders = df.groupby("evaluation_session_id").evaluation_order.apply(set)
    has1 = orders[orders.map(lambda v: 1 in v)].index
    skipped = {"duplicate_session_order_rows": int(df.attrs.get("n_dropped_dup_order", 0)),
               "sessions_without_vote_1": int(len(orders) - len(has1))}
    s1 = df[(df.evaluation_order == 1) & df.evaluation_session_id.isin(set(has1))] \
           .set_index("evaluation_session_id")
    s2 = df[df.evaluation_order == 2].set_index("evaluation_session_id")
    s2 = s2.reindex(s1.index)                                    # NaN where Z_1 = 0
    z1 = s2["id"].notna()

    rows = []
    for sid, r in s1.iterrows():
        z = bool(z1.loc[sid]); q = s2.loc[sid]
        rows.append({
            "episode_id": sid, "fold": fold_of(sid), "week": r["week"],
            "ts1": r["ts"].isoformat(), "Z1": int(z),
            "id1": r["id"], "id2": (q["id"] if z else None),
            "actions": [r["pair"], (q["pair"] if z else None)],
            "models1": [r["model_a"], r["model_b"]],
            "models2": ([q["model_a"], q["model_b"]] if z else None),
            "winner1": r["winner"], "winner2": (q["winner"] if z else None),
            "reward1": int(r["R"]), "reward2": (int(q["R"]) if z else 0),
            "language": r["language"], "is_code": bool(r["is_code"]),
        })

    # behaviour policy: per-week pair frequencies over ALL votes of TRAIN sessions, both stages
    tr = {p["episode_id"] for p in rows if p["fold"] == "train"}
    trv = df[df.evaluation_session_id.isin(tr)]
    pib = (trv.groupby(["week", "pair"]).size() / trv.groupby("week").size()) \
            .unstack().reindex(columns=CATS).fillna(0.0)
    pi_b = {wk: {c: float(pib.loc[wk, c]) for c in CATS} for wk in pib.index}

    live = rows
    cells = Counter((p["fold"], p["week"], p["actions"][0]) for p in live)
    all_cells = [(f, w, c) for f in ("train", "dev", "test") for w in pib.index for c in CATS]
    summary = {
        "seed": SEED, "votes_in_release_after_dedup": int(n_votes),
        "sessions_total": int(len(orders)), "skipped": skipped,
        "eligible_sessions": len(live),
        "fold_rule": "sha256(seed|session_id) -> 60/20/20. SESSION-disjoint; the release "
                     "has no user id, so folds are NOT user-disjoint.",
        "sessions_by_fold": dict(Counter(p["fold"] for p in live)),
        "votes_by_fold": {f: int(sum(1 + p["Z1"] for p in live if p["fold"] == f))
                          for f in ("train", "dev", "test")},
        "Z1_sessions_by_fold": {f: int(sum(p["Z1"] for p in live if p["fold"] == f))
                                for f in ("train", "dev", "test")},
        "weeks": list(pib.index), "n_weeks": int(len(pib)),
        "pi_b_fit_on": "train fold, all votes, both stages",
        "pi_b_min_probability": float(pib.replace(0, np.nan).min().min()),
        "pi_b_by_week": pi_b,
        "cells_fold_week_pair1_min_count": int(min(cells.get(k, 0) for k in all_cells)),
        "empty_cells": [f"{f}/{w}/{c}" for (f, w, c) in all_cells if cells.get((f, w, c), 0) == 0],
        "dedup_rule": "duplicate (session, order) keys keep the earliest timestamp (screen_arena.load)",
        "note": "No reward is summarised by action, fold or model here. reward1/reward2 are "
                "stored per unit exactly as ReDial's manifest stores per-packet reward.",
    }
    json.dump(live, open(os.path.join(DATA, "arena_session_manifest.json"), "w"))
    json.dump(summary, open(os.path.join(DATA, "arena_manifest_summary.json"), "w"), indent=1)
    show = {k: v for k, v in summary.items() if k != "pi_b_by_week"}
    show["pi_b_by_week_first_two"] = dict(list(pi_b.items())[:2])
    print(json.dumps(show, indent=1))

if __name__ == "__main__":
    main()
