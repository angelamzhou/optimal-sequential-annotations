"""Coauthor-facing table: base vs June FT vs July FT, on the annotated probability sample.

Every labeler is scored on the SAME annotated notes with the SAME design weights, so all
differences are labeler disagreement -- no extrapolation or support assumptions.

Two things this script fixes relative to design_weighted_labeler_eval.py:

  1. LEAKAGE. 423 of the annotated notes are in the July FT's training split. That model is
     scored on HELD-OUT notes only (train ContactIDs dropped). The leaked number is printed
     alongside, clearly marked, so the size of the leak is visible rather than hidden.
  2. BASE AND JUNE FT. Those generations live in the 180k-row corpus keyed by clientID + note
     text, with no ContactID. They are linked to ContactID by (ClientID, sha1(note)) in order
     of occurrence -- the same linkage used to recover the gold ContactIDs.

Held-out weights are RE-NORMALIZED within each (band x tier) cell after dropping train notes,
so the design weighting stays valid on the reduced sample.

Prints aggregates only. Never echoes note text.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# All inputs are protected; paths below are relative to PROTECTED_DATA_ROOT (default: current directory).
REPO = Path(os.environ.get("PROTECTED_DATA_ROOT", ".")).resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from design_weighted_labeler_eval import (  # noqa: E402
    BANDS, RANK, SDIR, TIER_POOL, cohort_population, load_generations, wshares,
)
from safe_ppi_progress_counts import band_silver  # noqa: E402
from progress_surrogate_feasibility import CONTACT_ID, normalize_id_columns  # noqa: E402

FULL_SRC = REPO / "protected_data/casenotes_full.csv"
SPLIT_KEY = REPO / "protected_data/finetune_split/split_key.csv"
DOWNLOADS = REPO / "protected_data" / "llm_generations"
GEMMA_FILES = {
    "gemma_base": DOWNLOADS / "full-gemma3-base_merged_1a_labels.csv",
    "gemma_june_ft": DOWNLOADS / "full-gemma3-ft_merged_1a_labels.csv",
}
SAFE = REPO / "outputs" / "safe" / "ppi_progress"
N_BOOT = 2000
SEED = 20260713


def _hash(s) -> str:
    return hashlib.sha1(str(s).encode("utf8", "ignore")).hexdigest()


def _occurrence_key(client_ids, notes) -> list[str]:
    """(ClientID, note-hash, nth occurrence) -- unique even when a note repeats for a client."""
    seen: dict[str, int] = defaultdict(int)
    out = []
    for c, n in zip(client_ids, notes):
        k = f"{c}|{_hash(n)}"
        out.append(f"{k}|{seen[k]}")
        seen[k] += 1
    return out


def load_gemma_generations() -> dict[str, pd.DataFrame]:
    """Link the 180k-corpus gemma files (clientID + note, no ContactID) to ContactID."""
    full = pd.read_csv(FULL_SRC)
    full["k"] = _occurrence_key(full.ClientID, full.Notes)
    k2contact = dict(zip(full.k, full.ContactID.astype(str)))

    gens = {}
    for name, path in GEMMA_FILES.items():
        df = pd.read_csv(path, usecols=["clientID", "note", "primary_label"])
        df["k"] = _occurrence_key(df.clientID, df.note)
        df[CONTACT_ID] = df.k.map(k2contact)
        hit = df[CONTACT_ID].notna().mean()
        df = df[df[CONTACT_ID].notna()][[CONTACT_ID, "primary_label"]].rename(
            columns={"primary_label": name})
        df = normalize_id_columns(df, [CONTACT_ID]).drop_duplicates(CONTACT_ID)
        print(f"  {name:14s} linked to ContactID: {hit:.4%} of rows  ({len(df):,} unique)")
        gens[name] = df
    return gens


WORKBOOKS = [
    "annotation_workbook_main.csv",
    "annotation_workbook_clean_eval-annotated.csv",
    "annotation_workbook_clean_eval_batch2-annotated.csv",
]


def annotated_sample(gens, pop) -> pd.DataFrame:
    """All annotated notes from the SAME 1000-note designed draw, pooled across workbooks.

    The two clean-eval batches were annotated later but were drawn from the same
    sample_design.csv, so they carry the same strata and belong in the same estimator.
    (design_weighted_labeler_eval.py reads only the main workbook and so sees 397 of 677.)
    """
    labels = []
    for f in WORKBOOKS:
        wb = pd.read_csv(SDIR / f, dtype=str, keep_default_na=False,
                         usecols=[CONTACT_ID, "progress_label"])
        wb = normalize_id_columns(wb, [CONTACT_ID])
        wb = wb[wb["progress_label"].str.strip() != ""]
        wb["source"] = f
        labels.append(wb)
    wb = pd.concat(labels, ignore_index=True).drop_duplicates(CONTACT_ID, keep="first")

    design = normalize_id_columns(pd.read_csv(SDIR / "sample_design.csv", dtype=str), [CONTACT_ID])
    tiers = normalize_id_columns(pd.read_csv(SDIR / "triage_labels.csv", dtype=str), [CONTACT_ID])
    df = (design[[CONTACT_ID, "silver_band"]]
          .merge(tiers[[CONTACT_ID, "triage_consensus"]], on=CONTACT_ID)
          .merge(wb, on=CONTACT_ID, how="left"))
    df["tier"] = df["triage_consensus"].map(TIER_POOL)
    df["annotated"] = df["progress_label"].fillna("").str.strip() != ""
    df["human_b"] = band_silver(df["progress_label"])
    for name, g in gens.items():
        df = df.merge(g, on=CONTACT_ID, how="left")
        df[name + "_b"] = band_silver(df[name])
    return df


def reweight(ann: pd.DataFrame, pop: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """Design weights re-normalized within (band x tier) cells over the rows in `mask`."""
    W = pop["old_silver_b"].value_counts(normalize=True)
    sub = ann[mask]
    m_ht = sub.groupby(["silver_band", "tier"]).size()
    n_ht = ann.groupby(["silver_band", "tier"]).size()
    n_h = ann.groupby("silver_band").size()
    w = pd.Series(0.0, index=ann.index)
    for (h, t), m in m_ht.items():
        idx = sub.index[(sub["silver_band"] == h) & (sub["tier"] == t)]
        w.loc[idx] = W[h] * (n_ht[(h, t)] / n_h[h]) / m
    return w.to_numpy()


def score(ann, w, col, hr, sub: np.ndarray | None = None) -> dict:
    """Design-weighted agreement. `sub` restricts to a subgroup (Hajek-normalized within it)."""
    ok = ann[col].notna().to_numpy() & (w > 0)
    if sub is not None:
        ok = ok & sub
    ww = w[ok] / w[ok].sum()
    pr = ann.loc[ok, col].map(RANK).to_numpy(float)
    hh = hr[ok]
    d = pr - hh
    return {
        "n": int(ok.sum()),
        "band_acc": float((ww * (d == 0)).sum()),
        "signed_err": float((ww * d).sum()),
        "under": float((ww * (d < 0)).sum()),
        "over": float((ww * (d > 0)).sum()),
        "shares": wshares(ann.loc[ok, col], w[ok]),
    }


def boot_ci(ann, pop, col, hr, mask, rng, sub: np.ndarray | None = None) -> tuple[float, float]:
    cells = list(ann.groupby(["silver_band", "tier"]).indices.items())
    vals = []
    for _ in range(N_BOOT):
        take = []
        for _, idx in cells:
            take.extend(rng.choice(idx, size=len(idx), replace=True))
        b = ann.iloc[take].reset_index(drop=True)
        m = mask.to_numpy()[take]
        bw = reweight(b, pop, pd.Series(m, index=b.index))
        ok = b[col].notna().to_numpy() & (bw > 0)
        if sub is not None:
            ok = ok & sub[take]
        if ok.sum() < 5:
            continue
        ww = bw[ok] / bw[ok].sum()
        d = b.loc[ok, col].map(RANK).to_numpy(float) - hr[take][ok]
        vals.append(float((ww * d).sum()))
    if len(vals) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    gens = load_generations()
    print("\nlinking gemma generations (180k corpus) to ContactID:")
    gens.update(load_gemma_generations())

    pop = cohort_population(gens)
    sample = annotated_sample(gens, pop)
    ann = sample[sample["annotated"] & sample["human_b"].notna()].copy().reset_index(drop=True)

    split = pd.read_csv(SPLIT_KEY, dtype=str)
    split = normalize_id_columns(split, [CONTACT_ID])
    train_ids = set(split.loc[split.split == "train", CONTACT_ID])
    ann["in_train"] = ann[CONTACT_ID].isin(train_ids)
    heldout = ~ann["in_train"]
    allrows = pd.Series(True, index=ann.index)

    print(f"\nannotated probability sample: {len(ann)} notes")
    print(f"  in July FT train split      : {int(ann.in_train.sum())}  (excluded for July FT)")
    print(f"  held out                    : {int(heldout.sum())}")

    hr = ann["human_b"].map(RANK).to_numpy(float)

    specs = [
        ("human_b",        "Human (expert gold)",              allrows, True),
        ("gemma_base_b",   "Gemma3-12B base (no FT)",          allrows, True),
        ("gemma_june_ft_b","Gemma3-12B June FT",               allrows, True),
        ("old_silver_b",   "Old silver = paper's labels",      allrows, True),
        ("qwen_zeroshot_b","Qwen3-235B zero-shot",             allrows, True),
        ("july_ft_b",      "Gemma3-12B July FT (held-out)",    heldout, True),
        ("july_ft_b",      "  [July FT, leaked - do not cite]", allrows, False),
    ]

    rows = []
    for col, name, mask, clean in specs:
        if col not in ann:
            print(f"  !! missing column {col}, skipping {name}")
            continue
        w = reweight(ann, pop, mask)
        s = score(ann, w, col, hr)
        if col != "human_b":
            lo, hi = boot_ci(ann, pop, col, hr, mask, rng)
            s["ci"] = (lo, hi)
            s["sig"] = not (lo <= 0 <= hi)
        rows.append({"labeler": name, "clean": clean, **s})

    print("\n=== Design-weighted band shares (population-projected) ===")
    hdr = f"{'labeler':34s} " + " ".join(f"{b[:11]:>12s}" for b in BANDS)
    print(hdr)
    for r in rows:
        cells = " ".join(f"{100*r['shares'][b]:11.1f}%" for b in BANDS)
        print(f"{r['labeler']:34s} {cells}")

    print("\n=== Agreement with the human standard (design-weighted) ===")
    print(f"{'labeler':34s} {'n':>4s} {'band acc':>9s} {'signed err':>11s} "
          f"{'95% CI':>18s} {'biased?':>8s}")
    for r in rows:
        if r["labeler"].startswith("Human"):
            print(f"{r['labeler']:34s} {r['n']:4d} {'--':>9s} {'--':>11s} {'--':>18s} {'--':>8s}")
            continue
        lo, hi = r["ci"]
        flag = "YES" if r["sig"] else "no"
        print(f"{r['labeler']:34s} {r['n']:4d} {r['band_acc']:9.3f} {r['signed_err']:11.3f} "
              f"  [{lo:+.3f},{hi:+.3f}] {flag:>8s}")

    print("\nsigned error = labeler - human, in band units. negative = UNDER-codes progress.")
    print("'biased?' = does the 95% bootstrap CI exclude zero.")

    # ---- Breakout: EASY (human fine label <= 1.5) vs HARD (> 1.5) --------------------
    # 1.5 sits inside the conversation_need band (cutoff 1.75), so this splits the bottom
    # two bands (routine contact) from the top two (planning/paperwork + milestones).
    # Conditioning is on the HUMAN label -- the truth -- so this says where model error
    # lands as a function of what actually happened, not of what the model guessed.
    hfine = pd.to_numeric(ann["progress_label"], errors="coerce").to_numpy()
    easy = hfine <= 1.5
    hard = hfine > 1.5
    w_all = reweight(ann, pop, allrows)
    pop_hard = float(w_all[hard].sum() / w_all[w_all > 0].sum())
    print(f"\n=== Bias by difficulty: human label <= 1.5 (EASY) vs > 1.5 (HARD) ===")
    print(f"easy: {int(easy.sum())} notes   hard: {int(hard.sum())} notes   "
          f"(hard = {pop_hard:.1%} of the weighted population)")
    print(f"\n{'labeler':32s} {'n':>4s} {'EASY err':>9s} {'95% CI':>17s} | "
          f"{'n':>4s} {'HARD err':>9s} {'95% CI':>17s}")
    diff_rows = []
    for col, name, mask, clean in specs:
        if col == "human_b" or not clean:
            continue
        w = reweight(ann, pop, mask)
        rec = {"labeler": name}
        cells = []
        for tag, sub in (("easy", easy), ("hard", hard)):
            s = score(ann, w, col, hr, sub=sub)
            lo, hi = boot_ci(ann, pop, col, hr, mask, rng, sub=sub)
            rec |= {f"{tag}_n": s["n"], f"{tag}_err": s["signed_err"],
                    f"{tag}_lo": lo, f"{tag}_hi": hi}
            cells.append(f"{s['n']:4d} {s['signed_err']:9.3f}  [{lo:+.3f},{hi:+.3f}]")
        diff_rows.append(rec)
        print(f"{name:32s} {cells[0]} | {cells[1]}")

    print("\n(hard cases are rare, so hard-case CIs are wide; read the point estimates as a pattern.)")
    pd.DataFrame(diff_rows).to_csv(SAFE / "coauthor_bias_by_difficulty.csv", index=False)
    print(f"wrote {SAFE / 'coauthor_bias_by_difficulty.csv'}")

    SAFE.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([{
        "labeler": r["labeler"], "clean": r["clean"], "n": r["n"],
        "band_accuracy": r["band_acc"], "signed_error": r["signed_err"],
        "ci_low": r.get("ci", (np.nan, np.nan))[0], "ci_high": r.get("ci", (np.nan, np.nan))[1],
        "under_coded": r["under"], "over_coded": r["over"],
        **{f"share_{b}": r["shares"][b] for b in BANDS},
    } for r in rows])
    out.to_csv(SAFE / "coauthor_model_comparison.csv", index=False)
    print(f"\nwrote {SAFE / 'coauthor_model_comparison.csv'}")


if __name__ == "__main__":
    main()
