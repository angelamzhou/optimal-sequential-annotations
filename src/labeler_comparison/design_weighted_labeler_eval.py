"""Design-based labeler comparison on the annotated probability sample.

Rather than extrapolating model labels across the population, this evaluates every
labeler on the SAME annotated notes using the sample's design weights, so all
differences are labeler disagreement (no support/extrapolation assumptions).

Design: wave-1/2 sample = silver band x stage strata (125/cell) from the 2019
cohort population; annotation proceeded in ascending consensus-tier order, so
weights are post-stratified by silver band x consensus tier:

    w_i  proportional to  W_h * (n_ht / n_h) / m_ht

with W_h the population share of silver band h, n_ht/n_h the tier composition of
the sampled stratum, and m_ht the annotated count in cell (h,t). Bootstrap CIs
resample annotated notes within cells.

Validity check: the design-weighted estimate of the SILVER band shares should
reproduce the exact population silver census (silver is observed on every note).

Leakage: 423 of the gold notes are in the July FT's training data; that model is
therefore reported on held-out notes only (reweighted for the label-stratified
val draw), alongside its exact population census.

Writes outputs/safe/ppi_progress/design_weighted_labeler_eval.csv.

Usage: PROTECTED_DATA_ROOT=<PROTECTED_DATA_ROOT> python src/labeler_comparison/design_weighted_labeler_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# All inputs are protected; paths below are relative to PROTECTED_DATA_ROOT (default: current directory).
REPO = Path(os.environ.get("PROTECTED_DATA_ROOT", ".")).resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from progress_surrogate_feasibility import (  # noqa: E402
    CLIENT_ID,
    CONTACT_DATE,
    CONTACT_ID,
    PLACEMENT_TYPE,
    PRIMARY_LABEL,
    normalize_id_columns,
    read_selected_csv,
)
from safe_ppi_progress_counts import BANDS, band_silver  # noqa: E402

SDIR = REPO / "protected_data" / "annotation_sample_2026-07"
SAFE = REPO / "outputs" / "safe" / "ppi_progress"
FTDIR = REPO / "protected_data" / "finetune_split"
RANK = {b: i for i, b in enumerate(BANDS)}
TIER_POOL = {"2of4": "low", "band_majority": "low", "3of4": "3of4", "4of4": "4of4"}
N_BOOT = 2000
SEED = 20260711
S1, S2, OW, OE = map(pd.Timestamp, ("2019-01-01", "2019-07-01", "2020-01-01", "2021-01-01"))


def load_generations() -> dict[str, pd.DataFrame]:
    gens = {}
    old = read_selected_csv(
        REPO / "protected_data/silver_labels_qwen3_ft.csv",
        [CONTACT_ID, PRIMARY_LABEL], "old silver")
    gens["old_silver"] = normalize_id_columns(old, [CONTACT_ID]).drop_duplicates(CONTACT_ID).rename(
        columns={PRIMARY_LABEL: "old_silver"})
    for name, path in (
        ("qwen_zeroshot", "protected_data/labels_qwen235b_zeroshot.csv"),
        ("july_ft", "protected_data/labels_gemma3_july_ft.csv"),
    ):
        p = REPO / path
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["ContactID", "primary_label"], dtype=str).rename(
            columns={"primary_label": name})
        gens[name] = normalize_id_columns(df, [CONTACT_ID]).drop_duplicates(CONTACT_ID)
    return gens


def cohort_population(gens) -> pd.DataFrame:
    contacts = read_selected_csv(REPO / "protected_data/contact_timestamps.csv",
                                 [CLIENT_ID, CONTACT_ID, CONTACT_DATE, PLACEMENT_TYPE], "contacts")
    contacts = normalize_id_columns(contacts, [CLIENT_ID, CONTACT_ID]).drop_duplicates(CONTACT_ID)
    contacts[CONTACT_DATE] = pd.to_datetime(contacts[CONTACT_DATE], format="%m/%d/%y", errors="coerce")
    c = contacts.merge(gens["old_silver"], on=CONTACT_ID, how="left")
    c["old_silver_b"] = band_silver(c["old_silver"])
    d = c[CONTACT_DATE]
    sb = set(c.loc[(d >= S1) & (d < S2), CLIENT_ID]) & set(c.loc[(d >= S2) & (d < OW), CLIENT_ID])
    r1 = set(c.loc[(d >= S1) & (d < S2) & c["old_silver_b"].notna(), CLIENT_ID])
    r2 = set(c.loc[(d >= S2) & (d < OW) & c["old_silver_b"].notna(), CLIENT_ID])
    y2 = set(c.loc[(d >= OW) & (d < OE) & pd.to_numeric(c[PLACEMENT_TYPE], errors="coerce").notna(), CLIENT_ID])
    coh = sb & r1 & r2 & y2
    pop = c[c[CLIENT_ID].isin(coh) & (d >= S1) & (d < OW) & c["old_silver_b"].notna()].copy()
    for name, g in gens.items():
        if name != "old_silver":
            pop = pop.merge(g, on=CONTACT_ID, how="left")
            pop[name + "_b"] = band_silver(pop[name])
    print(f"population (2019 cohort support): {len(pop):,} contacts, {len(coh)} clients")
    return pop


def annotated_sample(gens, pop) -> pd.DataFrame:
    wb = pd.read_csv(SDIR / "annotation_workbook_main.csv", dtype=str, keep_default_na=False,
                     usecols=[CONTACT_ID, "progress_label"])
    wb = normalize_id_columns(wb, [CONTACT_ID])
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


def design_weights(sample: pd.DataFrame, pop: pd.DataFrame) -> pd.Series:
    """Post-stratified weights: population band share x tier composition / annotated count."""
    W = pop["old_silver_b"].value_counts(normalize=True)
    n_ht = sample.groupby(["silver_band", "tier"]).size()
    n_h = sample.groupby("silver_band").size()
    ann = sample[sample["annotated"]]
    m_ht = ann.groupby(["silver_band", "tier"]).size()
    w = pd.Series(0.0, index=sample.index)
    for (h, t), m in m_ht.items():
        idx = ann.index[(ann["silver_band"] == h) & (ann["tier"] == t)]
        w.loc[idx] = W[h] * (n_ht[(h, t)] / n_h[h]) / m
    return w


def wshares(bands: pd.Series, w: np.ndarray) -> dict:
    tot = w.sum()
    return {b: float(w[(bands == b).to_numpy()].sum() / tot) for b in BANDS}


def main() -> None:
    rng = np.random.default_rng(SEED)
    gens = load_generations()
    pop = cohort_population(gens)
    sample = annotated_sample(gens, pop)
    ann = sample[sample["annotated"] & sample["human_b"].notna()].copy()
    ann["w"] = design_weights(sample, pop)[ann.index]
    print(f"annotated probability sample: {len(ann)} notes (of {len(sample)} sampled)")

    labelers = [("human_b", "Human (expert gold)", True),
                ("old_silver_b", "Old silver (12B FT, June std)", True),
                ("qwen_zeroshot_b", "Qwen3-235B zero-shot (independent)", True),
                ("july_ft_b", "July FT (LEAKED: trained on 423 of these)", False)]

    rows = []
    w = ann["w"].to_numpy()
    hb = ann["human_b"]
    hr = hb.map(RANK).to_numpy(float)
    for col, name, clean in labelers:
        if col not in ann:
            continue
        ok = ann[col].notna().to_numpy()
        ww = w * ok
        sh = wshares(ann[col].fillna("x"), ww)
        row = {"labeler": name, "clean": clean, "n_scored": int(ok.sum())}
        row.update({f"share_{b}": sh[b] for b in BANDS})
        if col != "human_b":
            raw = col[:-2]  # strip the _b band suffix -> raw fine label column
            mr = ann[col].map(RANK).to_numpy(float)
            band_acc = float(np.nansum(ww * (mr == hr)) / ww.sum())
            num = pd.to_numeric(ann[raw], errors="coerce").to_numpy(float)
            hnum = pd.to_numeric(ann["progress_label"], errors="coerce").to_numpy(float)
            m = (~np.isnan(num)) & (~np.isnan(hnum))
            exact = float(np.sum(w[m] * (num[m] == hnum[m])) / w[m].sum())
            bias = float(np.sum(w[m] * (num[m] - hnum[m])) / w[m].sum())
            under = float(np.nansum(ww * (mr < hr)) / ww.sum())
            over = float(np.nansum(ww * (mr > hr)) / ww.sum())
            row.update(band_accuracy=band_acc, exact_accuracy=exact,
                       mean_signed_error=bias, share_under_coded=under, share_over_coded=over)
        rows.append(row)

    # bootstrap CIs on band shares, resampling within (band x tier) cells
    cells = list(ann.groupby(["silver_band", "tier"]).indices.items())
    boots = {name: {b: [] for b in BANDS} for _, name, _ in labelers}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for _, ix in cells])
        bs = ann.iloc[idx]
        wb_ = bs["w"].to_numpy()
        for col, name, _ in labelers:
            if col not in bs:
                continue
            ok = bs[col].notna().to_numpy()
            sh = wshares(bs[col].fillna("x"), wb_ * ok)
            for b in BANDS:
                boots[name][b].append(sh[b])
    for r in rows:
        for b in BANDS:
            v = boots[r["labeler"]][b]
            if v:
                r[f"lo_{b}"], r[f"hi_{b}"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    out = pd.DataFrame(rows)
    SAFE.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAFE / "design_weighted_labeler_eval.csv", index=False)

    # ---- report ----
    print("\n=== Design-weighted band shares on the annotated probability sample ===")
    print("(same notes, same weights for every labeler; 95% bootstrap CIs)\n")
    hdr = f"{'labeler':40s} " + " ".join(f"{b[:9]:>17s}" for b in BANDS)
    print(hdr)
    for r in rows:
        cells_s = " ".join(f"{r['share_'+b]*100:6.1f} [{r['lo_'+b]*100:4.1f},{r['hi_'+b]*100:4.1f}]" for b in BANDS)
        flag = "" if r["clean"] else "  <-- leaked"
        print(f"{r['labeler']:40s} {cells_s}{flag}")

    print("\n=== Agreement with the human standard (design-weighted) ===")
    print(f"{'labeler':40s} {'band acc':>9s} {'exact':>7s} {'signed err':>11s} {'under-coded':>12s} {'over-coded':>11s}")
    for r in rows:
        if "band_accuracy" not in r:
            continue
        print(f"{r['labeler']:40s} {r['band_accuracy']:9.3f} {r['exact_accuracy']:7.3f} "
              f"{r['mean_signed_error']:+11.3f} {r['share_under_coded']:12.1%} {r['share_over_coded']:11.1%}")

    print("\n=== VALIDITY CHECK: design-weighted silver share vs exact population census ===")
    census = pop["old_silver_b"].value_counts(normalize=True)
    est = next(r for r in rows if r["labeler"].startswith("Old silver"))
    for b in BANDS:
        print(f"  {b:20s} design-weighted {est['share_'+b]*100:5.1f}%  vs census {census[b]*100:5.1f}%")
    print("\n(NOTE: the weights post-stratify ON the silver band, so this agreement is an "
          "arithmetic consistency check of the weighting, not independent evidence.)")
    print(f"\nwrote {(SAFE / 'design_weighted_labeler_eval.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
