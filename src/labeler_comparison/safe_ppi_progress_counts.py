"""SAFE_OUTPUT: calibration/PPI-corrected progress-share estimates for count reports.

Combines the ContactID-keyed authoritative ground truth
(annotations/adjudication/contact_ground_truth, multi-annotator agreement rows)
with the deployed qwen3-ft silver labels to debias descriptive progress counts.

Design constraint discovered in the data: gold-annotated casenotes were sampled
archive-wide (2007-2019), with only ~22 pairs inside the 2019 cohort window. Exact
within-month PPI is therefore infeasible; instead we:

  1. estimate the band-level calibration matrix P(gold band | silver band) from all
     gold-silver pairs (n ~ 594), assuming the labeler's conditional error
     distribution is stable over time;
  2. correct each 2019 month's silver band distribution by that matrix,
     propagating calibration + sampling uncertainty via parametric bootstrap
     (Dirichlet rows x multinomial month counts);
  3. report sensitivity variants: rectifier restricted to 2016-2019 pairs, and
     exact PPI on the 2019 in-window pairs for the year-total (wide CI, honest).

Reads protected inputs column-selectively (never note text); writes only aggregate
cells with k<10 suppression.

Band mapping (deployed/paper schema vs v3 ground-truth schema):
  no_progress            silver {0}            gold {0, 0.25}
  conversation_need      silver {0.975}        gold {0.5, 0.75, 1, 1.5}   # team's merged eval bucket
  planning_paperwork     silver {2, 2.5}       gold {2, 2.5}
  milestones             silver {3, 3.5, 4}    gold {3, 3.5, 4}

Usage: python src/labeler_comparison/safe_ppi_progress_counts.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
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

SUPPRESS_K = 10
N_BOOT = 2000
SEED = 20260706
STAGE1 = (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-07-01"))
STAGE2 = (pd.Timestamp("2019-07-01"), pd.Timestamp("2020-01-01"))
OUTCOME = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"))
RECENT_YEARS = (2016, 2019)

BANDS = ["no_progress", "conversation_need", "planning_paperwork", "milestones"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--contacts-csv", default="protected_data/contact_timestamps.csv")
    p.add_argument(
        "--labels-csv",
        default="protected_data/silver_labels_qwen3_ft.csv",
    )
    p.add_argument(
        "--gold-csv",
        default="annotations/adjudication/contact_ground_truth/contact_plus_reconciled_ground_truth_input.csv",
    )
    # ContactID-keyed adjudicated disagreement rows (the contact_ground_truth build
    # drops their ContactIDs; this upstream file keeps them). Contains all gold-0
    # labels — annotators disagreed mostly at the no-progress boundary, so omitting
    # these selects on the label and breaks calibration.
    p.add_argument(
        "--gold-adjudicated-csv",
        default="annotations/adjudication/adjudicated_progress_ground_truth_combined_reconciled.csv",
    )
    # single-annotation rows with human-reviewed reconciled labels; ContactIDs
    # recovered by a note-hash link to ContactID (not included)
    p.add_argument(
        "--gold-single-csv",
        default="annotations/adjudication/contact_single_annotation_qc/reconciliation_summary/"
        "single_annotation_reconciled_with_contact_id.csv",
    )
    p.add_argument("--date-format", default="%m/%d/%y")
    p.add_argument("--output-dir", default="outputs/safe/ppi_progress")
    return p.parse_args()


def band_silver(v: pd.Series) -> pd.Series:
    x = pd.to_numeric(v, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    out[x <= 0.4] = "no_progress"
    out[(x > 0.4) & (x < 1.75)] = "conversation_need"
    out[(x >= 1.75) & (x < 2.75)] = "planning_paperwork"
    out[x >= 2.75] = "milestones"
    out[x.isna()] = pd.NA
    return out


def band_gold(v: pd.Series) -> pd.Series:
    x = pd.to_numeric(v, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    out[x <= 0.25] = "no_progress"
    out[(x >= 0.5) & (x <= 1.5)] = "conversation_need"
    out[(x >= 2.0) & (x <= 2.5)] = "planning_paperwork"
    out[x >= 3.0] = "milestones"
    out[(x < 0) | x.isna()] = pd.NA
    return out


def calibration_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    """silver band x gold band count matrix (rows: silver)."""
    mat = pd.crosstab(pairs["silver_band"], pairs["gold_band"])
    return mat.reindex(index=BANDS, columns=BANDS, fill_value=0)


def bootstrap_corrected(month_counts: pd.DataFrame, calib: pd.DataFrame, rng) -> dict:
    """Corrected band shares per month with bootstrap CIs.

    month_counts: month x silver band counts. calib: silver x gold counts.
    corrected_share_m(gold b') = sum_b share_silver_m(b) * P(b' | b)
    """
    months = month_counts.index
    n_m = month_counts.sum(axis=1).to_numpy()
    p_silver = (month_counts.to_numpy().T / n_m).T  # month x band

    calib_arr = calib.to_numpy().astype(float)
    row_tot = calib_arr.sum(axis=1, keepdims=True)
    p_cal = np.divide(calib_arr, row_tot, out=np.full_like(calib_arr, np.nan), where=row_tot > 0)
    point = p_silver @ p_cal  # month x gold band

    draws = np.empty((N_BOOT, len(months), len(BANDS)))
    for it in range(N_BOOT):
        cal_b = np.vstack([
            rng.dirichlet(calib_arr[i] + 0.5) if calib_arr[i].sum() > 0 else np.full(len(BANDS), np.nan)
            for i in range(len(BANDS))
        ])
        counts_b = np.vstack([rng.multinomial(int(n), p) for n, p in zip(n_m, p_silver)])
        draws[it] = (counts_b.T / counts_b.sum(axis=1)).T @ cal_b
    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)
    return dict(months=list(months), point=point, lo=lo, hi=hi, p_silver=p_silver, n_m=n_m)


def main() -> None:
    args = parse_args()
    outdir = REPO / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    contacts = read_selected_csv(
        REPO / args.contacts_csv, [CLIENT_ID, CONTACT_ID, CONTACT_DATE, PLACEMENT_TYPE], "contacts CSV"
    )
    silver = read_selected_csv(REPO / args.labels_csv, [CONTACT_ID, PRIMARY_LABEL], "silver labels CSV")
    # Column-selective reads: note column is protected and never loaded.
    gold_agree = pd.read_csv(REPO / args.gold_csv, usecols=["contact_id", "final_vetted_ground_truth"])
    gold_agree = gold_agree.dropna(subset=["contact_id"]).rename(columns={"contact_id": CONTACT_ID})
    gold_agree["gold_source"] = "multi_annotator_agreement"
    gold_adj = pd.read_csv(REPO / args.gold_adjudicated_csv, usecols=["ContactID", "final_vetted_ground_truth"])
    gold_adj = gold_adj.dropna(subset=["ContactID"])
    gold_adj["gold_source"] = "adjudicated_disagreement"
    gold_single = pd.read_csv(REPO / args.gold_single_csv).rename(
        columns={"contact_id": CONTACT_ID, "reconciled_ground_truth": "final_vetted_ground_truth"}
    )
    gold = pd.concat([gold_agree, gold_adj, gold_single], ignore_index=True)

    contacts = normalize_id_columns(contacts, [CLIENT_ID, CONTACT_ID])
    silver = normalize_id_columns(silver, [CONTACT_ID]).drop_duplicates(CONTACT_ID)
    gold = normalize_id_columns(gold, [CONTACT_ID]).drop_duplicates(CONTACT_ID)
    contacts[CONTACT_DATE] = pd.to_datetime(contacts[CONTACT_DATE], format=args.date_format, errors="coerce")

    merged = contacts.merge(silver, on=CONTACT_ID, how="left")
    merged["silver_band"] = band_silver(merged[PRIMARY_LABEL])
    gold["gold_band"] = band_gold(gold["final_vetted_ground_truth"])
    gold = gold[gold["gold_band"].notna()]

    d = merged[CONTACT_DATE]

    def in_window(s, w):
        return (s >= w[0]) & (s < w[1])

    # cohort reconstruction (same criteria as safe_trajectory_aggregates.py)
    stage_both = set(merged.loc[in_window(d, STAGE1), CLIENT_ID]) & set(merged.loc[in_window(d, STAGE2), CLIENT_ID])
    r1 = set(merged.loc[in_window(d, STAGE1) & merged["silver_band"].notna(), CLIENT_ID])
    r2 = set(merged.loc[in_window(d, STAGE2) & merged["silver_band"].notna(), CLIENT_ID])
    y2 = set(merged.loc[in_window(d, OUTCOME) & merged[PLACEMENT_TYPE].notna(), CLIENT_ID])
    cohort = stage_both & r1 & r2 & y2

    # gold-silver calibration pairs, archive-wide
    pairs = merged[merged["silver_band"].notna()].merge(gold.drop_duplicates(CONTACT_ID), on=CONTACT_ID, how="inner")
    pairs["year"] = pairs[CONTACT_DATE].dt.year
    recent = pairs[(pairs["year"] >= RECENT_YEARS[0]) & (pairs["year"] <= RECENT_YEARS[1])]
    vetted = pairs[pairs["gold_source"] != "single_annotation_reconciled"]
    inwin = pairs[in_window(pairs[CONTACT_DATE], (STAGE1[0], STAGE2[1])) & pairs[CLIENT_ID].isin(cohort)]
    print(f"cohort clients: {len(cohort)}")
    print(f"calibration pairs: all-gold n={len(pairs)}, vetted-only n={len(vetted)}, "
          f"{RECENT_YEARS[0]}-{RECENT_YEARS[1]} n={len(recent)}, cohort-2019-window n={len(inwin)}")
    print("pairs by gold source:", pairs["gold_source"].value_counts().to_dict())
    print(f"band agreement: all {(pairs['silver_band'] == pairs['gold_band']).mean():.3f}, "
          f"vetted {(vetted['silver_band'] == vetted['gold_band']).mean():.3f}, "
          f"recent {(recent['silver_band'] == recent['gold_band']).mean():.3f}")

    # calibration matrices (aggregate counts; suppress cells < K in the released file)
    for name, subset in (("all_years", pairs), ("recent", recent), ("vetted_only", vetted)):
        mat = calibration_counts(subset)
        rel = mat.copy().astype(float)
        rel[rel < SUPPRESS_K] = np.nan
        rel.to_csv(outdir / f"calibration_counts_{name}.csv")

    # monthly silver band counts for the 2019 cohort window
    frame = merged[
        merged[CLIENT_ID].isin(cohort)
        & in_window(d, (STAGE1[0], STAGE2[1]))
        & merged["silver_band"].notna()
    ].copy()
    frame["month"] = frame[CONTACT_DATE].dt.to_period("M").astype(str)
    month_counts = frame.pivot_table(index="month", columns="silver_band", values=CONTACT_ID,
                                     aggfunc="count", fill_value=0).reindex(columns=BANDS, fill_value=0)

    rows = []
    variants = {"calibrated_all_years": pairs, "calibrated_recent": recent,
                "calibrated_vetted_only": vetted}
    boots = {name: bootstrap_corrected(month_counts, calibration_counts(sub), rng)
             for name, sub in variants.items()}
    base = boots["calibrated_all_years"]
    for mi, month in enumerate(base["months"]):
        for bi, band in enumerate(BANDS):
            row = dict(month=month, band=band,
                       n_contacts=int(base["n_m"][mi]),
                       silver_share=base["p_silver"][mi, bi])
            for name, b in boots.items():
                row[f"{name}_share"] = b["point"][mi, bi]
                row[f"{name}_lo"] = b["lo"][mi, bi]
                row[f"{name}_hi"] = b["hi"][mi, bi]
            rows.append(row)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(outdir / "monthly_corrected_shares.csv", index=False)

    # exact PPI on the 2019 cohort-window pairs, year-total only (sensitivity)
    ppi_rows = []
    f_all = frame["silver_band"]
    for band in BANDS:
        fa = (f_all == band).to_numpy(float)
        fg = (inwin["silver_band"] == band).to_numpy(float)
        yg = (inwin["gold_band"] == band).to_numpy(float)
        n_g = len(inwin)
        rect = (yg - fg).mean() if n_g else np.nan
        se = np.sqrt(fa.var(ddof=1) / len(fa) + (yg - fg).var(ddof=1) / n_g) if n_g > 1 else np.nan
        ppi_rows.append(dict(period="2019", band=band, n_gold=n_g,
                             silver_share=fa.mean(),
                             ppi_share=fa.mean() + rect if n_g else np.nan,
                             ppi_lo=fa.mean() + rect - 1.96 * se if n_g > 1 else np.nan,
                             ppi_hi=fa.mean() + rect + 1.96 * se if n_g > 1 else np.nan,
                             suppressed=n_g < SUPPRESS_K))
    ppi = pd.DataFrame(ppi_rows)
    ppi.loc[ppi["suppressed"], ["ppi_share", "ppi_lo", "ppi_hi"]] = np.nan
    ppi.to_csv(outdir / "ppi_2019_total_inwindow.csv", index=False)

    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    manifest = REPO / "outputs" / "safe" / "output_manifest.csv"
    with open(manifest, "a", newline="") as f:
        csv.writer(f).writerow([
            dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + git_hash, git_hash,
            "python src/labeler_comparison/safe_ppi_progress_counts.py",
            "restricted_cohort_contacts+silver+gold_contact_ground_truth", len(cohort), "",
            "month,band", "calibration-corrected share cells; bootstrap CIs; PPI year-total", str(SEED),
            str(outdir.relative_to(REPO)) + "/*.csv", "safe_aggregate",
            f"Gold = multi-annotator agreement (ContactID-keyed, n={len(pairs)} pairs w/ silver). "
            "Monthly correction assumes stationary conditional labeler error; exact PPI year-total "
            f"uses n={len(inwin)} in-window pairs. k<{SUPPRESS_K} suppression on released count cells.",
        ])
    print(f"wrote aggregates to {outdir}")


if __name__ == "__main__":
    main()
