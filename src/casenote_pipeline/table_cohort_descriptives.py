"""Cohort descriptives for the SI: outreach-quintile cut points, casenote volume,
contact types, and the always-observed state S'_t.

Reads protected row-level inputs only through outreach_dtr_reward_only's own
loaders (same 775-client analysis cohort as the paper) and writes ONLY aggregate
tables to results/cohort_descriptives/. Cells with fewer than 10 clients are
suppressed. Nothing row-level is printed or written. These cohort aggregates (SI Tables S1-S2)
are NOT included in the public release; running this script requires the protected data.

Usage:
  python src/casenote_pipeline/table_cohort_descriptives.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_LABELS = "protected_data/labels.csv"
AGG = REPO / "results" / "cohort_descriptives"
SUPPRESS_K = 10
STAGE_MONTHS = 6


def q(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return dict(n=len(x), mean=x.mean(), sd=x.std(), p25=x.quantile(0.25),
                median=x.median(), p75=x.quantile(0.75))


def main() -> None:
    import outreach_dtr_reward_only as orw
    from progress_surrogate_feasibility import (
        CLIENT_ID, CONTACT_DATE, CONTACT_ID, STATE_COLUMNS, IntervalSpec, assign_interval,
    )

    sys.argv = [sys.argv[0], "--labels-csv", DEFAULT_LABELS, "--date-format", "%m/%d/%y"]
    args = orw.parse_args()
    features, contacts, labels = orw.load_inputs(args)
    full, _coverage = orw.build_interval_features(
        features, contacts, labels,
        IntervalSpec(pd.Timestamp(args.start_date), args.interval_months, args.num_decision_intervals),
        args.date_format,
    )
    dtr = orw.build_dtr_frame(features, contacts, labels, args)[0]
    cohort = set(dtr[CLIENT_ID].astype("string").str.strip())
    print(f"analysis cohort: {len(cohort)} clients")
    AGG.mkdir(parents=True, exist_ok=True)

    # 1. Quintile cut points. Bins are formed among all clients with >=1 contact in the stage
    #    (bin_nonzero_quintiles), so report the contact range of each bin on that population,
    #    plus how many analysis-cohort clients fall in each bin.
    rows = []
    for t in (1, 2):
        cnt, b = full[f"contacts_t{t}"], full[f"outreach_bin_t{t}"]
        for k in range(1, 6):
            sel = cnt[b == k]
            n_coh = int(dtr["A1" if t == 1 else "A2"].eq(k).sum())
            rows.append(dict(stage=t, quintile=k, contacts_min=int(sel.min()), contacts_max=int(sel.max()),
                             per_month_min=sel.min() / STAGE_MONTHS, per_month_max=sel.max() / STAGE_MONTHS,
                             n_binning_population=int(len(sel)), n_analysis_cohort=n_coh))
    cuts = pd.DataFrame(rows)
    cuts.loc[cuts["n_analysis_cohort"] < SUPPRESS_K, "n_analysis_cohort"] = pd.NA
    cuts.to_csv(AGG / "outreach_quintile_cutpoints.csv", index=False)
    print(cuts.to_string(index=False))

    # 2. Casenote volume per client (contacts = casenotes; one note per recorded contact).
    vol = []
    for t in (1, 2):
        vol.append(dict(window=f"stage {t} (6 months)", **q(dtr[f"contacts_t{t}"])))
    vol.append(dict(window="stages 1-2 (12 months)", **q(dtr["contacts_t1"] + dtr["contacts_t2"])))
    c = orw.normalize_id_columns(contacts, [CLIENT_ID, CONTACT_ID]).copy()
    c[CONTACT_DATE] = orw.parse_contact_dates(c[CONTACT_DATE], args.date_format)
    c = c[c[CLIENT_ID].isin(cohort)]
    two_year = c[c[CONTACT_DATE].ge("2019-01-01") & c[CONTACT_DATE].lt("2021-01-01")]
    per_client = two_year.groupby(CLIENT_ID).size().reindex(list(cohort), fill_value=0)
    vol.append(dict(window="two years (2019-2020)", **q(per_client)))
    vol = pd.DataFrame(vol)
    vol.insert(0, "count", "recorded contacts")
    vol["total"] = [int(dtr["contacts_t1"].sum()), int(dtr["contacts_t2"].sum()),
                    int((dtr["contacts_t1"] + dtr["contacts_t2"]).sum()), int(per_client.sum())]
    # Casenotes: recorded contacts that have a casenote in the label table (2019 only).
    lab = set(orw.normalize_id_columns(labels, [CONTACT_ID])[CONTACT_ID].dropna())
    spec = IntervalSpec(pd.Timestamp(args.start_date), args.interval_months, args.num_decision_intervals)
    cn = c.assign(stage=assign_interval(c[CONTACT_DATE], spec)).dropna(subset=["stage"])
    cn = cn[cn[CONTACT_ID].isin(lab)].drop_duplicates(CONTACT_ID)
    notes = cn.groupby([CLIENT_ID, "stage"]).size().unstack(fill_value=0).reindex(list(cohort), fill_value=0)
    nrows = []
    for t in (1, 2):
        col = notes[t] if t in notes.columns else pd.Series(0, index=notes.index)
        nrows.append(dict(count="casenotes", window=f"stage {t} (6 months)", **q(col), total=int(col.sum())))
    both = notes.sum(axis=1)
    nrows.append(dict(count="casenotes", window="stages 1-2 (12 months)", **q(both), total=int(both.sum())))
    vol = pd.concat([vol, pd.DataFrame(nrows)], ignore_index=True)
    vol.to_csv(AGG / "casenote_volume.csv", index=False)
    print(vol.to_string(index=False))

    # 3. Contact types (attempted vs successful) among cohort contacts in stages 1-2.
    ct = pd.read_csv(args.contacts_csv, usecols=[CLIENT_ID, CONTACT_ID, CONTACT_DATE, "ContactType"])
    ct = orw.normalize_id_columns(ct, [CLIENT_ID, CONTACT_ID])
    ct[CONTACT_DATE] = orw.parse_contact_dates(ct[CONTACT_DATE], args.date_format)
    ct = ct[ct[CLIENT_ID].isin(cohort)]
    ct["stage"] = assign_interval(ct[CONTACT_DATE], IntervalSpec(pd.Timestamp(args.start_date),
                                                                 args.interval_months, args.num_decision_intervals))
    ct = ct.dropna(subset=["stage"])
    def contact_group(x) -> str:
        x = str(x)
        if x.startswith("Yes-OnStreet"):
            return "reached: on street"
        if x.startswith("Yes-"):
            return "reached: other (housing site, office, phone, other)"
        if "Coll" in x:
            return "collateral contact (third party)"
        return "attempted, not reached"
    ct["ContactType"] = ct["ContactType"].map(contact_group)
    types = (ct.groupby(["stage", "ContactType"], dropna=False).agg(n_contacts=(CONTACT_ID, "count"),
                                                                   n_clients=(CLIENT_ID, "nunique"))
             .reset_index())
    types["share_of_stage_contacts"] = types["n_contacts"] / types.groupby("stage")["n_contacts"].transform("sum")
    types.loc[types["n_clients"] < SUPPRESS_K, ["n_contacts", "share_of_stage_contacts"]] = pd.NA
    types.to_csv(AGG / "contact_types.csv", index=False)
    print(types.to_string(index=False))

    # 4. Always-observed state S'_1: numeric -> mean/sd/median/IQR; categorical -> shares.
    f = orw.normalize_id_columns(features, [CLIENT_ID]).drop_duplicates(CLIENT_ID)
    f = f[f[CLIENT_ID].isin(cohort)].copy()
    # DateFirstSeen is a proleptic Gregorian ordinal (date.toordinal()); 0 marks missing.
    ordinal = pd.to_numeric(f["DateFirstSeen"], errors="coerce").where(lambda x: x > 0)
    f["years_known_before_2019"] = (pd.Timestamp("2019-01-01").toordinal() - ordinal) / 365.25
    # Program and zone names could identify the partner; report them as anonymous levels by size.
    for col, prefix in (("Program", "Program"), ("Zone_cat", "Zone")):
        order = f[col].astype("string").fillna("missing").value_counts().index
        f[col] = f[col].astype("string").fillna("missing").map(
            {lvl: f"{prefix} {chr(65 + i)}" for i, lvl in enumerate(order)})
    num_cols = ["years_known_before_2019", "num_contacts_before_2019"]
    cat_cols = [col for col in STATE_COLUMNS if col not in ("DateFirstSeen", "num_contacts_before_2019")]
    srows = []
    for col in num_cols:
        srows.append(dict(variable=col, level="", **q(f[col])))
    for col in cat_cols:
        vc = f[col].astype("string").fillna("missing").value_counts()
        for level, n in vc.items():
            srows.append(dict(variable=col, level=level, n=int(n) if n >= SUPPRESS_K else pd.NA,
                              share=n / len(f) if n >= SUPPRESS_K else pd.NA))
    state = pd.DataFrame(srows)
    state.to_csv(AGG / "state_summary_stage1.csv", index=False)
    print(state.to_string(index=False))


if __name__ == "__main__":
    main()
