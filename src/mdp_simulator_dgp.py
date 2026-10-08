#!/usr/bin/env python3
"""Simulator-grounded semi-synthetic DGP for validating the casenote annotation pipeline.

Fits a monthly outreach simulator (placement / progress / service transitions as a
function of state and outreach intensity) on the real monthly MDP panel, a
stage-level behavior policy on the real two-stage DTR frame, and a deliberately
misspecified linear silver-label imputer on the real 2019 gold labels. It then
rolls out synthetic cohorts in the exact CSV schema read by
``outreach_dtr_annotation_mc.py`` and computes the ground-truth value of the
pipeline's target policy (shift outreach up one quintile) by brute-force rollout.

Subcommands: ``fit``, ``generate``, ``truth``, ``marginals``. Generated cohorts
resample real baseline rows, so they are written under the gitignored
``protected_data/simulated`` directory; only aggregate outputs go to ``results``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import outreach_dtr_reward_only as orw  # noqa: E402
import progress_surrogate_feasibility as psf  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PANEL_CSV = REPO_ROOT / "protected_data/monthly_mdp_panel.csv"
DEFAULT_SILVER_CSV = REPO_ROOT / "protected_data/silver_labels_qwen3_ft.csv"
DEFAULT_SIM_DIR = REPO_ROOT / "protected_data/simulated"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results/mdp_simulator_validation"

PROGRESS_BINS = ["0", "0.25", "0.5-1.5", "2", "2.5", "3", "3.5", "4"]
LABEL_GRID = np.array([0.0, 0.25, 0.5, 0.75, 0.9375, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
SVC_BINS = ["svc0", "svc1", "svc2+"]
ACTION_BINS = ["1", "2-3", "4-7", "8+"]  # monthly bins with positive contacts
PLACEMENTS = [0, 1, 2, 3]
STATE_CAT_COLS = ["h", "p", "svcprev", "demo", "shelter", "hist", "action_bin"]
STATE_NUM_COLS = ["log_action_count", "log_prior_contacts", "log_prior_shelter"]
JAN_2019_PERIOD = 2019 * 12
MONTHS_PER_STAGE = 6
N_STAGES = 4  # two decision stages in 2019, two behavior stages in 2020
QUINTILES = np.array([1, 2, 3, 4, 5])


# ----------------------------------------------------------------------------- helpers
def merge_bin(value: float) -> str:
    if 0.5 <= value <= 1.5:
        return "0.5-1.5"
    return f"{value:g}"


def shelter_bin(count: np.ndarray) -> np.ndarray:
    out = np.where(count <= 0, "0", np.where(count <= 2, "1-2", "3+"))
    return out.astype(object)


def hist_bin(count: np.ndarray) -> np.ndarray:
    out = np.where(count <= 0, "0", np.where(count <= 3, "1-3", np.where(count <= 10, "4-10", "11+")))
    return out.astype(object)


def action_bin(count: np.ndarray) -> np.ndarray:
    out = np.where(count <= 1, "1", np.where(count <= 3, "2-3", np.where(count <= 7, "4-7", "8+")))
    return out.astype(object)


def svc_index_from_count(count: np.ndarray) -> np.ndarray:
    return np.where(count <= 0, 0, np.where(count == 1, 1, 2)).astype(int)


def parse_state(values: pd.Series) -> pd.DataFrame:
    rows = []
    for value in values.astype(str):
        parsed: dict[str, str] = {}
        for part in value.split("|"):
            if "=" in part:
                key, item = part.split("=", 1)
                parsed[key] = item
        rows.append(parsed)
    return pd.DataFrame(rows, index=values.index)


def sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Row-wise categorical draw from a (n, k) probability matrix."""
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0 + 1e-12
    u = rng.uniform(size=probs.shape[0])
    return (u[:, None] > cum).sum(axis=1)


def grid_round(values: np.ndarray) -> np.ndarray:
    idx = np.abs(values[:, None] - LABEL_GRID[None, :]).argmin(axis=1)
    return LABEL_GRID[idx]


def encode_baseline(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot categorical / boolean baseline covariates the way the pipeline sees them."""
    work = frame[psf.STATE_COLUMNS].copy()
    numeric = pd.DataFrame(index=work.index)
    numeric["DateFirstSeen"] = pd.to_numeric(work["DateFirstSeen"], errors="coerce").fillna(0.0) / 1e5
    numeric["log_num_contacts_before_2019"] = np.log1p(pd.to_numeric(work["num_contacts_before_2019"], errors="coerce").fillna(0.0))
    numeric["max_PlacementType_before_2019"] = pd.to_numeric(work["max_PlacementType_before_2019"], errors="coerce").fillna(0.0)
    for col in ["Housing_application_Yes", "Service_refusal_Yes", "Impt_documents_Yes", "Benefits_Yes"]:
        numeric[col] = work[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    cats = work[["Program", "BelievedChronic", "Gender_cat", "Race_cat", "Age_cat", "Zone_cat", "Ethnicity", "Was311Call", "Was911Call", "Removal958"]].astype(str)
    dummies = pd.get_dummies(cats, dtype=float)
    out = pd.concat([numeric, dummies], axis=1)
    if columns is not None:
        out = out.reindex(columns=columns, fill_value=0.0)
    return out


# ----------------------------------------------------------------------------- transition heads
def make_head(kind: str, gamma: float, seed: int, n_components: int = 300):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), STATE_CAT_COLS),
            ("num", StandardScaler(), STATE_NUM_COLS),
        ]
    )
    if kind == "linear":
        return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
    if kind == "poly2":
        return make_pipeline(
            pre,
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.3),
        )
    if kind == "rbf":
        return make_pipeline(
            pre,
            RBFSampler(gamma=gamma, n_components=n_components, random_state=seed),
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0),
        )
    raise ValueError(f"unknown simulator kind {kind}")


def transition_features(panel: pd.DataFrame) -> pd.DataFrame:
    state = parse_state(panel["state"])
    feats = pd.DataFrame(index=panel.index)
    for col in ["h", "p", "svcprev", "demo", "shelter", "hist"]:
        feats[col] = state[col].astype(str)
    feats["action_bin"] = panel["action_bin"].astype(str)
    feats["log_action_count"] = np.log1p(pd.to_numeric(panel["action_count"], errors="coerce").fillna(0.0))
    feats["log_prior_contacts"] = np.log1p(pd.to_numeric(panel["prior_contacts"], errors="coerce").fillna(0.0))
    feats["log_prior_shelter"] = np.log1p(pd.to_numeric(panel["prior_shelter_contacts"], errors="coerce").fillna(0.0))
    return feats


def fit_heads(panel: pd.DataFrame, kind: str, seed: int, gammas: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    active = panel[pd.to_numeric(panel["action_count"], errors="coerce") > 0].copy()
    feats = transition_features(active)
    targets = {
        "h": active["next_placement"].astype(int).astype(str).values,
        "p": active["next_progress"].astype(str).values,
        "svc": np.array(SVC_BINS, dtype=object)[svc_index_from_count(pd.to_numeric(active["current_service_count"], errors="coerce").fillna(0).values)],
    }
    clients = np.array(sorted(active["ClientID"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(clients)
    n_val = int(0.2 * len(clients))
    val_clients = set(clients[:n_val])
    val_mask = active["ClientID"].astype(str).isin(val_clients).values

    summary: dict[str, Any] = {"kind": kind, "rows_active": int(len(active)), "heads": {}}
    heads: dict[str, Any] = {}
    grid = gammas if kind == "rbf" else [0.0]
    for name, y in targets.items():
        best = None
        for gamma in grid:
            model = make_head(kind, gamma, seed)
            model.fit(feats[~val_mask], y[~val_mask])
            classes = [str(c) for c in model.classes_]
            nll = float(log_loss(y[val_mask], model.predict_proba(feats[val_mask]), labels=classes))
            if best is None or nll < best[0]:
                best = (nll, gamma, model)
        nll, gamma, _ = best
        final = make_head(kind, gamma, seed)
        final.fit(feats, y)
        heads[name] = final
        # calibration diagnostics on the validation fold with the val-fitted model
        val_model = best[2]
        probs = val_model.predict_proba(feats[val_mask])
        classes = [str(c) for c in val_model.classes_]
        start = feats.loc[val_mask, name if name != "svc" else "svcprev"].astype(str).values
        change_true = float((y[val_mask] != start).mean())
        same_idx = np.array([classes.index(s) if s in classes else -1 for s in start])
        p_same = np.where(same_idx >= 0, probs[np.arange(len(same_idx)), np.clip(same_idx, 0, None)], 0.0)
        change_pred = float((1.0 - p_same).mean())
        summary["heads"][name] = {
            "val_nll": nll,
            "gamma": gamma,
            "val_change_rate_true": change_true,
            "val_change_rate_predicted": change_pred,
            "classes": classes,
        }
    return heads, summary


# ----------------------------------------------------------------------------- behavior + emission
def real_dtr_with_silver_h2(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipe_args = argparse.Namespace(
        features_csv=str(args.features_csv), contacts_csv=str(args.contacts_csv), labels_csv=str(args.labels_csv),
        date_format=args.date_format, start_date="2019-01-01", interval_months=6, num_decision_intervals=2,
        placement_outcome_end_date="2021-01-01",
    )
    orw.HAVE_SKLEARN = False
    features, contacts, labels = orw.load_inputs(pipe_args)
    dtr, _coverage, _population = orw.build_dtr_frame(features, contacts, labels, pipe_args)
    spec = psf.IntervalSpec(pd.Timestamp("2019-01-01"), 6, 2)
    silver = orw.read_selected_csv(Path(args.silver_csv), [psf.CONTACT_ID, psf.PRIMARY_LABEL], "silver labels CSV")
    silver_feats, _ = psf.build_interval_features(features, contacts, silver, spec, args.date_format)
    swap_cols = [c for c in orw.h2_columns() if "progress" in c]
    merged = dtr[[psf.CLIENT_ID]].merge(silver_feats[[psf.CLIENT_ID, *swap_cols]], on=psf.CLIENT_ID, how="left")
    for col in swap_cols:
        dtr[col] = merged[col].values
    return dtr, features, contacts, labels


def stage2_history_features(base_encoded: pd.DataFrame, a_prev: np.ndarray, contacts_prev: np.ndarray,
                            silver_max: np.ndarray, silver_latest: np.ndarray, silver_mean: np.ndarray) -> pd.DataFrame:
    out = base_encoded.copy()
    for level in QUINTILES:
        out[f"A_prev_{level}"] = (a_prev == level).astype(float)
    out["log_contacts_prev"] = np.log1p(np.asarray(contacts_prev, dtype=float))
    for name, vals in [("silver_max_prev", silver_max), ("silver_latest_prev", silver_latest), ("silver_mean_prev", silver_mean)]:
        vals = np.asarray(vals, dtype=float)
        observed = np.isfinite(vals) & (vals > psf.MISSING_PROGRESS_VALUE + 1)
        out[name] = np.where(observed, vals, 0.0)
        out[f"{name}_observed"] = observed.astype(float)
    return out


def fit_behavior(dtr: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Stage-level behavior policy on the real analysis population.

    Action bins are re-derived with the pipeline's own equal-mass binning applied to the
    analysis population itself (not to the wider positive-stage-1 cohort the real run bins on),
    so that the pipeline's per-sample quintiles on a simulated pool coincide in distribution
    with the bins the generator draws from.
    """
    dtr = dtr.copy()
    dtr["A1"] = psf.bin_nonzero_quintiles(dtr["contacts_t1"]).astype(int).values
    dtr["A2"] = psf.bin_nonzero_quintiles(dtr["contacts_t2"]).astype(int).values
    base = encode_baseline(dtr)
    base_cols = list(base.columns)
    g1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    g1.fit(base.values, dtr["A1"].astype(int).values)
    h2 = stage2_history_features(
        base, dtr["A1"].astype(int).values, dtr["contacts_t1"].values,
        dtr["max_progress_t1"].values, dtr["latest_progress_t1"].values, dtr["mean_progress_t1"].values,
    )
    h2_cols = list(h2.columns)
    g2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    g2.fit(h2.values, dtr["A2"].astype(int).values)
    count_pmfs: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    cutpoints: dict[str, list[float]] = {}
    for stage in (1, 2):
        pmfs = {}
        counts = dtr[f"contacts_t{stage}"].astype(int)
        bins = dtr[f"A{stage}"].astype(int)
        for q in QUINTILES:
            vals = counts[bins == q].value_counts().sort_index()
            pmfs[int(q)] = (vals.index.to_numpy(dtype=int), (vals / vals.sum()).to_numpy(dtype=float))
        count_pmfs[str(stage)] = pmfs
        cutpoints[str(stage)] = [float(counts[bins == q].max()) for q in QUINTILES[:-1]]
    return {
        "g1": g1, "g2": g2, "base_cols": base_cols, "h2_cols": h2_cols,
        "count_pmfs": count_pmfs, "quintile_upper_cutpoints": cutpoints,
        "g1_classes": [int(c) for c in g1.classes_], "g2_classes": [int(c) for c in g2.classes_],
    }


def fit_dirichlet_alpha(rng: np.random.Generator, totals: np.ndarray, target_zero_fraction: float) -> float:
    best = (None, None)
    for alpha in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        probs = rng.dirichlet(np.full(MONTHS_PER_STAGE, alpha), size=len(totals))
        counts = np.array([rng.multinomial(int(t), p) for t, p in zip(totals, probs)])
        zero = float((counts == 0).mean())
        gap = abs(zero - target_zero_fraction)
        if best[0] is None or gap < best[0]:
            best = (gap, alpha)
    return float(best[1])


def fit_emission(labels: pd.DataFrame, date_format: str | None) -> dict[str, Any]:
    """P(contact label | end-of-month merged bin) and within-bin label pmfs from real 2019 gold labels."""
    work = labels.copy()
    work["v"] = pd.to_numeric(work[psf.PRIMARY_LABEL], errors="coerce")
    work.loc[work["v"].eq(psf.INVALID_LABEL_SENTINEL), "v"] = 0.0
    work = work[work["v"].between(0, 4)].copy()
    work["v"] = grid_round(work["v"].to_numpy(dtype=float))
    work["dt"] = psf.parse_contact_dates(work[psf.CONTACT_DATE], None)  # label table carries ISO dates
    work = work[(work["dt"] >= "2019-01-01") & (work["dt"] < "2020-01-01")].sort_values([psf.CLIENT_ID, "dt"])
    work["m"] = work["dt"].dt.month
    work["bin"] = work["v"].map(merge_bin)
    endbin = work.groupby([psf.CLIENT_ID, "m"])["bin"].transform("last")
    emission = {}
    for b in PROGRESS_BINS:
        vals = work.loc[endbin == b, "v"].value_counts()
        vals = vals.reindex(LABEL_GRID, fill_value=0).astype(float)
        if vals.sum() == 0:
            vals[:] = 1.0
        emission[b] = (vals / vals.sum()).to_numpy()
    within = {}
    for b in PROGRESS_BINS:
        vals = work.loc[work["bin"] == b, "v"].value_counts().reindex(LABEL_GRID, fill_value=0).astype(float)
        if vals.sum() == 0:
            members = np.array([merge_bin(g) == b for g in LABEL_GRID], dtype=float)
            vals[:] = members
        within[b] = (vals / vals.sum()).to_numpy()
    return {"emission": emission, "within_bin": within, "label_grid": LABEL_GRID.tolist()}


def silver_confusion(labels: pd.DataFrame, silver: pd.DataFrame) -> dict[str, Any]:
    gold = psf.normalize_id_columns(labels[[psf.CONTACT_ID, psf.PRIMARY_LABEL]], [psf.CONTACT_ID]).copy()
    silv = psf.normalize_id_columns(silver[[psf.CONTACT_ID, psf.PRIMARY_LABEL]], [psf.CONTACT_ID]).copy()
    for d in (gold, silv):
        d["v"] = pd.to_numeric(d[psf.PRIMARY_LABEL], errors="coerce")
        d.loc[d["v"].eq(psf.INVALID_LABEL_SENTINEL), "v"] = 0.0
        d.loc[~d["v"].between(0, 4), "v"] = np.nan
    m = gold.groupby(psf.CONTACT_ID)["v"].max().to_frame("g").join(silv.groupby(psf.CONTACT_ID)["v"].max().to_frame("s"), how="inner").dropna()
    m["g"] = grid_round(m["g"].to_numpy()); m["s"] = grid_round(m["s"].to_numpy())
    rows = {}
    for g in LABEL_GRID:
        sub = m.loc[m["g"] == g, "s"].value_counts().reindex(LABEL_GRID, fill_value=0).astype(float)
        if sub.sum() == 0:
            sub[g] = 1.0
        rows[float(g)] = (sub / sub.sum()).to_numpy()
    return {
        "confusion": rows,
        "real_exact_agreement": float((m["g"] == m["s"]).mean()),
        "real_corr": float(np.corrcoef(m["g"], m["s"])[0, 1]),
        "real_rmse": float(np.sqrt(((m["g"] - m["s"]) ** 2).mean())),
        "real_silver_sd": float(m["s"].std()),
        "real_gold_sd": float(m["g"].std()),
        "matched": int(len(m)),
    }


def imputer_features(base_encoded: pd.DataFrame, month_index: np.ndarray, month_count: np.ndarray, cum_contacts: np.ndarray) -> np.ndarray:
    extra = np.column_stack([
        np.asarray(month_index, dtype=float) / 12.0,
        np.log1p(np.asarray(month_count, dtype=float)),
        np.log1p(np.asarray(cum_contacts, dtype=float)),
    ])
    return np.column_stack([base_encoded.values, extra])


def history_features(prev: np.ndarray, cummean: np.ndarray, last3: np.ndarray, cummax: np.ndarray) -> np.ndarray:
    """Client progress-history summaries (as of before the current contact/month); NaN -> 0 plus a missing flag."""
    H = np.column_stack([np.asarray(prev, float), np.asarray(cummean, float), np.asarray(last3, float), np.asarray(cummax, float)])
    miss = np.isnan(H).any(axis=1)
    return np.column_stack([np.nan_to_num(H, nan=0.0), miss.astype(float)])


def fit_history_imputer(features: pd.DataFrame, labels: pd.DataFrame, base_cols: list[str], seed: int) -> dict[str, Any]:
    """History-augmented misspecified linear imputer: gold label ~ covariates + month + counts + the client's own
    past-label history (previous label, running mean, mean of last three, running max, all before the contact).

    Fit on the real 2019 gold labels with real (gold) history; the simulator applies it with the client's gold
    history up to the previous month. Real-data ceiling: corr ~0.43 with the gold label (vs 0.20 covariates-only
    and 0.82 for the real LLM silver), so this is the intermediate-informativeness silver.
    """
    work = labels.copy()
    work[psf.CLIENT_ID] = work[psf.CLIENT_ID].astype(str).str.strip()
    work["v"] = pd.to_numeric(work[psf.PRIMARY_LABEL], errors="coerce")
    work.loc[work["v"].eq(psf.INVALID_LABEL_SENTINEL), "v"] = 0.0
    work = work[work["v"].between(0, 4)].copy()
    work["dt"] = psf.parse_contact_dates(work[psf.CONTACT_DATE], None)
    work = work.sort_values([psf.CLIENT_ID, "dt"])
    g = work.groupby(psf.CLIENT_ID)["v"]
    work["prev"] = g.shift(1)
    work["cummean"] = g.transform(lambda x: x.shift(1).expanding().mean())
    work["last3"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    work["cummax"] = g.transform(lambda x: x.shift(1).expanding().max())
    work["cum"] = work.groupby(psf.CLIENT_ID).cumcount()
    work = work[(work["dt"] >= "2019-01-01") & (work["dt"] < "2020-01-01")].copy()
    work["m"] = work["dt"].dt.month
    work["month_count"] = work.groupby([psf.CLIENT_ID, "m"])["v"].transform("size")
    feats = psf.normalize_id_columns(features, [psf.CLIENT_ID])
    work = work.merge(feats[[psf.CLIENT_ID, *psf.STATE_COLUMNS]], on=psf.CLIENT_ID, how="inner")
    base = encode_baseline(work, base_cols)
    x = np.column_stack([
        imputer_features(base, work["m"].values, work["month_count"].values, work["cum"].values),
        history_features(work["prev"].values, work["cummean"].values, work["last3"].values, work["cummax"].values),
    ])
    y = work["v"].to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y)
    pred = model.predict(x)
    return {"model": model, "fit_r2_real": float(1.0 - ((y - pred) ** 2).mean() / y.var()), "pred_sd_real": float(pred.std()), "n_fit": int(len(y))}


def fit_silver_imputer(features: pd.DataFrame, labels: pd.DataFrame, base_cols: list[str], date_format: str | None,
                       target_silver_sd: float, seed: int) -> dict[str, Any]:
    """Deliberately misspecified linear imputer: gold label ~ baseline covariates + month + monthly count + cumulative contacts.

    Fit on the REAL 2019 gold labels so it is a fixed function of pre-annotation information; it never sees the
    simulator's latent placement/progress state, so it is misspecified for the simulator by construction.
    """
    work = labels.copy()
    work[psf.CLIENT_ID] = work[psf.CLIENT_ID].astype(str).str.strip()
    work["v"] = pd.to_numeric(work[psf.PRIMARY_LABEL], errors="coerce")
    work.loc[work["v"].eq(psf.INVALID_LABEL_SENTINEL), "v"] = 0.0
    work = work[work["v"].between(0, 4)].copy()
    work["dt"] = psf.parse_contact_dates(work[psf.CONTACT_DATE], None)  # label table carries ISO dates
    work = work[(work["dt"] >= "2019-01-01") & (work["dt"] < "2020-01-01")].sort_values([psf.CLIENT_ID, "dt"])
    work["m"] = work["dt"].dt.month
    work["month_count"] = work.groupby([psf.CLIENT_ID, "m"])["v"].transform("size")
    work["cum"] = work.groupby(psf.CLIENT_ID).cumcount()
    feats = psf.normalize_id_columns(features, [psf.CLIENT_ID])
    work = work.merge(feats[[psf.CLIENT_ID, *psf.STATE_COLUMNS]], on=psf.CLIENT_ID, how="inner")
    base = encode_baseline(work, base_cols)
    x = imputer_features(base, work["m"].values, work["month_count"].values, work["cum"].values)
    y = work["v"].to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y)
    pred = model.predict(x)
    r2 = float(1.0 - ((y - pred) ** 2).mean() / y.var())
    noise_sd = float(np.sqrt(max(target_silver_sd**2 - pred.var(), 0.0)))
    return {"model": model, "fit_r2_real": r2, "pred_sd_real": float(pred.std()), "noise_sd": noise_sd, "n_fit": int(len(y))}


# ----------------------------------------------------------------------------- initial-state pool
def build_initial_pool(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    jan = panel[panel["period_id"] == JAN_2019_PERIOD].copy()
    jan["ClientID"] = jan["ClientID"].astype(str).str.strip()
    state = parse_state(jan["state"])
    init = pd.DataFrame({
        psf.CLIENT_ID: jan["ClientID"].values,
        "h0": state["h"].astype(int).values,
        "p0": state["p"].astype(str).values,
        "svc0": state["svcprev"].astype(str).values,
        "demo": state["demo"].astype(str).values,
        "prior_contacts0": pd.to_numeric(jan["prior_contacts"], errors="coerce").fillna(0).astype(int).values,
        "prior_shelter0": pd.to_numeric(jan["prior_shelter_contacts"], errors="coerce").fillna(0).astype(int).values,
    })
    feats = psf.normalize_id_columns(features, [psf.CLIENT_ID])
    pool = init.merge(feats[[psf.CLIENT_ID, *psf.STATE_COLUMNS]], on=psf.CLIENT_ID, how="inner")
    return pool.reset_index(drop=True)


# ----------------------------------------------------------------------------- rollout engine
class Rollout:
    """Vectorized 24-month rollout for a set of clients drawn from the initial-state pool."""

    def __init__(self, bundle: dict[str, Any], rng: np.random.Generator, pool_idx: np.ndarray, shift_quintiles: int,
                 label_coverage: float, silver_mode: str, emit_contacts: bool, silver_beta: float = 0.0,
                 pipeline_policy: dict[str, Any] | None = None, propensity_floor: float = 0.0):
        self.b = bundle
        self.propensity_floor = float(propensity_floor)
        # optional: define the target policy exactly as the pipeline does (shift of ITS fitted ridge-softmax
        # propensity, quintile bins = pipeline qcut on the pool) instead of the shift of the true behavior policy
        self.pipeline_policy = pipeline_policy
        self.silver_beta = float(silver_beta) if silver_mode == "impute_mix" else 0.0
        self.rng = rng
        self.n = len(pool_idx)
        self.shift = int(shift_quintiles)
        self.coverage = float(label_coverage)
        self.silver_mode = silver_mode
        self.emit = emit_contacts
        pool = bundle["initial_pool"].iloc[pool_idx].reset_index(drop=True)
        self.pool = pool
        self.base_encoded = encode_baseline(pool, bundle["behavior"]["base_cols"])
        self.h = pool["h0"].to_numpy(dtype=int)
        self.p = pool["p0"].to_numpy(dtype=object)
        self.svc = np.array([SVC_BINS.index(s) if s in SVC_BINS else 0 for s in pool["svc0"]], dtype=int)
        self.demo = pool["demo"].to_numpy(dtype=object)
        self.prior_contacts = pool["prior_contacts0"].to_numpy(dtype=int).copy()
        self.prior_shelter = pool["prior_shelter0"].to_numpy(dtype=int).copy()
        self.stage_total = np.zeros((N_STAGES, self.n), dtype=int)
        self.stage_quintile = np.zeros((N_STAGES, self.n), dtype=int)
        self.stage_gold_max = np.full((N_STAGES, self.n), -np.inf)
        self.stage_silver_max = np.full((N_STAGES, self.n), -np.inf)
        self.stage_silver_min = np.full((N_STAGES, self.n), np.inf)
        self.stage_silver_sum = np.zeros((N_STAGES, self.n))
        self.stage_silver_cnt = np.zeros((N_STAGES, self.n))
        self.stage_silver_latest = np.full((N_STAGES, self.n), np.nan)
        # client gold-label history (all labeled contacts so far) for the history-augmented silver imputer
        self.hist_cnt = np.zeros(self.n)
        self.hist_sum = np.zeros(self.n)
        self.hist_max = np.full(self.n, np.nan)
        self.hist_prev = np.full(self.n, np.nan)
        self.hist_last3 = np.full((self.n, 3), np.nan)
        self.g_true = np.full((2, self.n, len(QUINTILES)), np.nan)  # true behavior propensities at the realized H1 / H2
        self.h_path = np.zeros((25, self.n), dtype=int)
        self.h_path[0] = self.h
        self.monthly_counts = np.zeros((24, self.n), dtype=int)
        self.contact_frames: list[pd.DataFrame] = []
        self.label_frames: list[pd.DataFrame] = []
        self.silver_frames: list[pd.DataFrame] = []
        self.contact_counter = 0

    # -- behavior / target policy at stage level
    def draw_stage(self, stage: int) -> None:
        beh = self.b["behavior"]
        if stage == 0:
            probs = beh["g1"].predict_proba(self.base_encoded.values)
            classes = np.array(beh["g1_classes"])
        else:
            prev = stage - 1
            cnt = self.stage_silver_cnt[prev]
            mean = np.where(cnt > 0, self.stage_silver_sum[prev] / np.maximum(cnt, 1), psf.MISSING_PROGRESS_VALUE)
            smax = np.where(cnt > 0, self.stage_silver_max[prev], psf.MISSING_PROGRESS_VALUE)
            latest = np.where(cnt > 0, self.stage_silver_latest[prev], psf.MISSING_PROGRESS_VALUE)
            feats = stage2_history_features(self.base_encoded, self.stage_quintile[prev], self.stage_total[prev], smax, latest, mean)
            feats = feats.reindex(columns=beh["h2_cols"], fill_value=0.0)
            probs = beh["g2"].predict_proba(feats.values)
            classes = np.array(beh["g2_classes"])
        if stage < 2:
            full = np.zeros((self.n, len(QUINTILES)))
            for j, cls in enumerate(classes):
                full[:, int(cls) - 1] = probs[:, j]
            self.g_true[stage] = full
        q = classes[sample_categorical(self.rng, probs)]
        if stage < 2 and self.shift > 0 and self.propensity_floor > 0 and self.pipeline_policy is None:
            # Pipeline-consistent evaluation policy: the pipeline clips propensities at the floor and renormalizes
            # before shifting, so redraw the behavior quintile from the floored policy for the shift.
            floored = np.maximum(probs, self.propensity_floor)
            floored = floored / floored.sum(axis=1, keepdims=True)
            q = classes[sample_categorical(self.rng, floored)]
        pmf_stage = "1" if stage == 0 else "2"
        pmfs = beh["count_pmfs"][pmf_stage]
        if stage < 2 and self.shift > 0 and self.pipeline_policy is not None and self.pipeline_policy.get("true_g_pool_bins"):
            # Diagnostic: TRUE behavior policy, but the shift acts on the POOL's quintile bins (the bins the pipeline's
            # per-sample qcut converges to), with totals drawn from the pool's within-bin count distributions.
            pp = self.pipeline_policy
            totals_beh = np.zeros(self.n, dtype=int)
            for level in QUINTILES:
                mask = q == level
                if mask.any():
                    support, weights = pmfs[int(level)]
                    totals_beh[mask] = self.rng.choice(support, size=int(mask.sum()), p=weights)
            cuts = np.asarray(pp["cutpoints"][pmf_stage], dtype=float)
            q = np.minimum(1 + np.searchsorted(cuts, totals_beh, side="left") + self.shift, 5)
            pmfs = pp["count_pmfs"][pmf_stage]
        elif stage < 2 and self.shift > 0 and self.pipeline_policy is not None:
            pp = self.pipeline_policy
            frame = self.pipeline_history_frame(stage)
            g_hat = orw.predict_action_probs(pp[f"model{stage + 1}"], pp[f"cols{stage + 1}"], frame, pp[f"columns{stage + 1}"], pp["floor"])
            pi_hat = orw.shift_up_probs(g_hat, self.shift)
            q = QUINTILES[sample_categorical(self.rng, pi_hat)]
            pmfs = pp["count_pmfs"][pmf_stage]  # pool empirical totals within PIPELINE bins
        elif stage < 2 and self.shift > 0:  # target policy acts on the two 2019 decision stages only
            q = np.minimum(q + self.shift, 5)
        totals = np.zeros(self.n, dtype=int)
        for level in QUINTILES:
            mask = q == level
            if mask.any():
                support, weights = pmfs[int(level)]
                totals[mask] = self.rng.choice(support, size=int(mask.sum()), p=weights)
        self.stage_quintile[stage] = q
        self.stage_total[stage] = totals
        alpha = self.b["dirichlet_alpha"]
        shares = self.rng.dirichlet(np.full(MONTHS_PER_STAGE, alpha), size=self.n)
        counts = np.zeros((MONTHS_PER_STAGE, self.n), dtype=int)
        for i in range(self.n):
            if totals[i] > 0:
                counts[:, i] = self.rng.multinomial(totals[i], shares[i])
        self.monthly_counts[stage * MONTHS_PER_STAGE:(stage + 1) * MONTHS_PER_STAGE] = counts

    def pipeline_history_frame(self, stage: int) -> pd.DataFrame:
        """Reconstruct the pipeline's H1 (stage 0) or H2 (stage 1) columns for the rolled-out clients."""
        frame = self.pool[psf.STATE_COLUMNS].copy().reset_index(drop=True)
        if stage == 0:
            return frame
        pp = self.pipeline_policy
        t = self.stage_total[0]
        cuts = np.asarray(pp["cutpoints"]["1"], dtype=float)
        a1 = 1 + np.searchsorted(cuts, t, side="left")  # pipeline quintile of stage-1 total under pool cutpoints
        frame["A1"] = a1.astype(int)
        frame["contacts_t1"] = t.astype(int)
        frame["outreach_bin_t1"] = a1.astype(int)
        cnt = self.stage_silver_cnt[0]
        obs = cnt > 0
        frame["max_progress_t1"] = np.where(obs, self.stage_silver_max[0], psf.MISSING_PROGRESS_VALUE)
        frame["mean_progress_t1"] = np.where(obs, self.stage_silver_sum[0] / np.maximum(cnt, 1), psf.MISSING_PROGRESS_VALUE)
        frame["latest_progress_t1"] = np.where(obs, self.stage_silver_latest[0], psf.MISSING_PROGRESS_VALUE)
        frame["progress_range_t1"] = np.where(obs, self.stage_silver_max[0] - self.stage_silver_min[0], 0.0)
        frame["progress_label_count_t1"] = cnt.astype(int)
        frame["progress_observed_t1"] = obs.astype(int)
        return frame

    # -- one month of transitions and emissions
    def step_month(self, month: int) -> None:
        counts = self.monthly_counts[month]
        active = np.flatnonzero(counts > 0)
        stage = month // MONTHS_PER_STAGE
        h_next = self.h.copy()
        p_next = self.p.copy()
        svc_next = self.svc.copy()
        if len(active) > 0:
            feats = pd.DataFrame({
                "h": self.h[active].astype(str),
                "p": self.p[active].astype(str),
                "svcprev": np.array(SVC_BINS, dtype=object)[self.svc[active]],
                "demo": self.demo[active],
                "shelter": shelter_bin(self.prior_shelter[active]),
                "hist": hist_bin(self.prior_contacts[active]),
                "action_bin": action_bin(counts[active]),
                "log_action_count": np.log1p(counts[active].astype(float)),
                "log_prior_contacts": np.log1p(self.prior_contacts[active].astype(float)),
                "log_prior_shelter": np.log1p(self.prior_shelter[active].astype(float)),
            })
            heads = self.b["heads"]
            hp = heads["h"].predict_proba(feats)
            h_next[active] = np.array([int(c) for c in heads["h"].classes_])[sample_categorical(self.rng, hp)]
            pp = heads["p"].predict_proba(feats)
            p_next[active] = np.array([str(c) for c in heads["p"].classes_], dtype=object)[sample_categorical(self.rng, pp)]
            sp = heads["svc"].predict_proba(feats)
            svc_next[active] = np.array([SVC_BINS.index(str(c)) for c in heads["svc"].classes_])[sample_categorical(self.rng, sp)]
            self.emit_month(month, stage, active, counts[active], h_next[active], p_next[active])
        self.prior_contacts += counts
        self.h = h_next
        self.p = p_next
        self.svc = svc_next
        self.h_path[month + 1] = self.h

    def emit_month(self, month: int, stage: int, active: np.ndarray, counts: np.ndarray, h_new: np.ndarray, p_new: np.ndarray) -> None:
        rng = self.rng
        total = int(counts.sum())
        owner = np.repeat(np.arange(len(active)), counts)  # index into `active`
        pos = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        is_last = pos == counts[owner] - 1
        # placement carried by each contact: start-of-month state except the last contact carries the new state
        placement = np.where(is_last, h_new[owner], self.h[active][owner])
        self.prior_shelter[active] += np.bincount(owner, weights=(placement == 1).astype(float), minlength=len(active)).astype(int)
        # gold labels: last contact ~ within-bin pmf of the new state; others ~ emission table given the new state
        em = self.b["emission"]
        gold = np.empty(total)
        for b in PROGRESS_BINS:
            mask_bin = (p_new[owner] == b)
            if not mask_bin.any():
                continue
            m_last = mask_bin & is_last
            m_other = mask_bin & ~is_last
            if m_last.any():
                gold[m_last] = rng.choice(LABEL_GRID, size=int(m_last.sum()), p=em["within_bin"][b])
            if m_other.any():
                gold[m_other] = rng.choice(LABEL_GRID, size=int(m_other.sum()), p=em["emission"][b])
        labeled = is_last | (rng.uniform(size=total) < self.coverage)
        # silver labels
        if self.silver_mode == "noisy_gold":
            # measurement-error silver: draw from the real gold->silver confusion row (matches the real LLM error structure)
            conf = self.b["silver_confusion"]["confusion"]
            silver = np.empty(total)
            for g in LABEL_GRID:
                m = gold == g
                if m.any():
                    silver[m] = rng.choice(LABEL_GRID, size=int(m.sum()), p=conf[float(g)])
        elif self.silver_mode == "impute_hist":
            # history-augmented misspecified imputer: covariates + the client's gold-label history up to the previous month
            imp = self.b["silver_history_imputer"]
            cummean = np.where(self.hist_cnt[active] > 0, self.hist_sum[active] / np.maximum(self.hist_cnt[active], 1), np.nan)
            last3 = np.nanmean(np.where(np.isnan(self.hist_last3[active]), np.nan, self.hist_last3[active]), axis=1) if len(active) else np.array([])
            x = np.column_stack([
                imputer_features(self.base_encoded.iloc[active], np.full(len(active), month % 12 + 1), counts, self.prior_contacts[active]),
                history_features(self.hist_prev[active], cummean, last3, self.hist_max[active]),
            ])
            pred = imp["model"].predict(x)[owner]
            real = self.b["silver_confusion"]
            noise_sd = float(np.sqrt(max(real["real_silver_sd"] ** 2 - imp["pred_sd_real"] ** 2, 0.0)))
            silver = grid_round(pred + rng.normal(scale=noise_sd, size=total))
        else:
            # imputation silver: misspecified linear imputer (fit on real data, covariates only), optionally mixed
            # with a fraction beta of the true label (beta=0 is pure imputation; beta~0.7 matches the real silver-gold corr)
            imp = self.b["silver_imputer"]
            x = imputer_features(self.base_encoded.iloc[active], np.full(len(active), month % 12 + 1), counts, self.prior_contacts[active])
            pred = imp["model"].predict(x)[owner]
            beta = self.silver_beta
            core = pred + beta * (gold - pred)
            real = self.b["silver_confusion"]
            core_var = (beta ** 2) * real["real_gold_sd"] ** 2 + ((1 - beta) ** 2) * imp["pred_sd_real"] ** 2
            noise_sd = float(np.sqrt(max(real["real_silver_sd"] ** 2 - core_var, 0.0)))
            silver = grid_round(core + rng.normal(scale=noise_sd, size=total))
        # stage aggregates over labeled contacts (labels are only observed on labeled contacts)
        idx = active[owner]
        lab_idx = idx[labeled]
        np.maximum.at(self.stage_gold_max[stage], lab_idx, gold[labeled])
        np.maximum.at(self.stage_silver_max[stage], lab_idx, silver[labeled])
        np.minimum.at(self.stage_silver_min[stage], lab_idx, silver[labeled])
        np.add.at(self.stage_silver_sum[stage], lab_idx, silver[labeled])
        np.add.at(self.stage_silver_cnt[stage], lab_idx, 1.0)
        last_idx = idx[is_last]
        self.stage_silver_latest[stage, last_idx] = silver[is_last]  # the last contact of the month is always labeled
        # gold history update (labeled contacts only): counts/sums/max vectorised; last-three buffer per client
        np.add.at(self.hist_cnt, lab_idx, 1.0)
        np.add.at(self.hist_sum, lab_idx, gold[labeled])
        cur_max = np.where(np.isnan(self.hist_max), -np.inf, self.hist_max)
        np.maximum.at(cur_max, lab_idx, gold[labeled])
        self.hist_max = np.where(np.isfinite(cur_max), cur_max, np.nan)
        self.hist_prev[last_idx] = gold[is_last]
        lab_owner = owner[labeled]
        lab_gold = gold[labeled]
        for j in np.unique(lab_owner):
            vals = lab_gold[lab_owner == j][-3:]
            buf = np.concatenate([self.hist_last3[active[j]], vals])
            buf = buf[~np.isnan(buf)][-3:]
            self.hist_last3[active[j]] = np.concatenate([np.full(3 - len(buf), np.nan), buf])
        if self.emit:
            year = 2019 + month // 12
            mon = month % 12 + 1
            days_in_month = pd.Timestamp(year=year, month=mon, day=1).days_in_month
            # contacts are ordered within each client-month block: sort random days per block
            day_sorted = np.empty(total, dtype=int)
            rand_days = rng.integers(0, days_in_month, size=total)
            for start, c in zip(np.cumsum(counts) - counts, counts):
                day_sorted[start:start + c] = np.sort(rand_days[start:start + c])
            dates = pd.to_datetime({"year": np.full(total, year), "month": np.full(total, mon), "day": day_sorted + 1})
            cids = np.arange(self.contact_counter, self.contact_counter + total)
            self.contact_counter += total
            contact_ids = np.char.add("SC", np.char.zfill(cids.astype(str), 9))
            client_ids = self.client_ids[idx]
            self.contact_frames.append(pd.DataFrame({
                psf.CLIENT_ID: client_ids, psf.CONTACT_ID: contact_ids,
                psf.CONTACT_DATE: dates.dt.strftime("%Y-%m-%d").values, psf.PLACEMENT_TYPE: placement.astype(int),
            }))
            self.label_frames.append(pd.DataFrame({psf.CONTACT_ID: contact_ids[labeled], psf.PRIMARY_LABEL: gold[labeled]}))
            self.silver_frames.append(pd.DataFrame({psf.CONTACT_ID: contact_ids[labeled], psf.PRIMARY_LABEL: silver[labeled]}))

    def run(self, client_ids: np.ndarray | None = None) -> None:
        self.client_ids = client_ids if client_ids is not None else np.array([f"SIM{i:06d}" for i in range(self.n)], dtype=object)
        for stage in range(N_STAGES):
            self.draw_stage(stage)
            for month in range(stage * MONTHS_PER_STAGE, (stage + 1) * MONTHS_PER_STAGE):
                self.step_month(month)

    def outcomes(self) -> pd.DataFrame:
        r1 = np.where(np.isfinite(self.stage_gold_max[0]), self.stage_gold_max[0], np.nan)
        r2 = np.where(np.isfinite(self.stage_gold_max[1]), self.stage_gold_max[1], np.nan)
        baseline_2020 = self.h_path[12]
        y2_any = (self.h_path[13:25].max(axis=0) > baseline_2020).astype(float)
        year2_contacts = self.monthly_counts[12:24].sum(axis=0)
        out = pd.DataFrame({
            "R1_progress": r1, "R2_progress": r2, "progress_total": r1 + r2,
            "y2_any_placement_improvement": np.where(year2_contacts > 0, y2_any, np.nan),
            "A1": self.stage_quintile[0], "A2": self.stage_quintile[1],
            "contacts_t1": self.stage_total[0], "contacts_t2": self.stage_total[1],
            "h_start": self.h_path[0], "h_end_2019": self.h_path[12],
        })
        for stage in (0, 1):
            for level in QUINTILES:
                out[f"oracle_g{stage + 1}_{level}"] = self.g_true[stage, :, level - 1]
        return out


# ----------------------------------------------------------------------------- subcommands
def cmd_fit(args: argparse.Namespace) -> None:
    t0 = time.time()
    panel = pd.read_csv(args.panel_csv, low_memory=False)
    panel["ClientID"] = panel["ClientID"].astype(str).str.strip()
    print(f"[fit] panel rows={len(panel)} clients={panel['ClientID'].nunique()}")
    heads, head_summary = fit_heads(panel, args.simulator_kind, args.seed, [float(g) for g in args.rbf_gammas.split(",")])
    print(f"[fit] heads done in {time.time() - t0:.1f}s: " + json.dumps({k: round(v['val_nll'], 4) for k, v in head_summary['heads'].items()}))
    dtr, features, contacts, labels = real_dtr_with_silver_h2(args)
    behavior = fit_behavior(dtr, args.seed)
    rng = np.random.default_rng(args.seed)
    totals = np.concatenate([dtr["contacts_t1"].to_numpy(dtype=int), dtr["contacts_t2"].to_numpy(dtype=int)])
    alpha = fit_dirichlet_alpha(rng, totals, args.target_zero_month_fraction)
    labels_full = orw.read_selected_csv(Path(args.labels_csv), [psf.CLIENT_ID, psf.CONTACT_ID, psf.CONTACT_DATE, psf.PRIMARY_LABEL], "labels CSV")
    emission = fit_emission(labels_full, args.date_format)
    silver = orw.read_selected_csv(Path(args.silver_csv), [psf.CONTACT_ID, psf.PRIMARY_LABEL], "silver labels CSV")
    confusion = silver_confusion(labels_full, silver)
    imputer = fit_silver_imputer(features, labels_full, behavior["base_cols"], args.date_format, confusion["real_silver_sd"], args.seed)
    hist_imputer = fit_history_imputer(features, labels_full, behavior["base_cols"], args.seed)
    pool = build_initial_pool(panel, features)
    real_marginals = {
        "n_analysis_rows": int(len(dtr)),
        "contacts_t1_quantiles": dtr["contacts_t1"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist(),
        "contacts_t2_quantiles": dtr["contacts_t2"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist(),
        "R1_mean": float(dtr["R1_progress"].mean()), "R1_sd": float(dtr["R1_progress"].std()),
        "R2_mean": float(dtr["R2_progress"].mean()), "R2_sd": float(dtr["R2_progress"].std()),
        "progress_total_mean": float((dtr["R1_progress"] + dtr["R2_progress"]).mean()),
        "y2_any_rate": float(dtr["Y2_any_placement_improvement"].mean()),
        "baseline_placement_dist": dtr["max_PlacementType_before_2019"].value_counts(normalize=True).sort_index().round(3).to_dict(),
        "A1_dist": dtr["A1"].value_counts(normalize=True).sort_index().round(3).to_dict(),
        "quintile_upper_cutpoints": behavior["quintile_upper_cutpoints"],
        "zero_month_fraction_target": args.target_zero_month_fraction,
        "label_coverage_real": 0.70,
        "silver": {k: v for k, v in confusion.items() if k != "confusion"},
        "behavior_mean_value_progress_total": float((dtr["R1_progress"] + dtr["R2_progress"]).mean()),
    }
    bundle = {
        "kind": args.simulator_kind, "heads": heads, "head_summary": head_summary, "behavior": behavior,
        "dirichlet_alpha": alpha, "emission": emission, "silver_confusion": confusion, "silver_imputer": imputer,
        "silver_history_imputer": hist_imputer, "initial_pool": pool, "real_marginals": real_marginals, "seed": args.seed, "panel_csv": str(args.panel_csv),
    }
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.bundle)
    summary = {
        "kind": args.simulator_kind, "heads": head_summary, "dirichlet_alpha": alpha,
        "initial_pool_clients": int(len(pool)),
        "silver_imputer": {k: v for k, v in imputer.items() if k != "model"},
        "silver_history_imputer": {k: v for k, v in hist_imputer.items() if k != "model"},
        "silver_real": real_marginals["silver"],
        "behavior": {"g1_classes": behavior["g1_classes"], "g2_classes": behavior["g2_classes"], "cutpoints": behavior["quintile_upper_cutpoints"]},
        "real_marginals": real_marginals, "elapsed_s": time.time() - t0,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


def cmd_generate(args: argparse.Namespace) -> None:
    t0 = time.time()
    bundle = joblib.load(args.bundle)
    rng = np.random.default_rng(args.seed)
    pool_idx = rng.integers(0, len(bundle["initial_pool"]), size=args.n_clients)
    ro = Rollout(bundle, rng, pool_idx, shift_quintiles=0, label_coverage=args.label_coverage, silver_mode=args.silver_mode, emit_contacts=True, silver_beta=args.silver_beta)
    ro.run()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = ro.pool[psf.STATE_COLUMNS].copy()
    features.insert(0, psf.CLIENT_ID, ro.client_ids)
    features.to_csv(args.out_dir / "features.csv", index=False)
    contacts = pd.concat(ro.contact_frames, ignore_index=True)
    contacts.to_csv(args.out_dir / "contacts.csv", index=False)
    pd.concat(ro.label_frames, ignore_index=True).to_csv(args.out_dir / "gold_labels.csv", index=False)
    pd.concat(ro.silver_frames, ignore_index=True).to_csv(args.out_dir / "silver_labels.csv", index=False)
    out = ro.outcomes()
    out.insert(0, psf.CLIENT_ID, ro.client_ids)
    out.to_csv(args.out_dir / "simulator_outcomes.csv", index=False)
    meta = {"bundle": str(args.bundle), "seed": args.seed, "n_clients": args.n_clients, "label_coverage": args.label_coverage,
            "silver_mode": args.silver_mode, "silver_beta": args.silver_beta, "contacts": int(len(contacts)), "elapsed_s": time.time() - t0}
    (args.out_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def fit_pipeline_policy(pool_dir: Path, floor: float = 0.02, learner: str = "ridge_softmax", propensity_c: float = 1.0) -> dict[str, Any]:
    """Fit the pipeline's own behavior-propensity models (ridge-softmax on one-hot H1 / H2, silver H2 summaries)
    on a generated pool, plus the pool's pipeline quintile bins and within-bin count distributions."""
    pipe_args = argparse.Namespace(
        features_csv=str(pool_dir / "features.csv"), contacts_csv=str(pool_dir / "contacts.csv"),
        labels_csv=str(pool_dir / "gold_labels.csv"), date_format="%Y-%m-%d", start_date="2019-01-01",
        interval_months=6, num_decision_intervals=2, placement_outcome_end_date="2021-01-01",
        n_estimators=200, min_samples_leaf=10, propensity_learner=learner, propensity_c=propensity_c,
    )
    orw.HAVE_SKLEARN = False
    features, contacts, labels = orw.load_inputs(pipe_args)
    dtr, _, _ = orw.build_dtr_frame(features, contacts, labels, pipe_args)
    spec = psf.IntervalSpec(pd.Timestamp("2019-01-01"), 6, 2)
    silver = orw.read_selected_csv(pool_dir / "silver_labels.csv", [psf.CONTACT_ID, psf.PRIMARY_LABEL], "silver labels CSV")
    silver_feats, _ = psf.build_interval_features(features, contacts, silver, spec, "%Y-%m-%d")
    swap_cols = [c for c in orw.h2_columns() if "progress" in c]
    merged = dtr[[psf.CLIENT_ID]].merge(silver_feats[[psf.CLIENT_ID, *swap_cols]], on=psf.CLIENT_ID, how="left")
    for col in swap_cols:
        dtr[col] = merged[col].values
    model1, cols1 = orw.fit_classifier(dtr, orw.h1_columns(), dtr["A1"], pipe_args, seed=0)
    model2, cols2 = orw.fit_classifier(dtr, orw.h2_columns(), dtr["A2"], pipe_args, seed=1)
    cutpoints, pmfs = {}, {}
    for stage in (1, 2):
        counts = dtr[f"contacts_t{stage}"].astype(int)
        bins = dtr[f"outreach_bin_t{stage}"].astype(int)
        cutpoints[str(stage)] = [float(counts[bins == q].max()) for q in QUINTILES[:-1]]
        pmfs[str(stage)] = {}
        for q in QUINTILES:
            vals = counts[bins == q].value_counts().sort_index()
            pmfs[str(stage)][int(q)] = (vals.index.to_numpy(dtype=int), (vals / vals.sum()).to_numpy(dtype=float))
    return {"model1": model1, "cols1": cols1, "columns1": orw.h1_columns(), "model2": model2, "cols2": cols2,
            "columns2": orw.h2_columns(), "cutpoints": cutpoints, "count_pmfs": pmfs, "floor": floor, "n_pool": int(len(dtr))}


def cmd_truth(args: argparse.Namespace) -> None:
    t0 = time.time()
    bundle = joblib.load(args.bundle)
    pipeline_policy = fit_pipeline_policy(args.pipeline_policy_pool, learner=args.propensity_learner, propensity_c=args.propensity_c) if args.pipeline_policy_pool else None
    if pipeline_policy is not None and args.true_g_pool_bins:
        pipeline_policy["true_g_pool_bins"] = True
    if pipeline_policy is not None:
        print(f"[truth] target policy = shift of the PIPELINE's fitted propensity on {args.pipeline_policy_pool} (n={pipeline_policy['n_pool']}, cutpoints {pipeline_policy['cutpoints']})")
    results: dict[str, Any] = {"bundle": str(args.bundle), "seed": args.seed, "n_rollouts": args.n_rollouts,
                               "shift_quintiles": args.shift_quintiles, "label_coverage": args.label_coverage, "silver_mode": args.silver_mode, "silver_beta": args.silver_beta,
                               "pipeline_policy_pool": str(args.pipeline_policy_pool) if args.pipeline_policy_pool else None, "propensity_floor": args.propensity_floor,
                               "propensity_learner": args.propensity_learner if args.pipeline_policy_pool else None,
                               "propensity_c": args.propensity_c if args.pipeline_policy_pool else None}
    for name, shift in [("target_policy", args.shift_quintiles), ("behavior_policy", 0)]:
        rng = np.random.default_rng(args.seed + (0 if shift else 1))
        chunks = []
        remaining = args.n_rollouts
        while remaining > 0:
            m = min(remaining, args.chunk)
            pool_idx = rng.integers(0, len(bundle["initial_pool"]), size=m)
            ro = Rollout(bundle, rng, pool_idx, shift_quintiles=shift, label_coverage=args.label_coverage, silver_mode=args.silver_mode, emit_contacts=False, silver_beta=args.silver_beta, pipeline_policy=pipeline_policy, propensity_floor=args.propensity_floor)
            ro.run()
            chunks.append(ro.outcomes())
            remaining -= m
        out = pd.concat(chunks, ignore_index=True)
        block = {}
        for target in ["progress_total", "y2_any_placement_improvement", "R1_progress", "R2_progress"]:
            vals = out[target].dropna().to_numpy(dtype=float)
            block[target] = {"value": float(vals.mean()), "mc_se": float(vals.std(ddof=1) / np.sqrt(len(vals))), "n": int(len(vals))}
        block["A1_dist"] = out["A1"].value_counts(normalize=True).sort_index().round(4).to_dict()
        block["A2_dist"] = out["A2"].value_counts(normalize=True).sort_index().round(4).to_dict()
        block["missing_reward_rate"] = float(out["progress_total"].isna().mean())
        results[name] = block
        print(f"[truth] {name}: progress_total={block['progress_total']['value']:.4f} y2_any={block['y2_any_placement_improvement']['value']:.4f} ({time.time() - t0:.0f}s)")
    results["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))


def cmd_marginals(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.bundle)
    real = bundle["real_marginals"]
    pipe_args = argparse.Namespace(
        features_csv=str(args.pool_dir / "features.csv"), contacts_csv=str(args.pool_dir / "contacts.csv"),
        labels_csv=str(args.pool_dir / "gold_labels.csv"), date_format="%Y-%m-%d", start_date="2019-01-01",
        interval_months=6, num_decision_intervals=2, placement_outcome_end_date="2021-01-01",
    )
    orw.HAVE_SKLEARN = False
    features, contacts, labels = orw.load_inputs(pipe_args)
    dtr, coverage, population = orw.build_dtr_frame(features, contacts, labels, pipe_args)
    sim_out = pd.read_csv(args.pool_dir / "simulator_outcomes.csv")
    merged = dtr[[psf.CLIENT_ID, "R1_progress", "R2_progress", "Y2_any_placement_improvement", "A1", "A2"]].merge(sim_out, on=psf.CLIENT_ID, suffixes=("_pipe", "_sim"))
    consistency = {
        "R1_match": float((merged["R1_progress_pipe"] == merged["R1_progress_sim"]).mean()),
        "R2_match": float((merged["R2_progress_pipe"] == merged["R2_progress_sim"]).mean()),
        "Y2_match": float((merged["Y2_any_placement_improvement"] == merged["y2_any_placement_improvement"]).mean()),
        "A1_match_pipeline_qcut_vs_sim_quintile": float((merged["A1_pipe"] == merged["A1_sim"]).mean()),
        "A2_match_pipeline_qcut_vs_sim_quintile": float((merged["A2_pipe"] == merged["A2_sim"]).mean()),
    }
    sim = {
        "n_pool_clients": int(len(features)), "n_analysis_rows": int(len(dtr)),
        "population": population.iloc[0].to_dict(),
        "contacts_t1_quantiles": dtr["contacts_t1"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist(),
        "contacts_t2_quantiles": dtr["contacts_t2"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist(),
        "R1_mean": float(dtr["R1_progress"].mean()), "R1_sd": float(dtr["R1_progress"].std()),
        "R2_mean": float(dtr["R2_progress"].mean()), "R2_sd": float(dtr["R2_progress"].std()),
        "progress_total_mean": float((dtr["R1_progress"] + dtr["R2_progress"]).mean()),
        "y2_any_rate": float(dtr["Y2_any_placement_improvement"].mean()),
        "baseline_placement_dist": dtr["max_PlacementType_before_2019"].value_counts(normalize=True).sort_index().round(3).to_dict(),
        "A1_dist": dtr["A1"].value_counts(normalize=True).sort_index().round(3).to_dict(),
        "pipeline_qcut_upper_cutpoints": {str(s): [float(dtr.loc[dtr[f'outreach_bin_t{s}'] == q, f'contacts_t{s}'].max()) for q in QUINTILES[:-1]] for s in (1, 2)},
        "label_coverage": coverage["label_coverage_rate"].round(3).tolist(),
        "zero_contact_month_fraction_2019": None,
    }
    c = contacts.copy()
    c[psf.CONTACT_DATE] = pd.to_datetime(c[psf.CONTACT_DATE])
    c19 = c[(c[psf.CONTACT_DATE] >= "2019-01-01") & (c[psf.CONTACT_DATE] < "2020-01-01")]
    cm = c19.groupby([psf.CLIENT_ID, c19[psf.CONTACT_DATE].dt.month]).size().unstack(fill_value=0).reindex(features[psf.CLIENT_ID], fill_value=0)
    sim["zero_contact_month_fraction_2019"] = float((cm == 0).mean().mean())
    # silver agreement on the pool
    silver = pd.read_csv(args.pool_dir / "silver_labels.csv")
    gs = labels.merge(silver, on=psf.CONTACT_ID, suffixes=("_g", "_s"))
    g = pd.to_numeric(gs[f"{psf.PRIMARY_LABEL}_g"]); s = pd.to_numeric(gs[f"{psf.PRIMARY_LABEL}_s"])
    gen_meta = json.loads((args.pool_dir / "generation_meta.json").read_text())
    sim["silver_mode"] = gen_meta.get("silver_mode") + (f" (beta={gen_meta.get('silver_beta')})" if gen_meta.get("silver_mode") == "impute_mix" else "")
    sim["silver"] = {"exact_agreement": float((g == s).mean()), "corr": float(np.corrcoef(g, s)[0, 1]), "rmse": float(np.sqrt(((g - s) ** 2).mean())), "silver_sd": float(s.std()), "gold_sd": float(g.std())}
    lines = ["# Simulator pool vs real analysis population", "", f"Bundle: `{args.bundle}`  ", f"Pool: `{args.pool_dir}`", "",
             "| Quantity | Real | Simulator |", "|---|---|---|"]
    def row(label, r, s_):
        lines.append(f"| {label} | {r} | {s_} |")
    row("Analysis rows", real["n_analysis_rows"], sim["n_analysis_rows"])
    row("contacts_t1 q10/25/50/75/90", real["contacts_t1_quantiles"], sim["contacts_t1_quantiles"])
    row("contacts_t2 q10/25/50/75/90", real["contacts_t2_quantiles"], sim["contacts_t2_quantiles"])
    row("Quintile upper cutpoints (t1)", real["quintile_upper_cutpoints"]["1"], sim["pipeline_qcut_upper_cutpoints"]["1"])
    row("Quintile upper cutpoints (t2)", real["quintile_upper_cutpoints"]["2"], sim["pipeline_qcut_upper_cutpoints"]["2"])
    row("R1 mean (sd)", f"{real['R1_mean']:.3f} ({real['R1_sd']:.3f})", f"{sim['R1_mean']:.3f} ({sim['R1_sd']:.3f})")
    row("R2 mean (sd)", f"{real['R2_mean']:.3f} ({real['R2_sd']:.3f})", f"{sim['R2_mean']:.3f} ({sim['R2_sd']:.3f})")
    row("progress_total mean", f"{real['progress_total_mean']:.3f}", f"{sim['progress_total_mean']:.3f}")
    row("Y2 any placement improvement rate", f"{real['y2_any_rate']:.3f}", f"{sim['y2_any_rate']:.3f}")
    row("Baseline placement dist", real["baseline_placement_dist"], sim["baseline_placement_dist"])
    row("A1 dist", real["A1_dist"], sim["A1_dist"])
    row("Zero-contact month fraction (2019)", real["zero_month_fraction_target"], f"{sim['zero_contact_month_fraction_2019']:.3f}")
    row("Label coverage (t1, t2)", real["label_coverage_real"], sim["label_coverage"])
    rs = real["silver"]; ss = sim["silver"]
    row("Silver mode", "qwen3-ft LLM labels", sim["silver_mode"])
    row("Silver exact agreement", f"{rs['real_exact_agreement']:.3f}", f"{ss['exact_agreement']:.3f}")
    row("Silver-gold corr", f"{rs['real_corr']:.3f}", f"{ss['corr']:.3f}")
    row("Silver-gold RMSE", f"{rs['real_rmse']:.3f}", f"{ss['rmse']:.3f}")
    row("Silver sd / gold sd", f"{rs['real_silver_sd']:.3f} / {rs['real_gold_sd']:.3f}", f"{ss['silver_sd']:.3f} / {ss['gold_sd']:.3f}")
    lines += ["", "## Pipeline-vs-simulator consistency (same clients)", ""]
    for k, v in consistency.items():
        lines.append(f"- {k}: {v:.4f}")
    lines += ["", "## Pipeline population accounting", ""]
    for k, v in sim["population"].items():
        lines.append(f"- {k}: {v}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.with_suffix(".json")).write_text(json.dumps({"real": real, "simulator": sim, "consistency": consistency}, indent=2, default=str))
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit the simulator, behavior policy, emission tables and silver imputer.")
    fit.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL_CSV)
    fit.add_argument("--features-csv", default=orw.DEFAULT_FEATURES_CSV)
    fit.add_argument("--contacts-csv", default=orw.DEFAULT_CONTACTS_CSV)
    fit.add_argument("--labels-csv", default=orw.DEFAULT_LABELS_CSV)
    fit.add_argument("--silver-csv", default=str(DEFAULT_SILVER_CSV))
    fit.add_argument("--date-format", default="%m/%d/%y")
    fit.add_argument("--simulator-kind", choices=["linear", "poly2", "rbf"], default="rbf")
    fit.add_argument("--rbf-gammas", default="0.005,0.01,0.02,0.05,0.1")
    fit.add_argument("--target-zero-month-fraction", type=float, default=0.455)
    fit.add_argument("--seed", type=int, default=20260910)
    fit.add_argument("--bundle", type=Path, default=DEFAULT_SIM_DIR / "simulator_rbf.joblib")
    fit.add_argument("--summary", type=Path, default=DEFAULT_RESULTS_DIR / "fit_summary_rbf.json")
    fit.set_defaults(func=cmd_fit)

    gen = sub.add_parser("generate", help="Roll out a synthetic cohort under the behavior policy in the pipeline's CSV schema.")
    gen.add_argument("--bundle", type=Path, required=True)
    gen.add_argument("--n-clients", type=int, default=8000)
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--label-coverage", type=float, default=0.7)
    gen.add_argument("--silver-mode", choices=["impute", "impute_mix", "impute_hist", "noisy_gold"], default="impute_mix")
    gen.add_argument("--silver-beta", type=float, default=0.7, help="impute_mix only: weight on the true label inside the imputation silver.")
    gen.add_argument("--out-dir", type=Path, required=True)
    gen.set_defaults(func=cmd_generate)

    tr = sub.add_parser("truth", help="Ground-truth value of the shift-up-one-quintile policy by rollout.")
    tr.add_argument("--bundle", type=Path, required=True)
    tr.add_argument("--n-rollouts", type=int, default=100_000)
    tr.add_argument("--chunk", type=int, default=20_000)
    tr.add_argument("--seed", type=int, default=2)
    tr.add_argument("--shift-quintiles", type=int, default=1)
    tr.add_argument("--label-coverage", type=float, default=0.7)
    tr.add_argument("--silver-mode", choices=["impute", "impute_mix", "impute_hist", "noisy_gold"], default="impute_mix")
    tr.add_argument("--silver-beta", type=float, default=0.7)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--pipeline-policy-pool", type=Path, default=None,
                    help="Diagnostic: define the target policy as the pipeline does (shift of its fitted propensity on this pool, pool qcut bins) instead of the true behavior shift.")
    tr.add_argument("--propensity-learner", choices=["ridge_softmax", "multinomial"], default="ridge_softmax")
    tr.add_argument("--propensity-c", type=float, default=1.0)
    tr.add_argument("--propensity-floor", type=float, default=0.0, help="Clip the true behavior propensities at this floor (renormalized) before the shift, as the pipeline does (its default is 0.02). 0 = unfloored true policy.")
    tr.add_argument("--true-g-pool-bins", action="store_true", help="With --pipeline-policy-pool: shift the TRUE behavior policy within the pool's quintile bins (isolates the bin-definition effect).")
    tr.set_defaults(func=cmd_truth)

    mg = sub.add_parser("marginals", help="Compare a generated pool (through the real pipeline's frame builder) with the real analysis population.")
    mg.add_argument("--bundle", type=Path, required=True)
    mg.add_argument("--pool-dir", type=Path, required=True)
    mg.add_argument("--out", type=Path, required=True)
    mg.set_defaults(func=cmd_marginals)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
