#!/usr/bin/env python3
"""Resumable outreach DTR Monte Carlo for adaptive reward annotation.

This script is protected-data safe by design: it reads the same projected columns
as ``outreach_dtr_reward_only.py`` and writes aggregate MC outputs only. It
does not write client identifiers, case-note text, raw labels, or row examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency in lightweight environments.
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None
    delayed = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import outreach_dtr_reward_only as orw  # noqa: E402
from sampler_names import (  # noqa: E402
    ADAPTIVE_STAGEWISE,
    ADAPTIVE_TRAJECTORY,
    UNCERTAINTY_PROPORTIONAL,
    CANONICAL as CANONICAL_SAMPLERS,
    canonical_cli_sampler,
    canonicalize_raw_driver_output,
)
import progress_surrogate_feasibility as psf  # noqa: E402
from t2_optimal_allocation import solve as solve_prefix_allocation  # noqa: E402


DEFAULT_BUDGETS = "0.30,0.40,0.50,0.60,0.70,0.80"
DEFAULT_SILVER_LABELS_CSV = "protected_data/silver_labels.csv"
CORE_BATCH_FILES = ("results.csv", "aggregate_metrics.csv", "run_metadata.csv")
EPS = 1e-8
SILVER_VAR_DELTA = 1e-3
_INTERRUPTED = False


def silver_stage_dispersion(
    contacts: pd.DataFrame,
    silver: pd.DataFrame,
    spec: "psf.IntervalSpec",
    date_format: str | None,
) -> pd.DataFrame:
    """Per-client, per-interval dispersion of SILVER progress labels.

    Zero-gold-cost variance proxy: silver labels exist for the full cohort, so
    this is pre-annotation (F0-measurable) design information. -1 sentinels map
    to 0 like the gold guard; single-label intervals get variance 0.
    """
    contacts = psf.normalize_id_columns(contacts, [psf.CLIENT_ID, psf.CONTACT_ID]).copy()
    contacts[psf.CONTACT_DATE] = psf.parse_contact_dates(contacts[psf.CONTACT_DATE], date_format)
    contacts = contacts.dropna(subset=[psf.CONTACT_DATE, psf.CLIENT_ID, psf.CONTACT_ID])
    contacts["interval"] = psf.assign_interval(contacts[psf.CONTACT_DATE], spec)
    contacts = contacts.dropna(subset=["interval"])
    contacts["interval"] = contacts["interval"].astype(int)

    silver = psf.normalize_id_columns(silver, [psf.CONTACT_ID]).copy()
    values = pd.to_numeric(silver[psf.PRIMARY_LABEL], errors="coerce")
    values = values.mask(values.eq(psf.INVALID_LABEL_SENTINEL), 0.0)
    silver["silver_numeric"] = values.where(values.between(psf.VALID_PROGRESS_MIN, psf.VALID_PROGRESS_MAX))
    silver = silver.dropna(subset=[psf.CONTACT_ID]).groupby(psf.CONTACT_ID, as_index=False)["silver_numeric"].max()

    merged = contacts.merge(silver, on=psf.CONTACT_ID, how="left").dropna(subset=["silver_numeric"])
    stats = (
        merged.groupby([psf.CLIENT_ID, "interval"])["silver_numeric"]
        .agg(silver_var=lambda s: float(np.var(s.to_numpy(dtype=float))), silver_count="size")
        .reset_index()
    )
    out = stats.pivot(index=psf.CLIENT_ID, columns="interval", values="silver_var")
    out.columns = [f"silver_var_t{int(c)}" for c in out.columns]
    return out.reset_index()


@dataclass(frozen=True)
class McTarget:
    name: str
    stage1_col: str
    stage2_col: str
    reveal_mode: str
    description: str


@dataclass(frozen=True)
class FoldContext:
    train_idx: np.ndarray
    test_idx: np.ndarray
    q1_train: np.ndarray
    q2_train: np.ndarray
    q1_test: np.ndarray
    q2_test: np.ndarray
    rho1_test: np.ndarray
    rho12_test: np.ndarray


@dataclass(frozen=True)
class BehaviorContext:
    folds: tuple[FoldContext, ...]
    rho1: np.ndarray
    rho12: np.ndarray


class WeightedRidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.constant_: float | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            self.constant_ = 0.0
            return
        if len(y) < 2 or x.shape[1] == 0:
            self.constant_ = float(np.average(y, weights=sample_weight)) if sample_weight is not None else float(np.mean(y))
            return
        x_mat = np.asarray(x, dtype=float)
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        if sample_weight is None:
            weights = np.ones(len(y), dtype=float)
        else:
            weights = np.maximum(np.asarray(sample_weight, dtype=float), EPS)
        sqrt_w = np.sqrt(weights)
        x_w = x_mat * sqrt_w[:, None]
        y_w = y * sqrt_w
        penalty = self.alpha * np.eye(x_mat.shape[1])
        penalty[0, 0] = 0.0
        try:
            self.coef_ = np.linalg.pinv(x_w.T @ x_w + penalty) @ x_w.T @ y_w
        except np.linalg.LinAlgError:
            self.constant_ = float(np.average(y, weights=weights))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(x), self.constant_, dtype=float)
        if self.coef_ is None:
            raise RuntimeError("WeightedRidgeRegressor must be fit before predict")
        x_mat = np.asarray(x, dtype=float)
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        return x_mat @ self.coef_


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def signal_handler(_signum: int, _frame: object) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_dir_name(start_rep: int, reps: int) -> str:
    return f"reps_{start_rep:06d}_{start_rep + reps - 1:06d}"


def batch_complete(batch_dir: Path) -> bool:
    manifest = batch_dir / "batch_manifest.json"
    return manifest.exists() and all((batch_dir / name).exists() and (batch_dir / name).stat().st_size > 0 for name in CORE_BATCH_FILES)


def completed_manifests(batches_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not batches_dir.exists():
        return manifests
    for manifest_path in sorted(batches_dir.glob("*/batch_manifest.json")):
        if not batch_complete(manifest_path.parent):
            continue
        manifest = load_json(manifest_path)
        manifest["batch_dir"] = str(manifest_path.parent)
        manifests.append(manifest)
    return sorted(manifests, key=lambda row: (int(row["start_rep"]), int(row["reps"])))


def contiguous_completed_reps(manifests: Iterable[dict[str, Any]], start: int = 0) -> int:
    cursor = start
    for manifest in manifests:
        start = int(manifest["start_rep"])
        reps = int(manifest["reps"])
        if start < cursor:
            cursor = max(cursor, start + reps)
            continue
        if start != cursor:
            break
        cursor += reps
    return cursor


def available_targets() -> dict[str, McTarget]:
    return {
        "progress_total": McTarget(
            name="progress_total",
            stage1_col="R1_progress",
            stage2_col="R2_progress",
            reveal_mode="two_stage_bundle",
            description="R1 progress plus R2 progress",
        ),
        "y2_any_placement_improvement": McTarget(
            name="y2_any_placement_improvement",
            stage1_col="zero_reward",
            stage2_col="Y2_any_placement_improvement",
            reveal_mode="terminal_only",
            description="Indicator for any year-2 placement improvement from baseline",
        ),
    }


def select_targets(names: list[str]) -> list[McTarget]:
    choices = available_targets()
    missing = sorted(set(names) - set(choices))
    if missing:
        raise ValueError(f"Unknown targets: {missing}. Available: {sorted(choices)}")
    return [choices[name] for name in names]


def fit_weighted_ridge(
    train: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    alpha: float,
) -> tuple[WeightedRidgeRegressor, pd.Index]:
    x_train = orw.encode_frame(train, columns)
    model = WeightedRidgeRegressor(alpha=alpha)
    model.fit(x_train, y, sample_weight=sample_weight)
    return model, x_train.columns


def predict_weighted_ridge(
    model: WeightedRidgeRegressor,
    encoded_columns: pd.Index,
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    x = orw.encoded_like(frame, columns, encoded_columns)
    return np.asarray(model.predict(x), dtype=float)


def predict_action_values(
    model: WeightedRidgeRegressor,
    encoded_columns: pd.Index,
    base: pd.DataFrame,
    columns: list[str],
    action_col: str,
) -> np.ndarray:
    values = np.zeros((len(base), len(orw.ACTION_LEVELS)), dtype=float)
    for j, action in enumerate(orw.ACTION_LEVELS):
        modified = base.copy()
        modified[action_col] = int(action)
        values[:, j] = predict_weighted_ridge(model, encoded_columns, modified, columns)
    return values


def probabilities_from_priority(priority: np.ndarray, budget: float, floor: float) -> np.ndarray:
    if not 0.0 < budget <= 1.0:
        raise ValueError("Budgets must be fractions in (0, 1].")
    if floor < 0.0 or floor >= budget:
        raise ValueError("--annotation-floor must be nonnegative and smaller than each budget.")
    priority = np.maximum(np.asarray(priority, dtype=float), EPS)
    scaled = priority / max(float(np.mean(priority)), EPS)

    def probs(scale: float) -> np.ndarray:
        return np.clip(scale * scaled, floor, 1.0)

    lo = 0.0
    hi = 1.0
    while float(np.mean(probs(hi))) < budget and hi < 1e8:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(np.mean(probs(mid))) < budget:
            lo = mid
        else:
            hi = mid
    return probs(hi)


def ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(np.square(weights)))
    return float(np.square(np.sum(weights)) / denom) if denom > 0 else float("nan")


def summarize_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_max": float(np.max(values)),
    }


def oracle_propensities(frame: pd.DataFrame, floor: float) -> tuple[np.ndarray, np.ndarray]:
    out = []
    for stage in (1, 2):
        cols = [f"oracle_g{stage}_{level}" for level in orw.ACTION_LEVELS]
        raw = np.maximum(frame[cols].to_numpy(dtype=float), floor)
        out.append(raw / raw.sum(axis=1, keepdims=True))
    return out[0], out[1]


def prepare_behavior_context(dtr: pd.DataFrame, args: argparse.Namespace, seed: int) -> BehaviorContext:
    n = len(dtr)
    folds = orw.choose_folds(n, args.n_folds)
    kfold = orw.SimpleKFold(n_splits=folds, shuffle=True, random_state=seed)
    h1 = orw.h1_columns()
    h2 = orw.h2_columns()
    g1 = np.zeros((n, len(orw.ACTION_LEVELS)), dtype=float)
    g2 = np.zeros_like(g1)
    q1 = np.zeros_like(g1)
    q2 = np.zeros_like(g1)
    fold_contexts: list[FoldContext] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dtr)):
        train = dtr.iloc[train_idx].copy()
        test = dtr.iloc[test_idx].copy()
        fold_seed = seed + 1009 * fold_idx
        beh1, beh1_cols = orw.fit_classifier(train, h1, train["A1"], args, fold_seed)
        beh2, beh2_cols = orw.fit_classifier(train, h2, train["A2"], args, fold_seed + 1)
        g1_train = orw.predict_action_probs(beh1, beh1_cols, train, h1, args.propensity_floor)
        g2_train = orw.predict_action_probs(beh2, beh2_cols, train, h2, args.propensity_floor)
        g1_test = orw.predict_action_probs(beh1, beh1_cols, test, h1, args.propensity_floor)
        g2_test = orw.predict_action_probs(beh2, beh2_cols, test, h2, args.propensity_floor)
        if getattr(args, "oracle_propensity_csv", None):
            # Simulator-validation diagnostic only: replace the fitted behavior propensities with the
            # data-generating ones (columns oracle_g{1,2}_{1..5} merged onto dtr in load_dtr_frame).
            g1_train, g2_train = oracle_propensities(train, args.propensity_floor)
            g1_test, g2_test = oracle_propensities(test, args.propensity_floor)
        q1_train = orw.shift_up_probs(g1_train, orw.stage_shift(args, 1))
        q2_train = orw.shift_up_probs(g2_train, orw.stage_shift(args, 2))
        q1_test = orw.shift_up_probs(g1_test, orw.stage_shift(args, 1))
        q2_test = orw.shift_up_probs(g2_test, orw.stage_shift(args, 2))
        g1[test_idx] = g1_test
        g2[test_idx] = g2_test
        q1[test_idx] = q1_test
        q2[test_idx] = q2_test
        g1_obs = orw.observed_probability(g1_test, test["A1"])
        g2_obs = orw.observed_probability(g2_test, test["A2"])
        q1_obs = orw.observed_probability(q1_test, test["A1"])
        q2_obs = orw.observed_probability(q2_test, test["A2"])
        rho1_test = orw.safe_ratio(q1_obs, g1_obs, args.propensity_floor)
        rho12_test = rho1_test * orw.safe_ratio(q2_obs, g2_obs, args.propensity_floor)
        fold_contexts.append(
            FoldContext(
                train_idx=train_idx,
                test_idx=test_idx,
                q1_train=q1_train,
                q2_train=q2_train,
                q1_test=q1_test,
                q2_test=q2_test,
                rho1_test=rho1_test,
                rho12_test=rho12_test,
            )
        )

    rho1 = orw.safe_ratio(
        orw.observed_probability(q1, dtr["A1"]),
        orw.observed_probability(g1, dtr["A1"]),
        args.propensity_floor,
    )
    rho12 = rho1 * orw.safe_ratio(
        orw.observed_probability(q2, dtr["A2"]),
        orw.observed_probability(g2, dtr["A2"]),
        args.propensity_floor,
    )
    return BehaviorContext(tuple(fold_contexts), rho1=rho1, rho12=rho12)


def crossfit_residual_variance(
    dtr: pd.DataFrame,
    context: BehaviorContext,
    outcome_col: str,
    columns: list[str],
    args: argparse.Namespace,
    seed: int,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Covariate-only conditional-variance predictions, cross-fitted.

    The annotation priority must be measurable w.r.t. pre-annotation data, so a
    held-out client's own realized reward can never enter that client's score.
    Within each outer training fold, an inner 2-fold split produces out-of-fold
    squared residuals; a second ridge model fit to those residual squares then
    predicts the conditional variance on the outer test fold from covariates.

    When ``train_mask`` is given (the charged pilot batch), only masked rows may
    serve as variance-model training data — the labels the design has actually
    paid for. Folds whose pilot training set is too small fall back to a
    constant (floor-level) variance prediction for their test clients.
    """
    n = len(dtr)
    var_pred = np.zeros(n, dtype=float)
    for fold_idx, fold in enumerate(context.folds):
        train_idx = np.asarray(fold.train_idx)
        if train_mask is not None:
            train_idx = train_idx[np.asarray(train_mask, dtype=bool)[train_idx]]
        if len(train_idx) < 8:
            continue  # var_pred stays 0 for this fold's test rows; floored below
        train = dtr.iloc[train_idx].copy()
        test = dtr.iloc[fold.test_idx].copy()
        y_train = pd.to_numeric(train[outcome_col], errors="coerce").to_numpy(dtype=float)
        m = len(train)
        rng = np.random.default_rng(seed + 1009 * fold_idx)
        perm = rng.permutation(m)
        halves = (perm[: m // 2], perm[m // 2 :])
        residual_sq_train = np.zeros(m, dtype=float)
        for fit_idx, pred_idx in ((halves[0], halves[1]), (halves[1], halves[0])):
            if len(fit_idx) == 0 or len(pred_idx) == 0:
                continue
            inner_model, inner_encoded = fit_weighted_ridge(
                train.iloc[fit_idx],
                columns,
                y_train[fit_idx],
                sample_weight=None,
                alpha=args.ridge_alpha,
            )
            inner_pred = predict_weighted_ridge(inner_model, inner_encoded, train.iloc[pred_idx], columns)
            residual_sq_train[pred_idx] = np.square(y_train[pred_idx] - inner_pred)
        var_model, var_encoded = fit_weighted_ridge(
            train,
            columns,
            residual_sq_train,
            sample_weight=None,
            alpha=args.ridge_alpha,
        )
        var_pred[fold.test_idx] = predict_weighted_ridge(var_model, var_encoded, test, columns)
    floor = float(np.quantile(var_pred, 0.05)) if len(var_pred) else 0.0
    return np.maximum(var_pred, max(floor, EPS))


def _outcome_range(dtr: pd.DataFrame, outcome_col: str) -> tuple[float, float, bool]:
    values = pd.to_numeric(dtr[outcome_col], errors="coerce").dropna()
    binary = bool(set(np.unique(values.to_numpy(dtype=float))) <= {0.0, 1.0})
    if binary:
        return 0.0, 1.0, True
    return float(psf.VALID_PROGRESS_MIN), float(psf.VALID_PROGRESS_MAX), False


def crossfit_link_variance(
    dtr: pd.DataFrame,
    context: BehaviorContext,
    outcome_col: str,
    columns: list[str],
    args: argparse.Namespace,
    seed: int,
    train_mask: np.ndarray | None = None,
    silver_col: str | None = None,
) -> np.ndarray:
    """Mean-variance-link conditional variances, cross-fitted, pilot-charged.

    Binary outcomes: Var(Y|x) = m(x)(1-m(x)) exactly - no residual regression.
    Bounded [0,4] progress rewards: quasi-binomial Var ~ phi * m(4-m)/4 with the
    scalar dispersion phi calibrated from pooled pilot out-of-fold residuals.
    With ``silver_col`` (link_silver), the dispersion becomes
    phi(x) = exp(alpha + beta * log(silver_var(x) + delta)) fit on the same
    pilot residual ratios - the heteroskedasticity shape comes from the free
    silver signal; the pilot calibrates two scalars.
    """
    n = len(dtr)
    lo, hi, binary = _outcome_range(dtr, outcome_col)
    span = hi - lo
    y_all = pd.to_numeric(dtr[outcome_col], errors="coerce").to_numpy(dtype=float)
    silver_all = (
        np.maximum(pd.to_numeric(dtr[silver_col], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0)
        if silver_col is not None
        else None
    )

    def vlink(m: np.ndarray) -> np.ndarray:
        m = np.clip(m, lo + 0.02 * span, hi - 0.02 * span)
        return (m - lo) * (hi - m) / span

    pilot_idx = np.flatnonzero(train_mask) if train_mask is not None else np.arange(n)

    # Dispersion calibration from a pooled pilot half-split (skipped for binary
    # without silver: the link is exact, phi = 1).
    phi = 1.0
    alpha_beta: tuple[float, float] | None = None
    if not binary:
        rng = np.random.default_rng(seed + 977)
        perm = rng.permutation(len(pilot_idx))
        halves = (pilot_idx[perm[: len(perm) // 2]], pilot_idx[perm[len(perm) // 2 :]])
        oof_resid2, oof_vlink, oof_silver = [], [], []
        for fit_idx, pred_idx in ((halves[0], halves[1]), (halves[1], halves[0])):
            if len(fit_idx) < 8 or len(pred_idx) == 0:
                continue
            model, encoded = fit_weighted_ridge(dtr.iloc[fit_idx], columns, y_all[fit_idx], sample_weight=None, alpha=args.ridge_alpha)
            m_pred = predict_weighted_ridge(model, encoded, dtr.iloc[pred_idx], columns)
            oof_resid2.append(np.square(y_all[pred_idx] - np.clip(m_pred, lo, hi)))
            oof_vlink.append(vlink(m_pred))
            if silver_all is not None:
                oof_silver.append(silver_all[pred_idx])
        if oof_resid2:
            resid2 = np.concatenate(oof_resid2)
            base = np.maximum(np.concatenate(oof_vlink), EPS)
            if silver_col is not None and len(oof_silver) > 0:
                z = np.log(np.concatenate(oof_silver) + SILVER_VAR_DELTA)
                target_log = np.log((resid2 + SILVER_VAR_DELTA) / base)
                design = np.column_stack([np.ones_like(z), z])
                coef, *_ = np.linalg.lstsq(design, target_log, rcond=None)
                alpha_beta = (float(coef[0]), float(coef[1]))
            else:
                phi = float(np.sum(resid2) / max(np.sum(base), EPS))
                phi = float(np.clip(phi, 0.05, 20.0))

    var_pred = np.zeros(n, dtype=float)
    fallback_mean = float(np.nanmean(y_all[pilot_idx])) if len(pilot_idx) else float(np.nanmean(y_all))
    for fold_idx, fold in enumerate(context.folds):
        train_idx = np.asarray(fold.train_idx)
        if train_mask is not None:
            train_idx = train_idx[np.asarray(train_mask, dtype=bool)[train_idx]]
        if len(train_idx) < 8:
            m_test = np.full(len(fold.test_idx), fallback_mean)
        else:
            model, encoded = fit_weighted_ridge(dtr.iloc[train_idx], columns, y_all[train_idx], sample_weight=None, alpha=args.ridge_alpha)
            m_test = predict_weighted_ridge(model, encoded, dtr.iloc[fold.test_idx], columns)
        base = vlink(np.asarray(m_test, dtype=float))
        if alpha_beta is not None and silver_all is not None:
            a, b = alpha_beta
            scale = np.exp(np.clip(a + b * np.log(silver_all[fold.test_idx] + SILVER_VAR_DELTA), -6.0, 6.0))
            var_pred[fold.test_idx] = scale * base
        else:
            var_pred[fold.test_idx] = phi * base
    floor = float(np.quantile(var_pred, 0.05)) if len(var_pred) else 0.0
    return np.maximum(var_pred, max(floor, EPS))


def stage_variance(
    dtr: pd.DataFrame,
    context: BehaviorContext,
    outcome_col: str,
    columns: list[str],
    args: argparse.Namespace,
    seed: int,
    variance_mask: np.ndarray | None,
    silver_col: str,
) -> np.ndarray:
    estimator = args.variance_estimator
    if estimator == "silver_only":
        # Pure zero-gold-cost design signal: silver-label dispersion, available
        # for the whole cohort before any annotation. No pilot required.
        return np.maximum(
            pd.to_numeric(dtr[silver_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) + SILVER_VAR_DELTA,
            EPS,
        )
    if estimator in {"link", "link_silver"}:
        return crossfit_link_variance(
            dtr, context, outcome_col, columns, args, seed,
            train_mask=variance_mask,
            silver_col=silver_col if estimator == "link_silver" else None,
        )
    return crossfit_residual_variance(dtr, context, outcome_col, columns, args, seed, train_mask=variance_mask)


def target_priority(
    dtr: pd.DataFrame,
    context: BehaviorContext,
    target: McTarget,
    args: argparse.Namespace,
    seed: int,
    variance_mask: np.ndarray | None = None,
    return_stage_signals: bool = False,
    return_uncertainty: bool = False,
) -> np.ndarray:
    # Theorem form (thm-twostage-optimal-allocation): pi* proportional to
    # sqrt(mu_1^2 sigma_1^2 + mu_{1:2}^2 sigma_2^2) on the bundle boundary
    # q_{1:2} = lambda_1, and sqrt(mu_{1:2}^2 sigma_2^2) for terminal-only.
    # Legacy form kept for sensitivity: density ratios at power one inside
    # the sqrt (the pre-2026-08-30 behavior).
    rho1 = np.maximum(context.rho1, 0.0)
    rho12 = np.maximum(context.rho12, 0.0)
    if args.priority_form == "legacy":
        overlap1, overlap12 = rho1, rho12
    else:
        overlap1, overlap12 = np.square(rho1), np.square(rho12)
    if target.reveal_mode == "two_stage_bundle":
        v1 = stage_variance(dtr, context, target.stage1_col, orw.q1_columns(), args, seed + 11, variance_mask, silver_col="silver_var_t1")
        v2 = stage_variance(dtr, context, target.stage2_col, orw.q2_columns(), args, seed + 12, variance_mask, silver_col="silver_var_t2")
        priority = np.sqrt(np.maximum(v1, EPS) * overlap1 + np.maximum(v2, EPS) * overlap12)
        if return_uncertainty:
            # active-inference baseline: same variance signal, density ratios dropped
            return np.sqrt(np.maximum(v1, EPS) + np.maximum(v2, EPS))
        if return_stage_signals:
            # per-stage design signals s_1^2 = v1*rho1^2, s_2^2 = v2*rho12^2 for the stagewise (prefix) allocation
            return priority, np.maximum(v1, EPS) * np.maximum(overlap1, EPS), np.maximum(v2, EPS) * np.maximum(overlap12, EPS)
        return priority
    v2 = stage_variance(dtr, context, target.stage2_col, orw.q2_columns(), args, seed + 13, variance_mask, silver_col="silver_var_t2")
    priority = np.sqrt(np.maximum(v2, EPS) * overlap12)
    if return_uncertainty:
        return np.sqrt(np.maximum(v2, EPS))
    if return_stage_signals:
        return priority, None, None
    return priority


def annotation_design(
    target: McTarget,
    sampler: str,
    budget: float,
    priority: np.ndarray,
    uniform: np.ndarray,
    args: argparse.Namespace,
    pilot_revealed: np.ndarray | None = None,
    pilot_prob: float = 0.0,
    stage_signals: tuple[np.ndarray | None, np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    n = len(priority)
    pilot_count = 0
    if sampler == "random":
        probabilities = np.full(n, budget, dtype=float)
        revealed = uniform < probabilities
    elif sampler == ADAPTIVE_STAGEWISE and target.reveal_mode == "two_stage_bundle":
        # Stagewise (prefix) allocation of Algorithm 1 / Theorem (T=2): solve the joint problem
        # min E[s1^2/lambda1 + s2^2/Lambda12] s.t. E[lambda1 + Lambda12] <= 2*budget, floor <= Lambda12 <= lambda1,
        # with F0-measurable per-stage signals (single F1 branch: revealing R1 reveals no state here).
        # Equal expected cost to the bundle samplers, which spend 2*budget labels per client on average.
        if stage_signals is None or stage_signals[0] is None:
            raise ValueError("adaptive_stagewise needs per-stage signals; only two_stage_bundle targets provide them.")
        s1, s2 = stage_signals
        if pilot_revealed is not None and pilot_prob > 0.0:
            residual_mean = (budget - pilot_prob) / max(1.0 - pilot_prob, EPS)
            if residual_mean <= args.annotation_floor:
                raise ValueError("Pilot fraction leaves no residual budget above the annotation floor.")
        else:
            residual_mean = budget
        alloc = solve_prefix_allocation(s1, s2[:, None], np.ones((n, 1)), 2.0 * residual_mean, floor=args.annotation_floor)
        lam1_ad, cum2_ad = np.asarray(alloc.lambda1, dtype=float), np.asarray(alloc.cumulative2[:, 0], dtype=float)
        if pilot_revealed is not None and pilot_prob > 0.0:
            prob1 = pilot_prob + (1.0 - pilot_prob) * lam1_ad
            prob12 = pilot_prob + (1.0 - pilot_prob) * cum2_ad
            pilot_count = int(np.sum(pilot_revealed))
        else:
            prob1, prob12 = lam1_ad, cum2_ad
        u1, u2 = uniform[0], uniform[1]
        c1 = (u1 < lam1_ad)
        c2 = c1 & (u2 < cum2_ad / np.maximum(lam1_ad, EPS))
        if pilot_revealed is not None and pilot_prob > 0.0:
            c1 = c1 | pilot_revealed
            c2 = c2 | pilot_revealed
        if not np.any(c1):
            c1[int(np.argmax(prob1))] = True
        diagnostics = {
            "annotation_budget": float(budget),
            "sampler": sampler,
            "pilot_prob": float(pilot_prob),
            "pilot_count": pilot_count,
            "target_reveal_mode": target.reveal_mode,
            "annotated_count": int(np.sum(c1)),
            "stage2_annotated_count": int(np.sum(c2)),
            "observed_annotation_cost": float(np.sum(c1) + np.sum(c2)),
            "expected_annotation_cost": float(np.sum(prob1 + prob12)),
            "full_annotation_cost": float(2.0 * n),
            "realized_cost_fraction": float((np.sum(c1) + np.sum(c2)) / (2.0 * n)),
            "mean_stage2_given_stage1": float(np.mean(cum2_ad / np.maximum(lam1_ad, EPS))),
            "prefix_multiplier": float(alloc.multiplier),
        }
        return c1, c2, prob1, prob12, diagnostics
    elif sampler in (ADAPTIVE_TRAJECTORY, ADAPTIVE_STAGEWISE, UNCERTAINTY_PROPORTIONAL):
        # Trajectory-wise (bundle) design; for terminal-only targets the stagewise design coincides with it.
        if pilot_revealed is not None and pilot_prob > 0.0:
            # Charged two-batch design: the pilot batch (random, probability
            # pilot_prob) paid for the variance-model labels; the remaining
            # budget is allocated by the priority among unpiloted clients so
            # that total expected cost equals the budget.
            residual_mean = (budget - pilot_prob) / max(1.0 - pilot_prob, EPS)
            if residual_mean <= args.annotation_floor:
                raise ValueError("Pilot fraction leaves no residual budget above the annotation floor.")
            pi2 = probabilities_from_priority(priority, budget=residual_mean, floor=args.annotation_floor)
            probabilities = pilot_prob + (1.0 - pilot_prob) * pi2
            revealed = pilot_revealed | ((uniform < pi2) & ~pilot_revealed)
            pilot_count = int(np.sum(pilot_revealed))
        else:
            probabilities = probabilities_from_priority(priority, budget=budget, floor=args.annotation_floor)
            revealed = uniform < probabilities
    else:
        raise ValueError(f"Unknown sampler {sampler!r}")

    if not np.any(revealed):
        revealed[int(np.argmax(probabilities))] = True

    if target.reveal_mode == "two_stage_bundle":
        c1 = revealed.copy()
        c2 = revealed.copy()
        prob1 = probabilities.copy()
        prob2 = probabilities.copy()
        full_cost = 2.0
        observed_cost = c1.astype(float) + c2.astype(float)
        expected_cost = prob1 + prob2
    else:
        c1 = np.ones(n, dtype=bool)
        c2 = revealed.copy()
        prob1 = np.ones(n, dtype=float)
        prob2 = probabilities.copy()
        full_cost = 1.0
        observed_cost = c2.astype(float)
        expected_cost = prob2

    diagnostics: dict[str, float | str] = {
        "annotation_budget": float(budget),
        "sampler": sampler,
        "pilot_prob": float(pilot_prob if sampler in (ADAPTIVE_TRAJECTORY, ADAPTIVE_STAGEWISE, UNCERTAINTY_PROPORTIONAL) else 0.0),
        "pilot_revealed_count": pilot_count,
        "target_reveal_mode": target.reveal_mode,
        "full_revealable_cost": full_cost,
        "expected_reveal_cost": float(np.mean(expected_cost)),
        "observed_reveal_cost": float(np.mean(observed_cost)),
        "expected_reveal_fraction": float(np.mean(expected_cost) / full_cost),
        "observed_reveal_fraction": float(np.mean(observed_cost) / full_cost),
        "stage1_reveal_rate": float(np.mean(c1)),
        "stage2_reveal_rate": float(np.mean(c2)),
        "stage1_probability_ess": ess(prob1),
        "stage2_probability_ess": ess(prob2),
        "priority_cv": float(np.std(priority) / max(np.mean(priority), EPS)),
        **summarize_array(probabilities, "annotation_probability"),
        **summarize_array(priority, "priority"),
    }
    return c1, c2, prob1, prob2, diagnostics


def estimate_target(
    dtr: pd.DataFrame,
    context: BehaviorContext,
    target: McTarget,
    c1: np.ndarray,
    c2: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    n = len(dtr)
    plugin_scores = np.zeros(n, dtype=float)
    drl_scores = np.zeros(n, dtype=float)
    q2_cols = orw.q2_columns()
    q1_cols = orw.q1_columns()

    for fold_idx, fold in enumerate(context.folds):
        train = dtr.iloc[fold.train_idx].copy()
        test = dtr.iloc[fold.test_idx].copy()
        train_c2 = c2[fold.train_idx]
        train_prob2 = np.maximum(prob2[fold.train_idx], EPS)
        idx2 = np.flatnonzero(train_c2)
        if len(idx2) == 0:
            idx2 = np.array([int(np.argmax(train_prob2))])
        y2_train = pd.to_numeric(train.iloc[idx2][target.stage2_col], errors="coerce").to_numpy(dtype=float)
        q2_model, q2_encoded = fit_weighted_ridge(
            train.iloc[idx2],
            q2_cols,
            y2_train,
            sample_weight=1.0 / train_prob2[idx2],
            alpha=args.ridge_alpha,
        )

        q2_train_values = predict_action_values(q2_model, q2_encoded, train, q2_cols, "A2")
        v2_train = orw.weighted_action_mean(fold.q2_train, q2_train_values)

        if target.stage1_col == "zero_reward":
            idx1 = np.arange(len(train))
            y1_target = v2_train
            weight1 = np.ones(len(train), dtype=float)
        else:
            train_c1 = c1[fold.train_idx]
            train_prob1 = np.maximum(prob1[fold.train_idx], EPS)
            idx1 = np.flatnonzero(train_c1)
            if len(idx1) == 0:
                idx1 = np.array([int(np.argmax(train_prob1))])
            z1_train = pd.to_numeric(train.iloc[idx1][target.stage1_col], errors="coerce").to_numpy(dtype=float)
            y1_target = z1_train + v2_train[idx1]
            weight1 = 1.0 / train_prob1[idx1]

        q1_model, q1_encoded = fit_weighted_ridge(
            train.iloc[idx1],
            q1_cols,
            y1_target,
            sample_weight=weight1,
            alpha=args.ridge_alpha,
        )

        q2_test_values = predict_action_values(q2_model, q2_encoded, test, q2_cols, "A2")
        v2_test = orw.weighted_action_mean(fold.q2_test, q2_test_values)
        q2_obs_test = predict_weighted_ridge(q2_model, q2_encoded, test, q2_cols)
        q1_test_values = predict_action_values(q1_model, q1_encoded, test, q1_cols, "A1")
        v1_test = orw.weighted_action_mean(fold.q1_test, q1_test_values)
        q1_obs_test = predict_weighted_ridge(q1_model, q1_encoded, test, q1_cols)

        z1_test = pd.to_numeric(test[target.stage1_col], errors="coerce").to_numpy(dtype=float)
        z2_test = pd.to_numeric(test[target.stage2_col], errors="coerce").to_numpy(dtype=float)
        test_c1 = c1[fold.test_idx].astype(float)
        test_c2 = c2[fold.test_idx].astype(float)
        test_prob1 = np.maximum(prob1[fold.test_idx], EPS)
        test_prob2 = np.maximum(prob2[fold.test_idx], EPS)
        r1_ipw = test_c1 / test_prob1 * z1_test
        r2_residual_ipw = test_c2 / test_prob2 * (z2_test - q2_obs_test)
        signal = (
            v1_test
            + fold.rho1_test * (r1_ipw + v2_test - q1_obs_test)
            + fold.rho12_test * r2_residual_ipw
        )
        plugin_scores[fold.test_idx] = v1_test
        drl_scores[fold.test_idx] = signal

    estimates = {
        "plugin": float(np.mean(plugin_scores)),
        "drl": float(np.mean(drl_scores)),
    }
    diagnostics = {
        "plugin_score_sd": float(np.std(plugin_scores, ddof=1)) if n > 1 else 0.0,
        "drl_score_sd": float(np.std(drl_scores, ddof=1)) if n > 1 else 0.0,
        "mean_abs_drl_correction": float(np.mean(np.abs(drl_scores - plugin_scores))),
    }
    return estimates, diagnostics


def run_replication(rep: int, dtr_full: pd.DataFrame, targets: list[McTarget], budgets: list[float], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed + 1_000_003 * rep)
    if args.sample_size > len(dtr_full):
        raise ValueError(f"--sample-size {args.sample_size} exceeds available analysis rows {len(dtr_full)}.")
    sample_idx = rng.choice(len(dtr_full), size=args.sample_size, replace=False)
    dtr = dtr_full.iloc[sample_idx].copy().reset_index(drop=True)
    context = prepare_behavior_context(dtr, args, seed=args.seed + 17_003 * rep)
    rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets):
        full_c1 = np.ones(len(dtr), dtype=bool)
        full_c2 = np.ones(len(dtr), dtype=bool)
        full_prob = np.ones(len(dtr), dtype=float)
        full_estimates, full_diag = estimate_target(
            dtr,
            context,
            target,
            full_c1,
            full_c2,
            full_prob,
            full_prob,
            args,
            seed=args.seed + 31_337 * rep + target_index,
        )
        sampler_uniforms = {
            "random": rng.uniform(size=len(dtr)),
            ADAPTIVE_TRAJECTORY: rng.uniform(size=len(dtr)),  # formerly keyed "stagewise_adaptive"; draw order unchanged
        }
        pilot_uniform = rng.uniform(size=len(dtr))
        samplers = [canonical_cli_sampler(s) for s in parse_csv_strings(args.samplers)]
        if ADAPTIVE_STAGEWISE in samplers:
            sampler_uniforms[ADAPTIVE_STAGEWISE] = (rng.uniform(size=len(dtr)), rng.uniform(size=len(dtr)))  # drawn AFTER the shared ones: existing samplers' streams unchanged
        if UNCERTAINTY_PROPORTIONAL in samplers:
            # separate generator: the shared rng stream (and hence every other sampler and target) is unchanged
            sampler_uniforms[UNCERTAINTY_PROPORTIONAL] = np.random.default_rng([args.seed, rep, target_index, 4242]).uniform(size=len(dtr))
        if args.pilot_fraction <= 0.0:
            uncharged_priority, uncharged_s1, uncharged_s2 = target_priority(dtr, context, target, args, seed=args.seed + 77_777 * rep + target_index, return_stage_signals=True)
            if UNCERTAINTY_PROPORTIONAL in samplers:
                uncharged_uncertainty = target_priority(dtr, context, target, args, seed=args.seed + 77_777 * rep + target_index, return_uncertainty=True)
        for budget in budgets:
            if args.pilot_fraction > 0.0:
                pilot_prob = float(args.pilot_fraction * budget)
                pilot_revealed = pilot_uniform < pilot_prob
                priority, stage_s1, stage_s2 = target_priority(
                    dtr,
                    context,
                    target,
                    args,
                    seed=args.seed + 77_777 * rep + target_index,
                    variance_mask=pilot_revealed,
                    return_stage_signals=True,
                )
                if UNCERTAINTY_PROPORTIONAL in samplers:
                    uncertainty = target_priority(
                        dtr, context, target, args,
                        seed=args.seed + 77_777 * rep + target_index,
                        variance_mask=pilot_revealed,
                        return_uncertainty=True,
                    )
            else:
                pilot_prob = 0.0
                pilot_revealed = None
                priority, stage_s1, stage_s2 = uncharged_priority, uncharged_s1, uncharged_s2
                if UNCERTAINTY_PROPORTIONAL in samplers:
                    uncertainty = uncharged_uncertainty
            for sampler in samplers:
                terminal_only_stagewise = sampler == ADAPTIVE_STAGEWISE and target.reveal_mode != "two_stage_bundle"
                # Terminal-only targets: the stagewise design coincides with the single-stage (trajectory) design.
                sampler_uniform = sampler_uniforms[ADAPTIVE_TRAJECTORY] if terminal_only_stagewise else sampler_uniforms[sampler]
                c1, c2, prob1, prob2, design_diag = annotation_design(
                    target,
                    sampler=sampler,
                    budget=budget,
                    priority=uncertainty if sampler == UNCERTAINTY_PROPORTIONAL else priority,
                    uniform=sampler_uniform,
                    args=args,
                    pilot_revealed=pilot_revealed,
                    pilot_prob=pilot_prob,
                    stage_signals=(stage_s1, stage_s2),
                )
                partial_estimates, partial_diag = estimate_target(
                    dtr,
                    context,
                    target,
                    c1,
                    c2,
                    prob1,
                    prob2,
                    args,
                    seed=args.seed + 91_919 * rep + 101 * target_index + int(round(100 * budget)),
                )
                for estimator in parse_csv_strings(args.estimators):
                    benchmark = full_estimates[estimator]
                    estimate = partial_estimates[estimator]
                    error = estimate - benchmark
                    score_sd = float(partial_diag.get(f"{estimator}_score_sd", float("nan")))
                    model_based_se = score_sd / math.sqrt(len(dtr)) if np.isfinite(score_sd) else float("nan")
                    interval_width = 2.0 * 1.96 * model_based_se if np.isfinite(model_based_se) else float("nan")
                    full_score_sd = float(full_diag.get(f"{estimator}_score_sd", float("nan")))
                    full_model_based_se = full_score_sd / math.sqrt(len(dtr)) if np.isfinite(full_score_sd) else float("nan")
                    full_interval_width = 2.0 * 1.96 * full_model_based_se if np.isfinite(full_model_based_se) else float("nan")
                    row: dict[str, Any] = {
                        "rep": rep,
                        "priority_form": args.priority_form,
                        "variance_estimator": args.variance_estimator,
                        "h2_progress_source": args.h2_progress_source,
                        "sample_size": int(args.sample_size),
                        "target": target.name,
                        "target_description": target.description,
                        "budget": float(budget),
                        "sampler": sampler,
                        "estimator": estimator,
                        "estimate": estimate,
                        "full_reward_benchmark": benchmark,
                        "error": error,
                        "abs_error": abs(error),
                        "squared_error": error * error,
                        "model_based_se": model_based_se,
                        "model_based_ci_lower_95": estimate - 1.96 * model_based_se if np.isfinite(model_based_se) else float("nan"),
                        "model_based_ci_upper_95": estimate + 1.96 * model_based_se if np.isfinite(model_based_se) else float("nan"),
                        "model_based_ci_width_95": interval_width,
                        "full_model_based_se": full_model_based_se,
                        "full_model_based_ci_width_95": full_interval_width,
                        "interval_width_ratio_vs_full": interval_width / full_interval_width if np.isfinite(interval_width) and full_interval_width > 0 else float("nan"),
                        "full_plugin_estimate": full_estimates["plugin"],
                        "full_drl_estimate": full_estimates["drl"],
                        **design_diag,
                        **partial_diag,
                        **{f"full_{key}": value for key, value in full_diag.items()},
                    }
                    rows.append(row)
    return rows


def aggregate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target", "budget", "sampler", "estimator"]
    results = results.copy()
    if "model_based_ci_width_95" not in results.columns:
        results["model_based_ci_width_95"] = np.nan
    if "full_model_based_ci_width_95" not in results.columns:
        results["full_model_based_ci_width_95"] = np.nan
    if "interval_width_ratio_vs_full" not in results.columns:
        results["interval_width_ratio_vs_full"] = np.nan
    for estimator in ("plugin", "drl"):
        mask = results["estimator"].eq(estimator)
        if f"{estimator}_score_sd" in results.columns:
            se = pd.to_numeric(results.loc[mask, f"{estimator}_score_sd"], errors="coerce") / np.sqrt(
                pd.to_numeric(results.loc[mask, "sample_size"], errors="coerce")
            )
            missing = mask & results["model_based_ci_width_95"].isna()
            results.loc[missing, "model_based_ci_width_95"] = 2.0 * 1.96 * se.loc[missing]
        if f"full_{estimator}_score_sd" in results.columns:
            full_se = pd.to_numeric(results.loc[mask, f"full_{estimator}_score_sd"], errors="coerce") / np.sqrt(
                pd.to_numeric(results.loc[mask, "sample_size"], errors="coerce")
            )
            missing = mask & results["full_model_based_ci_width_95"].isna()
            results.loc[missing, "full_model_based_ci_width_95"] = 2.0 * 1.96 * full_se.loc[missing]
    missing_ratio = results["interval_width_ratio_vs_full"].isna()
    denom = pd.to_numeric(results["full_model_based_ci_width_95"], errors="coerce")
    num = pd.to_numeric(results["model_based_ci_width_95"], errors="coerce")
    results.loc[missing_ratio & denom.gt(0), "interval_width_ratio_vs_full"] = num[missing_ratio & denom.gt(0)] / denom[missing_ratio & denom.gt(0)]

    rows: list[dict[str, Any]] = []
    for key, group in results.groupby(group_cols, sort=True):
        target, budget, sampler, estimator = key
        squared = group["squared_error"].to_numpy(dtype=float)
        errors = group["error"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(squared)))
        rows.append(
            {
                "target": target,
                "budget": float(budget),
                "sampler": sampler,
                "estimator": estimator,
                "n_reps": int(len(group)),
                "mean_estimate": float(group["estimate"].mean()),
                "mean_full_reward_benchmark": float(group["full_reward_benchmark"].mean()),
                "bias_vs_full_reward": float(np.mean(errors)),
                "sd_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                "rmse": rmse,
                "mae": float(group["abs_error"].mean()),
                "mc_se_rmse_delta": float(np.std(squared, ddof=1) / math.sqrt(len(squared)) / max(2.0 * rmse, EPS)) if len(squared) > 1 else 0.0,
                "mean_model_based_ci_width_95": float(pd.to_numeric(group["model_based_ci_width_95"], errors="coerce").mean()),
                "mean_full_model_based_ci_width_95": float(pd.to_numeric(group["full_model_based_ci_width_95"], errors="coerce").mean()),
                "mean_interval_width_ratio_vs_full": float(pd.to_numeric(group["interval_width_ratio_vs_full"], errors="coerce").mean()),
                "mean_expected_reveal_fraction": float(group["expected_reveal_fraction"].mean()),
                "mean_observed_reveal_fraction": float(group["observed_reveal_fraction"].mean()),
                "mean_stage1_reveal_rate": float(group["stage1_reveal_rate"].mean()),
                "mean_stage2_reveal_rate": float(group["stage2_reveal_rate"].mean()),
                "mean_priority_cv": float(group["priority_cv"].mean()),
                "mean_stage1_probability_ess": float(group["stage1_probability_ess"].mean()),
                "mean_stage2_probability_ess": float(group["stage2_probability_ess"].mean()),
            }
        )
    metrics = pd.DataFrame.from_records(rows)
    if metrics.empty:
        return metrics
    random_rmse = (
        metrics[metrics["sampler"].eq("random")][["target", "budget", "estimator", "rmse"]]
        .rename(columns={"rmse": "random_rmse"})
    )
    metrics = metrics.merge(random_rmse, on=["target", "budget", "estimator"], how="left")
    metrics["rmse_reduction_vs_random_pct"] = 100.0 * (metrics["random_rmse"] - metrics["rmse"]) / metrics["random_rmse"]
    return metrics.sort_values(group_cols).reset_index(drop=True)


def add_batch_columns(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    start_rep = int(manifest["start_rep"])
    reps = int(manifest["reps"])
    frame.insert(0, "mc_batch", str(manifest["batch_name"]))
    frame.insert(1, "batch_start_rep", start_rep)
    frame.insert(2, "batch_reps", reps)
    frame["global_rep"] = frame["rep"].astype(int)
    frame["batch_rep"] = frame["global_rep"] - start_rep
    return frame


def combine_batches(output_dir: Path, manifests: list[dict[str, Any]], start: int = 0) -> int:
    if not manifests:
        return 0
    frames = []
    metadata_rows = []
    for manifest in manifests:
        batch_dir = Path(manifest["batch_dir"])
        # Raw driver output only: legacy names map unambiguously (stagewise_adaptive was the bundle design).
        frames.append(add_batch_columns(canonicalize_raw_driver_output(pd.read_csv(batch_dir / "results.csv")), manifest))
        metadata = pd.read_csv(batch_dir / "run_metadata.csv")
        row = metadata.iloc[0].to_dict() if not metadata.empty else {}
        row.update({k: manifest.get(k) for k in ("batch_name", "start_rep", "reps", "completed_at", "batch_dir")})
        metadata_rows.append(row)
    results = pd.concat(frames, ignore_index=True).sort_values(
        ["global_rep", "target", "budget", "sampler", "estimator"]
    )
    results.to_csv(output_dir / "results.csv", index=False)
    aggregate_metrics(results).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(metadata_rows).sort_values(["start_rep", "reps"]).to_csv(output_dir / "batch_run_metadata.csv", index=False)
    completed = contiguous_completed_reps(manifests, start=start)
    pd.DataFrame.from_records(
        [
            {
                "completed_reps": completed,
                "completed_batches": len(manifests),
                "combined_at": utc_now(),
            }
        ]
    ).to_csv(output_dir / "incremental_run_metadata.csv", index=False)
    return completed


def write_summary(output_dir: Path, args: argparse.Namespace, completed_reps: int, target_reps: int, elapsed: float) -> None:
    lines = [
        "# Outreach DTR Annotation Monte Carlo",
        "",
        f"Status: {'completed' if completed_reps >= target_reps else 'partial'}",
        f"Updated at: {utc_now()}",
        f"Elapsed wrapper seconds: {elapsed:.3f}",
        "",
        "## Design",
        f"- Sample size: `{args.sample_size}`",
        f"- Completed contiguous MC reps: `{completed_reps}`",
        f"- Target MC reps: `{target_reps}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Budgets: `{args.budgets}`",
        f"- Targets: `{args.targets}`",
        f"- Estimators: `{args.estimators}`",
        f"- Jobs: `{args.jobs}`",
        f"- Learner: `{args.learner}`",
        f"- Priority form: `{args.priority_form}` (theorem = squared density ratios; legacy = power-one)",
        f"- Pilot fraction: `{args.pilot_fraction}` (fraction of budget spent on the random pilot that pays for variance-model labels; 0 = uncharged)",
        f"- Variance estimator: `{args.variance_estimator}`",
        f"- H2 progress-summary source: `{args.h2_progress_source}` (silver = LLM-derived, deployable information set; gold = legacy leak)",
        "",
        "## Outputs",
        "- `results.csv`: combined row-level MC estimates without identifiers.",
        "- `aggregate_metrics.csv`: bias, RMSE, MAE, interval widths, reveal rates, ESS, and RMSE reduction vs random.",
        "- `incremental_status.json`: resumable status file.",
        "- `batches/reps_*/`: per-batch logs and raw aggregate outputs.",
    ]
    (output_dir / "mc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(
    dtr: pd.DataFrame,
    targets: list[McTarget],
    budgets: list[float],
    args: argparse.Namespace,
    output_dir: Path,
    start_rep: int,
    reps: int,
) -> int:
    batch_name = batch_dir_name(start_rep, reps)
    batch_dir = output_dir / "batches" / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "batch_name": batch_name,
        "start_rep": start_rep,
        "reps": reps,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "launched_at": utc_now(),
    }
    write_json_atomic(batch_dir / "batch_manifest.json", manifest)
    started = time.perf_counter()
    reps_to_run = list(range(start_rep, start_rep + reps))
    if args.jobs == 1 or Parallel is None:
        nested_rows = [run_replication(rep, dtr, targets, budgets, args) for rep in reps_to_run]
    else:
        nested_rows = Parallel(n_jobs=args.jobs, prefer="threads")(
            delayed(run_replication)(rep, dtr, targets, budgets, args) for rep in reps_to_run
        )
    rows = [row for rep_rows in nested_rows for row in rep_rows]
    results = pd.DataFrame.from_records(rows)
    results.to_csv(batch_dir / "results.csv", index=False)
    aggregate_metrics(results).to_csv(batch_dir / "aggregate_metrics.csv", index=False)
    elapsed = time.perf_counter() - started
    pd.DataFrame.from_records(
        [
            {
                "batch_name": batch_name,
                "start_rep": start_rep,
                "reps": reps,
                "elapsed_wall_seconds": elapsed,
                "n_rows": int(len(results)),
                "status": "completed",
            }
        ]
    ).to_csv(batch_dir / "run_metadata.csv", index=False)
    manifest["completed_at"] = utc_now()
    manifest["return_code"] = 0
    write_json_atomic(batch_dir / "batch_manifest.json", manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default=orw.DEFAULT_FEATURES_CSV)
    parser.add_argument("--contacts-csv", default=orw.DEFAULT_CONTACTS_CSV)
    parser.add_argument("--labels-csv", default=orw.DEFAULT_LABELS_CSV)
    parser.add_argument("--date-format", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--target-reps", type=int, default=20)
    parser.add_argument("--add-reps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--targets", default="progress_total,y2_any_placement_improvement")
    parser.add_argument("--estimators", default="plugin,drl")
    parser.add_argument("--samplers", default="random,adaptive_stagewise",
                        help="Comma list from random, adaptive_stagewise (paper's stagewise T=2 allocation; annotation may stop after stage 1), adaptive_trajectory (trajectory-wise/bundle design), uncertainty_proportional (active-inference baseline: sqrt of summed stage variances, no density ratios). Deprecated aliases: stagewise_adaptive=adaptive_trajectory, prefix_adaptive=adaptive_stagewise. See sampler_names.py.")
    parser.add_argument("--rep-start", type=int, default=0,
                        help="First global replication index for this process (shards: reps are seeded by global index, so shards merge exactly).")
    parser.add_argument("--learner", choices=["ridge"], default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--annotation-floor", type=float, default=0.01)
    parser.add_argument("--propensity-floor", type=float, default=0.02)
    parser.add_argument("--shift-quintiles", type=int, default=1)
    parser.add_argument("--shift-quintiles-stage1", type=int, default=None, help="Stage-1 shift (default: --shift-quintiles).")
    parser.add_argument("--shift-quintiles-stage2", type=int, default=None, help="Stage-2 shift (default: --shift-quintiles).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--interval-months", type=int, default=6)
    parser.add_argument("--num-decision-intervals", type=int, default=2)
    parser.add_argument("--placement-outcome-end-date", default="2021-01-01")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    parser.add_argument(
        "--priority-form",
        choices=["legacy", "theorem"],
        default="theorem",
        help="Overlap power in the stagewise priority: 'theorem' uses squared density ratios (paper theorem); 'legacy' the pre-2026-08-30 power-one form.",
    )
    parser.add_argument(
        "--pilot-fraction",
        type=float,
        default=0.3,
        help="Fraction of each budget spent on a random pilot batch that pays for the variance-model labels (charged two-batch design). 0 reproduces the uncharged behavior where the variance model sees all labels for free.",
    )
    parser.add_argument(
        "--variance-estimator",
        choices=["ridge", "link", "link_silver", "silver_only"],
        default="ridge",
        help="Conditional-variance signal for the priority: 'ridge' regression on residual squares; 'link' mean-variance link (exact m(1-m) for binary targets, quasi-binomial with pilot-calibrated scalar dispersion for bounded progress rewards); 'link_silver' link with dispersion shaped by zero-cost silver-label variance (two pilot-calibrated scalars); 'silver_only' raw silver dispersion, no gold labels used for the design (run with --pilot-fraction 0).",
    )
    parser.add_argument(
        "--silver-labels-csv",
        default=DEFAULT_SILVER_LABELS_CSV,
        help="Silver (LLM) labels used for the silver dispersion signal when --variance-estimator is link_silver or silver_only, and for silver-derived H2 progress summaries when --h2-progress-source is silver.",
    )
    parser.add_argument(
        "--propensity-learner",
        choices=["ridge_softmax", "multinomial"],
        default="ridge_softmax",
        help="Behavior-propensity learner: 'ridge_softmax' (paper runs; softmax over ridge linear probabilities, near-uniform) or 'multinomial' (standardized multinomial logistic regression).",
    )
    parser.add_argument("--propensity-c", type=float, default=1.0, help="Inverse L2 penalty for --propensity-learner multinomial (smaller = stronger regularization).")
    parser.add_argument(
        "--oracle-propensity-csv",
        default=None,
        help="Simulator-validation diagnostic only: CSV with ClientID and oracle_g{1,2}_{1..5} (true behavior propensities at the observed histories) used in place of the fitted propensities for the target policy and the DRL weights. Never set on real data.",
    )
    parser.add_argument(
        "--h2-progress-source",
        choices=["gold", "silver"],
        default="silver",
        help="Source of the stage-1 progress summaries entering the stage-2 history: 'silver' (LLM labels, available for every casenote; the deployable information set) or 'gold' (pre-2026-09 behavior, which leaked the stage-1 gold reward into the always-observed history).",
    )
    return parser


def args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    return config


def apply_saved_config_for_add_reps(args: argparse.Namespace) -> argparse.Namespace:
    config_path = args.output_dir / "run_config.json"
    if args.add_reps <= 0 or not config_path.exists():
        return args
    saved = load_json(config_path)
    keep = {"output_dir", "add_reps", "target_reps"}
    for key, value in saved.items():
        if key in keep:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    if "priority_form" not in saved:
        # Run directories created before the priority-form flag used the
        # legacy power-one priority; resume them as such, never as theorem.
        args.priority_form = "legacy"
        print(f"[resume] {config_path} predates --priority-form; continuing with priority_form=legacy", file=sys.stderr)
    if "pilot_fraction" not in saved:
        # Pre-pilot run directories used the uncharged variance model.
        args.pilot_fraction = 0.0
        print(f"[resume] {config_path} predates --pilot-fraction; continuing with pilot_fraction=0", file=sys.stderr)
    if "variance_estimator" not in saved:
        args.variance_estimator = "ridge"
        print(f"[resume] {config_path} predates --variance-estimator; continuing with ridge", file=sys.stderr)
    if "h2_progress_source" not in saved:
        args.h2_progress_source = "gold"
        print(f"[resume] {config_path} predates --h2-progress-source; continuing with gold H2 features", file=sys.stderr)
    return args


def load_dtr_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.num_decision_intervals != 2:
        raise ValueError("This MC driver currently supports exactly two decision intervals.")
    orw.HAVE_SKLEARN = False
    if args.synthetic_smoke_test:
        features, contacts, labels = orw.synthetic_inputs(args.seed, n_clients=max(args.sample_size + 50, 300))
    else:
        features, contacts, labels = orw.load_inputs(args)
    dtr, coverage, population = orw.build_dtr_frame(features, contacts, labels, args)
    if dtr.empty:
        raise ValueError("No rows remain after DTR frame construction.")
    needs_silver_variance = args.variance_estimator in {"link_silver", "silver_only"}
    needs_silver_h2 = args.h2_progress_source == "silver"
    if needs_silver_variance or needs_silver_h2:
        if args.synthetic_smoke_test:
            if needs_silver_variance:
                for col in ("silver_var_t1", "silver_var_t2"):
                    dtr[col] = 1.0
            # No silver table in the synthetic smoke; the H2 swap is a no-op there.
        else:
            spec = psf.IntervalSpec(pd.Timestamp(args.start_date), args.interval_months, args.num_decision_intervals)
            silver = orw.read_selected_csv(Path(args.silver_labels_csv), [psf.CONTACT_ID, psf.PRIMARY_LABEL], "silver labels CSV")
            if needs_silver_variance:
                silver_stats = silver_stage_dispersion(contacts, silver, spec, args.date_format)
                dtr = dtr.merge(silver_stats, on=psf.CLIENT_ID, how="left")
                for col in ("silver_var_t1", "silver_var_t2"):
                    if col not in dtr.columns:
                        dtr[col] = np.nan
                    fill = float(np.nanmedian(pd.to_numeric(dtr[col], errors="coerce"))) if dtr[col].notna().any() else 0.0
                    dtr[col] = pd.to_numeric(dtr[col], errors="coerce").fillna(fill)
            if needs_silver_h2:
                # Replace the gold-label-derived stage-1 progress summaries in the
                # "always observed" stage-2 history with SILVER-derived versions:
                # in deployment these summaries come from the LLM labels available
                # for every casenote, while gold annotation reveals only the
                # rewards. Rewards R1/R2 were already computed from gold inside
                # build_dtr_frame and are untouched by this swap.
                silver_feats, _ = psf.build_interval_features(features, contacts, silver, spec, args.date_format)
                swap_cols = [c for c in orw.h2_columns() if "progress" in c]
                missing = [c for c in swap_cols if c not in silver_feats.columns]
                if missing:
                    raise ValueError(f"Silver interval features missing H2 columns: {missing}")
                merged = dtr[[psf.CLIENT_ID]].merge(
                    silver_feats[[psf.CLIENT_ID, *swap_cols]], on=psf.CLIENT_ID, how="left"
                )
                for col in swap_cols:
                    dtr[col] = merged[col].values
    if getattr(args, "oracle_propensity_csv", None):
        cols = [f"oracle_g{stage}_{level}" for stage in (1, 2) for level in orw.ACTION_LEVELS]
        oracle = orw.read_selected_csv(Path(args.oracle_propensity_csv), [psf.CLIENT_ID, *cols], "oracle propensity CSV")
        oracle = psf.normalize_id_columns(oracle, [psf.CLIENT_ID]).drop_duplicates(psf.CLIENT_ID)
        merged = dtr[[psf.CLIENT_ID]].merge(oracle, on=psf.CLIENT_ID, how="left")
        if merged[cols].isna().any().any():
            raise ValueError("oracle propensity CSV does not cover every analysis client")
        for col in cols:
            dtr[col] = merged[col].values
    return dtr, coverage, population


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_saved_config_for_add_reps(args)
    budgets = parse_csv_floats(args.budgets)
    targets = select_targets(parse_csv_strings(args.targets))
    estimators = parse_csv_strings(args.estimators)
    if any(estimator not in {"plugin", "drl"} for estimator in estimators):
        raise ValueError("--estimators must contain only plugin,drl for this driver.")
    for s in parse_csv_strings(args.samplers):
        canonical_cli_sampler(s)  # raises on unknown names; warns on deprecated aliases
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.target_reps <= 0:
        raise ValueError("--target-reps must be positive.")
    if args.add_reps < 0:
        raise ValueError("--add-reps must be nonnegative.")

    output_dir = args.output_dir
    batches_dir = output_dir / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    started = time.perf_counter()
    if not (output_dir / "run_config.json").exists() or args.add_reps == 0:
        write_json_atomic(output_dir / "run_config.json", args_to_config(args))

    dtr, coverage, population = load_dtr_frame(args)
    population.to_csv(output_dir / "analysis_population.csv", index=False)
    coverage.to_csv(output_dir / "interval_coverage.csv", index=False)

    manifests = completed_manifests(batches_dir)
    completed = contiguous_completed_reps(manifests, start=args.rep_start)
    target_reps = completed + args.add_reps if args.add_reps else args.rep_start + args.target_reps
    if target_reps < completed:
        target_reps = completed
    if manifests:
        completed = combine_batches(output_dir, manifests, start=args.rep_start)

    write_json_atomic(
        output_dir / "incremental_status.json",
        {
            "status": "running",
            "output_dir": str(output_dir),
            "sample_size": args.sample_size,
            "target_reps": target_reps,
            "completed_reps": completed,
            "completed_batches": len(manifests),
            "batch_size": args.batch_size,
            "budgets": args.budgets,
            "targets": args.targets,
            "estimators": args.estimators,
            "jobs": args.jobs,
            "started_at": utc_now(),
        },
    )

    cursor = completed
    while cursor < target_reps:
        if _INTERRUPTED:
            break
        reps = min(args.batch_size, target_reps - cursor)
        batch_dir = batches_dir / batch_dir_name(cursor, reps)
        if not batch_complete(batch_dir):
            print(
                f"[{utc_now()}] start batch {batch_dir.name} "
                f"({cursor}/{target_reps} reps complete)",
                flush=True,
            )
            write_json_atomic(
                output_dir / "incremental_status.json",
                {
                    "status": "running",
                    "output_dir": str(output_dir),
                    "target_reps": target_reps,
                    "completed_reps": cursor,
                    "current_batch": batch_dir.name,
                    "batch_size": reps,
                    "wrapper_elapsed_seconds": time.perf_counter() - started,
                },
            )
            run_batch(dtr, targets, budgets, args, output_dir, cursor, reps)
        manifests = completed_manifests(batches_dir)
        cursor = combine_batches(output_dir, manifests, start=args.rep_start)
        print(
            f"[{utc_now()}] completed {cursor}/{target_reps} reps; "
            f"outputs={output_dir}",
            flush=True,
        )
        write_summary(output_dir, args, cursor, target_reps, time.perf_counter() - started)

    manifests = completed_manifests(batches_dir)
    completed = combine_batches(output_dir, manifests, start=args.rep_start) if manifests else 0
    elapsed = time.perf_counter() - started
    status = "interrupted" if _INTERRUPTED and completed < target_reps else "completed"
    write_summary(output_dir, args, completed, target_reps, elapsed)
    write_json_atomic(
        output_dir / "incremental_status.json",
        {
            "status": status,
            "output_dir": str(output_dir),
            "sample_size": args.sample_size,
            "target_reps": target_reps,
            "completed_reps": completed,
            "completed_batches": len(manifests),
            "batch_size": args.batch_size,
            "wrapper_elapsed_seconds": elapsed,
            "finished_at": utc_now(),
        },
    )
    print(f"{status.capitalize()} {completed} contiguous MC reps at n={args.sample_size} in {elapsed:.3f}s")
    print(f"Outputs: {output_dir}")
    return 130 if status == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
