#!/usr/bin/env python3
"""Two-stage outreach DTR value estimation with reward-only annotation priorities.

This script is designed to be safe for the protected casenote data: it reads only documented
analytical columns via ``usecols`` and writes aggregate diagnostics only. It never
prints client rows, identifiers, notes, raw LLM outputs, or justifications.

The primary analysis population is clients with positive outreach in both of the
first two six-month stages. Actions are positive outreach quintiles 1..5. The
evaluation policy shifts each positive quintile up by one, with the top quintile
remaining fixed: delta(a)=min(a+1, 5).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on the execution environment.
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import KFold
    HAVE_SKLEARN = True
except ImportError:  # Lightweight deterministic fallback for smoke tests/minimal containers.
    RandomForestClassifier = None
    RandomForestRegressor = None
    HAVE_SKLEARN = False

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_surrogate_feasibility import (  # noqa: E402
    CLIENT_ID,
    CONTACT_DATE,
    CONTACT_ID,
    DEFAULT_CONTACTS_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_LABELS_CSV,
    IntervalSpec,
    PLACEMENT_TYPE,
    PRIMARY_LABEL,
    STATE_COLUMNS,
    build_interval_features,
    derive_outcome_from_contacts,
    normalize_id_columns,
    parse_contact_dates,
    read_selected_csv,
)

ACTION_LEVELS = np.array([1, 2, 3, 4, 5], dtype=int)
EPS = 1e-6


class SimpleKFold:
    def __init__(self, n_splits: int, shuffle: bool, random_state: int):
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)

    def split(self, frame: pd.DataFrame):
        n = len(frame)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        folds = np.array_split(indices, self.n_splits)
        for test_idx in folds:
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=False)
            yield train_idx, test_idx


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        x_mat = np.asarray(x, dtype=float)
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        y_vec = np.asarray(y, dtype=float)
        penalty = self.alpha * np.eye(x_mat.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(x_mat.T @ x_mat + penalty) @ x_mat.T @ y_vec

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegressor must be fit before predict")
        x_mat = np.asarray(x, dtype=float)
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        return x_mat @ self.coef_


class ConstantBinaryClassifier:
    def __init__(self, probability: float | None = None):
        self.probability = probability
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=float)
        self.probability = float(np.mean(y_arr)) if len(y_arr) else 0.0

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.probability is None:
            raise RuntimeError("ConstantBinaryClassifier must be fit before predict_proba")
        p = float(np.clip(self.probability, EPS, 1.0 - EPS))
        return np.column_stack([np.full(len(x), 1.0 - p), np.full(len(x), p)])


class BinaryRidgeClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_ = np.array([0, 1], dtype=int)
        self.model_ = RidgeRegressor(alpha=alpha)

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        self.model_.fit(x, np.asarray(y, dtype=float))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.model_.predict(x), EPS, 1.0 - EPS)
        return np.column_stack([1.0 - p, p])


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


class RidgeSoftmaxClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_ = ACTION_LEVELS.copy()
        self.model_ = RidgeRegressor(alpha=alpha)

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=int)
        y_onehot = np.column_stack([(y_arr == cls).astype(float) for cls in self.classes_])
        x_mat = np.asarray(x, dtype=float)
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        penalty = self.alpha * np.eye(x_mat.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.pinv(x_mat.T @ x_mat + penalty) @ x_mat.T @ y_onehot
        self.model_.coef_ = coef

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model_.coef_ is None:
            raise RuntimeError("RidgeSoftmaxClassifier must be fit before predict_proba")
        x_mat = np.asarray(x, dtype=float)
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        scores = x_mat @ self.model_.coef_
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    stage1_col: str
    stage2_col: str
    description: str
    terminal_progress_mode: str = "none"
    annotation_target: bool = True


TARGETS = (
    TargetSpec(
        "progress_total",
        "R1_progress",
        "R2_progress",
        "R1 max progress plus stage-2 max progress",
    ),
    TargetSpec(
        "placement_total",
        "R1_progress",
        "Y2_placement",
        "R1 max progress plus year-2 max placement",
    ),
    TargetSpec(
        "combined_total",
        "R1_progress",
        "R2_progress_plus_Y2_placement",
        "R1 max progress plus stage-2 max progress plus year-2 max placement",
    ),
    TargetSpec(
        "y2_placement_only",
        "zero_reward",
        "Y2_placement",
        "Year-2 max placement only; stage-1 progress summaries enter H2; no R2 progress predictor",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "y2_placement_time_weighted",
        "zero_reward",
        "Y2_placement_time_weighted",
        "Time-weighted average PlacementType during the year-2 outcome window",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "y2_placement_improvement_speed",
        "zero_reward",
        "Y2_placement_improvement_speed",
        "Time-to-first placement improvement score: 1 - days-to-improvement / outcome-window days",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "y2_placement_improvement_from_baseline",
        "zero_reward",
        "Y2_placement_improvement_from_baseline",
        "Maximum year-2 PlacementType minus placement baseline at outcome-window start",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "y2_any_placement_improvement",
        "zero_reward",
        "Y2_any_placement_improvement",
        "Indicator that year-2 PlacementType ever improves relative to placement baseline",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "r1_plus_y2_placement_time_weighted",
        "R1_progress",
        "Y2_placement_time_weighted",
        "R1 max progress plus time-weighted average year-2 PlacementType",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "r1_plus_y2_placement_improvement_speed",
        "R1_progress",
        "Y2_placement_improvement_speed",
        "R1 max progress plus year-2 placement improvement speed score",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "r1_plus_y2_placement_improvement_from_baseline",
        "R1_progress",
        "Y2_placement_improvement_from_baseline",
        "R1 max progress plus max year-2 placement improvement from baseline",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
    TargetSpec(
        "r1_plus_y2_any_placement_improvement",
        "R1_progress",
        "Y2_any_placement_improvement",
        "R1 max progress plus indicator for any year-2 placement improvement from baseline",
        terminal_progress_mode="none",
        annotation_target=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--contacts-csv", default=DEFAULT_CONTACTS_CSV)
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", default="results/outreach_dtr_reward_only")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--date-format", default=None)
    parser.add_argument("--interval-months", type=int, default=6)
    parser.add_argument("--num-decision-intervals", type=int, default=2)
    parser.add_argument("--placement-outcome-end-date", default="2021-01-01")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--propensity-floor", type=float, default=0.02)
    parser.add_argument(
        "--shift-quintiles",
        type=int,
        default=1,
        help="Positive outreach-quintile shift for the evaluation policy: delta(a)=min(a+shift, 5).",
    )
    parser.add_argument("--shift-quintiles-stage1", type=int, default=None,
                        help="Stage-1 shift (default: --shift-quintiles). 0 keeps the behavior policy at stage 1.")
    parser.add_argument("--shift-quintiles-stage2", type=int, default=None,
                        help="Stage-2 shift (default: --shift-quintiles). 0 keeps the behavior policy at stage 2.")
    parser.add_argument("--annotation-budget", type=float, default=1.0, help="Expected reward labels per unit; full two-stage cost is 2.")
    parser.add_argument("--annotation-floor", type=float, default=0.01)
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    return parser.parse_args()


def require_columns(actual_columns: Iterable[str], required_columns: Iterable[str], label: str) -> None:
    actual = set(actual_columns)
    missing = sorted(set(required_columns) - actual)
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [CLIENT_ID, *STATE_COLUMNS]
    contact_columns = [CLIENT_ID, CONTACT_ID, CONTACT_DATE, PLACEMENT_TYPE]
    label_columns = [CONTACT_ID, PRIMARY_LABEL]
    features = read_selected_csv(Path(args.features_csv), feature_columns, "features CSV")
    contacts = read_selected_csv(Path(args.contacts_csv), contact_columns, "contacts CSV")
    labels = read_selected_csv(Path(args.labels_csv), label_columns, "labels CSV")
    return features, contacts, labels


def synthetic_inputs(seed: int, n_clients: int = 600) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    client_ids = np.array([f"SYN{i:04d}" for i in range(n_clients)], dtype=object)
    x0 = rng.normal(size=n_clients)
    x1 = rng.normal(size=n_clients)
    baseline_place = np.clip(np.rint(1.0 + 0.6 * x0 + rng.normal(scale=0.7, size=n_clients)), 0, 4).astype(int)
    features = pd.DataFrame(
        {
            CLIENT_ID: client_ids,
            "DateFirstSeen": pd.Timestamp("2018-01-01").toordinal() + rng.integers(0, 365, size=n_clients),
            "Housing_application_Yes": rng.integers(0, 2, size=n_clients),
            "Service_refusal_Yes": rng.integers(0, 2, size=n_clients),
            "Impt_documents_Yes": rng.integers(0, 2, size=n_clients),
            "Benefits_Yes": rng.integers(0, 2, size=n_clients),
            "num_contacts_before_2019": rng.poisson(4, size=n_clients),
            "max_PlacementType_before_2019": baseline_place,
            "Program": rng.choice(["A", "B", "C"], size=n_clients),
            "BelievedChronic": rng.choice(["Yes", "No"], size=n_clients),
            "Gender_cat": rng.choice(["F", "M", "Other"], size=n_clients),
            "Race_cat": rng.choice(["R1", "R2", "R3"], size=n_clients),
            "Age_cat": rng.choice(["18-34", "35-54", "55+"], size=n_clients),
            "Zone_cat": rng.choice(["Z1", "Z2", "Z3", "Z4"], size=n_clients),
            "Ethnicity": rng.choice(["E1", "E2", "Unknown"], size=n_clients),
            "Was311Call": rng.choice(["Yes", "No"], size=n_clients),
            "Was911Call": rng.choice(["Yes", "No"], size=n_clients),
            "Removal958": rng.choice(["Yes", "No"], size=n_clients),
        }
    )

    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    contact_idx = 0
    for i, client_id in enumerate(client_ids):
        for stage, start in [(1, pd.Timestamp("2019-01-01")), (2, pd.Timestamp("2019-07-01"))]:
            # Some zero outreach clients are included to exercise exclusion logic.
            lam = max(0.05, 4.0 + 1.1 * x0[i] + 0.6 * x1[i] + 0.8 * (stage == 2))
            n_contacts = int(rng.poisson(lam))
            if rng.random() < 0.08:
                n_contacts = 0
            for _ in range(n_contacts):
                contact_idx += 1
                contact_id = f"C{contact_idx:06d}"
                day = int(rng.integers(0, 182))
                date = start + pd.Timedelta(days=day)
                action_signal = min(n_contacts, 12) / 12.0
                progress_mean = 0.5 + 1.2 * action_signal + 0.5 * x0[i] - 0.25 * baseline_place[i]
                if stage == 2:
                    progress_mean += 0.35 * x1[i]
                progress = float(np.clip(progress_mean + rng.normal(scale=0.9), 0, 4))
                rows.append(
                    {
                        CLIENT_ID: client_id,
                        CONTACT_ID: contact_id,
                        CONTACT_DATE: date.strftime("%Y-%m-%d"),
                        PLACEMENT_TYPE: int(np.clip(round(baseline_place[i] + 0.3 * progress + rng.normal()), 0, 4)),
                    }
                )
                labels.append({CONTACT_ID: contact_id, PRIMARY_LABEL: progress})

        # Year-2 administrative placement contacts.
        n_y = int(rng.integers(1, 4))
        latent_place = baseline_place[i] + 0.5 * x0[i] + 0.3 * x1[i] + rng.normal(scale=0.8)
        for _ in range(n_y):
            contact_idx += 1
            contact_id = f"C{contact_idx:06d}"
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
            rows.append(
                {
                    CLIENT_ID: client_id,
                    CONTACT_ID: contact_id,
                    CONTACT_DATE: date.strftime("%Y-%m-%d"),
                    PLACEMENT_TYPE: int(np.clip(round(latent_place), 0, 4)),
                }
            )
            # Labels in the outcome year are not used for decision-period progress rewards.
    return features, pd.DataFrame(rows), pd.DataFrame(labels)


def max_progress_reward(analysis: pd.DataFrame, stage: int) -> pd.Series:
    value = pd.to_numeric(analysis[f"max_progress_t{stage}"], errors="coerce")
    observed = pd.to_numeric(analysis[f"progress_observed_t{stage}"], errors="coerce").fillna(0).gt(0)
    return value.where(observed, np.nan)


def derive_placement_trajectory_outcomes(
    features: pd.DataFrame,
    contacts: pd.DataFrame,
    spec: IntervalSpec,
    date_format: str | None,
    outcome_end_date: str | None,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    require_columns(contacts.columns, [CLIENT_ID, CONTACT_DATE, PLACEMENT_TYPE], "contacts for placement trajectory outcomes")
    outcome_start = spec.end_date
    outcome_end = pd.Timestamp(outcome_end_date) if outcome_end_date else None
    if outcome_end is None:
        raise ValueError("placement trajectory outcomes require --placement-outcome-end-date.")
    window_days = int((outcome_end - outcome_start).days)
    if window_days <= 0:
        raise ValueError("placement outcome end date must be after the decision-window end date.")

    feature_cols = [CLIENT_ID]
    if "max_PlacementType_before_2019" in features.columns:
        feature_cols.append("max_PlacementType_before_2019")
    feature_base = normalize_id_columns(features[feature_cols], [CLIENT_ID]).drop_duplicates(CLIENT_ID).copy()
    feature_base["feature_baseline"] = pd.to_numeric(
        feature_base.get("max_PlacementType_before_2019"), errors="coerce"
    )

    contacts_work = normalize_id_columns(contacts, [CLIENT_ID]).copy()
    contacts_work[CONTACT_DATE] = parse_contact_dates(contacts_work[CONTACT_DATE], date_format)
    contacts_work[PLACEMENT_TYPE] = pd.to_numeric(contacts_work[PLACEMENT_TYPE], errors="coerce")
    contacts_work = contacts_work.dropna(subset=[CLIENT_ID, CONTACT_DATE, PLACEMENT_TYPE])
    contacts_work = contacts_work[contacts_work[CONTACT_DATE].lt(outcome_end)].copy()
    contacts_work = contacts_work.sort_values([CLIENT_ID, CONTACT_DATE])

    pre_contacts = contacts_work[contacts_work[CONTACT_DATE].lt(outcome_start)]
    if pre_contacts.empty:
        pre_baseline = pd.DataFrame(columns=[CLIENT_ID, "pre_outcome_baseline"])
    else:
        pre_baseline = (
            pre_contacts.groupby(CLIENT_ID, as_index=False)
            .tail(1)[[CLIENT_ID, PLACEMENT_TYPE]]
            .rename(columns={PLACEMENT_TYPE: "pre_outcome_baseline"})
        )

    outcome_contacts = contacts_work[
        contacts_work[CONTACT_DATE].ge(outcome_start) & contacts_work[CONTACT_DATE].lt(outcome_end)
    ].copy()
    max_outcome = (
        outcome_contacts.groupby(CLIENT_ID)[PLACEMENT_TYPE].max().rename("Y2_placement").reset_index()
    )
    base = feature_base.merge(max_outcome, on=CLIENT_ID, how="inner").merge(pre_baseline, on=CLIENT_ID, how="left")

    rows: list[dict[str, float | str]] = []
    baseline_source_counts = {"pre_outcome_contact": 0, "feature_baseline": 0, "first_outcome_contact": 0}
    clients_with_improvement = 0
    grouped = {client_id: group for client_id, group in outcome_contacts.groupby(CLIENT_ID)}
    for rec in base.to_dict(orient="records"):
        client_id = rec[CLIENT_ID]
        client_contacts = grouped.get(client_id)
        if client_contacts is None or client_contacts.empty:
            continue

        baseline = rec.get("pre_outcome_baseline")
        source = "pre_outcome_contact"
        if pd.isna(baseline):
            baseline = rec.get("feature_baseline")
            source = "feature_baseline"
        if pd.isna(baseline):
            baseline = float(client_contacts.iloc[0][PLACEMENT_TYPE])
            source = "first_outcome_contact"
        baseline = float(baseline)
        baseline_source_counts[source] += 1

        area = 0.0
        cursor = outcome_start
        state = baseline
        first_improvement_days = float(window_days)
        improved = False
        for _, contact in client_contacts.iterrows():
            contact_date = contact[CONTACT_DATE]
            duration = max(0, int((contact_date - cursor).days))
            area += duration * state
            placement = float(contact[PLACEMENT_TYPE])
            if not improved and placement > baseline:
                first_improvement_days = float(max(0, int((contact_date - outcome_start).days)))
                improved = True
            state = placement
            cursor = contact_date
        area += max(0, int((outcome_end - cursor).days)) * state
        if improved:
            clients_with_improvement += 1

        rows.append(
            {
                CLIENT_ID: client_id,
                "Y2_placement": float(rec["Y2_placement"]),
                "Y2_placement_baseline": baseline,
                "Y2_placement_time_weighted": float(area / window_days),
                "Y2_days_to_first_placement_improvement": first_improvement_days,
                "Y2_placement_improvement_speed": float(1.0 - first_improvement_days / window_days),
                "Y2_placement_improvement_from_baseline": float(rec["Y2_placement"] - baseline),
                "Y2_any_placement_improvement": float(improved),
            }
        )

    metadata = {
        "derived_outcome": True,
        "derived_outcome_column": "Y2_placement",
        "outcome_start_date": str(outcome_start.date()),
        "outcome_end_date": str(outcome_end.date()),
        "placement_outcome_window_days": window_days,
        "clients_with_derived_outcome": int(len(rows)),
        "post_observation_contacts_for_outcome": int(len(outcome_contacts)),
        "clients_with_pre_outcome_baseline_contact": int(baseline_source_counts["pre_outcome_contact"]),
        "clients_using_feature_baseline": int(baseline_source_counts["feature_baseline"]),
        "clients_using_first_outcome_as_baseline": int(baseline_source_counts["first_outcome_contact"]),
        "clients_with_placement_improvement": int(clients_with_improvement),
    }
    return pd.DataFrame.from_records(rows), metadata


def build_dtr_frame(
    features: pd.DataFrame,
    contacts: pd.DataFrame,
    labels: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = IntervalSpec(pd.Timestamp(args.start_date), args.interval_months, args.num_decision_intervals)
    analysis, coverage = build_interval_features(features, contacts, labels, spec, args.date_format)
    y_feature_cols = [CLIENT_ID, "max_PlacementType_before_2019"]
    y_feature_cols = [col for col in y_feature_cols if col in analysis.columns]
    y_features = analysis[y_feature_cols].drop_duplicates(CLIENT_ID).copy()
    y_frame, y_meta = derive_placement_trajectory_outcomes(
        y_features,
        contacts,
        spec,
        args.date_format,
        outcome_end_date=args.placement_outcome_end_date,
    )
    analysis = analysis.merge(y_frame, on=CLIENT_ID, how="left")
    analysis["A1"] = pd.to_numeric(analysis["outreach_bin_t1"], errors="coerce")
    analysis["A2"] = pd.to_numeric(analysis["outreach_bin_t2"], errors="coerce")
    analysis["R1_progress"] = max_progress_reward(analysis, 1)
    analysis["R2_progress"] = max_progress_reward(analysis, 2)
    placement_outcome_cols = [
        "Y2_placement",
        "Y2_placement_time_weighted",
        "Y2_days_to_first_placement_improvement",
        "Y2_placement_improvement_speed",
        "Y2_placement_improvement_from_baseline",
        "Y2_any_placement_improvement",
    ]
    for col in placement_outcome_cols:
        analysis[col] = pd.to_numeric(analysis[col], errors="coerce")
    analysis["R2_progress_plus_Y2_placement"] = analysis["R2_progress"] + analysis["Y2_placement"]
    analysis["zero_reward"] = 0.0

    n_initial = len(analysis)
    stage1_positive = analysis["A1"].isin(ACTION_LEVELS)
    stage2_positive = analysis["A2"].isin(ACTION_LEVELS)
    positive = stage1_positive & stage2_positive
    n_positive = int(positive.sum())
    reward_columns = ["R1_progress", "R2_progress", "Y2_placement"]
    reward_complete = analysis[reward_columns].notna().all(axis=1)
    keep = positive & reward_complete
    dtr = analysis.loc[keep].copy().reset_index(drop=True)
    if not dtr.empty:
        dtr["A1"] = dtr["A1"].astype(int)
        dtr["A2"] = dtr["A2"].astype(int)
        dtr["R2_progress_plus_Y2_placement"] = dtr["R2_progress"] + dtr["Y2_placement"]
    population = pd.DataFrame.from_records(
        [
            {
                "initial_clients": int(n_initial),
                "positive_outreach_stage1": int(stage1_positive.sum()),
                "positive_outreach_stage2": int(stage2_positive.sum()),
                "positive_outreach_both_stages": n_positive,
                "complete_reward_rows": int(keep.sum()),
                "excluded_zero_or_missing_outreach": int(n_initial - n_positive),
                "excluded_missing_reward_among_positive": int((positive & ~reward_complete).sum()),
                "r1_progress_observed_rows": int(analysis["R1_progress"].notna().sum()),
                "r2_progress_observed_rows": int(analysis["R2_progress"].notna().sum()),
                "y2_placement_observed_rows": int(analysis["Y2_placement"].notna().sum()),
                "y2_time_weighted_observed_rows": int(analysis["Y2_placement_time_weighted"].notna().sum()),
                "y2_improvement_speed_observed_rows": int(analysis["Y2_placement_improvement_speed"].notna().sum()),
                "y2_improvement_from_baseline_observed_rows": int(analysis["Y2_placement_improvement_from_baseline"].notna().sum()),
                "y2_any_improvement_observed_rows": int(analysis["Y2_any_placement_improvement"].notna().sum()),
                "r1_missing_among_positive": int((positive & analysis["R1_progress"].isna()).sum()),
                "r2_missing_among_positive": int((positive & analysis["R2_progress"].isna()).sum()),
                "y2_missing_among_positive": int((positive & analysis["Y2_placement"].isna()).sum()),
                **y_meta,
            }
        ]
    )
    return dtr, coverage, population


def stage_shift(args: argparse.Namespace, stage: int) -> int:
    """Per-stage evaluation-policy shift; falls back to the common --shift-quintiles."""
    value = getattr(args, f"shift_quintiles_stage{stage}", None)
    return int(args.shift_quintiles if value is None else value)


def shift_up_probs(g: np.ndarray, shift_quintiles: int) -> np.ndarray:
    shift = int(shift_quintiles)
    if shift < 0 or shift >= len(ACTION_LEVELS):
        raise ValueError(f"--shift-quintiles must be between 0 and {len(ACTION_LEVELS) - 1}.")
    q = np.zeros_like(g, dtype=float)
    for source_idx in range(len(ACTION_LEVELS)):
        target_idx = min(source_idx + shift, len(ACTION_LEVELS) - 1)
        q[:, target_idx] += g[:, source_idx]
    return q


def encode_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = frame[columns].copy()
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            values = pd.to_numeric(x[col], errors="coerce")
            fill = values.median()
            if pd.isna(fill):
                fill = 0.0
            x[col] = values.fillna(fill)
        else:
            x[col] = x[col].astype("string").fillna("Unknown")
    return pd.get_dummies(x, dummy_na=False)


def encoded_like(frame: pd.DataFrame, columns: list[str], encoded_columns: pd.Index) -> pd.DataFrame:
    return encode_frame(frame, columns).reindex(columns=encoded_columns, fill_value=0)


class MultinomialLogitClassifier:
    """Standardized multinomial logistic regression (sklearn) with the ACTION_LEVELS class layout.

    Alternative propensity learner to RidgeSoftmaxClassifier, whose softmax over fitted linear
    probabilities in [0, 1] compresses the propensities toward uniform (simulator validation,
    2026-09-10). Selected with --propensity-learner multinomial; the default is unchanged.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 3000):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
        self.classes_ = ACTION_LEVELS.copy()

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        self.model_.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
        self.classes_ = np.asarray(self.model_[-1].classes_, dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(x, dtype=float))


def fit_classifier(train: pd.DataFrame, columns: list[str], y: pd.Series, args: argparse.Namespace, seed: int):
    x_train = encode_frame(train, columns)
    learner = getattr(args, "propensity_learner", "ridge_softmax")
    if learner == "multinomial":
        model = MultinomialLogitClassifier(C=float(getattr(args, "propensity_c", 1.0)))
        model.fit(x_train, y.astype(int))
        return model, x_train.columns
    if HAVE_SKLEARN:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )
    else:
        model = RidgeSoftmaxClassifier(alpha=5.0)
    model.fit(x_train, y.astype(int))
    return model, x_train.columns


def predict_action_probs(model, encoded_columns: pd.Index, frame: pd.DataFrame, columns: list[str], floor: float) -> np.ndarray:
    x = encoded_like(frame, columns, encoded_columns)
    raw = np.zeros((len(frame), len(ACTION_LEVELS)), dtype=float)
    pred = model.predict_proba(x)
    for col_idx, cls in enumerate(model.classes_):
        if int(cls) in ACTION_LEVELS:
            raw[:, int(cls) - 1] = pred[:, col_idx]
    raw = np.maximum(raw, floor)
    return raw / raw.sum(axis=1, keepdims=True)


def fit_binary_classifier(train: pd.DataFrame, columns: list[str], y: pd.Series, args: argparse.Namespace, seed: int):
    x_train = encode_frame(train, columns)
    y_bin = pd.to_numeric(y, errors="coerce").fillna(0).gt(0).astype(int)
    if y_bin.nunique() < 2:
        model = ConstantBinaryClassifier(float(y_bin.mean()))
    elif HAVE_SKLEARN:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )
    else:
        model = BinaryRidgeClassifier(alpha=5.0)
    model.fit(x_train, y_bin)
    return model, x_train.columns


def predict_positive_probability(model, encoded_columns: pd.Index, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = encoded_like(frame, columns, encoded_columns)
    pred = model.predict_proba(x)
    if 1 in set(int(cls) for cls in model.classes_):
        one_idx = [int(cls) for cls in model.classes_].index(1)
        prob = pred[:, one_idx]
    else:
        prob = np.zeros(len(frame), dtype=float)
    return np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)


def fit_regressor(train: pd.DataFrame, columns: list[str], y: pd.Series, args: argparse.Namespace, seed: int):
    x_train = encode_frame(train, columns)
    if HAVE_SKLEARN:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )
    else:
        model = RidgeRegressor(alpha=5.0)
    model.fit(x_train, pd.to_numeric(y, errors="coerce"))
    return model, x_train.columns


def predict_regressor(model, encoded_columns: pd.Index, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = encoded_like(frame, columns, encoded_columns)
    return np.asarray(model.predict(x), dtype=float)


def predict_action_values(model, encoded_columns: pd.Index, base: pd.DataFrame, columns: list[str], action_col: str) -> np.ndarray:
    values = np.zeros((len(base), len(ACTION_LEVELS)), dtype=float)
    for j, action in enumerate(ACTION_LEVELS):
        modified = base.copy()
        modified[action_col] = int(action)
        values[:, j] = predict_regressor(model, encoded_columns, modified, columns)
    return values


def observed_probability(prob: np.ndarray, actions: pd.Series) -> np.ndarray:
    idx = actions.astype(int).to_numpy() - 1
    return prob[np.arange(len(actions)), idx]


def weighted_action_mean(q: np.ndarray, action_values: np.ndarray) -> np.ndarray:
    return np.sum(q * action_values, axis=1)


def safe_ratio(num: np.ndarray, den: np.ndarray, floor: float) -> np.ndarray:
    return num / np.maximum(den, floor)


def h1_columns() -> list[str]:
    return list(STATE_COLUMNS)


def h2_columns() -> list[str]:
    return [
        *STATE_COLUMNS,
        "A1",
        "contacts_t1",
        "outreach_bin_t1",
        "max_progress_t1",
        "mean_progress_t1",
        "latest_progress_t1",
        "progress_range_t1",
        "progress_label_count_t1",
        "progress_observed_t1",
    ]


def q1_columns() -> list[str]:
    return [*STATE_COLUMNS, "A1"]


def q2_columns() -> list[str]:
    return [*h2_columns(), "A2"]


def choose_folds(n: int, requested: int) -> int:
    if n < 4:
        raise ValueError("Need at least four rows for cross-fitting.")
    return max(2, min(requested, n))


def wald_interval(value: float, se: float, z: float = 1.96) -> tuple[float, float]:
    return float(value - z * se), float(value + z * se)


def crossfit_behavior_and_values(
    dtr: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    n = len(dtr)
    folds = choose_folds(n, args.n_folds)
    kfold = (KFold(n_splits=folds, shuffle=True, random_state=args.random_state) if HAVE_SKLEARN else SimpleKFold(n_splits=folds, shuffle=True, random_state=args.random_state))
    h1 = h1_columns()
    h2 = h2_columns()
    q1_cols = q1_columns()
    q2_cols = q2_columns()

    g1 = np.zeros((n, len(ACTION_LEVELS)))
    g2 = np.zeros((n, len(ACTION_LEVELS)))
    q1 = np.zeros_like(g1)
    q2 = np.zeros_like(g2)
    target_scores: dict[str, dict[str, np.ndarray]] = {
        target.name: {
            "plugin": np.zeros(n),
            "drl": np.zeros(n),
            "observed": pd.to_numeric(dtr[target.stage1_col], errors="coerce").to_numpy(dtype=float)
            + pd.to_numeric(dtr[target.stage2_col], errors="coerce").to_numpy(dtype=float),
        }
        for target in TARGETS
    }
    prediction_scores: dict[str, dict[str, np.ndarray]] = {
        target.name: {
            "q2_actual": np.full(n, np.nan),
            "q2_pred": np.full(n, np.nan),
        }
        for target in TARGETS
    }

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dtr)):
        train = dtr.iloc[train_idx].copy()
        test = dtr.iloc[test_idx].copy()
        seed = args.random_state + 1009 * fold_idx

        beh1, beh1_cols = fit_classifier(train, h1, train["A1"], args, seed)
        beh2, beh2_cols = fit_classifier(train, h2, train["A2"], args, seed + 1)
        g1_train = predict_action_probs(beh1, beh1_cols, train, h1, args.propensity_floor)
        g2_train = predict_action_probs(beh2, beh2_cols, train, h2, args.propensity_floor)
        g1_test = predict_action_probs(beh1, beh1_cols, test, h1, args.propensity_floor)
        g2_test = predict_action_probs(beh2, beh2_cols, test, h2, args.propensity_floor)
        q1_train = shift_up_probs(g1_train, stage_shift(args, 1))
        q2_train = shift_up_probs(g2_train, stage_shift(args, 2))
        q1_test = shift_up_probs(g1_test, stage_shift(args, 1))
        q2_test = shift_up_probs(g2_test, stage_shift(args, 2))
        g1[test_idx] = g1_test
        g2[test_idx] = g2_test
        q1[test_idx] = q1_test
        q2[test_idx] = q2_test

        for target in TARGETS:
            z1_train = pd.to_numeric(train[target.stage1_col], errors="coerce")
            z2_train = pd.to_numeric(train[target.stage2_col], errors="coerce")
            q2_model, q2_encoded = fit_regressor(train, q2_cols, z2_train, args, seed + 11)
            q2_train_values = predict_action_values(q2_model, q2_encoded, train, q2_cols, "A2")
            v2_train = weighted_action_mean(q2_train, q2_train_values)
            q1_target_train = z1_train.to_numpy(dtype=float) + v2_train
            q1_model, q1_encoded = fit_regressor(train, q1_cols, pd.Series(q1_target_train), args, seed + 12)

            q2_test_values = predict_action_values(q2_model, q2_encoded, test, q2_cols, "A2")
            v2_test = weighted_action_mean(q2_test, q2_test_values)
            q2_obs_test = predict_regressor(q2_model, q2_encoded, test, q2_cols)
            q1_test_values = predict_action_values(q1_model, q1_encoded, test, q1_cols, "A1")
            v1_test = weighted_action_mean(q1_test, q1_test_values)
            q1_obs_test = predict_regressor(q1_model, q1_encoded, test, q1_cols)

            z1_test = pd.to_numeric(test[target.stage1_col], errors="coerce").to_numpy(dtype=float)
            z2_test = pd.to_numeric(test[target.stage2_col], errors="coerce").to_numpy(dtype=float)
            g1_obs = observed_probability(g1_test, test["A1"])
            g2_obs = observed_probability(g2_test, test["A2"])
            q1_obs = observed_probability(q1_test, test["A1"])
            q2_obs = observed_probability(q2_test, test["A2"])
            rho1 = safe_ratio(q1_obs, g1_obs, args.propensity_floor)
            rho12 = rho1 * safe_ratio(q2_obs, g2_obs, args.propensity_floor)
            drl = v1_test + rho1 * (z1_test + v2_test - q1_obs_test) + rho12 * (z2_test - q2_obs_test)
            target_scores[target.name]["plugin"][test_idx] = v1_test
            target_scores[target.name]["drl"][test_idx] = drl
            prediction_scores[target.name]["q2_actual"][test_idx] = z2_test
            prediction_scores[target.name]["q2_pred"][test_idx] = q2_obs_test

    prediction_diagnostics = prediction_diagnostics_from_scores(prediction_scores)

    rows: list[dict[str, float | int | str]] = []
    for target in TARGETS:
        observed = target_scores[target.name]["observed"]
        observed_value = float(np.mean(observed))
        observed_se = float(np.std(observed, ddof=1) / math.sqrt(n))
        observed_lower, observed_upper = wald_interval(observed_value, observed_se)
        rows.append(
            {
                "target": target.name,
                "description": target.description,
                "stage1_col": target.stage1_col,
                "stage2_col": target.stage2_col,
                "terminal_progress_mode": target.terminal_progress_mode,
                "annotation_target": target.annotation_target,
                "estimator": "observed_behavior_mean",
                "value": observed_value,
                "se": observed_se,
                "ci_lower_95": observed_lower,
                "ci_upper_95": observed_upper,
                "difference_vs_observed": 0.0,
                "difference_se": 0.0,
                "difference_ci_lower_95": 0.0,
                "difference_ci_upper_95": 0.0,
                "n": n,
            }
        )
        for estimator in ["plugin", "drl"]:
            scores = target_scores[target.name][estimator]
            value = float(np.mean(scores))
            se = float(np.std(scores, ddof=1) / math.sqrt(n))
            diff_scores = scores - observed
            diff = value - observed_value
            diff_se = float(np.std(diff_scores, ddof=1) / math.sqrt(n))
            lower, upper = wald_interval(value, se)
            diff_lower, diff_upper = wald_interval(diff, diff_se)
            rows.append(
                {
                    "target": target.name,
                    "description": target.description,
                    "stage1_col": target.stage1_col,
                    "stage2_col": target.stage2_col,
                    "terminal_progress_mode": target.terminal_progress_mode,
                    "annotation_target": target.annotation_target,
                    "estimator": f"shift_up_{estimator}",
                    "value": value,
                    "se": se,
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "difference_vs_observed": diff,
                    "difference_se": diff_se,
                    "difference_ci_lower_95": diff_lower,
                    "difference_ci_upper_95": diff_upper,
                    "n": n,
                }
            )
    value_estimates = pd.DataFrame.from_records(rows)
    nuisances = {"g1": g1, "g2": g2, "q1": q1, "q2": q2}
    overlap = overlap_diagnostics(dtr, nuisances, args)
    return value_estimates, overlap, nuisances, prediction_diagnostics


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    auc = (float(np.sum(ranks[y == 1])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1.0)
    return float(np.sum(precision * y_sorted) / n_pos)


def binary_log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_prediction_diagnostics_from_scores(target: str, description: str, actual: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], EPS, 1.0 - EPS)
    if len(y) == 0:
        return pd.DataFrame()
    prevalence = float(np.mean(y))
    brier = float(np.mean(np.square(y - p)))
    baseline_brier = float(np.mean(np.square(y - prevalence)))
    return pd.DataFrame.from_records([
        {
            "target": target,
            "description": description,
            "model_stage": "q2_observed_action_classifier",
            "n": int(len(y)),
            "prevalence": prevalence,
            "mean_predicted_probability": float(np.mean(p)),
            "brier_score": brier,
            "baseline_brier_score": baseline_brier,
            "brier_skill_score": float(1.0 - brier / baseline_brier) if baseline_brier > 0 else float("nan"),
            "log_loss": binary_log_loss(y, p),
            "roc_auc": binary_auc(y, p),
            "average_precision": average_precision_binary(y, p),
        }
    ])


def binary_calibration_deciles(target: str, actual: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], EPS, 1.0 - EPS)
    if len(y) == 0:
        return pd.DataFrame()
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, min(n_bins, len(order)))
    rows: list[dict[str, float | int | str]] = []
    for bin_idx, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        observed_rate = float(np.mean(y[idx]))
        mean_pred = float(np.mean(p[idx]))
        rows.append(
            {
                "target": target,
                "decile": int(bin_idx),
                "n": int(len(idx)),
                "predicted_min": float(np.min(p[idx])),
                "predicted_max": float(np.max(p[idx])),
                "mean_predicted_probability": mean_pred,
                "observed_event_rate": observed_rate,
                "events": int(np.sum(y[idx])),
                "calibration_error": float(observed_rate - mean_pred),
            }
        )
    return pd.DataFrame.from_records(rows)


def regression_diagnostics_row(target: str, description: str, model_stage: str, actual: np.ndarray, pred: np.ndarray) -> dict[str, float | int | str]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    resid = y - p
    centered = y - float(np.mean(y))
    sse = float(np.sum(np.square(resid)))
    tss = float(np.sum(np.square(centered)))
    return {
        "target": target,
        "description": description,
        "model_stage": model_stage,
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(np.square(resid)))) if len(y) else float("nan"),
        "mae": float(np.mean(np.abs(resid))) if len(y) else float("nan"),
        "mean_actual": float(np.mean(y)) if len(y) else float("nan"),
        "sd_actual": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "mean_predicted": float(np.mean(p)) if len(y) else float("nan"),
        "baseline_mean_rmse": float(np.sqrt(np.mean(np.square(centered)))) if len(y) else float("nan"),
        "r2_oos_vs_mean": float(1.0 - sse / tss) if tss > 0 else float("nan"),
    }


def crossfit_binary_prediction_diagnostics(
    dtr: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "Y2_any_placement_improvement" not in dtr.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    n = len(dtr)
    folds = choose_folds(n, args.n_folds)
    kfold = (
        KFold(n_splits=folds, shuffle=True, random_state=args.random_state + 313)
        if HAVE_SKLEARN
        else SimpleKFold(n_splits=folds, shuffle=True, random_state=args.random_state + 313)
    )
    q2_cols = q2_columns()
    actual_binary = pd.to_numeric(dtr["Y2_any_placement_improvement"], errors="coerce").fillna(0).gt(0).astype(int).to_numpy()
    binary_prob = np.full(n, np.nan)
    two_part_targets = {
        "y2_placement_improvement_speed": (
            "Y2_placement_improvement_speed",
            "Two-part prediction for placement improvement speed: P(any improvement) times positive-outcome mean",
        ),
        "y2_placement_improvement_from_baseline": (
            "Y2_placement_improvement_from_baseline",
            "Two-part prediction for placement improvement magnitude: P(any improvement) times positive-outcome mean",
        ),
    }
    two_part_pred = {name: np.full(n, np.nan) for name in two_part_targets}

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dtr)):
        train = dtr.iloc[train_idx].copy()
        test = dtr.iloc[test_idx].copy()
        seed = args.random_state + 4001 * fold_idx
        binary_model, binary_encoded = fit_binary_classifier(
            train, q2_cols, train["Y2_any_placement_improvement"], args, seed
        )
        p_test = predict_positive_probability(binary_model, binary_encoded, test, q2_cols)
        binary_prob[test_idx] = p_test
        train_positive = train[pd.to_numeric(train["Y2_any_placement_improvement"], errors="coerce").fillna(0).gt(0)].copy()
        for name, (col, _description) in two_part_targets.items():
            if len(train_positive) >= 2 and pd.to_numeric(train_positive[col], errors="coerce").notna().any():
                amount_model, amount_encoded = fit_regressor(
                    train_positive, q2_cols, pd.to_numeric(train_positive[col], errors="coerce"), args, seed + len(name)
                )
                amount_pred = predict_regressor(amount_model, amount_encoded, test, q2_cols)
            else:
                fallback = float(pd.to_numeric(train[col], errors="coerce").mean())
                amount_model = ConstantRegressor(0.0 if pd.isna(fallback) else fallback)
                amount_pred = amount_model.predict(test)
            two_part_pred[name][test_idx] = p_test * amount_pred

    binary_description = "Classifier diagnostics for any year-2 placement improvement from baseline"
    binary_diag = binary_prediction_diagnostics_from_scores(
        "y2_any_placement_improvement", binary_description, actual_binary, binary_prob
    )
    calibration = binary_calibration_deciles("y2_any_placement_improvement", actual_binary, binary_prob)
    rows: list[dict[str, float | int | str]] = []
    for name, (col, description) in two_part_targets.items():
        rows.append(
            regression_diagnostics_row(
                name,
                description,
                "q2_observed_action_two_part",
                pd.to_numeric(dtr[col], errors="coerce").to_numpy(dtype=float),
                two_part_pred[name],
            )
        )
    return binary_diag, calibration, pd.DataFrame.from_records(rows)


def prediction_diagnostics_from_scores(
    prediction_scores: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    targets_by_name = {target.name: target for target in TARGETS}
    for target_name, arrays in prediction_scores.items():
        target = targets_by_name[target_name]
        actual = np.asarray(arrays["q2_actual"], dtype=float)
        pred = np.asarray(arrays["q2_pred"], dtype=float)
        mask = np.isfinite(actual) & np.isfinite(pred)
        actual = actual[mask]
        pred = pred[mask]
        if len(actual) == 0:
            continue
        resid = actual - pred
        centered = actual - float(np.mean(actual))
        sse = float(np.sum(np.square(resid)))
        tss = float(np.sum(np.square(centered)))
        rows.append(
            {
                "target": target.name,
                "description": target.description,
                "stage2_col": target.stage2_col,
                "terminal_progress_mode": target.terminal_progress_mode,
                "annotation_target": target.annotation_target,
                "model_stage": "q2_observed_action",
                "n": int(len(actual)),
                "rmse": float(np.sqrt(np.mean(np.square(resid)))),
                "mae": float(np.mean(np.abs(resid))),
                "mean_actual": float(np.mean(actual)),
                "sd_actual": float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0,
                "mean_predicted": float(np.mean(pred)),
                "baseline_mean_rmse": float(np.sqrt(np.mean(np.square(centered)))),
                "r2_oos_vs_mean": float(1.0 - sse / tss) if tss > 0 else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(np.square(weights)))
    return float(np.square(np.sum(weights)) / denom) if denom > 0 else float("nan")


def overlap_diagnostics(dtr: pd.DataFrame, nuisances: dict[str, np.ndarray], args: argparse.Namespace) -> pd.DataFrame:
    g1 = nuisances["g1"]
    g2 = nuisances["g2"]
    q1 = nuisances["q1"]
    q2 = nuisances["q2"]
    g1_obs = observed_probability(g1, dtr["A1"])
    g2_obs = observed_probability(g2, dtr["A2"])
    q1_obs = observed_probability(q1, dtr["A1"])
    q2_obs = observed_probability(q2, dtr["A2"])
    rho1 = safe_ratio(q1_obs, g1_obs, args.propensity_floor)
    rho12 = rho1 * safe_ratio(q2_obs, g2_obs, args.propensity_floor)
    rows = [
        {
            "n": len(dtr),
            "propensity_floor": args.propensity_floor,
            "shift_quintiles": int(args.shift_quintiles),
            "stage1_ratio_ess": ess(rho1),
            "stage12_ratio_ess": ess(rho12),
            **summarize_array(g1_obs, "g1_obs"),
            **summarize_array(g2_obs, "g2_obs"),
            **summarize_array(q1_obs, "q1_obs"),
            **summarize_array(q2_obs, "q2_obs"),
            **summarize_array(rho1, "rho1"),
            **summarize_array(rho12, "rho12"),
        }
    ]
    return pd.DataFrame.from_records(rows)


def crossfit_reward_residual_variances(dtr: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    n = len(dtr)
    folds = choose_folds(n, args.n_folds)
    kfold = (KFold(n_splits=folds, shuffle=True, random_state=args.random_state + 77) if HAVE_SKLEARN else SimpleKFold(n_splits=folds, shuffle=True, random_state=args.random_state + 77))
    specs = {
        "R1_progress": (q1_columns(), "R1_progress", "A1"),
        "R2_progress": (q2_columns(), "R2_progress", "A2"),
        "Y2_placement": (q2_columns(), "Y2_placement", "A2"),
        "R2_progress_plus_Y2_placement": (q2_columns(), "R2_progress_plus_Y2_placement", "A2"),
    }
    residual_sq = {name: np.zeros(n, dtype=float) for name in specs}
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dtr)):
        train = dtr.iloc[train_idx].copy()
        test = dtr.iloc[test_idx].copy()
        seed = args.random_state + 2003 * fold_idx
        for name, (cols, outcome, _action_col) in specs.items():
            y_train = pd.to_numeric(train[outcome], errors="coerce")
            model, encoded_cols = fit_regressor(train, cols, y_train, args, seed + len(name))
            pred = predict_regressor(model, encoded_cols, test, cols)
            y_test = pd.to_numeric(test[outcome], errors="coerce").to_numpy(dtype=float)
            residual_sq[name][test_idx] = np.square(y_test - pred)
    for name in residual_sq:
        floor = float(np.quantile(residual_sq[name], 0.05)) if len(residual_sq[name]) else 0.0
        residual_sq[name] = np.maximum(residual_sq[name], max(floor, 1e-8))
    return residual_sq


def allocation_from_priority(raw1: np.ndarray, raw2: np.ndarray, budget: float, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not 0.0 < budget <= 2.0:
        raise ValueError("--annotation-budget must be in (0, 2], where 2 is full two-stage annotation cost.")
    if floor < 0.0 or floor >= 1.0:
        raise ValueError("--annotation-floor must be in [0, 1).")
    if budget < 2.0 * floor:
        raise ValueError("annotation budget is too small for the requested floor and two-stage cost.")
    raw1 = np.maximum(np.asarray(raw1, dtype=float), EPS)
    raw2 = np.maximum(np.asarray(raw2, dtype=float), EPS)

    def probs(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam1 = np.clip(scale * raw1, floor, 1.0)
        q12_pre = np.clip(scale * raw2, floor, 1.0)
        q12 = np.minimum(lam1, q12_pre)
        lam2 = q12 / np.maximum(lam1, EPS)
        return lam1, q12, lam2

    lo = 0.0
    hi = 1.0
    while True:
        lam1, q12, _ = probs(hi)
        if float(np.mean(lam1 + q12)) >= budget or hi > 1e8:
            break
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lam1, q12, _ = probs(mid)
        if float(np.mean(lam1 + q12)) < budget:
            lo = mid
        else:
            hi = mid
    lam1, q12, lam2 = probs(hi)
    return lam1, q12, lam2, hi


def annotation_priority_summary(dtr: pd.DataFrame, nuisances: dict[str, np.ndarray], args: argparse.Namespace) -> pd.DataFrame:
    residual_var = crossfit_reward_residual_variances(dtr, args)
    g1_obs = observed_probability(nuisances["g1"], dtr["A1"])
    g2_obs = observed_probability(nuisances["g2"], dtr["A2"])
    q1_obs = observed_probability(nuisances["q1"], dtr["A1"])
    q2_obs = observed_probability(nuisances["q2"], dtr["A2"])
    rho1 = safe_ratio(q1_obs, g1_obs, args.propensity_floor)
    rho12 = rho1 * safe_ratio(q2_obs, g2_obs, args.propensity_floor)

    rows: list[dict[str, float | int | str]] = []
    for target in TARGETS:
        if not target.annotation_target:
            continue
        sigma1 = np.sqrt(residual_var[target.stage1_col])
        sigma2 = np.sqrt(residual_var[target.stage2_col])
        raw1 = np.maximum(rho1 * sigma1, EPS)
        raw2 = np.maximum(rho12 * sigma2, EPS)
        lam1, q12, lam2, scale = allocation_from_priority(
            raw1,
            raw2,
            budget=args.annotation_budget,
            floor=args.annotation_floor,
        )
        rows.append(
            {
                "target": target.name,
                "description": target.description,
                "annotation_budget": args.annotation_budget,
                "annotation_floor": args.annotation_floor,
                "scale": scale,
                "expected_stage1_annotation_rate": float(np.mean(lam1)),
                "expected_both_stage_annotation_rate_q12": float(np.mean(q12)),
                "expected_stage2_given_stage1_rate": float(np.mean(lam2)),
                "expected_reward_labels_per_unit": float(np.mean(lam1 + q12)),
                "stage1_annotation_ess": ess(lam1),
                "both_stage_annotation_ess": ess(q12),
                **summarize_array(raw1, "raw_stage1_priority"),
                **summarize_array(raw2, "raw_stage2_priority"),
                **summarize_array(lam1, "lambda1"),
                **summarize_array(q12, "q12"),
                **summarize_array(lam2, "lambda2"),
            }
        )
    return pd.DataFrame.from_records(rows)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(shown.columns)
    rows = shown.astype(str).values.tolist()
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep, *[fmt(row) for row in rows]])


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    value_estimates: pd.DataFrame,
    overlap: pd.DataFrame,
    annotation: pd.DataFrame,
    population: pd.DataFrame,
    prediction_diagnostics: pd.DataFrame,
    binary_prediction_diagnostics: pd.DataFrame,
    binary_calibration_deciles: pd.DataFrame,
    two_part_prediction_diagnostics: pd.DataFrame,
    elapsed: float,
) -> None:
    lines = [
        "# Outreach DTR Reward-Only Analysis",
        "",
        "Synthetic smoke test" if args.synthetic_smoke_test else "Protected-data run; aggregate outputs only",
        "",
        "## Estimand",
        "- Analytic population: clients with positive outreach bins in both stages.",
        f"- Evaluation policy: shift positive outreach up `{args.shift_quintiles}` quintile(s), with quintile 5 fixed.",
        "- Values include progress totals, max-placement outcomes, time-weighted placement, placement-improvement speed, improvement from baseline, and any placement improvement; gamma=1.",
        "- Pure placement targets use zero stage-1 reward; R1-plus-placement targets include stage-1 progress as an explicit reward component.",
        "- H2 includes stage-1 progress summaries but no R2 progress predictor for terminal placement outcomes.",
        "- Annotation priority: reward-only STAIX-style forward-monotone allocation for reward-label targets only.",
        "",
        "## Files",
        "- `value_estimates.csv`: observed, plugin, and DRL values with SEs and 95% Wald intervals.",
        "- `overlap_diagnostics.csv`: behavior/evaluation policy ratio summaries.",
        "- `annotation_priority_summary.csv`: aggregate reward-only annotation probabilities.",
        "- `prediction_diagnostics.csv`: aggregate cross-fitted observed-action nuisance prediction diagnostics.",
        "- `binary_prediction_diagnostics.csv`: classifier metrics for the binary any-placement-improvement outcome.",
        "- `binary_calibration_deciles.csv`: observed event rates by predicted-risk decile for the binary outcome.",
        "- `two_part_prediction_diagnostics.csv`: hurdle/two-part diagnostics for continuous placement-improvement outcomes.",
        "- `analysis_population.csv`: aggregate inclusion/exclusion counts.",
        "- `interval_coverage.csv`: aggregate contact/progress label coverage by interval.",
        "",
        f"Elapsed wall-clock seconds: `{elapsed:.3f}`.",
        "",
        "## Population",
        "",
        frame_to_markdown(population),
        "",
        "## Value estimates",
        "",
        frame_to_markdown(value_estimates),
        "",
        "## Overlap diagnostics",
        "",
        frame_to_markdown(overlap),
        "",
        "## Prediction diagnostics",
        "",
        frame_to_markdown(prediction_diagnostics),
        "",
        "## Binary prediction diagnostics",
        "",
        frame_to_markdown(binary_prediction_diagnostics),
        "",
        "## Binary calibration deciles",
        "",
        frame_to_markdown(binary_calibration_deciles),
        "",
        "## Two-part prediction diagnostics",
        "",
        frame_to_markdown(two_part_prediction_diagnostics),
        "",
        "## Annotation priority",
        "",
        frame_to_markdown(annotation[[
            "target",
            "expected_stage1_annotation_rate",
            "expected_both_stage_annotation_rate_q12",
            "expected_stage2_given_stage1_rate",
            "expected_reward_labels_per_unit",
        ]]),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_decision_intervals != 2:
        raise ValueError("This DTR implementation currently supports exactly two decision intervals.")
    if args.synthetic_smoke_test:
        features, contacts, labels = synthetic_inputs(args.random_state)
    else:
        features, contacts, labels = load_inputs(args)

    dtr, coverage, population = build_dtr_frame(features, contacts, labels, args)
    population.to_csv(output_dir / "analysis_population.csv", index=False)
    coverage.to_csv(output_dir / "interval_coverage.csv", index=False)
    if dtr.empty:
        elapsed = time.perf_counter() - start
        pd.DataFrame.from_records(
            [
                {
                    "elapsed_wall_seconds": elapsed,
                    "timing_source": "script_perf_counter",
                    "synthetic_smoke_test": bool(args.synthetic_smoke_test),
                    "n_analysis_rows": 0,
                    "n_folds": 0,
                    "n_estimators": int(args.n_estimators),
                    "shift_quintiles": int(args.shift_quintiles),
                    "status": "failed_empty_analysis_population",
                }
            ]
        ).to_csv(output_dir / "run_metadata.csv", index=False)
        raise ValueError(
            "No rows remain after positive-outreach and complete-reward filtering. "
            f"Wrote aggregate diagnostics to {output_dir / 'analysis_population.csv'} "
            f"and {output_dir / 'interval_coverage.csv'}."
        )

    value_estimates, overlap, nuisances, prediction_diagnostics = crossfit_behavior_and_values(dtr, args)
    annotation = annotation_priority_summary(dtr, nuisances, args)
    binary_prediction_diagnostics, binary_calibration_deciles, two_part_prediction_diagnostics = (
        crossfit_binary_prediction_diagnostics(dtr, args)
    )

    value_estimates.to_csv(output_dir / "value_estimates.csv", index=False)
    overlap.to_csv(output_dir / "overlap_diagnostics.csv", index=False)
    prediction_diagnostics.to_csv(output_dir / "prediction_diagnostics.csv", index=False)
    binary_prediction_diagnostics.to_csv(output_dir / "binary_prediction_diagnostics.csv", index=False)
    binary_calibration_deciles.to_csv(output_dir / "binary_calibration_deciles.csv", index=False)
    two_part_prediction_diagnostics.to_csv(output_dir / "two_part_prediction_diagnostics.csv", index=False)
    annotation.to_csv(output_dir / "annotation_priority_summary.csv", index=False)
    elapsed = time.perf_counter() - start
    pd.DataFrame.from_records(
        [
            {
                "elapsed_wall_seconds": elapsed,
                "timing_source": "script_perf_counter",
                "synthetic_smoke_test": bool(args.synthetic_smoke_test),
                "n_analysis_rows": int(len(dtr)),
                "n_folds": int(choose_folds(len(dtr), args.n_folds)),
                "n_estimators": int(args.n_estimators),
                "shift_quintiles": int(args.shift_quintiles),
                "status": "completed",
            }
        ]
    ).to_csv(output_dir / "run_metadata.csv", index=False)
    write_summary(
        output_dir,
        args,
        value_estimates,
        overlap,
        annotation,
        population,
        prediction_diagnostics,
        binary_prediction_diagnostics,
        binary_calibration_deciles,
        two_part_prediction_diagnostics,
        elapsed,
    )
    print(f"Wrote aggregate DTR outputs to {output_dir}")
    print(f"Analysis rows: {len(dtr)}")
    print(f"Elapsed wall-clock seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
