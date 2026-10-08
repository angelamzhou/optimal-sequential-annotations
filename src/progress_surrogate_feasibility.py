#!/usr/bin/env python3
"""Feasibility analysis for progress labels as outcome-model surrogates.

This script intentionally avoids loading case-note text and raw model output text.
It reads only analytical columns needed to construct client-level interval features
and aggregate model metrics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    from sklearn.model_selection import GridSearchCV, train_test_split
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    RandomForestClassifier = None
    RandomForestRegressor = None
    accuracy_score = None
    f1_score = None
    mean_squared_error = None
    GridSearchCV = None
    train_test_split = None
    HAVE_SKLEARN = False

CLIENT_ID = "ClientID"
CONTACT_ID = "ContactID"
CONTACT_DATE = "ContactDate"
PRIMARY_LABEL = "primary_label"
PLACEMENT_TYPE = "PlacementType"
MISSING_PROGRESS_VALUE = -999.0
INVALID_LABEL_SENTINEL = -1.0
VALID_PROGRESS_MIN = 0.0
VALID_PROGRESS_MAX = 4.0

DEFAULT_FEATURES_CSV = "protected_data/cohort_features_outcome_cat.csv"
DEFAULT_CONTACTS_CSV = "protected_data/contact_timestamps.csv"
DEFAULT_LABELS_CSV = (
    "protected_data/labels.csv"
)
DEFAULT_OUTCOME = "max_PlacementType_2019_2021"

STATE_COLUMNS = [
    "DateFirstSeen",
    "Housing_application_Yes",
    "Service_refusal_Yes",
    "Impt_documents_Yes",
    "Benefits_Yes",
    "num_contacts_before_2019",
    "max_PlacementType_before_2019",
    "Program",
    "BelievedChronic",
    "Gender_cat",
    "Race_cat",
    "Age_cat",
    "Zone_cat",
    "Ethnicity",
    "Was311Call",
    "Was911Call",
    "Removal958",
]


@dataclass(frozen=True)
class IntervalSpec:
    start_date: pd.Timestamp
    interval_months: int
    num_intervals: int

    @property
    def end_date(self) -> pd.Timestamp:
        return self.start_date + pd.DateOffset(months=self.interval_months * self.num_intervals)

    def bounds(self) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
        out = []
        for idx in range(1, self.num_intervals + 1):
            start = self.start_date + pd.DateOffset(months=self.interval_months * (idx - 1))
            end = self.start_date + pd.DateOffset(months=self.interval_months * idx)
            out.append((idx, start, end))
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit RF outcome-model comparisons for long-term placement using "
            "baseline state, outreach/action bins, and interval progress summaries."
        )
    )
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--contacts-csv", default=DEFAULT_CONTACTS_CSV)
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    parser.add_argument("--outcome-column", default=DEFAULT_OUTCOME)
    parser.add_argument(
        "--derive-outcome-from-contacts",
        action="store_true",
        help=(
            "Derive Y as max PlacementType from contacts strictly after the progress/action "
            "observation window. This avoids temporal overlap between R and Y."
        ),
    )
    parser.add_argument(
        "--derived-outcome-column",
        default="max_PlacementType_after_observation",
        help="Column name to use when deriving Y from post-observation contacts.",
    )
    parser.add_argument(
        "--outcome-start-date",
        default=None,
        help="Optional outcome start date. Defaults to the end of the A/R observation window.",
    )
    parser.add_argument(
        "--outcome-end-date",
        default=None,
        help="Optional exclusive outcome end date for derived Y. Defaults to no upper bound.",
    )
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument(
        "--date-format",
        default=None,
        help=(
            "Optional strptime format for ContactDate, e.g. %Y-%m-%d or %m/%d/%y. "
            "Defaults to pandas mixed-format parsing."
        ),
    )
    parser.add_argument("--interval-months", type=int, default=6)
    parser.add_argument("--num-intervals", type=int, default=4)
    parser.add_argument("--output-dir", default="results/progress_surrogate_feasibility")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--model-kind",
        choices=["classifier", "regressor", "both"],
        default="both",
        help="RF estimator family to run. Default runs both classifier and regressor.",
    )
    parser.add_argument(
        "--hyperparam-mode",
        choices=["default", "grid"],
        default="grid",
        help="RF hyperparameter strategy. Grid search is the default and optimizes CV RMSE on the training split.",
    )
    parser.add_argument("--grid-cv", type=int, default=3)
    parser.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help="Run on generated non-sensitive synthetic data instead of CSV inputs.",
    )
    return parser.parse_args()


def require_columns(actual_columns: Iterable[str], required_columns: Iterable[str], label: str) -> None:
    actual = set(actual_columns)
    missing = sorted(set(required_columns) - actual)
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_selected_csv(path: Path, required_columns: list[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    wanted = set(required_columns)
    id_dtype = {col: "string" for col in wanted if col in {CLIENT_ID, CONTACT_ID}}
    df = pd.read_csv(path, usecols=lambda col: col in wanted, dtype=id_dtype)
    require_columns(df.columns, required_columns, label)
    return df


def normalize_id_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            values = out[col].astype("string").str.strip()
            out[col] = values.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)
    return out


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [CLIENT_ID, *STATE_COLUMNS]
    if not args.derive_outcome_from_contacts:
        feature_columns.append(args.outcome_column)
    features = read_selected_csv(Path(args.features_csv), feature_columns, "features CSV")
    contact_columns = [CLIENT_ID, CONTACT_ID, CONTACT_DATE]
    if args.derive_outcome_from_contacts:
        contact_columns.append(PLACEMENT_TYPE)
    contacts = read_selected_csv(Path(args.contacts_csv), contact_columns, "contacts CSV")
    labels = read_selected_csv(Path(args.labels_csv), [CONTACT_ID, PRIMARY_LABEL], "labels CSV")
    return features, contacts, labels


def synthetic_inputs(outcome_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_clients = 100
    client_ids = [f"C{i:03d}" for i in range(n_clients)]
    features = pd.DataFrame(
        {
            CLIENT_ID: client_ids,
            "DateFirstSeen": pd.Timestamp("2018-01-01").toordinal()
            + rng.integers(0, 365, size=n_clients),
            "Housing_application_Yes": rng.integers(0, 2, size=n_clients),
            "Service_refusal_Yes": rng.integers(0, 2, size=n_clients),
            "Impt_documents_Yes": rng.integers(0, 2, size=n_clients),
            "Benefits_Yes": rng.integers(0, 2, size=n_clients),
            "num_contacts_before_2019": rng.integers(0, 20, size=n_clients),
            "max_PlacementType_before_2019": rng.integers(0, 4, size=n_clients),
            "Program": rng.choice(["A", "B"], size=n_clients),
            "BelievedChronic": rng.choice(["Yes", "No"], size=n_clients),
            "Gender_cat": rng.choice(["Female", "Male"], size=n_clients),
            "Race_cat": rng.choice(["Black/African American", "White/Caucasian", "Other"], size=n_clients),
            "Age_cat": rng.choice(["18-34", "35-54", "55+"], size=n_clients),
            "Zone_cat": rng.choice(["Z1", "Z2", "Z3"], size=n_clients),
            "Ethnicity": rng.choice(["Hispanic", "Non-Hispanic", "Unknown"], size=n_clients),
            "Was311Call": rng.choice(["Yes", "No"], size=n_clients),
            "Was911Call": rng.choice(["Yes", "No"], size=n_clients),
            "Removal958": rng.choice(["Yes", "No"], size=n_clients),
        }
    )

    contacts_rows = []
    labels_rows = []
    contact_idx = 0
    for client_idx, client_id in enumerate(client_ids):
        n_contacts = int(rng.integers(0, 13))
        max_progress = -1.0
        mean_progress_terms = []
        for _ in range(n_contacts):
            contact_idx += 1
            contact_id = f"K{contact_idx:04d}"
            day_offset = int(rng.integers(0, 730))
            progress = float(rng.choice([-1, 0, 0.5, 1, 2, 2.5, 3, 4]))
            max_progress = max(max_progress, progress)
            mean_progress_terms.append(progress)
            contacts_rows.append(
                {
                    CLIENT_ID: client_id,
                    CONTACT_ID: contact_id,
                    CONTACT_DATE: pd.Timestamp("2019-01-01") + pd.Timedelta(days=day_offset),
                    PLACEMENT_TYPE: min(3, max(0, int(round(progress)))),
                }
            )
            labels_rows.append({CONTACT_ID: contact_id, PRIMARY_LABEL: progress})
        baseline = int(features.loc[client_idx, "max_PlacementType_before_2019"])
        mean_progress = float(np.mean(mean_progress_terms)) if mean_progress_terms else -1.0
        score = baseline + (max_progress >= 2.5) + (max_progress >= 4) + (mean_progress >= 2.0)
        features.loc[client_idx, outcome_column] = min(3, max(0, int(score)))

    contacts = pd.DataFrame(contacts_rows, columns=[CLIENT_ID, CONTACT_ID, CONTACT_DATE, PLACEMENT_TYPE])
    labels = pd.DataFrame(labels_rows, columns=[CONTACT_ID, PRIMARY_LABEL])
    features[outcome_column] = features[outcome_column].astype(int)
    return features, contacts, labels


def assign_interval(contact_dates: pd.Series, spec: IntervalSpec) -> pd.Series:
    interval = pd.Series(pd.NA, index=contact_dates.index, dtype="Int64")
    for idx, start, end in spec.bounds():
        mask = contact_dates.ge(start) & contact_dates.lt(end)
        interval.loc[mask] = idx
    return interval


def bin_nonzero_quintiles(counts: pd.Series) -> pd.Series:
    counts = counts.fillna(0).astype(float)
    bins = pd.Series(0, index=counts.index, dtype="int64")
    nonzero = counts[counts > 0]
    if nonzero.empty:
        return bins
    n_unique = int(nonzero.nunique())
    if n_unique == 1:
        bins.loc[nonzero.index] = 1
        return bins
    q = min(5, n_unique)
    cut = pd.qcut(nonzero, q=q, labels=False, duplicates="drop")
    bins.loc[nonzero.index] = cut.astype(int) + 1
    return bins


def parse_contact_dates(contact_dates: pd.Series, date_format: str | None) -> pd.Series:
    if date_format:
        return pd.to_datetime(contact_dates, errors="coerce", format=date_format)
    return pd.to_datetime(contact_dates, errors="coerce", format="mixed")


def derive_outcome_from_contacts(
    features: pd.DataFrame,
    contacts: pd.DataFrame,
    spec: IntervalSpec,
    date_format: str | None,
    outcome_column: str,
    outcome_start_date: str | None,
    outcome_end_date: str | None,
) -> tuple[pd.DataFrame, dict]:
    require_columns(contacts.columns, [CLIENT_ID, CONTACT_DATE, PLACEMENT_TYPE], "contacts for derived outcome")
    contacts_for_y = normalize_id_columns(contacts, [CLIENT_ID]).copy()
    contacts_for_y[CONTACT_DATE] = parse_contact_dates(contacts_for_y[CONTACT_DATE], date_format)
    contacts_for_y[PLACEMENT_TYPE] = pd.to_numeric(contacts_for_y[PLACEMENT_TYPE], errors="coerce")
    contacts_for_y = contacts_for_y.dropna(subset=[CLIENT_ID, CONTACT_DATE, PLACEMENT_TYPE])

    start = pd.Timestamp(outcome_start_date) if outcome_start_date else spec.end_date
    end = pd.Timestamp(outcome_end_date) if outcome_end_date else None
    mask = contacts_for_y[CONTACT_DATE].ge(start)
    if end is not None:
        mask &= contacts_for_y[CONTACT_DATE].lt(end)
    contacts_for_y = contacts_for_y.loc[mask].copy()

    outcome = contacts_for_y.groupby(CLIENT_ID)[PLACEMENT_TYPE].max().rename(outcome_column).reset_index()
    out = normalize_id_columns(features, [CLIENT_ID]).drop(columns=[outcome_column], errors="ignore")
    out = out.merge(outcome, on=CLIENT_ID, how="inner")
    metadata = {
        "derived_outcome": True,
        "derived_outcome_column": outcome_column,
        "outcome_start_date": str(start.date()),
        "outcome_end_date": str(end.date()) if end is not None else "none",
        "clients_with_derived_outcome": int(out[outcome_column].notna().sum()),
        "post_observation_contacts_for_outcome": int(len(contacts_for_y)),
    }
    return out, metadata


def build_interval_features(
    features: pd.DataFrame,
    contacts: pd.DataFrame,
    labels: pd.DataFrame,
    spec: IntervalSpec,
    date_format: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = normalize_id_columns(features, [CLIENT_ID])
    contacts = normalize_id_columns(contacts, [CLIENT_ID, CONTACT_ID])
    labels = normalize_id_columns(labels, [CONTACT_ID])

    contacts = contacts.copy()
    contacts[CONTACT_DATE] = parse_contact_dates(contacts[CONTACT_DATE], date_format)
    contacts = contacts.dropna(subset=[CONTACT_DATE, CLIENT_ID, CONTACT_ID])
    contacts["interval"] = assign_interval(contacts[CONTACT_DATE], spec)
    contacts = contacts.dropna(subset=["interval"])
    contacts["interval"] = contacts["interval"].astype(int)

    labels = labels.copy()
    progress_numeric = pd.to_numeric(labels[PRIMARY_LABEL], errors="coerce")
    # -1 is the annotation pipeline's invalid-label sentinel; treated as no progress.
    progress_numeric = progress_numeric.mask(progress_numeric.eq(INVALID_LABEL_SENTINEL), 0.0)
    valid = progress_numeric.between(VALID_PROGRESS_MIN, VALID_PROGRESS_MAX)
    labels["progress_numeric"] = progress_numeric.where(valid)
    labels = labels.dropna(subset=[CONTACT_ID])
    labels_by_contact = labels.groupby(CONTACT_ID, as_index=False)["progress_numeric"].max()

    contacts_with_labels = contacts.merge(labels_by_contact, on=CONTACT_ID, how="left")
    clients = pd.Index(features[CLIENT_ID].dropna().astype("string").str.strip().unique(), name=CLIENT_ID)

    interval_features = pd.DataFrame(index=clients)
    max_progress_by_interval: dict[int, pd.Series] = {}
    latest_progress_by_interval: dict[int, pd.Series] = {}
    coverage_rows = []
    for idx, _start, _end in spec.bounds():
        interval_contacts = contacts[contacts["interval"] == idx]
        counts = interval_contacts.groupby(CLIENT_ID).size().reindex(clients, fill_value=0)
        interval_features[f"contacts_t{idx}"] = counts.astype(int).values
        interval_features[f"outreach_bin_t{idx}"] = bin_nonzero_quintiles(counts).values

        labeled = contacts_with_labels[contacts_with_labels["interval"] == idx]
        labeled_progress = labeled.dropna(subset=["progress_numeric"])
        observed_counts = labeled_progress.groupby(CLIENT_ID).size().reindex(clients, fill_value=0)
        progress_max = labeled_progress.groupby(CLIENT_ID)["progress_numeric"].max().reindex(clients)
        progress_min = labeled_progress.groupby(CLIENT_ID)["progress_numeric"].min().reindex(clients)
        progress_mean = labeled_progress.groupby(CLIENT_ID)["progress_numeric"].mean().reindex(clients)
        progress_latest = (
            labeled_progress.sort_values([CLIENT_ID, CONTACT_DATE])
            .groupby(CLIENT_ID)["progress_numeric"]
            .last()
            .reindex(clients)
        )
        progress_range = progress_max - progress_min
        any_ge_2_5 = labeled_progress.assign(flag=labeled_progress["progress_numeric"].ge(2.5).astype(int))
        any_ge_3 = labeled_progress.assign(flag=labeled_progress["progress_numeric"].ge(3).astype(int))
        any_eq_4 = labeled_progress.assign(flag=labeled_progress["progress_numeric"].eq(4).astype(int))

        max_progress_by_interval[idx] = progress_max
        latest_progress_by_interval[idx] = progress_latest
        interval_features[f"max_progress_t{idx}"] = progress_max.fillna(MISSING_PROGRESS_VALUE).astype(float).values
        interval_features[f"mean_progress_t{idx}"] = progress_mean.fillna(MISSING_PROGRESS_VALUE).astype(float).values
        interval_features[f"latest_progress_t{idx}"] = progress_latest.fillna(MISSING_PROGRESS_VALUE).astype(float).values
        interval_features[f"progress_range_t{idx}"] = progress_range.fillna(0).astype(float).values
        interval_features[f"progress_label_count_t{idx}"] = observed_counts.astype(int).values
        interval_features[f"progress_observed_t{idx}"] = (observed_counts > 0).astype(int).values
        interval_features[f"any_progress_ge_2_5_t{idx}"] = (
            any_ge_2_5.groupby(CLIENT_ID)["flag"].max().reindex(clients, fill_value=0).astype(int).values
        )
        interval_features[f"any_progress_ge_3_t{idx}"] = (
            any_ge_3.groupby(CLIENT_ID)["flag"].max().reindex(clients, fill_value=0).astype(int).values
        )
        interval_features[f"any_progress_eq_4_t{idx}"] = (
            any_eq_4.groupby(CLIENT_ID)["flag"].max().reindex(clients, fill_value=0).astype(int).values
        )

        total_contacts = int(counts.sum())
        labeled_contacts = int(observed_counts.sum())
        coverage_rows.append(
            {
                "interval": idx,
                "contacts": total_contacts,
                "labeled_contacts": labeled_contacts,
                "clients_with_contacts": int((counts > 0).sum()),
                "clients_with_progress": int((observed_counts > 0).sum()),
                "label_coverage_rate": labeled_contacts / total_contacts if total_contacts else 0.0,
            }
        )

    max_progress_frame = pd.DataFrame(max_progress_by_interval).reindex(clients)
    latest_progress_frame = pd.DataFrame(latest_progress_by_interval).reindex(clients)
    cumulative_max = max_progress_frame.cummax(axis=1)
    latest_carry_forward = latest_progress_frame.ffill(axis=1)
    for idx in range(1, spec.num_intervals + 1):
        interval_features[f"cumulative_max_progress_t{idx}"] = (
            cumulative_max[idx].fillna(MISSING_PROGRESS_VALUE).astype(float).values
        )
        interval_features[f"latest_progress_carry_forward_t{idx}"] = (
            latest_carry_forward[idx].fillna(MISSING_PROGRESS_VALUE).astype(float).values
        )
        interval_features[f"ever_progress_ge_2_5_t{idx}"] = (
            cumulative_max[idx].ge(2.5).fillna(False).astype(int).values
        )
        interval_features[f"ever_progress_ge_3_t{idx}"] = (
            cumulative_max[idx].ge(3).fillna(False).astype(int).values
        )
        interval_features[f"ever_progress_eq_4_t{idx}"] = (
            cumulative_max[idx].eq(4).fillna(False).astype(int).values
        )

    for idx in range(2, spec.num_intervals + 1):
        prev_max = max_progress_by_interval[idx - 1]
        curr_max = max_progress_by_interval[idx]
        change = curr_max - prev_max
        observed_change = curr_max.notna() & prev_max.notna()
        interval_features[f"max_progress_change_t{idx}_vs_t{idx - 1}"] = change.fillna(0).astype(float).values
        interval_features[f"max_progress_change_observed_t{idx}_vs_t{idx - 1}"] = observed_change.astype(int).values

    interval_features = interval_features.reset_index()
    analysis = features.merge(interval_features, on=CLIENT_ID, how="left")
    for col in interval_features.columns:
        if col != CLIENT_ID:
            analysis[col] = analysis[col].fillna(0)
    return analysis, pd.DataFrame(coverage_rows)


def encode_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = frame[feature_columns].copy()
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
            fill_value = x[col].median()
            if pd.isna(fill_value):
                fill_value = 0
            x[col] = x[col].fillna(fill_value)
        else:
            x[col] = x[col].astype("string").fillna("Unknown")
    return pd.get_dummies(x, dummy_na=False)


def rounded_class_predictions(predictions: np.ndarray, y_train: pd.Series) -> np.ndarray:
    lower = int(y_train.min())
    upper = int(y_train.max())
    return np.rint(np.clip(predictions, lower, upper)).astype(int)


def build_estimator(estimator: str, n_estimators: int, random_state: int):
    if estimator == "classifier":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if estimator == "regressor":
        return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    raise ValueError(f"Unknown estimator: {estimator}")


def small_param_grid(estimator: str, n_estimators: int) -> dict:
    if estimator == "classifier":
        return {
            "n_estimators": [n_estimators],
            "max_depth": [None, 8],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", None],
        }
    if estimator == "regressor":
        return {
            "n_estimators": [n_estimators],
            "max_depth": [None, 8],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", None],
        }
    raise ValueError(f"Unknown estimator: {estimator}")


def fit_rf(
    estimator: str,
    hyperparam_config: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    random_state: int,
    grid_cv: int,
):
    if not HAVE_SKLEARN:
        raise SystemExit(
            "Missing dependency: scikit-learn is required for progress_surrogate_feasibility modeling. "
            "Install it in the analysis environment before running this script."
        )
    if hyperparam_config == "default":
        model = build_estimator(estimator, n_estimators, random_state)
        model.fit(x_train, y_train.astype(float) if estimator == "regressor" else y_train)
        return model, "", np.nan

    base = build_estimator(estimator, n_estimators, random_state)
    if estimator == "classifier":
        min_class_count = int(y_train.value_counts().min())
        cv = min(grid_cv, min_class_count)
    else:
        cv = grid_cv
    if cv < 2:
        model = build_estimator(estimator, n_estimators, random_state)
        model.fit(x_train, y_train.astype(float) if estimator == "regressor" else y_train)
        return model, "grid_skipped_cv_lt_2", np.nan

    search = GridSearchCV(
        estimator=base,
        param_grid=small_param_grid(estimator, n_estimators),
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train.astype(float) if estimator == "regressor" else y_train)
    return search.best_estimator_, str(search.best_params_), float(search.best_score_)


def fit_model(
    analysis: pd.DataFrame,
    feature_columns: list[str],
    outcome_column: str,
    feature_set: str,
    estimator: str,
    hyperparam_config: str,
    random_state: int,
    test_size: float,
    n_estimators: int,
    grid_cv: int,
) -> dict:
    model_frame = analysis[[outcome_column, *feature_columns]].dropna(subset=[outcome_column]).copy()
    y_numeric = pd.to_numeric(model_frame[outcome_column], errors="coerce")
    keep = y_numeric.notna()
    model_frame = model_frame.loc[keep].copy()
    y = y_numeric.loc[keep].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"{estimator} {feature_set} needs at least two outcome classes after filtering")

    x = encode_features(model_frame, feature_columns)
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 and len(class_counts) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    rf, best_params, best_cv_score = fit_rf(
        estimator,
        hyperparam_config,
        x_train,
        y_train,
        n_estimators,
        random_state,
        grid_cv,
    )
    if estimator == "classifier":
        y_pred_for_rmse = rf.predict(x_test)
        y_pred_class = y_pred_for_rmse.astype(int)
    elif estimator == "regressor":
        y_pred_for_rmse = rf.predict(x_test)
        y_pred_class = rounded_class_predictions(y_pred_for_rmse, y_train)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    rmse = math.sqrt(mean_squared_error(y_test, y_pred_for_rmse))
    rmse_rounded = math.sqrt(mean_squared_error(y_test, y_pred_class))
    return {
        "estimator": estimator,
        "hyperparam_config": hyperparam_config,
        "feature_set": feature_set,
        "model": f"{estimator}/{hyperparam_config}: {feature_set}",
        "n_rows": int(len(model_frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features_encoded": int(x.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred_class)),
        "macro_f1": float(f1_score(y_test, y_pred_class, average="macro", zero_division=0)),
        "rmse": float(rmse),
        "rmse_rounded": float(rmse_rounded),
        "grid_best_cv_neg_rmse": best_cv_score,
        "best_params": best_params,
    }


def max_only_columns(num_intervals: int) -> list[str]:
    cols = []
    for idx in range(1, num_intervals + 1):
        cols.extend([f"max_progress_t{idx}", f"progress_observed_t{idx}"])
    return cols


def moments_columns(num_intervals: int) -> list[str]:
    cols = []
    for idx in range(1, num_intervals + 1):
        cols.extend(
            [
                f"mean_progress_t{idx}",
                f"latest_progress_t{idx}",
                f"progress_range_t{idx}",
                f"progress_label_count_t{idx}",
                f"progress_observed_t{idx}",
            ]
        )
    return cols


def threshold_columns(num_intervals: int) -> list[str]:
    cols = []
    for idx in range(1, num_intervals + 1):
        cols.extend(
            [
                f"any_progress_ge_2_5_t{idx}",
                f"any_progress_ge_3_t{idx}",
                f"any_progress_eq_4_t{idx}",
                f"progress_observed_t{idx}",
            ]
        )
    return cols


def trajectory_columns(num_intervals: int) -> list[str]:
    cols = []
    for idx in range(1, num_intervals + 1):
        cols.extend(
            [
                f"cumulative_max_progress_t{idx}",
                f"latest_progress_carry_forward_t{idx}",
                f"ever_progress_ge_2_5_t{idx}",
                f"ever_progress_ge_3_t{idx}",
                f"ever_progress_eq_4_t{idx}",
            ]
        )
    for idx in range(2, num_intervals + 1):
        cols.extend(
            [
                f"max_progress_change_t{idx}_vs_t{idx - 1}",
                f"max_progress_change_observed_t{idx}_vs_t{idx - 1}",
            ]
        )
    return cols


def rich_columns(num_intervals: int) -> list[str]:
    seen = []
    for col in [
        *max_only_columns(num_intervals),
        *moments_columns(num_intervals),
        *threshold_columns(num_intervals),
        *trajectory_columns(num_intervals),
    ]:
        if col not in seen:
            seen.append(col)
    return seen


def model_feature_sets(num_intervals: int) -> dict[str, list[str]]:
    a_cols = [f"outreach_bin_t{i}" for i in range(1, num_intervals + 1)]
    return {
        "S only (descriptive, non-causal)": list(STATE_COLUMNS),
        "S + A": [*STATE_COLUMNS, *a_cols],
        "S + A + R max": [*STATE_COLUMNS, *a_cols, *max_only_columns(num_intervals)],
        "S + A + R moments": [*STATE_COLUMNS, *a_cols, *moments_columns(num_intervals)],
        "S + A + R thresholds": [*STATE_COLUMNS, *a_cols, *threshold_columns(num_intervals)],
        "S + A + R trajectory": [*STATE_COLUMNS, *a_cols, *trajectory_columns(num_intervals)],
        "S + A + R rich": [*STATE_COLUMNS, *a_cols, *rich_columns(num_intervals)],
    }


def estimator_kinds(model_kind: str) -> list[str]:
    if model_kind == "both":
        return ["classifier", "regressor"]
    return [model_kind]


def hyperparam_configs(hyperparam_mode: str) -> list[str]:
    return [hyperparam_mode]


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["delta_accuracy_vs_S_plus_A"] = np.nan
    out["delta_macro_f1_vs_S_plus_A"] = np.nan
    out["delta_rmse_vs_S_plus_A"] = np.nan
    out["delta_rmse_rounded_vs_S_plus_A"] = np.nan
    for (estimator, hyperparam_config), group in out.groupby(["estimator", "hyperparam_config"]):
        baseline = group[group["feature_set"] == "S + A"]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        mask = (out["estimator"] == estimator) & (out["hyperparam_config"] == hyperparam_config)
        out.loc[mask, "delta_accuracy_vs_S_plus_A"] = out.loc[mask, "accuracy"] - base["accuracy"]
        out.loc[mask, "delta_macro_f1_vs_S_plus_A"] = out.loc[mask, "macro_f1"] - base["macro_f1"]
        out.loc[mask, "delta_rmse_vs_S_plus_A"] = out.loc[mask, "rmse"] - base["rmse"]
        out.loc[mask, "delta_rmse_rounded_vs_S_plus_A"] = (
            out.loc[mask, "rmse_rounded"] - base["rmse_rounded"]
        )
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.4f}")
        else:
            display[col] = display[col].astype(str)
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def best_r_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    r_rows = metrics[metrics["feature_set"].str.startswith("S + A + R")].copy()
    if r_rows.empty:
        return r_rows
    idx = r_rows.groupby(["estimator", "hyperparam_config"])["delta_rmse_vs_S_plus_A"].idxmin()
    return r_rows.loc[idx].sort_values(["estimator", "hyperparam_config"])


def write_summary(
    path: Path,
    args: argparse.Namespace,
    spec: IntervalSpec,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    outcome_metadata: dict,
) -> None:
    primary_cols = [
        "estimator",
        "hyperparam_config",
        "feature_set",
        "delta_accuracy_vs_S_plus_A",
        "delta_macro_f1_vs_S_plus_A",
        "delta_rmse_vs_S_plus_A",
        "delta_rmse_rounded_vs_S_plus_A",
        "best_params",
    ]
    lines = [
        "# Progress Surrogate Feasibility Summary",
        "",
        "## Configuration",
        f"- Outcome column: `{args.outcome_column}`",
        f"- Derived outcome from contacts: `{bool(outcome_metadata.get('derived_outcome', False))}`",
        f"- Outcome start date: `{outcome_metadata.get('outcome_start_date', 'existing outcome column')}`",
        f"- Outcome end date: `{outcome_metadata.get('outcome_end_date', 'existing outcome column')}`",
        f"- Clients with derived outcome: `{outcome_metadata.get('clients_with_derived_outcome', 'n/a')}`",
        f"- Start date: `{spec.start_date.date()}`",
        f"- End date: `{spec.end_date.date()}`",
        f"- Interval months: `{args.interval_months}`",
        f"- Number of intervals: `{args.num_intervals}`",
        f"- Date format: `{args.date_format or 'mixed'}`",
        f"- Model kind: `{args.model_kind}`",
        f"- Hyperparameter mode: `{args.hyperparam_mode}`",
        f"- Synthetic smoke test: `{bool(args.synthetic_smoke_test)}`",
        "",
        "## Best R State Construction By RMSE Delta",
        markdown_table(best_r_rows(metrics)[primary_cols]),
        "",
        "## Metrics",
        markdown_table(metrics),
        "",
        "## Label Coverage By Interval",
        markdown_table(coverage),
        "",
        "## Notes",
        "- `S only` is descriptive and non-causal because it does not adjust for outreach/action history.",
        "- `S + A` is the baseline causal outcome model for this feasibility pass.",
        "- Each `S + A + R ...` row tests one label-derived state construction after adjusting for outreach/action history.",
        "- Grid tuning uses a small RF grid optimized for CV RMSE on the training split.",
        "- For regressors, `rmse` uses continuous predictions and accuracy/F1 use rounded/clipped predictions.",
        "- This script does not load case-note text, raw model outputs, or justifications.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    if args.interval_months <= 0 or args.num_intervals <= 0:
        raise ValueError("interval-months and num-intervals must be positive")
    if args.grid_cv <= 0:
        raise ValueError("grid-cv must be positive")
    spec = IntervalSpec(pd.Timestamp(args.start_date), args.interval_months, args.num_intervals)

    if args.synthetic_smoke_test:
        features, contacts, labels = synthetic_inputs(args.outcome_column)
    else:
        features, contacts, labels = load_inputs(args)

    outcome_metadata = {"derived_outcome": False}
    if args.derive_outcome_from_contacts:
        args.outcome_column = args.derived_outcome_column
        features, outcome_metadata = derive_outcome_from_contacts(
            features,
            contacts,
            spec,
            args.date_format,
            args.outcome_column,
            args.outcome_start_date,
            args.outcome_end_date,
        )

    require_columns(features.columns, [CLIENT_ID, args.outcome_column, *STATE_COLUMNS], "features")
    require_columns(contacts.columns, [CLIENT_ID, CONTACT_ID, CONTACT_DATE], "contacts")
    require_columns(labels.columns, [CONTACT_ID, PRIMARY_LABEL], "labels")

    analysis, coverage = build_interval_features(features, contacts, labels, spec, args.date_format)
    feature_sets = model_feature_sets(args.num_intervals)
    metric_rows = []
    for estimator in estimator_kinds(args.model_kind):
        for hyperparam_config in hyperparam_configs(args.hyperparam_mode):
            for feature_set, feature_columns in feature_sets.items():
                require_columns(
                    analysis.columns,
                    feature_columns,
                    f"analysis frame for {estimator} {hyperparam_config} {feature_set}",
                )
                metric_rows.append(
                    fit_model(
                        analysis,
                        feature_columns,
                        args.outcome_column,
                        feature_set,
                        estimator,
                        hyperparam_config,
                        args.random_state,
                        args.test_size,
                        args.n_estimators,
                        args.grid_cv,
                    )
                )
    metrics = add_deltas(pd.DataFrame(metric_rows))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    coverage_path = output_dir / "coverage.csv"
    summary_path = output_dir / "summary.md"
    metrics.to_csv(metrics_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    write_summary(summary_path, args, spec, metrics, coverage, outcome_metadata)

    print(f"analysis_rows={len(analysis)}")
    if outcome_metadata.get("derived_outcome"):
        print(f"derived_outcome_column={outcome_metadata['derived_outcome_column']}")
        print(f"outcome_start_date={outcome_metadata['outcome_start_date']}")
        print(f"outcome_end_date={outcome_metadata['outcome_end_date']}")
        print(f"post_observation_contacts_for_outcome={outcome_metadata['post_observation_contacts_for_outcome']}")
    print(f"metrics_csv={metrics_path}")
    print(f"coverage_csv={coverage_path}")
    print(f"summary_md={summary_path}")
    for row in best_r_rows(metrics).itertuples(index=False):
        print(
            f"best_r estimator={row.estimator} hyperparam_config={row.hyperparam_config} "
            f"feature_set={row.feature_set!r} "
            f"delta_accuracy_vs_S_plus_A={row.delta_accuracy_vs_S_plus_A:.4f} "
            f"delta_macro_f1_vs_S_plus_A={row.delta_macro_f1_vs_S_plus_A:.4f} "
            f"delta_rmse_vs_S_plus_A={row.delta_rmse_vs_S_plus_A:.4f}"
        )
    return metrics, coverage, output_dir


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error={exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
