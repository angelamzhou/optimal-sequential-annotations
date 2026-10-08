#!/usr/bin/env python3
"""Paper-style reporting metrics for optimized annotation experiments.

Reads an `aggregate_metrics.csv` file from `optimized_annotation_krr_eval.py`
and writes fixed-budget gains plus interpolated budget-saved summaries.

Budget saved follows the paper's reporting idea: estimate the random annotation
budget needed to reach the adaptive method's metric value, then report the
relative reduction in annotation budget. The CI-width metric uses the
model-based 95% CI width emitted by `optimized_annotation_krr_eval.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_SAMPLER = "random_50pct"
DEFAULT_ADAPTIVE_SAMPLER = "protocol_theorem_reward_variance_50pct"


def _lower_is_better_metric(frame: pd.DataFrame, metric: str) -> np.ndarray:
    if metric == "rmse":
        return frame["rmse"].to_numpy(dtype=float)
    if metric == "mse":
        return np.square(frame["rmse"].to_numpy(dtype=float))
    if metric == "model_ci_width":
        if "mean_model_based_ci_width_95" not in frame.columns:
            raise ValueError("aggregate metrics must include mean_model_based_ci_width_95 for model_ci_width")
        return frame["mean_model_based_ci_width_95"].to_numpy(dtype=float)
    raise ValueError(f"Unknown metric {metric!r}")


def _metric_column(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = _lower_is_better_metric(frame, metric)
    return pd.Series(values, index=frame.index, name=metric)


def _interp_random_budget(
    random_curve: pd.DataFrame,
    target_value: float,
    metric: str,
) -> tuple[float, str]:
    curve = random_curve.sort_values("budget").copy()
    budgets = curve["budget"].to_numpy(dtype=float)
    values = _lower_is_better_metric(curve, metric)
    finite = np.isfinite(budgets) & np.isfinite(values)
    budgets = budgets[finite]
    values = values[finite]
    if len(budgets) == 0 or not np.isfinite(target_value):
        return np.nan, "missing"

    # Use the monotone envelope: the best random metric achieved by each budget.
    values = np.minimum.accumulate(values)
    if target_value >= values[0]:
        return float(budgets[0]), "left_censored_random_already_matches"
    if target_value < values[-1]:
        return float(budgets[-1]), "right_censored_better_than_max_random"

    for j in range(1, len(budgets)):
        high_value = values[j - 1]
        low_value = values[j]
        if high_value >= target_value >= low_value:
            if abs(high_value - low_value) <= 1e-12:
                return float(budgets[j]), "flat_segment"
            frac = (high_value - target_value) / (high_value - low_value)
            budget = budgets[j - 1] + frac * (budgets[j] - budgets[j - 1])
            return float(budget), "interpolated"
    return float(budgets[-1]), "fallback"


def fixed_budget_gains(aggregate: pd.DataFrame, adaptive_samplers: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["budget", "sample_size", "estimator"]
    for keys, group in aggregate.groupby(group_cols, sort=True):
        budget, sample_size, estimator = keys
        random = group[group["sampler"] == RANDOM_SAMPLER]
        if random.empty:
            continue
        random_row = random.iloc[0]
        for sampler in adaptive_samplers:
            adaptive = group[group["sampler"] == sampler]
            if adaptive.empty:
                continue
            adaptive_row = adaptive.iloc[0]
            random_rmse = float(random_row["rmse"])
            adaptive_rmse = float(adaptive_row["rmse"])
            random_mse = random_rmse * random_rmse
            adaptive_mse = adaptive_rmse * adaptive_rmse
            random_width = float(random_row["mean_model_based_ci_width_95"])
            adaptive_width = float(adaptive_row["mean_model_based_ci_width_95"])
            rows.append(
                {
                    "budget": float(budget),
                    "sample_size": int(sample_size),
                    "estimator": estimator,
                    "adaptive_sampler": sampler,
                    "random_rmse": random_rmse,
                    "adaptive_rmse": adaptive_rmse,
                    "rmse_reduction_pct": 100.0 * (random_rmse - adaptive_rmse) / random_rmse,
                    "random_mse": random_mse,
                    "adaptive_mse": adaptive_mse,
                    "mse_gain_pct": 100.0 * (random_mse - adaptive_mse) / random_mse,
                    "random_model_ci_width_95": random_width,
                    "adaptive_model_ci_width_95": adaptive_width,
                    "model_ci_width_reduction_pct": 100.0 * (random_width - adaptive_width) / random_width,
                    "random_sd": float(random_row["sd"]),
                    "adaptive_sd": float(adaptive_row["sd"]),
                    "adaptive_mean_annotation_ess": float(adaptive_row["mean_annotation_ess"]),
                    "adaptive_prob_min": float(adaptive_row["mean_prob_min"]),
                    "adaptive_prob_max": float(adaptive_row["mean_prob_max"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def budget_saved(aggregate: pd.DataFrame, adaptive_samplers: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["rmse", "mse", "model_ci_width"]
    for (sample_size, estimator), group in aggregate.groupby(["sample_size", "estimator"], sort=True):
        random_curve = group[group["sampler"] == RANDOM_SAMPLER].sort_values("budget")
        if random_curve.empty:
            continue
        for sampler in adaptive_samplers:
            adaptive_curve = group[group["sampler"] == sampler].sort_values("budget")
            if adaptive_curve.empty:
                continue
            for _, adaptive_row in adaptive_curve.iterrows():
                adaptive_budget = float(adaptive_row["budget"])
                for metric in metrics:
                    target = float(_metric_column(pd.DataFrame([adaptive_row]), metric).iloc[0])
                    random_budget, status = _interp_random_budget(random_curve, target, metric)
                    absolute_saved = random_budget - adaptive_budget
                    relative_saved = (
                        100.0 * absolute_saved / random_budget
                        if np.isfinite(random_budget) and random_budget > 0
                        else np.nan
                    )
                    rows.append(
                        {
                            "sample_size": int(sample_size),
                            "estimator": estimator,
                            "adaptive_sampler": sampler,
                            "metric": metric,
                            "adaptive_budget": adaptive_budget,
                            "adaptive_metric_value": target,
                            "random_budget_to_match": random_budget,
                            "absolute_budget_saved": absolute_saved,
                            "budget_saved_pct_of_random": relative_saved,
                            "status": status,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def summarize_budget_saved(budget_saved_frame: pd.DataFrame) -> pd.DataFrame:
    if budget_saved_frame.empty:
        return budget_saved_frame
    return (
        budget_saved_frame.groupby(
            ["adaptive_sampler", "metric", "adaptive_budget", "estimator"],
            as_index=False,
        )
        .agg(
            mean_budget_saved_pct=("budget_saved_pct_of_random", "mean"),
            median_budget_saved_pct=("budget_saved_pct_of_random", "median"),
            min_budget_saved_pct=("budget_saved_pct_of_random", "min"),
            max_budget_saved_pct=("budget_saved_pct_of_random", "max"),
            positive_cells=("budget_saved_pct_of_random", lambda x: int(np.sum(np.asarray(x) > 0))),
            cells=("budget_saved_pct_of_random", "size"),
            statuses=("status", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["metric", "mean_budget_saved_pct"], ascending=[True, False])
    )


def summarize_fixed_budget(fixed_frame: pd.DataFrame) -> pd.DataFrame:
    if fixed_frame.empty:
        return fixed_frame
    return (
        fixed_frame.groupby(["adaptive_sampler", "budget", "estimator"], as_index=False)
        .agg(
            mean_rmse_reduction_pct=("rmse_reduction_pct", "mean"),
            mean_mse_gain_pct=("mse_gain_pct", "mean"),
            mean_model_ci_width_reduction_pct=("model_ci_width_reduction_pct", "mean"),
            positive_rmse_cells=("rmse_reduction_pct", lambda x: int(np.sum(np.asarray(x) > 0))),
            cells=("rmse_reduction_pct", "size"),
            mean_random_rmse=("random_rmse", "mean"),
            mean_adaptive_rmse=("adaptive_rmse", "mean"),
            mean_adaptive_ess=("adaptive_mean_annotation_ess", "mean"),
        )
        .sort_values(["mean_mse_gain_pct", "mean_rmse_reduction_pct"], ascending=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adaptive-samplers",
        default=DEFAULT_ADAPTIVE_SAMPLER,
        help="Comma-separated adaptive samplers to compare against random_50pct.",
    )
    args = parser.parse_args()

    aggregate = pd.read_csv(args.aggregate_metrics)
    adaptive_samplers = [part.strip() for part in args.adaptive_samplers.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = fixed_budget_gains(aggregate, adaptive_samplers)
    saved = budget_saved(aggregate, adaptive_samplers)
    fixed_summary = summarize_fixed_budget(fixed)
    saved_summary = summarize_budget_saved(saved)

    fixed.to_csv(args.output_dir / "fixed_budget_gains.csv", index=False)
    fixed_summary.to_csv(args.output_dir / "fixed_budget_gains_summary.csv", index=False)
    saved.to_csv(args.output_dir / "budget_saved.csv", index=False)
    saved_summary.to_csv(args.output_dir / "budget_saved_summary.csv", index=False)

    with (args.output_dir / "paper_style_metrics_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Fixed-budget gains\n")
        handle.write(fixed_summary.to_string(index=False))
        handle.write("\n\nBudget saved\n")
        handle.write(saved_summary.to_string(index=False))
        handle.write("\n")

    print(f"Wrote paper-style metrics to {args.output_dir}")
    print("\nFixed-budget gains:")
    print(fixed_summary.to_string(index=False))
    print("\nBudget saved:")
    print(saved_summary.to_string(index=False))


if __name__ == "__main__":
    main()
