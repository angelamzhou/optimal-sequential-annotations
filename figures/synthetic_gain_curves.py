#!/usr/bin/env python3
"""Fixed-budget RMSE reduction of the stagewise design vs random, synthetic data (main-text numbers for Fig. 3).

Paper text: "42% at a 20% budget ... 12% at 80%" (DRL rows, per-regime stagewise designs).

Usage: python figures/synthetic_gain_curves.py --adaptive-mode per_regime_stagewise
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def _metric_and_se(random_sq: pd.Series, adaptive_sq: pd.Series) -> dict[str, float]:
    paired = pd.DataFrame({"random_sq": random_sq, "adaptive_sq": adaptive_sq}).dropna()
    n_mc = len(paired)
    if n_mc < 2:
        raise ValueError("Need at least two paired MC replications to compute SEs")
    random_mse = float(paired["random_sq"].mean())
    adaptive_mse = float(paired["adaptive_sq"].mean())
    cov = np.cov(paired[["random_sq", "adaptive_sq"]].to_numpy().T, ddof=1) / n_mc

    mse_gain = 100.0 * (1.0 - adaptive_mse / random_mse)
    grad_mse = np.array([100.0 * adaptive_mse / (random_mse**2), -100.0 / random_mse])
    mse_gain_se = float(np.sqrt(max(grad_mse @ cov @ grad_mse, 0.0)))

    random_rmse = float(np.sqrt(random_mse))
    adaptive_rmse = float(np.sqrt(adaptive_mse))
    rmse_reduction = 100.0 * (1.0 - adaptive_rmse / random_rmse)
    grad_rmse = np.array(
        [
            50.0 * adaptive_rmse / (random_mse * random_rmse),
            -50.0 / (adaptive_rmse * random_rmse),
        ]
    )
    rmse_reduction_se = float(np.sqrt(max(grad_rmse @ cov @ grad_rmse, 0.0)))

    return {
        "n_mc": n_mc,
        "random_mse": random_mse,
        "adaptive_mse": adaptive_mse,
        "random_rmse": random_rmse,
        "adaptive_rmse": adaptive_rmse,
        "mse_gain_pct": mse_gain,
        "mse_gain_pct_se": mse_gain_se,
        "rmse_reduction_pct": rmse_reduction,
        "rmse_reduction_pct_se": rmse_reduction_se,
    }


def _paired_squared_errors(results: pd.DataFrame, panel: str) -> pd.DataFrame:
    if panel == "Hard policy d=(0,0)":
        results = results[results["d1"].eq(0) & results["d2"].eq(0)].copy()
    elif panel != "Four-policy average":
        raise ValueError(f"Unknown panel: {panel}")

    grouped = results.groupby(["budget", "estimator", "rep", "sampler"], as_index=False).agg(
        squared_error=("squared_error", "mean")
    )
    return grouped.pivot(
        index=["budget", "estimator", "rep"], columns="sampler", values="squared_error"
    ).reset_index()


def _load_frame(run_dir: Path, sample_size: int, adaptive_mode: str) -> pd.DataFrame:
    results = pd.read_csv(run_dir / "results.csv")
    results = results[
        results["sample_size"].eq(sample_size) & results["estimator"].isin(["drl", "plugin"])
    ].copy()
    if adaptive_mode in ("per_regime", "per_regime_stagewise"):
        # Each regime is evaluated under its own matched campaign: trajectory-wise
        # (protocol_theorem_regime_*) or forward-monotone stagewise (protocol_stagewise_regime_*).
        prefix = "protocol_stagewise_regime_" if adaptive_mode == "per_regime_stagewise" else "protocol_theorem_regime_"
        matched = prefix + results["d1"].astype(int).astype(str) + results["d2"].astype(int).astype(str) + "_50pct"
        keep = results["sampler"].eq("random_50pct") | results["sampler"].eq(matched)
        results = results[keep].copy()
        results.loc[results["sampler"].ne("random_50pct"), "sampler"] = "adaptive"
    else:
        results = results[results["sampler"].isin(["random_50pct", "protocol_theorem_reward_variance_50pct"])].copy()
        results.loc[results["sampler"].ne("random_50pct"), "sampler"] = "adaptive"

    rows: list[dict[str, float | int | str]] = []
    for panel in ["Hard policy d=(0,0)", "Four-policy average"]:
        paired = _paired_squared_errors(results, panel)
        for (budget, estimator), group in paired.groupby(["budget", "estimator"]):
            metrics = _metric_and_se(group["random_50pct"], group["adaptive"])
            rows.append(
                {
                    "budget": float(budget),
                    "sample_size": sample_size,
                    "estimator": str(estimator),
                    "panel": panel,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(frame: pd.DataFrame, metric: str, ylabel: str, title: str, output: Path) -> None:
    colors = {"drl": "#1f77b4", "plugin": "#8c564b"}
    random_color = "#6b7280"
    labels = {"drl": "DRL", "plugin": "Plugin"}
    linestyles = {"drl": "-", "plugin": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.22, wspace=0.18)
    for ax, panel in zip(axes, ["Hard policy d=(0,0)", "Four-policy average"]):
        subset = frame[frame["panel"].eq(panel)]
        for estimator in ["drl", "plugin"]:
            group = subset[subset["estimator"].eq(estimator)].sort_values("budget")
            ax.errorbar(
                100.0 * group["budget"],
                group[metric],
                yerr=group[f"{metric}_se"],
                color=colors[estimator],
                linestyle=linestyles[estimator],
                marker="o",
                markersize=4.0,
                linewidth=2.0,
                capsize=2.5,
                elinewidth=1.0,
                label=labels[estimator],
            )
        ax.axhline(0, color=random_color, linewidth=1.3, linestyle=":", label="Random")
        ax.set_title(panel, fontsize=10)
        ax.set_xlabel("Annotation budget (%)")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.text(0.11, 0.06, "Error bars are Monte Carlo SEs over paired replications.", fontsize=7.5, color="#374151")
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    print(f"Wrote {output.with_suffix('.pdf')}")
    print(f"Wrote {output.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parents[1] / "aggregates" / "synthetic",
                        help="Directory with the synthetic results.csv (default: the shipped trimmed copy).")
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument(
        "--adaptive-mode",
        choices=["shared", "per_regime", "per_regime_stagewise"],
        default="per_regime_stagewise",
        help="per_regime_stagewise (paper; default): each regime under its own stagewise design. shared / per_regime need trajectory-wise arms not in the shipped trimmed results.",
    )
    args = parser.parse_args()

    frame = _load_frame(args.run_dir, args.sample_size, args.adaptive_mode)
    if frame.empty:
        raise ValueError(f"No rows found for sample_size={args.sample_size}")

    suffix = {"shared": "", "per_regime": "_per_regime", "per_regime_stagewise": "_per_regime_stagewise"}[args.adaptive_mode]
    output_stem = args.output_stem or f"fixed_budget_gain_curves_n{args.sample_size}{suffix}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_data = args.out_dir / f"{output_stem}_plot_data.csv"
    frame.to_csv(plot_data, index=False)
    print(f"Wrote {plot_data}")

    _plot_metric(
        frame,
        metric="rmse_reduction_pct",
        ylabel="RMSE reduction vs random (%)",
        title=f"Fixed-budget RMSE reduction at n={args.sample_size}",
        output=args.out_dir / f"{output_stem}_rmse_reduction",
    )
    _plot_metric(
        frame,
        metric="mse_gain_pct",
        ylabel="MSE gain vs random (%)",
        title=f"Fixed-budget MSE gain at n={args.sample_size}",
        output=args.out_dir / f"{output_stem}_mse_gain",
    )


if __name__ == "__main__":
    main()
