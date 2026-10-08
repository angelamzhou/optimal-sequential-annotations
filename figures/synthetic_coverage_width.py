#!/usr/bin/env python3
"""DRL coverage and mean model-based CI width vs budget, synthetic data, policy d=(0,0) (main-text numbers for Fig. 3).

Usage: python figures/synthetic_coverage_width.py [--d1 0 --d2 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

DEFAULT_ADAPTIVE_SAMPLER = "protocol_stagewise_regime_00_50pct"
COLORS = {"Random": "#6b7280", "Stagewise": "#1f77b4"}
LINESTYLES = {"Random": "--", "Stagewise": "-"}


def _load_frame(run_dir: Path, sample_size: int, d1: int, d2: int, adaptive_sampler: str) -> pd.DataFrame:
    samplers = ["random_50pct", adaptive_sampler]
    results = pd.read_csv(run_dir / "results.csv")
    results = results[
        results["sample_size"].eq(sample_size)
        & results["estimator"].eq("drl")
        & results["sampler"].isin(samplers)
        & results["d1"].eq(d1)
        & results["d2"].eq(d2)
    ].copy()
    if results.empty:
        raise ValueError(f"No DRL rows found for n={sample_size}, d=({d1},{d2})")

    rows: list[dict[str, float | int | str]] = []
    rep_col = "global_rep" if "global_rep" in results.columns else "rep"
    for (budget, sampler), group in results.groupby(["budget", "sampler"]):
        # Keep one value per MC replication for a single target regime. If a future
        # output has duplicate rows per rep, average within rep before summarizing.
        rep_level = group.groupby(rep_col, as_index=False).agg(
            covered=("model_based_ci_covers_truth", "mean"),
            ci_width=("model_based_ci_width_95", "mean"),
        )
        coverage_values = rep_level["covered"].dropna().to_numpy(dtype=float)
        width_values = rep_level["ci_width"].dropna().to_numpy(dtype=float)
        n_mc = int(len(coverage_values))
        if n_mc < 2:
            raise ValueError("Need at least two MC replications to compute uncertainty")
        coverage = float(np.mean(coverage_values))
        coverage_se = float(np.sqrt(max(coverage * (1.0 - coverage), 0.0) / n_mc))
        width = float(np.mean(width_values))
        width_se = float(np.std(width_values, ddof=1) / np.sqrt(len(width_values))) if len(width_values) > 1 else float("nan")
        label = "Random" if str(sampler) == "random_50pct" else "Stagewise"
        rows.append(
            {
                "budget": float(budget),
                "sample_size": sample_size,
                "estimator": "drl",
                "sampler": str(sampler),
                "sampler_label": label,
                "d1": d1,
                "d2": d2,
                "panel": f"Hard policy d=({d1},{d2})",
                "coverage": coverage,
                "coverage_se": coverage_se,
                "mean_ci_width": width,
                "mean_ci_width_se": width_se,
                "n_mc": n_mc,
            }
        )
    return pd.DataFrame(rows).sort_values(["sampler_label", "budget"])


def _plot(frame: pd.DataFrame, output: Path, sample_size: int, d1: int, d2: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.25), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.22, wspace=0.28)

    for sampler_label in ["Random", "Stagewise"]:
        group = frame[frame["sampler_label"].eq(sampler_label)].sort_values("budget")
        axes[0].errorbar(
            100.0 * group["budget"],
            group["coverage"],
            yerr=group["coverage_se"],
            color=COLORS[sampler_label],
            linestyle=LINESTYLES[sampler_label],
            marker="o" if sampler_label == "Stagewise" else None,
            markersize=4.0,
            linewidth=2.0,
            capsize=2.5,
            elinewidth=1.0,
            label=sampler_label,
        )
        axes[1].errorbar(
            100.0 * group["budget"],
            group["mean_ci_width"],
            yerr=group["mean_ci_width_se"],
            color=COLORS[sampler_label],
            linestyle=LINESTYLES[sampler_label],
            marker="o" if sampler_label == "Stagewise" else None,
            markersize=4.0,
            linewidth=2.0,
            capsize=2.5,
            elinewidth=1.0,
            label=sampler_label,
        )

    axes[0].axhline(0.95, color="#111827", linewidth=0.9, linestyle=":")
    axes[0].set_ylim(0.75, 1.01)
    axes[0].set_title("Coverage", fontsize=10)
    axes[0].set_ylabel("Empirical 95% CI coverage")

    axes[1].set_title("Mean CI width", fontsize=10)
    axes[1].set_ylabel("Mean 95% CI width")
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.set_xlabel("Annotation budget (%)")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"DRL inference diagnostics at n={sample_size}, hard policy d=({d1},{d2})", fontsize=11)
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
    parser.add_argument("--d1", type=int, default=0)
    parser.add_argument("--d2", type=int, default=0)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument(
        "--adaptive-sampler",
        default=DEFAULT_ADAPTIVE_SAMPLER,
        help="Adaptive sampler to plot against random (e.g. protocol_theorem_regime_00_50pct for the matched per-regime campaign).",
    )
    args = parser.parse_args()

    frame = _load_frame(args.run_dir, args.sample_size, args.d1, args.d2, args.adaptive_sampler)
    output_stem = args.output_stem or f"drl_coverage_ci_width_n{args.sample_size}_d{args.d1}{args.d2}_per_regime_stagewise"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_data = args.out_dir / f"{output_stem}_plot_data.csv"
    frame.to_csv(plot_data, index=False)
    print(f"Wrote {plot_data}")
    _plot(frame, args.out_dir / output_stem, args.sample_size, args.d1, args.d2)


if __name__ == "__main__":
    main()
