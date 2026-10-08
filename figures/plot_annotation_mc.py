#!/usr/bin/env python3
"""Plot casenote annotation Monte Carlo RMSE and interval-width curves (Fig. 4, Fig. S4).

Two input modes:
  --plot-data CSV  budget-level plot data (shipped in aggregates/casenote/); regenerates the panels.
  --run-dir DIR    a full Monte Carlo run directory (aggregate_metrics.csv + per-replication results.csv,
                   produced from the protected data by src/outreach_dtr_annotation_mc.py); also writes
                   the budget-level <prefix>_plot_data.csv.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sampler_names import ADAPTIVE_STAGEWISE, ADAPTIVE_TRAJECTORY, canonicalize_saved  # noqa: E402

# Keyed by canonical sampler names (sampler_names.py). With a single adaptive design the
# legend reads "Adaptive annotation"; with both, the designs are named explicitly.
SAMPLER_ORDER = ["random", ADAPTIVE_STAGEWISE, ADAPTIVE_TRAJECTORY]
METHOD_LABELS = {
    "random": "Random annotation",
    ADAPTIVE_STAGEWISE: "Adaptive, stagewise",
    ADAPTIVE_TRAJECTORY: "Adaptive, trajectory-wise",
}
SINGLE_ADAPTIVE_LABEL = "Adaptive annotation"
COLORS = {"random": "#6b7280", ADAPTIVE_STAGEWISE: "#d62728", ADAPTIVE_TRAJECTORY: "#1f77b4"}
MARKERS = {"random": "o", ADAPTIVE_STAGEWISE: "^", ADAPTIVE_TRAJECTORY: "s"}
LINESTYLES = {"random": "--", ADAPTIVE_STAGEWISE: "-.", ADAPTIVE_TRAJECTORY: "-"}
SINGLE_ADAPTIVE_STYLE = {"color": "#1f77b4", "marker": "s", "linestyle": "-"}


def curve_style(sampler: str, plotted: list[str]) -> tuple[str, str, str, str]:
    """(label, color, marker, linestyle); one adaptive curve uses the paper's single-design style."""
    adaptive = [x for x in plotted if x != "random"]
    if sampler != "random" and len(adaptive) == 1:
        st = SINGLE_ADAPTIVE_STYLE
        return SINGLE_ADAPTIVE_LABEL, st["color"], st["marker"], st["linestyle"]
    return METHOD_LABELS[sampler], COLORS[sampler], MARKERS[sampler], LINESTYLES[sampler]


TARGET_LABELS = {
    "progress_total": "Progress",
    "y2_any_placement_improvement": "Any Placement Impr.",
}


def mc_se(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def add_reconstructed_interval_widths(results: pd.DataFrame, estimator: str) -> pd.DataFrame:
    results = results.copy()
    if "model_based_ci_width_95" not in results.columns:
        results["model_based_ci_width_95"] = np.nan
    missing = results["model_based_ci_width_95"].isna()
    score_col = f"{estimator}_score_sd"
    if score_col in results.columns and "sample_size" in results.columns:
        score_sd = pd.to_numeric(results.loc[missing, score_col], errors="coerce")
        sample_size = pd.to_numeric(results.loc[missing, "sample_size"], errors="coerce")
        results.loc[missing, "model_based_ci_width_95"] = 2.0 * 1.96 * score_sd / np.sqrt(sample_size)
    return results


def load_plot_data(run_dir: Path, target: str, estimator: str, legacy_stagewise_adaptive: str | None = None) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "aggregate_metrics.csv")
    # Filter to the plotted target first: legacy-name ambiguity only matters for two-stage targets.
    subset = canonicalize_saved(metrics[metrics["target"].eq(target) & metrics["estimator"].eq(estimator)], legacy_stagewise_adaptive)
    if subset.empty:
        raise ValueError(f"No rows found for target={target!r}, estimator={estimator!r}.")
    subset["rmse_mc_se"] = pd.to_numeric(subset["mc_se_rmse_delta"], errors="coerce")

    results_path = run_dir / "results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        results = canonicalize_saved(results[results["target"].eq(target) & results["estimator"].eq(estimator)], legacy_stagewise_adaptive)
        results = add_reconstructed_interval_widths(results, estimator)
        width_se = (
            results.groupby(["budget", "sampler"], as_index=False)["model_based_ci_width_95"]
            .agg(mean_model_based_ci_width_95_mc_se=mc_se)
        )
        subset = subset.merge(width_se, on=["budget", "sampler"], how="left")
    else:
        subset["mean_model_based_ci_width_95_mc_se"] = np.nan
    return subset.sort_values(["sampler", "budget"]).reset_index(drop=True)


def plot_two_curves(
    frame: pd.DataFrame,
    y_col: str,
    yerr_col: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    samplers: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.24)
    present = set(frame["sampler"])
    wanted = samplers if samplers is not None else SAMPLER_ORDER
    plotted = [x for x in SAMPLER_ORDER if x in wanted and x in present]
    missing = [x for x in wanted if x not in present]
    if missing:
        raise ValueError(f"Requested samplers not in data: {missing}; available: {sorted(present)}")
    for sampler in plotted:
        label, color, marker, linestyle = curve_style(sampler, plotted)
        group = frame[frame["sampler"].eq(sampler)].sort_values("budget")
        yerr = pd.to_numeric(group[yerr_col], errors="coerce").to_numpy(dtype=float)
        if np.all(~np.isfinite(yerr)):
            yerr = None
        ax.errorbar(
            100.0 * group["budget"],
            group[y_col],
            yerr=yerr,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5,
            linewidth=1.5,
            elinewidth=0.8,
            capsize=2.5,
            capthick=1.0,
            label=label,
        )
    ax.set_xlabel("Annotation budget (%)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--plot-data", type=Path, help="Budget-level plot-data CSV (e.g. aggregates/casenote/placement_plot_data.csv).")
    src.add_argument("--run-dir", type=Path, help="Full MC run directory with aggregate_metrics.csv and results.csv (protected pipeline).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: output/ for --plot-data, the run dir for --run-dir).")
    parser.add_argument("--target", default="y2_any_placement_improvement")
    parser.add_argument("--estimator", default="drl")
    parser.add_argument("--output-prefix", default="casenote_annotation_mc_any_improvement_drl_link")
    parser.add_argument("--adaptive-samplers", default=ADAPTIVE_STAGEWISE,
                        help="Comma list of adaptive designs to plot beside random: adaptive_stagewise (paper main figure), adaptive_trajectory, or both.")
    parser.add_argument("--legacy-stagewise-adaptive", choices=["trajectory", "stagewise"], default=None,
                        help="Meaning of legacy 'stagewise_adaptive' rows for two-stage targets: trajectory (raw driver output) or stagewise (hand-relabelled fig_stagewise / *_stagewise_fig dirs). Required only if such rows exist.")
    args = parser.parse_args()
    samplers = ["random", *[x.strip() for x in args.adaptive_samplers.split(",") if x.strip()]]

    target_label = TARGET_LABELS.get(args.target, args.target.replace("_", " ").title())
    if args.plot_data is not None:
        out_dir = args.out_dir or Path("output")
        frame = pd.read_csv(args.plot_data)
        frame = frame[frame["target"].eq(args.target) & frame["estimator"].eq(args.estimator)]
        if frame.empty:
            raise ValueError(f"No rows for target={args.target!r}, estimator={args.estimator!r} in {args.plot_data}.")
    else:
        out_dir = args.out_dir or args.run_dir
        frame = load_plot_data(args.run_dir, args.target, args.estimator, args.legacy_stagewise_adaptive)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is not None:
        plot_data = out_dir / f"{args.output_prefix}_plot_data.csv"
        frame.to_csv(plot_data, index=False)
        print(f"Wrote {plot_data}")

    plot_two_curves(
        frame,
        y_col="rmse",
        yerr_col="rmse_mc_se",
        ylabel="RMSE vs full-reward benchmark",
        title=f"{target_label}: RMSE",
        output_stem=out_dir / f"{args.output_prefix}_rmse",
        samplers=samplers,
    )
    plot_two_curves(
        frame,
        y_col="mean_model_based_ci_width_95",
        yerr_col="mean_model_based_ci_width_95_mc_se",
        ylabel="Mean 95% interval width",
        title=f"{target_label}: CI Width",
        output_stem=out_dir / f"{args.output_prefix}_interval_width",
        samplers=samplers,
    )


if __name__ == "__main__":
    main()
