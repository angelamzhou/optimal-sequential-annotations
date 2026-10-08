#!/usr/bin/env python3
"""Appendix panels for the simulator-grounded validation, in the Fig. 4 style (2.5 x 2.5 in).

For each target (placement, progress) and the DRL estimator: RMSE against the true policy value,
empirical coverage of the true policy value by the model-based 95% CI, and mean CI width, random vs
adaptive annotation, over budgets. Error bars are Monte Carlo standard errors across replications.

Default mode plots from the shipped per-budget summaries (aggregates/simulator_validation/):
    python figures/plot_simulator_validation.py [--out-dir output]
Full mode (--from-runs) recomputes the summaries from per-replication simulator runs and the simulator's
true policy values; those inputs are derived from a simulator fit to the protected casenote data and
are not distributed:
    python figures/plot_simulator_validation.py --from-runs --placement-run DIR --progress-run DIR --truth JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR), str(SCRIPT_DIR.parent / "src")]
from plot_annotation_mc import curve_style  # noqa: E402
from sampler_names import ADAPTIVE_STAGEWISE, ADAPTIVE_TRAJECTORY, TWO_STAGE_TARGETS, canonicalize_saved  # noqa: E402

PANELS = {
    "rmse": ("RMSE", "RMSE vs. true value"),
    "coverage": ("Coverage", "95% CI coverage of truth"),
    "width": ("CI width", "Mean 95% CI width"),
}


def summarize(results: pd.DataFrame, truth: float) -> pd.DataFrame:
    rows = []
    for (budget, sampler), g in results.groupby(["budget", "sampler"]):
        est = pd.to_numeric(g["estimate"], errors="coerce")
        err2 = (est - truth) ** 2
        n = int(err2.notna().sum())
        rmse = float(np.sqrt(err2.mean()))
        lo = pd.to_numeric(g["model_based_ci_lower_95"], errors="coerce")
        hi = pd.to_numeric(g["model_based_ci_upper_95"], errors="coerce")
        cov = ((lo <= truth) & (hi >= truth)).astype(float)
        width = pd.to_numeric(g["model_based_ci_width_95"], errors="coerce")
        rows.append({
            "budget": float(budget), "sampler": sampler,
            "rmse": rmse, "rmse_se": float(err2.std(ddof=1) / (2.0 * rmse * np.sqrt(n))) if rmse > 0 else 0.0,
            "coverage": float(cov.mean()), "coverage_se": float(np.sqrt(cov.mean() * (1 - cov.mean()) / n)),
            "width": float(width.mean()), "width_se": float(width.std(ddof=1) / np.sqrt(n)),
        })
    return pd.DataFrame(rows)


def plot_panel(frame: pd.DataFrame, key: str, title: str, out_stem: Path, nominal: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.subplots_adjust(left=0.26, right=0.98, top=0.84, bottom=0.24)
    plotted = ["random", ADAPTIVE_STAGEWISE]
    for sampler in plotted:
        g = frame[frame["sampler"].eq(sampler)].sort_values("budget")
        if g.empty:
            raise ValueError(f"No {sampler!r} rows; available: {sorted(set(frame['sampler']))}")
        label, color, marker, linestyle = curve_style(sampler, plotted)
        ax.errorbar(100.0 * g["budget"], g[key], yerr=g[f"{key}_se"], color=color, linestyle=linestyle,
                    marker=marker, markersize=3.5, linewidth=1.5, elinewidth=0.8, capsize=2.5, capthick=1.0,
                    label=label)
    if nominal is not None:
        ax.axhline(nominal, color="#9ca3af", linewidth=0.8, linestyle=":")
        ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Annotation budget (%)", fontsize=8)
    ax.set_ylabel(PANELS[key][1], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(frameon=False, loc="upper right" if key != "coverage" else "lower left", fontsize=7)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {out_stem.with_suffix('.pdf')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=SCRIPT_DIR.parent / "aggregates" / "simulator_validation")
    parser.add_argument("--from-runs", action="store_true", help="Recompute summaries from per-replication runs (protected inputs).")
    parser.add_argument("--placement-run", type=Path, default=Path("results/mdp_simulator_validation/mc_rbf_n8000_placement_link_pilot30_oracleg"))
    # Raw run containing the paper's stagewise design (prefix_adaptive); the older default
    # (mc_rbf_n8000_progress_silveronly_oracleg) holds only the trajectory-wise design.
    parser.add_argument("--progress-run", type=Path, default=Path("results/mdp_simulator_validation/mc_rbf_n8000_progress_silveronly_oracleg_prefix"))
    parser.add_argument("--legacy-stagewise-adaptive", choices=["trajectory", "stagewise"], default="trajectory",
                        help="Meaning of legacy 'stagewise_adaptive' rows in the progress run (default trajectory: raw driver output).")
    parser.add_argument("--truth", type=Path, default=Path("results/mdp_simulator_validation/truth_rbf_impute_mix.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--estimator", default="drl")
    args = parser.parse_args()
    import json
    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth = json.loads(args.truth.read_text())["target_policy"] if args.from_runs else None
    for label, run, target in [("placement", args.placement_run, "y2_any_placement_improvement"), ("progress", args.progress_run, "progress_total")]:
        if args.from_runs:
            results = pd.read_csv(run / "results.csv", low_memory=False)
            results = results[results["target"].eq(target) & results["estimator"].eq(args.estimator)]
            results = canonicalize_saved(results, args.legacy_stagewise_adaptive)
            if target not in TWO_STAGE_TARGETS:  # terminal-only target: the two adaptive designs coincide
                results["sampler"] = results["sampler"].replace({ADAPTIVE_TRAJECTORY: ADAPTIVE_STAGEWISE})
            summary = summarize(results, float(truth[target]["value"]))
            summary.to_csv(args.out_dir / f"sim_validation_{label}_summary.csv", index=False)
        else:
            summary = pd.read_csv(args.summary_dir / f"sim_validation_{label}_summary.csv")
        pretty = "Any Placement" if label == "placement" else "Progress"
        for key, (title, _) in PANELS.items():
            plot_panel(summary, key, f"{pretty}: {title}", args.out_dir / f"sim_validation_{label}_{key}", nominal=0.95 if key == "coverage" else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
