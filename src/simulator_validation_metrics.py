#!/usr/bin/env python3
"""Score annotation-MC runs on simulator-generated cohorts against simulator ground truth.

Reads ``results.csv`` from one or more ``outreach_dtr_annotation_mc.py`` run directories
and a truth JSON from ``mdp_simulator_dgp.py truth``. For every (run, target, estimator,
budget, sampler) cell it reports bias, RMSE and coverage of the model-based 95% CI **against
the true target-policy value** (the quantity the real-data experiment cannot observe), plus
the adaptive-vs-random RMSE and width reductions in the same format as the paper figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampler_names import ADAPTIVE_STAGEWISE, canonicalize_saved  # noqa: E402

TARGET_LABELS = {"progress_total": "Progress", "y2_any_placement_improvement": "Any Placement"}


def load_runs(run_dirs: list[Path], legacy_stagewise_adaptive: str | None = None) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = canonicalize_saved(pd.read_csv(path, low_memory=False), legacy_stagewise_adaptive)
        frame["run"] = run_dir.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cell_metrics(results: pd.DataFrame, truth: dict) -> pd.DataFrame:
    rows = []
    keys = ["run", "target", "estimator", "budget", "sampler"]
    for key, group in results.groupby(keys):
        run, target, estimator, budget, sampler = key
        tv = float(truth["target_policy"][target]["value"])
        est = pd.to_numeric(group["estimate"], errors="coerce")
        lo = pd.to_numeric(group["model_based_ci_lower_95"], errors="coerce")
        hi = pd.to_numeric(group["model_based_ci_upper_95"], errors="coerce")
        width = pd.to_numeric(group["model_based_ci_width_95"], errors="coerce")
        bench = pd.to_numeric(group["full_reward_benchmark"], errors="coerce")
        ok = est.notna()
        rows.append({
            "run": run, "target": target, "estimator": estimator, "budget": float(budget), "sampler": sampler,
            "reps": int(ok.sum()), "truth": tv,
            "mean_estimate": float(est[ok].mean()),
            "bias_vs_truth": float((est[ok] - tv).mean()),
            "rmse_vs_truth": float(np.sqrt(((est[ok] - tv) ** 2).mean())),
            "mc_sd": float(est[ok].std(ddof=1)),
            "coverage_truth_95": float(((lo <= tv) & (hi >= tv))[ok & lo.notna()].mean()),
            "mean_ci_width": float(width[ok].mean()),
            "mean_se": float(pd.to_numeric(group["model_based_se"], errors="coerce")[ok].mean()),
            "sd_to_mean_se_ratio": float(est[ok].std(ddof=1) / pd.to_numeric(group["model_based_se"], errors="coerce")[ok].mean()),
            "rmse_vs_full_benchmark": float(np.sqrt(((est[ok] - bench[ok]) ** 2).mean())),
            "full_benchmark_rmse_vs_truth": float(np.sqrt(((bench[ok] - tv) ** 2).mean())),
            "full_benchmark_bias_vs_truth": float((bench[ok] - tv).mean()),
        })
    return pd.DataFrame(rows).sort_values(["run", "target", "estimator", "budget", "sampler"]).reset_index(drop=True)


def comparison(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, target, estimator, budget), group in cells.groupby(["run", "target", "estimator", "budget"]):
        by = group.set_index("sampler")
        if not {"random", ADAPTIVE_STAGEWISE} <= set(by.index):
            continue
        r, a = by.loc["random"], by.loc[ADAPTIVE_STAGEWISE]
        rows.append({
            "run": run, "target": target, "estimator": estimator, "budget": budget, "reps": int(min(r["reps"], a["reps"])),
            "truth": r["truth"],
            "rmse_random": r["rmse_vs_truth"], "rmse_adaptive": a["rmse_vs_truth"],
            "rmse_reduction_pct": 100.0 * (1.0 - a["rmse_vs_truth"] / r["rmse_vs_truth"]) if r["rmse_vs_truth"] > 0 else np.nan,
            "bias_random": r["bias_vs_truth"], "bias_adaptive": a["bias_vs_truth"],
            "coverage_random": r["coverage_truth_95"], "coverage_adaptive": a["coverage_truth_95"],
            "width_random": r["mean_ci_width"], "width_adaptive": a["mean_ci_width"],
            "width_reduction_pct": 100.0 * (1.0 - a["mean_ci_width"] / r["mean_ci_width"]) if r["mean_ci_width"] > 0 else np.nan,
            "sd_se_ratio_random": r["sd_to_mean_se_ratio"], "sd_se_ratio_adaptive": a["sd_to_mean_se_ratio"],
            "rmse_reduction_vs_full_benchmark_pct": 100.0 * (1.0 - a["rmse_vs_full_benchmark"] / r["rmse_vs_full_benchmark"]) if r["rmse_vs_full_benchmark"] > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = ""
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            cells.append(f"{val:.{digits}f}" if isinstance(val, (float, np.floating)) and np.isfinite(val) else str(val))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def plot_panels(comp: pd.DataFrame, cells: pd.DataFrame, out_prefix: Path, estimator: str) -> list[Path]:
    paths = []
    for (run, target), group in comp[comp["estimator"] == estimator].groupby(["run", "target"]):
        group = group.sort_values("budget")
        sub = cells[(cells["run"] == run) & (cells["target"] == target) & (cells["estimator"] == estimator)]
        fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6))
        label = TARGET_LABELS.get(target, target)
        for sampler, style in [("random", dict(color="tab:red", ls="--", marker="s")), (ADAPTIVE_STAGEWISE, dict(color="tab:blue", ls="-", marker="o"))]:
            s = sub[sub["sampler"] == sampler].sort_values("budget")
            axes[0].plot(s["budget"], s["rmse_vs_truth"], label="random" if sampler == "random" else "adaptive", ms=3, lw=1, **style)
            axes[1].plot(s["budget"], s["coverage_truth_95"], ms=3, lw=1, **style)
            axes[2].plot(s["budget"], s["mean_ci_width"], ms=3, lw=1, **style)
        axes[3].plot(group["budget"], group["rmse_reduction_pct"], color="black", marker="o", ms=3, lw=1)
        axes[3].axhline(0, color="gray", lw=0.6)
        axes[1].axhline(0.95, color="gray", lw=0.6, ls=":")
        axes[1].set_ylim(0.5, 1.02)
        titles = [f"{label}: RMSE vs truth", "Coverage of truth (95% CI)", "CI width", "RMSE reduction (%)"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("budget", fontsize=7)
            ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{run} / {estimator.upper()}", fontsize=8)
        fig.tight_layout()
        path = out_prefix.with_name(f"{out_prefix.name}_{run}_{target}_{estimator}.png")
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="MC run directory (repeatable).")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--legacy-stagewise-adaptive", choices=["trajectory", "stagewise"], default=None,
                        help="Meaning of legacy 'stagewise_adaptive' rows for two-stage targets (see sampler_names.py); required only if present.")
    args = parser.parse_args(argv)

    truth = json.loads(args.truth.read_text())
    results = load_runs(args.run_dir, args.legacy_stagewise_adaptive)
    cells = cell_metrics(results, truth)
    comp = comparison(cells)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "cell_metrics_vs_truth.csv", index=False)
    comp.to_csv(args.out_dir / "adaptive_vs_random_vs_truth.csv", index=False)
    figs = []
    for estimator in sorted(comp["estimator"].unique()):
        figs += plot_panels(comp, cells, args.out_dir / "validation", estimator)

    lines = ["# Simulator-grounded validation: MC estimates vs simulator truth", ""]
    lines.append(f"Truth file: `{args.truth}` (target policy shift = {truth.get('shift_quintiles')}, {truth.get('n_rollouts')} rollouts)")
    lines.append("")
    lines.append("| Target | Truth (target policy) | MC SE | Behavior-policy value |")
    lines.append("|---|---|---|---|")
    for target in ["progress_total", "y2_any_placement_improvement"]:
        t = truth["target_policy"][target]; b = truth["behavior_policy"][target]
        lines.append(f"| {TARGET_LABELS.get(target, target)} | {t['value']:.4f} | {t['mc_se']:.4f} | {b['value']:.4f} |")
    lines.append("")
    lines.append("Values above are simulator quantities; they are not estimates of the real-data effects.")
    lines.append("")
    for (run, target, estimator), group in comp.groupby(["run", "target", "estimator"]):
        lines.append(f"## {run} / {TARGET_LABELS.get(target, target)} / {estimator}")
        lines.append("")
        lines.append(markdown_table(group.sort_values("budget"), [
            "budget", "reps", "rmse_random", "rmse_adaptive", "rmse_reduction_pct", "bias_random", "bias_adaptive",
            "coverage_random", "coverage_adaptive", "width_random", "width_adaptive", "width_reduction_pct",
            "sd_se_ratio_random", "sd_se_ratio_adaptive",
        ]))
        full = cells[(cells["run"] == run) & (cells["target"] == target) & (cells["estimator"] == estimator)]
        lines.append(f"Full-annotation benchmark on the same 600-client samples: RMSE vs truth {full['full_benchmark_rmse_vs_truth'].mean():.4f}, bias {full['full_benchmark_bias_vs_truth'].mean():+.4f}.")
        lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in figs:
        lines.append(f"- `{fig.name}`")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
