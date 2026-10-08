"""Table 2: optimized design vs the uncertainty-proportional (active-inference) baseline, casenote data.

Prints the LaTeX tabular body: change in DRL interval width and RMSE relative to random annotation (%),
at budgets 30/50/70%. Optimized design = the design shown in Fig. 4 (stagewise for progress,
trajectory-wise with a first batch for placement).

Default input is the shipped per-budget summary aggregates/casenote/active_inference_summary.csv
(DRL RMSE and mean interval width per task x budget x sampler, 100 replications of 600 clients).
With --from-runs, the same summary is recomputed from the per-replication outputs of the protected
pipeline (src/casenote_pipeline/merge_uncprop_shards.py):
    python figures/table_active_inference.py --from-runs results/casenote_progress_uncprop results/casenote_placement_uncprop

Usage: python figures/table_active_inference.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "aggregates" / "casenote" / "active_inference_summary.csv"
BUDGETS = (0.3, 0.5, 0.7)
ROWS = [  # (outcome label, task, target, optimized sampler)
    ("Progress", "progress", "progress_total", "adaptive_stagewise"),
    ("Placement", "placement", "y2_any_placement_improvement", "adaptive_trajectory"),
]
DESIGNS = [("optimized", "Optimized"), ("uncertainty_proportional", "Uncertainty-prop.")]


def summary_from_runs(progress_run: Path, placement_run: Path) -> pd.DataFrame:
    parts = []
    for task, run, target in [("progress", progress_run, "progress_total"),
                              ("placement", placement_run, "y2_any_placement_improvement")]:
        d = pd.read_csv(Path(run) / "results.csv", low_memory=False)
        d = d[d.estimator.eq("drl") & d.target.eq(target)]
        g = d.groupby(["budget", "sampler"]).agg(
            rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            mean_model_based_ci_width_95=("model_based_ci_width_95", "mean")).reset_index()
        parts.append(g.assign(task=task, target=target))
    return pd.concat(parts, ignore_index=True)


def relative(summary: pd.DataFrame, task: str, sampler: str) -> tuple[list[float], list[float]]:
    g = summary[summary.task.eq(task)].set_index(["budget", "sampler"])
    width, rmse = [], []
    for b in BUDGETS:
        base, row = g.loc[(b, "random")], g.loc[(b, sampler)]
        width.append(100 * (row.mean_model_based_ci_width_95 / base.mean_model_based_ci_width_95 - 1))
        rmse.append(100 * (row.rmse / base.rmse - 1))
    return width, rmse


def fmt(v: float) -> str:
    r = int(np.floor(v + 0.5))  # half away from zero for positives; avoids banker's rounding
    return "$0$" if r == 0 else f"${r:+d}$"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--from-runs", nargs=2, metavar=("PROGRESS_RUN", "PLACEMENT_RUN"), default=None)
    args = ap.parse_args()
    summary = summary_from_runs(*args.from_runs) if args.from_runs else pd.read_csv(args.summary)
    lines = []
    for label, task, target, optimized in ROWS:
        for i, (key, name) in enumerate(DESIGNS):
            width, rmse = relative(summary, task, optimized if key == "optimized" else key)
            lead = label if i == 0 else ""
            lines.append(f"{lead:<10}& {name:<18}& " + " & ".join(fmt(v) for v in width + rmse) + r" \\")
        lines.append(r"\addlinespace" if label != ROWS[-1][0] else "")
    print("\n".join(line for line in lines if line))


if __name__ == "__main__":
    main()
