"""SI Fig. S4 numbers: stagewise vs trajectory-wise design on the casenote progress task.

Per budget: RMSE ratio (stagewise / trajectory-wise) with a paired bootstrap 95% CI over the 100
replications (4,000 resamples, seed 0), the mean interval-width ratio, the mean per-client conditional
stage-2 annotation probability given stage 1, and the realized share of annotated clients whose stage 2
was also annotated (stagewise design).

Default input is the precomputed, budget-level aggregates/casenote/bundle_stats.csv (the per-replication
casenote results are not distributed). With --results-dir, recomputes from a full run directory of the
protected pipeline (results/casenote_progress/results.csv).

Usage: python figures/bundle_stats.py [--results-dir results/casenote_progress]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent / "aggregates" / "casenote" / "bundle_stats.csv"
ALIASES = {"stagewise_adaptive": "adaptive_trajectory", "prefix_adaptive": "adaptive_stagewise"}


def from_results(run_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(run_dir / "results.csv")
    d["sampler"] = d["sampler"].replace(ALIASES)
    d = d[(d.target == "progress_total") & (d.estimator == "drl")]
    rng = np.random.default_rng(0)
    rows = []
    for b, g in d.groupby("budget"):
        sw = g[g.sampler == "adaptive_stagewise"].set_index("rep").sort_index()
        tr = g[g.sampler == "adaptive_trajectory"].set_index("rep").loc[sw.index]
        ratio = np.sqrt((sw.error ** 2).mean() / (tr.error ** 2).mean())
        reps = sw.index.to_numpy()
        bs = []
        for _ in range(4000):
            k = rng.choice(len(reps), len(reps))
            bs.append(np.sqrt((sw.error.to_numpy()[k] ** 2).mean() / (tr.error.to_numpy()[k] ** 2).mean()))
        rows.append(dict(budget=b, n_reps=len(reps), rmse_ratio=ratio, rmse_ratio_lo=np.percentile(bs, 2.5),
                         rmse_ratio_hi=np.percentile(bs, 97.5),
                         width_ratio=sw.model_based_ci_width_95.mean() / tr.model_based_ci_width_95.mean(),
                         stage2_given_stage1=sw.mean_stage2_given_stage1.mean(),
                         realized_stage2_share=sw.stage2_annotated_count.sum() / sw.annotated_count.sum()))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", type=Path, default=DEFAULT)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    s = from_results(args.results_dir) if args.results_dir else pd.read_csv(args.stats)
    out = pd.DataFrame(dict(budget=s.budget, rmse_ratio=s.rmse_ratio.round(2), lo=s.rmse_ratio_lo.round(2),
                            hi=s.rmse_ratio_hi.round(2), width_ratio=s.width_ratio.round(3),
                            stage2_given_stage1=s.stage2_given_stage1.round(2),
                            realized_stage2_share=s.realized_stage2_share.round(2)))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
