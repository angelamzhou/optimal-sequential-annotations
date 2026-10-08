"""Table S3: LMArena RMSE and mean model-based 95% interval width (x 1e-3), random vs adaptive
annotation, equal-exposure and tilt policies (the quantities plotted in Fig. 5). Also prints the
main-text LMArena numbers derived from the same summary.

Input (shipped): aggregates/arena/charged_v2_summary_error_based.csv, produced by
src/arena/summarize_charged.py from the per-replication results (aggregates/arena/charged_v2_results.csv).
Interval width = 2 x 1.96 x mean model-based SE.

Usage: python figures/table_arena.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from table_formatting import rhalf  # noqa: E402

BUDGETS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def main() -> None:
    s = pd.read_csv(HERE.parent / "aggregates" / "arena" / "charged_v2_summary_error_based.csv")
    s["width"] = 2 * 1.96 * s["mean_se"]
    g = s.set_index(["pi_e", "budget_fraction", "arm"])
    for b in BUDGETS:
        cells = []
        for pi in ("equal", "tilt"):
            for metric, dec in (("rmse", 2), ("width", 1)):
                for arm in ("uniform", "learned"):
                    cells.append(f"{rhalf(1e3 * g.loc[(pi, b, arm), metric], dec):.{dec}f}")
        print(f"{b} & " + " & ".join(cells) + r" \\")
    eq = g.loc["equal"]
    red = [100 * (1 - eq.loc[(b, "learned"), "rmse"] / eq.loc[(b, "uniform"), "rmse"]) for b in BUDGETS]
    w30 = 100 * (1 - eq.loc[(0.3, "learned"), "width"] / eq.loc[(0.3, "uniform"), "width"])
    print(f"\n% equal exposure: RMSE reduction {min(red):.1f}-{max(red):.1f}% across budgets; "
          f"width reduction at 30%: {w30:.1f}%; replications: {int(s.n_reps.min())}")


if __name__ == "__main__":
    main()
