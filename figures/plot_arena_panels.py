"""Arena charged-protocol figure panels, drawn by the paper's own plotting code.

Builds the same `frame` that figures/plot_annotation_mc.py::load_plot_data builds for Fig. 4
(columns sampler / budget / rmse / rmse_mc_se / mean_model_based_ci_width_95 / ..._mc_se) and hands
it to that module's `plot_two_curves`, so every idiom -- 2.5 x 2.5 in panel, METHOD_LABELS, COLORS,
MARKERS, LINESTYLES, fontsizes, grid, legend, pdf + 300-dpi png -- is the manuscript's by
construction. Arms map to the paper's two samplers: uniform -> "random", learned ->
"adaptive_stagewise". RMSE is against each replication's own full-annotation benchmark, with the
delta-method MC SE used by src/outreach_dtr_annotation_mc.py (`mc_se_rmse_delta`); width SE is the
plain MC SE of the per-rep model-based width. No coverage panel on real data, as in Fig. 4.

    python figures/plot_arena_panels.py [--results aggregates/arena/charged_v2_results.csv] [--out-dir output]
      -> arena_charged_<pi_e>_{rmse,interval_width}.{pdf,png}, arena_charged_<pi_e>_plot_data.csv
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from plot_annotation_mc import mc_se, plot_two_curves  # noqa: E402

ARM_TO_SAMPLER = {"uniform": "random", "learned": "adaptive_stagewise"}
SAMPLERS = ["random", "adaptive_stagewise"]
POLICY_LABELS = {"equal": "Equal exposure", "tilt": "Tilt open-weight $\\times$1.5"}
EPS = 1e-12


def load_plot_data(results: pd.DataFrame, pi_e: str) -> pd.DataFrame:
    sub = results[results["pi_e"].eq(pi_e)].copy()
    sub["sampler"] = sub["arm"].map(ARM_TO_SAMPLER)
    sub["budget"] = sub["budget_fraction"].astype(float)
    sub["model_based_ci_width_95"] = 2.0 * 1.96 * sub["se"].astype(float)
    rows = []
    for (sampler, budget), g in sub.groupby(["sampler", "budget"]):
        squared = g["error"].to_numpy(dtype=float) ** 2
        rmse = float(np.sqrt(squared.mean()))
        rows.append({
            "sampler": sampler, "budget": budget, "reps": len(g), "rmse": rmse,
            "rmse_mc_se": float(np.std(squared, ddof=1) / math.sqrt(len(squared)) / max(2.0 * rmse, EPS)) if len(squared) > 1 else 0.0,
            "mean_model_based_ci_width_95": float(g["model_based_ci_width_95"].mean()),
            "mean_model_based_ci_width_95_mc_se": mc_se(g["model_based_ci_width_95"]),
        })
    return pd.DataFrame(rows).sort_values(["sampler", "budget"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE.parent / "aggregates" / "arena" / "charged_v2_results.csv",
                    help="Per-replication LMArena MC results (public data; shipped).")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    for pi_e in sorted(results["pi_e"].unique()):
        frame = load_plot_data(results, pi_e)
        frame.to_csv(out / f"arena_charged_{pi_e}_plot_data.csv", index=False)
        label = POLICY_LABELS.get(pi_e, pi_e)
        plot_two_curves(frame, y_col="rmse", yerr_col="rmse_mc_se",
                        ylabel="RMSE vs full-reward benchmark", title=f"{label}: RMSE",
                        output_stem=out / f"arena_charged_{pi_e}_rmse", samplers=SAMPLERS)
        plot_two_curves(frame, y_col="mean_model_based_ci_width_95", yerr_col="mean_model_based_ci_width_95_mc_se",
                        ylabel="Mean 95% interval width", title=f"{label}: CI Width",
                        output_stem=out / f"arena_charged_{pi_e}_interval_width", samplers=SAMPLERS)


if __name__ == "__main__":
    main()
