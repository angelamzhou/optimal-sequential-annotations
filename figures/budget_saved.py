"""Label savings at equal interval width (abstract and main text: 26-31% placement, 25-28% progress,
37-48% LMArena).

For each adaptive budget B_a, the random-annotation budget B_r that matches the adaptive design's mean
model-based 95% CI width is found by linear interpolation of the random design's monotone-envelope width
curve (src/paper_style_budget_metrics.budget_saved); saved = 100 (B_r - B_a) / B_r. Cells where the
adaptive design is narrower than random at the largest budget are right-censored and flagged.

Inputs (shipped, budget-level): aggregates/casenote/{placement,progress}_plot_data.csv (Fig. 4) and
aggregates/arena/charged_v2_summary_error_based.csv (Fig. 5, equal-exposure policy; width = 2 x 1.96 x
mean model-based SE).

Usage: python figures/budget_saved.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paper_style_budget_metrics as psm  # noqa: E402

AGG = ROOT / "aggregates"
PANELS = [  # (label, plot data, target, adaptive design)
    ("placement", AGG / "casenote" / "placement_plot_data.csv", "y2_any_placement_improvement", "adaptive_trajectory"),
    ("progress", AGG / "casenote" / "progress_plot_data.csv", "progress_total", "adaptive_stagewise"),
]


def casenote_saved() -> pd.DataFrame:
    rows = []
    for label, path, target, adaptive in PANELS:
        a = pd.read_csv(path)
        a = a[(a.target == target) & (a.estimator == "drl")].copy()
        a["sample_size"] = 600
        s = psm.budget_saved(a, [adaptive])
        s = s[s.metric == "model_ci_width"]
        rows += [dict(panel=label, budget=r.adaptive_budget, saved_pct=r.budget_saved_pct_of_random, status=r.status)
                 for r in s.itertuples()]
    return pd.DataFrame(rows)


def arena_saved() -> pd.DataFrame:
    s = pd.read_csv(AGG / "arena" / "charged_v2_summary_error_based.csv")
    s = s[s.pi_e == "equal"].rename(columns={"budget_fraction": "budget", "arm": "sampler"})
    s["mean_model_based_ci_width_95"] = 2 * 1.96 * s["mean_se"]
    s["sampler"] = s["sampler"].replace({"uniform": "random"})
    s["sample_size"], s["estimator"] = 21685, "drl"
    out = psm.budget_saved(s, ["learned"])
    out = out[out.metric == "model_ci_width"]
    return pd.DataFrame([dict(panel="arena (equal exposure)", budget=r.adaptive_budget,
                              saved_pct=r.budget_saved_pct_of_random, status=r.status) for r in out.itertuples()])


def main() -> None:
    psm.RANDOM_SAMPLER = "random"
    out = pd.concat([casenote_saved(), arena_saved()], ignore_index=True)
    print(out.pivot_table(index="budget", columns="panel", values="saved_pct").round(1).to_string())
    cens = out[out.status != "interpolated"]
    print("\nnon-interpolated cells:\n", cens.round(1).to_string(index=False) if len(cens) else "none")
    for label, budgets in [("placement", (0.3, 0.4, 0.5)), ("progress", (0.3, 0.4, 0.5, 0.6)),
                           ("arena (equal exposure)", (0.3, 0.4, 0.5))]:
        v = out[(out.panel == label) & out.budget.round(2).isin(budgets)].saved_pct
        print(f"{label}: {v.min():.2f}-{v.max():.2f}% fewer labels at budgets {', '.join(f'{b:.0%}' for b in budgets)}")


if __name__ == "__main__":
    main()
