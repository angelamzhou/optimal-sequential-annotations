"""Summarise a charged-protocol results.csv on the ERROR against each rep's own full-annotation
benchmark (the casenote driver's convention), not on sd(estimate): the per-rep subsample moves the benchmark,
so sd(estimate) mixes target variation with annotation noise.
    python summarize_charged.py charged_v1
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
D = Path(sys.argv[1]); r = pd.read_csv(D / "results.csv"); n_reps = r.rep.nunique()
g = r.groupby(["pi_e", "budget_fraction", "arm"]).agg(
    n_reps=("estimate", "size"), rmse=("error", lambda e: float(np.sqrt(np.mean(e ** 2)))),
    sd_error=("error", lambda e: float(e.std(ddof=1))), bias=("error", "mean"),
    bias_mcse=("error", lambda e: float(e.std(ddof=1) / np.sqrt(len(e)))), coverage=("covered", "mean"),
    mean_se=("se", "mean"), mean_real_cost=("real_cost", "mean"), pilot=("pilot_sessions", "mean")).reset_index()
g["se_over_sd_error"] = g.mean_se / g.sd_error
g.to_csv(D / "summary_error_based.csv", index=False)
print(f"{D.name}: {n_reps} reps | coverage MCSE ≈ {np.sqrt(.95*.05/n_reps):.3f} | RMSE MCSE ≈ {100/np.sqrt(2*n_reps):.0f}% relative\n")
for pe in g.pi_e.unique():
    print(f"=== pi_e = {pe} ===")
    print(f"{'B':>5} {'pilot':>6} | {'RMSE U':>7} {'RMSE L':>7} {'RATIO':>6} {'var red':>8} | {'bias L ±mcse':>16} | {'cov U':>6} {'cov L':>6} | {'se/sd U':>7} {'se/sd L':>7} | {'cost U':>7} {'cost L':>7}")
    for b, gb in g[g.pi_e == pe].groupby("budget_fraction"):
        U = gb[gb.arm == "uniform"].iloc[0]; L = gb[gb.arm == "learned"].iloc[0]; rt = L.rmse / U.rmse
        print(f"{b:>5} {L.pilot:>6.0f} | {U.rmse*1e3:>7.3f} {L.rmse*1e3:>7.3f} {rt:>6.3f} {100*(1-rt**2):>+7.0f}% | {L.bias*1e3:>+7.3f} ± {L.bias_mcse*1e3:.3f} | {U.coverage:>6.3f} {L.coverage:>6.3f} | {U.se_over_sd_error:>7.2f} {L.se_over_sd_error:>7.2f} | {U.mean_real_cost:>7.0f} {L.mean_real_cost:>7.0f}")
    print()
print("se/sd: the model-based SE is sd(contributions)/sqrt(n) and targets the POPULATION variance of the estimate around Phi;")
print("the error here is against the in-sample full-annotation benchmark, which removes unit-sampling variance, so se/sd > 1 and")
print("coverage vs the benchmark is conservative by construction (the casenote driver's convention, same interpretation).")
