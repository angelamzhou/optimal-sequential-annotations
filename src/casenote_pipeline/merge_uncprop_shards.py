"""Merge the uncertainty-proportional (active-inference) ablation shards into the paper runs.

The uncertainty-proportional cluster job (2026-10-04) reran 'random' and the new 'uncertainty_proportional'
sampler with the same seeds/configs as the four paper runs below. The new sampler draws from its own
RNG, so every other sampler's rows are unchanged; 'random' is the merge check. Stored runs are not
modified: merged results go to <stored run>_uncprop/results.csv.

Usage: python src/casenote_pipeline/merge_uncprop_shards.py SHARD_ROOT   (run from the repo root)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from sampler_names import UNCERTAINTY_PROPORTIONAL, canonicalize_raw_driver_output  # noqa: E402

SIM = "results/mdp_simulator_validation/"
ARMS = {  # cluster arm dir -> stored paper run
    "sim_placement": SIM + "mc_rbf_n8000_placement_link_pilot30_oracleg",
    "sim_progress": SIM + "mc_rbf_n8000_progress_silveronly_oracleg_prefix",
    "real_progress": "results/casenote_progress",
    "real_placement": "results/casenote_placement",
}
KEYS = ["rep", "target", "budget", "sampler", "estimator"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.estimator.eq("drl")]
    g = d.groupby(["target", "budget", "sampler"]).agg(
        rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        width=("model_based_ci_width_95", "mean"),
    ).reset_index()
    base = g[g.sampler.eq("random")].set_index(["target", "budget"])[["rmse", "width"]]
    g = g.join(base, on=["target", "budget"], rsuffix="_random")
    g["rmse_vs_random"] = g.rmse / g.rmse_random - 1
    g["width_vs_random"] = g.width / g.width_random - 1
    return g.drop(columns=["rmse_random", "width_random"])


def main() -> None:
    root = Path(sys.argv[1])
    for arm, stored in ARMS.items():
        shards = sorted((root / arm).glob("shard_*/results.csv"))
        new = canonicalize_raw_driver_output(pd.concat([pd.read_csv(f) for f in shards], ignore_index=True))
        reps = sorted(new.rep.unique())
        assert reps == list(range(100)), f"{arm}: reps {reps[:3]}..{reps[-3:]} (n={len(reps)})"
        assert not new.duplicated(KEYS).any(), f"{arm}: duplicate keys"
        old = canonicalize_raw_driver_output(pd.read_csv(Path(stored) / "results.csv"))
        m = new[new.sampler.eq("random")].merge(old, on=KEYS, suffixes=("", "_old"))
        assert len(m) == int(new.sampler.eq("random").sum()), f"{arm}: random rows missing from stored run"
        diff = float(np.abs(m.estimate - m.estimate_old).max())
        bench = float(np.abs(m.full_reward_benchmark - m.full_reward_benchmark_old).max())
        assert diff < 1e-3 and bench < 1e-3, f"{arm}: random rows do not reproduce the stored run ({diff:.2e})"
        merged = pd.concat([old, new[new.sampler.eq(UNCERTAINTY_PROPORTIONAL)]], ignore_index=True)
        out = Path(stored + "_uncprop")
        out.mkdir(parents=True, exist_ok=True)
        merged.sort_values(KEYS).to_csv(out / "results.csv", index=False)
        summary = summarize(merged)
        summary.to_csv(out / "summary_drl_vs_random.csv", index=False)
        print(f"\n== {arm}: {len(shards)} shards, reps 0-99; random reproduces stored run "
              f"(max|est diff|={diff:.1e}, max|benchmark diff|={bench:.1e}) -> {out}")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
