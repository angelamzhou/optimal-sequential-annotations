"""Merge the sharded casenote Monte Carlo runs (Fig. 4) and audit them.

Each cluster array task ran outreach_dtr_annotation_mc.py for 5 replications of one arm and wrote
SHARD_ROOT/<arm>/shard_XX/results.csv (see slurm/casenote_mc_array.sbatch). Replications are seeded by
their global index, so the merge is exact. Checks: every rep 0..99 appears exactly once per arm, rows
per rep are constant, and there are no duplicate keys.

Usage: python src/casenote_pipeline/merge_mc_shards.py SHARD_ROOT   (run from the repo root)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from outreach_dtr_annotation_mc import aggregate_metrics  # noqa: E402

ARMS = {  # cluster arm dir -> merged run dir
    "placement_link": "results/casenote_placement",
    "progress_silveronly": "results/casenote_progress",
}
KEYS = ["rep", "target", "budget", "sampler", "estimator"]


def main() -> None:
    root = Path(sys.argv[1])
    for arm, out in ARMS.items():
        shards = sorted((root / arm).glob("shard_*/results.csv"))
        df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
        reps = sorted(df.rep.unique())
        assert reps == list(range(100)), f"{arm}: reps {reps[:3]}..{reps[-3:]} (n={len(reps)})"
        per_rep = df.groupby("rep").size()
        assert per_rep.nunique() == 1, f"{arm}: uneven rows per rep"
        assert not df.duplicated(KEYS).any(), f"{arm}: duplicate keys"
        Path(out).mkdir(parents=True, exist_ok=True)
        df.sort_values(KEYS).to_csv(Path(out) / "results.csv", index=False)
        aggregate_metrics(df).to_csv(Path(out) / "aggregate_metrics.csv", index=False)
        print(f"{arm}: {len(shards)} shards, {len(df)} rows, reps 0-99, {per_rep.iloc[0]} rows/rep -> {out}")


if __name__ == "__main__":
    main()
