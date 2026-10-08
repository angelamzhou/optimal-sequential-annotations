"""Merge sharded run_arena_charged.py outputs (results_reps{a}-{b}.csv) into <dest>/results.csv.
Reps are seeded (SEED, rep), so the merge is exact; asserts the rep set is contiguous and unique.
    python src/arena/merge_shards.py charged_v2   (shards in $ARENA_DATA/charged_v2/)"""
import sys, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena_paths import DATA  # noqa: E402
dest = DATA / (sys.argv[1] if len(sys.argv) > 1 else "charged_v2")
parts = sorted(glob.glob(str(dest / "results_reps*.csv")))
res = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).sort_values(["rep", "pi_e", "budget_fraction", "arm"])
reps = sorted(res.rep.unique()); assert reps == list(range(reps[0], reps[-1] + 1)), "non-contiguous reps"
per = res.groupby("rep").size(); assert per.nunique() == 1, "uneven rows per rep"
res.to_csv(dest / "results.csv", index=False)
print(f"merged {len(parts)} shards -> {dest/'results.csv'}: {len(reps)} reps x {per.iloc[0]} rows = {len(res)}")
