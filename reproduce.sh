#!/usr/bin/env bash
# Regenerate every figure, table and text number that can be rebuilt from the shipped aggregates.
# Outputs go to output/ (figures as PDF + PNG, plot data, and printed tables in output/*.txt).
# Run from the repository root:  bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${OUT:-output}
mkdir -p "$OUT"
run() { echo "+ $*" >&2; "$@"; }

# Fig. 3 and Fig. S3 (synthetic; plus the main-text numbers for Fig. 3)
run $PY figures/fig_simulated_panels.py --out-dir "$OUT"
run $PY figures/synthetic_gain_curves.py --out-dir "$OUT"
run $PY figures/synthetic_coverage_width.py --out-dir "$OUT"

# Fig. 4 (casenote: placement, progress) and Fig. S4 (progress, stagewise vs trajectory-wise)
run $PY figures/plot_annotation_mc.py --plot-data aggregates/casenote/placement_plot_data.csv \
    --target y2_any_placement_improvement --adaptive-samplers adaptive_trajectory \
    --output-prefix casenote_annotation_mc_any_improvement_drl_link --out-dir "$OUT"
run $PY figures/plot_annotation_mc.py --plot-data aggregates/casenote/progress_plot_data.csv \
    --target progress_total --adaptive-samplers adaptive_stagewise \
    --output-prefix casenote_annotation_mc_progress_drl_silveronly --out-dir "$OUT"
run $PY figures/plot_annotation_mc.py --plot-data aggregates/casenote/progress_plot_data.csv \
    --target progress_total --adaptive-samplers adaptive_stagewise,adaptive_trajectory \
    --output-prefix casenote_annotation_mc_progress_drl_stagewise_vs_trajectory --out-dir "$OUT"

# Fig. 5 (LMArena) and Fig. S5 (simulator-grounded validation)
run $PY figures/plot_arena_panels.py --out-dir "$OUT"
run $PY figures/plot_simulator_validation.py --out-dir "$OUT"

# Tables
run $PY figures/table_llm_judge.py        | tee "$OUT/table1_llm_judge.txt"
run $PY figures/table_active_inference.py | tee "$OUT/table2_active_inference.txt"
run $PY figures/table_arena.py            | tee "$OUT/tableS3_arena.txt"
run $PY figures/bundle_stats.py           | tee "$OUT/figS4_bundle_stats.txt"
run $PY figures/budget_saved.py           | tee "$OUT/budget_saved.txt"
