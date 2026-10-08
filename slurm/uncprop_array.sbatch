#!/bin/bash
#SBATCH --job-name=uncprop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --array=0-79
#SBATCH --output=slurm_logs/%x_%A_%a.out
# Add --account / --partition for your cluster. Run from the repository root.
# REQUIRES THE PROTECTED CASENOTE DATA (not distributed); set PROTECTED_DATA to its directory.
# Uncertainty-proportional (active-inference) ablation behind Table 2. Same seeds/configs as the
# paper runs: arm 0 simulator placement, arm 1 simulator progress (simulated cohorts drawn from the
# simulator fit on the protected data), arm 2 casenote progress, arm 3 casenote placement.
# 'random' is rerun as a merge check against the stored runs; the new sampler uses its own RNG, so
# all other samplers' streams are unchanged. 20 shards x 5 reps per arm. Merge with
#   python src/casenote_pipeline/merge_uncprop_shards.py results/uncprop_shards
set -euo pipefail
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
P=${PROTECTED_DATA:?set PROTECTED_DATA to the protected data directory}
t=${SLURM_ARRAY_TASK_ID}
arm=$(( t / 20 )); shard=$(( t % 20 )); start=$(( shard * 5 )); sh=$(printf %02d $shard)
SAMPLERS=random,uncertainty_proportional
SHARD=(--rep-start $start --target-reps 5 --batch-size 5 --jobs 5 --sample-size 600 --budgets 0.30,0.40,0.50,0.60,0.70,0.80 --priority-form theorem)
S=$P/simulated/pool_rbf_n8000_s11
SIM=(--features-csv $S/features.csv --contacts-csv $S/contacts.csv --labels-csv $S/gold_labels.csv
  --silver-labels-csv $S/silver_labels.csv --oracle-propensity-csv $S/simulator_outcomes.csv --date-format %Y-%m-%d --seed 20260910)
REAL=(--features-csv "$P/cohort_features_outcome_cat.csv" --contacts-csv "$P/contact_timestamps.csv"
  --labels-csv "$P/labels_reconciled.csv" --silver-labels-csv "$P/silver_labels.csv"
  --date-format "%m/%d/%y" --h2-progress-source silver --seed 20260512 --propensity-learner ridge_softmax)
case $arm in
  0) $PY src/outreach_dtr_annotation_mc.py "${SIM[@]}" "${SHARD[@]}" --samplers $SAMPLERS \
       --targets y2_any_placement_improvement --variance-estimator link --pilot-fraction 0.3 --output-dir results/uncprop_shards/sim_placement/shard_$sh ;;
  1) $PY src/outreach_dtr_annotation_mc.py "${SIM[@]}" "${SHARD[@]}" --samplers $SAMPLERS \
       --targets progress_total --variance-estimator silver_only --pilot-fraction 0.0 --output-dir results/uncprop_shards/sim_progress/shard_$sh ;;
  2) $PY src/outreach_dtr_annotation_mc.py "${REAL[@]}" "${SHARD[@]}" --samplers $SAMPLERS \
       --targets progress_total --variance-estimator silver_only --pilot-fraction 0.0 --output-dir results/uncprop_shards/real_progress/shard_$sh ;;
  3) $PY src/outreach_dtr_annotation_mc.py "${REAL[@]}" "${SHARD[@]}" --samplers $SAMPLERS \
       --targets progress_total,y2_any_placement_improvement --variance-estimator link --pilot-fraction 0.3 --output-dir results/uncprop_shards/real_placement/shard_$sh ;;
esac
