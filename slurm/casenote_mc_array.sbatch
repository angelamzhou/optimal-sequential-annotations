#!/bin/bash
#SBATCH --job-name=casenote_mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=1:30:00
#SBATCH --array=0-39
#SBATCH --output=slurm_logs/%x_%A_%a.out
# Add --account / --partition for your cluster. Run from the repository root.
# REQUIRES THE PROTECTED CASENOTE DATA (not distributed); set PROTECTED_DATA to its directory.
# Casenote Monte Carlo behind Fig. 4 / Fig. S4 (100 replications of 600 clients):
# tasks 0-19 = placement arm, 20-39 = progress arm, 5 replications per task. Replications are
# seeded by their global index, so shards merge exactly:
#   python src/casenote_pipeline/merge_mc_shards.py results/casenote_shards
set -euo pipefail
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
P=${PROTECTED_DATA:?set PROTECTED_DATA to the protected data directory}
t=${SLURM_ARRAY_TASK_ID}
arm=$(( t / 20 )); shard=$(( t % 20 )); start=$(( shard * 5 ))
COMMON=(--features-csv "$P/cohort_features_outcome_cat.csv" --contacts-csv "$P/contact_timestamps.csv"
  --labels-csv "$P/labels_reconciled.csv" --silver-labels-csv "$P/silver_labels.csv"
  --date-format "%m/%d/%y" --h2-progress-source silver --priority-form theorem --seed 20260512
  --sample-size 600 --budgets 0.30,0.40,0.50,0.60,0.70,0.80 --rep-start $start --target-reps 5 --batch-size 5
  --jobs 5 --propensity-learner ridge_softmax)
if [ $arm -eq 0 ]; then
  $PY src/outreach_dtr_annotation_mc.py "${COMMON[@]}" --variance-estimator link --pilot-fraction 0.3 \
    --samplers random,adaptive_trajectory --targets progress_total,y2_any_placement_improvement \
    --output-dir results/casenote_shards/placement_link/shard_$(printf %02d $shard)
else
  $PY src/outreach_dtr_annotation_mc.py "${COMMON[@]}" --variance-estimator silver_only --pilot-fraction 0.0 \
    --samplers random,adaptive_trajectory,adaptive_stagewise --targets progress_total \
    --output-dir results/casenote_shards/progress_silveronly/shard_$(printf %02d $shard)
fi
