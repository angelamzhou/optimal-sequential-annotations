#!/bin/bash
#SBATCH --job-name=synth_stagewise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/%x_%j.out
# Add --account / --partition for your cluster. Run from the repository root.
# Synthetic Monte Carlo behind Fig. 3 and Fig. S3 (n = 1000, 100 replications, 7 budgets).
# --stagewise-arm adds the forward-monotone stagewise designs (protocol_stagewise_regime_*,
# joint T=2 solver) beside the trajectory-wise protocol_theorem_regime_* designs.
# About 25 minutes on 25 cores. Purely synthetic: no protected data are read.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
OUT=results/synthetic/paper_two_stage_n1000_theorem_pilot30_reps100_crossfit5_stagewise
$PY src/synthetic/run_incremental_mc_batches.py --output-dir "$OUT" \
  --sample-size 1000 --target-reps 100 --batch-size 25 \
  --budgets 0.20,0.30,0.40,0.50,0.60,0.70,0.80 --validation-reps 3 --jobs 25 \
  --common-oracle-n 1000 --estimators plugin,drl --dgp paper-two-stage --p 5 \
  --floor 0.01 --priority-exponent 0.50 --reward-noise-profile paper-two-stage \
  --heteroskedastic-strength 4.0 --behavior-min-prob 0.05 --behavior-logit-scale 1.5 \
  --seed 20260712 --pilot-fraction 0.3 --crossfit-folds 5 --stagewise-arm --heartbeat-seconds 60
