# Optimal Sequential Annotations for Off-Policy Evaluation: code and data release

Code and aggregate results for the paper *Optimal Sequential Annotations for Off-Policy Evaluation*
(PNAS submission). The paper studies how to spend a limited budget of expensive reward labels
(annotations) across the stages of a sequential decision problem so that a doubly robust off-policy
estimate of a target policy's value is as precise as possible. It illustrates the method on synthetic
data, on casenotes from a homelessness street-outreach program, and on LMArena chatbot battles.

On the casenote data, a stage's progress reward is the maximum progress label over a client's casenotes in that
six-month stage. The labels come from a fine-tuned language model, with the maximum-label casenote in each client-stage and a
sample of near-maximum casenotes ground-truthed (SI, *Reconciled progress reward*). These reconciled labels are the progress
reward in every casenote result and are part of the protected data. The placement
outcome does not use the progress labels.

## What is here

| Path | Contents |
|---|---|
| `src/` | The method and the experiment drivers: optimal two-stage allocation (`t2_optimal_allocation.py`), the casenote off-policy evaluation (`outreach_dtr_reward_only.py`) and annotation Monte Carlo (`outreach_dtr_annotation_mc.py`) drivers, the simulator used for validation (`mdp_simulator_dgp.py`), and the sampler names (`sampler_names.py`). |
| `src/synthetic/` | Synthetic-data Monte Carlo (Fig. 3, Fig. S3). Purely simulated; fully regenerable. |
| `src/arena/` | LMArena pipeline (Fig. 5, Table S3): manifest, reward-model and text features, charged batch-adaptive Monte Carlo. |
| `src/labeler_comparison/` | Design-weighted comparison of human and LLM labels (Table 1). Requires the protected data. |
| `src/casenote_pipeline/` | Merging the cluster shards of the casenote Monte Carlo, and the cohort descriptives (Tables S1 and S2). Requires the protected data. |
| `figures/` | One script per paper figure or table. Each one runs from the shipped aggregates. Also `compare_to_paper.py`. |
| `aggregates/` | Shipped inputs, grouped by experiment: `synthetic/`, `casenote/`, `simulator_validation/`, `arena/` and `labeler_comparison/`. See `RELEASE_MANIFEST.md` for every file's size, rows, columns and provenance. |
| `paper_figs/` | The figure PDFs as they appear in the paper, for comparison with regenerated output. |
| `slurm/` | Batch scripts for the Monte Carlo runs. Add your own account and partition. |
| `reproduce.sh` | Regenerates every figure, table and text number that can be rebuilt from `aggregates/`, writing to `output/`. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12 was used
pip install -r requirements.txt
bash reproduce.sh                                       # about 1 minute; writes output/
python figures/compare_to_paper.py                      # optional; needs poppler's pdftoppm
```

Run every command from the repository root.

## Paper figures and tables

Numbering follows the PDF. "Shipped" means the result regenerates from `aggregates/` alone.

| Paper item | Script | Command | Aggregate inputs | Regenerates from shipped data? |
|---|---|---|---|---|
| Table 1 (2019 casenote shares by progress level: human vs LLM) | `figures/table_llm_judge.py` | `python figures/table_llm_judge.py` | `labeler_comparison/labeler_shares_2019.csv`, `labeler_share_diffs_2019.csv` | Yes. The producer `src/labeler_comparison/table_labeler_shares.py` needs the protected data. |
| Fig. 1 (sequential annotation model), Fig. 2 (forward-monotone annotation) | none | none | none | Schematics drawn in the paper source |
| Fig. 3 (simulated data, policy (0,0): RMSE and interval width) | `figures/fig_simulated_panels.py` | `python figures/fig_simulated_panels.py` produces `synth_d00_{rmse,width}.pdf` | `synthetic/results.csv` | Yes. Rerun from scratch with `slurm/synthetic_stagewise.sbatch`. |
| Fig. 3 text (42% / 12% RMSE reduction, coverage, 12–41% narrower) | `figures/synthetic_gain_curves.py`, `figures/synthetic_coverage_width.py` | `python figures/synthetic_gain_curves.py`; `python figures/synthetic_coverage_width.py` | `synthetic/results.csv` | Yes |
| Fig. 4 (casenote data: placement and progress, RMSE and interval width) | `figures/plot_annotation_mc.py` | See `reproduce.sh`: run it with `--plot-data` on each of `aggregates/casenote/placement_plot_data.csv` (`--adaptive-samplers adaptive_trajectory`) and `progress_plot_data.csv` (`--adaptive-samplers adaptive_stagewise`) | `casenote/placement_plot_data.csv`, `casenote/progress_plot_data.csv` | Yes, from the budget-level plot data. The per-replication runs need the protected data (below). |
| Table 2 (optimized design vs uncertainty-proportional baseline) | `figures/table_active_inference.py` | `python figures/table_active_inference.py` | `casenote/active_inference_summary.csv` | Yes, from the budget-level summary |
| Fig. 5 (LMArena: equal exposure, RMSE and interval width; the tilt policy is reported only in Table S3) | `figures/plot_arena_panels.py` | `python figures/plot_arena_panels.py` | `arena/charged_v2_results.csv` | Yes |
| Fig. S1 (reward as cumulative history annotation) | none | none | none | Schematic in the paper source |
| Fig. S2 (batch-adaptive experiment with cross-fitting) | none | none | none | Schematic, `paper_figs/CSBAE-cross-fitting-onecol-v2.pdf` |
| Fig. S3 (simulated data, all four policies: RMSE, coverage, width) | `figures/fig_simulated_panels.py` | `python figures/fig_simulated_panels.py` produces `synth_d{00,01,10,11}_{rmse,coverage,width}.pdf` | `synthetic/results.csv` | Yes |
| Table S1 (outreach quintiles), Table S2 (cohort descriptives) | `src/casenote_pipeline/table_cohort_descriptives.py` | below | none shipped | **No.** These are cohort descriptives of the protected data. |
| SI, Reconciled progress reward | none | none | none shipped | **No.** It describes how the protected labels were ground-truthed. |
| Fig. S4 (progress task: stagewise vs trajectory-wise design) | `figures/plot_annotation_mc.py`, `figures/bundle_stats.py` | `reproduce.sh` (`--adaptive-samplers adaptive_stagewise,adaptive_trajectory`); `python figures/bundle_stats.py` | `casenote/progress_plot_data.csv`, `casenote/bundle_stats.csv` | Yes, from the budget-level aggregates. The paired bootstrap is precomputed. |
| Fig. S5 (simulator-grounded validation) | `figures/plot_simulator_validation.py` | `python figures/plot_simulator_validation.py` | `simulator_validation/sim_validation_{placement,progress}_summary.csv` | Yes, from the per-budget summaries. The simulator itself is fit to the protected data. |
| Table S3 (LMArena RMSE and interval width) | `figures/table_arena.py` | `python figures/table_arena.py` | `arena/charged_v2_summary_error_based.csv` | Yes |
| Text: equal-width label savings (26–31%, 25–28%, 37–48%) | `figures/budget_saved.py` | `python figures/budget_saved.py` | `casenote/*_plot_data.csv`, `arena/charged_v2_summary_error_based.csv` | Yes |
| Text: full-cohort policy effects (progress +0.30, 3.92 → 4.22; placement 22.5% → 27.9%, +5.4 pp; single-stage shifts) | none | read the CSVs directly | `casenote/value_estimates_{both_stages,stage1_only,stage2_only}.csv` | Yes, as shipped estimates. Producing them needs the protected data (below). |

The PDFs in `output/` match `paper_figs/` in their data. A raster comparison with `figures/compare_to_paper.py` finds identical pixels for Figs. 3, 4, S3 and S4 and the placement row of Fig. S5. Fig. 5 and the progress row of Fig. S5 differ only in text antialiasing (font subsetting). All regenerated plot-data CSVs equal the shipped references exactly.

## What needs the protected data

The casenote data are records of a homelessness-services nonprofit and cannot be shared (see the data
availability statement below). The following are released **as code only**: their inputs and their
per-client and per-replication outputs are withheld. `<PROTECTED>` stands for the directory holding the
protected files:

- `labels_reconciled.csv`: the reconciled progress labels used as the progress reward
- `labels.csv`: progress labels from the fine-tuned model (named in the shipped placement run configuration; the placement results do not depend on the progress labels)
- `silver_labels.csv`: base-model silver labels
- `cohort_features_outcome_cat.csv`
- `contact_timestamps.csv`

Most scripts default to `protected_data/<file>` under the repository root.

**Full-cohort policy effects (main text):**

```bash
python src/outreach_dtr_reward_only.py --labels-csv <PROTECTED>/labels_reconciled.csv --date-format "%m/%d/%y" --output-dir results/fig4
python src/outreach_dtr_reward_only.py --labels-csv <PROTECTED>/labels_reconciled.csv --date-format "%m/%d/%y" --output-dir results/fig4_stage1only --shift-quintiles-stage1 1 --shift-quintiles-stage2 0
python src/outreach_dtr_reward_only.py --labels-csv <PROTECTED>/labels_reconciled.csv --date-format "%m/%d/%y" --output-dir results/fig4_stage2only --shift-quintiles-stage1 0 --shift-quintiles-stage2 1
```

Each run writes `value_estimates.csv`, which is shipped as `aggregates/casenote/value_estimates_{both_stages,stage1_only,stage2_only}.csv`. The features and contacts files are read from their defaults, `protected_data/cohort_features_outcome_cat.csv` and `protected_data/contact_timestamps.csv`. Pass `--features-csv` and `--contacts-csv` to point elsewhere.

**Casenote annotation Monte Carlo (Fig. 4, Fig. S4).** This runs 100 replications of 600 clients, sharded on a cluster:

```bash
PROTECTED_DATA=<PROTECTED> sbatch slurm/casenote_mc_array.sbatch          # 40 tasks: placement (0-19), progress (20-39); reads labels_reconciled.csv
python src/casenote_pipeline/merge_mc_shards.py results/casenote_shards   # -> results/casenote_{placement,progress}/
python figures/plot_annotation_mc.py --run-dir results/casenote_placement --target y2_any_placement_improvement \
    --adaptive-samplers adaptive_trajectory --output-prefix casenote_annotation_mc_any_improvement_drl_link
python figures/plot_annotation_mc.py --run-dir results/casenote_progress --target progress_total \
    --adaptive-samplers adaptive_stagewise,adaptive_trajectory --output-prefix casenote_annotation_mc_progress_drl_stagewise_vs_trajectory
python figures/bundle_stats.py --results-dir results/casenote_progress
```

The run configurations are in `aggregates/casenote/{placement,progress}_run_config_shard00.json`.
The paper's casenote Monte Carlo numbers come from cluster job runs of these drivers. The per-replication outputs are withheld; only the budget-level summaries in `aggregates/casenote/` are released.
The drivers shipped here are the current versions. They differ from the as-run copies only in naming and refactoring; the sampler random-number streams are unchanged.

**Uncertainty-proportional baseline (Table 2):**

```bash
PROTECTED_DATA=<PROTECTED> sbatch slurm/uncprop_array.sbatch
python src/casenote_pipeline/merge_uncprop_shards.py results/uncprop_shards
python figures/table_active_inference.py --from-runs results/casenote_progress_uncprop results/casenote_placement_uncprop
```

**Simulator-grounded validation (Fig. S5).** The simulator is fit to the protected cohort, and its generated cohorts resample real baseline rows, so the simulator, the cohorts and the per-replication runs are all withheld. The commands below are reconstructed from the stored run configurations:

```bash
python src/mdp_simulator_dgp.py fit --simulator-kind rbf        # reads <PROTECTED> via the defaults in src/
python src/mdp_simulator_dgp.py generate --bundle protected_data/simulated/simulator_rbf.joblib --n-clients 8000 --seed 11 \
    --silver-mode impute_mix --silver-beta 0.7 --out-dir protected_data/simulated/pool_rbf_n8000_s11
python src/mdp_simulator_dgp.py truth --bundle protected_data/simulated/simulator_rbf.joblib --n-rollouts 100000 \
    --silver-mode impute_mix --out results/simulator_validation/truth_rbf_impute_mix.json
P=protected_data/simulated/pool_rbf_n8000_s11
SIM="--features-csv $P/features.csv --contacts-csv $P/contacts.csv --labels-csv $P/gold_labels.csv --silver-labels-csv $P/silver_labels.csv \
     --oracle-propensity-csv $P/simulator_outcomes.csv --date-format %Y-%m-%d --seed 20260910 --sample-size 600 --target-reps 100 --batch-size 25"
python src/outreach_dtr_annotation_mc.py $SIM --targets y2_any_placement_improvement --variance-estimator link --pilot-fraction 0.3 \
    --samplers random,adaptive_trajectory --output-dir results/simulator_validation/placement
python src/outreach_dtr_annotation_mc.py $SIM --targets progress_total --variance-estimator silver_only --pilot-fraction 0 \
    --samplers random,adaptive_trajectory,adaptive_stagewise --output-dir results/simulator_validation/progress
python figures/plot_simulator_validation.py --from-runs --placement-run results/simulator_validation/placement \
    --progress-run results/simulator_validation/progress --truth results/simulator_validation/truth_rbf_impute_mix.json
```

**Table 1 and cohort descriptives (Tables S1, S2):**

```bash
PROTECTED_DATA_ROOT=<PROTECTED_ROOT> python src/labeler_comparison/table_labeler_shares.py
python src/casenote_pipeline/table_cohort_descriptives.py
```

## Synthetic data (fully regenerable)

`aggregates/synthetic/results.csv` is a trimmed copy of the synthetic Monte Carlo output: DRL and plug-in rows for random annotation, the four stagewise designs and full annotation, keeping the 16 columns that the figures use. To regenerate the full run from scratch (about 25 minutes on 25 cores):

```bash
sbatch slurm/synthetic_stagewise.sbatch        # or run its python command directly
python figures/fig_simulated_panels.py --run-dir results/synthetic/paper_two_stage_n1000_theorem_pilot30_reps100_crossfit5_stagewise
```

## LMArena (public data)

The Monte Carlo results used for Fig. 5 and Table S3 are shipped (`aggregates/arena/charged_v2_results.csv`; no session identifiers).
The per-session manifest (about 48 MB) and the reward-model and TF-IDF feature files are not shipped. The summary values the paper quotes from the manifest are in `aggregates/arena/arena_paper_numbers.json`: Φ^πb, the date range and the model counts.

To rebuild every input from Hugging Face, set `ARENA_DATA` to a working directory (default `data/arena/`) and run the following.

1. Download [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) (CC-BY-4.0) to `$ARENA_DATA/arena140k.parquet`. You need columns `id, model_a, model_b, winner, evaluation_session_id, evaluation_order, language, is_code, timestamp, category_tag, conv_metadata`.
2. Build the manifest and policy menu:

   ```bash
   python src/arena/build_arena_manifest.py
   python src/arena/arena_policy_menu.py
   ```

3. Compute the reward-model scores with `Skywork/Skywork-Reward-V2-Llama-3.1-8B` on a GPU. Run `src/arena/score_reward_model.py` and move `arena_rm_scores.parquet` to `$ARENA_DATA`.
4. Build the hashed text features with `src/arena/hash_text_features.py`, producing `$ARENA_DATA/arena_text_hash/`.
5. Optionally, build the TF-IDF scores with `src/arena/build_arena_tfidf_scores.py`.
6. Run the Monte Carlo and summarize it:

   ```bash
   python src/arena/run_arena_charged.py --dest charged_v2 --reps 1000
   python src/arena/summarize_charged.py $ARENA_DATA/charged_v2
   ```

   To run in shards, use `--rep-start` and then `src/arena/merge_shards.py`.

`src/arena/ppe_tfidf_pilot.py` supplies the text model (`fit_text`). Its standalone pilot reads the public PPE-Human-Preference-V1 data from `$ARENA_DATA/ppe/`.

## Data availability statement

As stated in the paper:

> Code for all methods and analyses is publicly available at [repository URL]. The LMArena data are publicly
> available (https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k). The casenote data are
> restricted to protect client privacy and cannot be shared. Details of the simulator are available from the
> authors on request.

## Naming

Designs (samplers) use the names defined in `src/sampler_names.py`. `random` is uniform random annotation. `adaptive_stagewise` is the forward-monotone stagewise design of the T=2 theorem. `adaptive_trajectory` is the trajectory-wise design. `uncertainty_proportional` is the active-inference baseline.

## License

MIT (see `LICENSE`).
