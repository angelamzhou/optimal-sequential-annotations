#!/usr/bin/env python3
"""Run optimized annotation Monte Carlo replications in resumable batches.

The base runner writes a complete MC run at the end. This wrapper runs a fixed
sample size in replication batches, keeps pilot/validation settings fixed, and
uses --rep-offset so every completed batch contributes distinct global MC reps.
Rerunning with a larger --target-reps, or with --add-reps, only launches the
missing batches and then recomputes root summaries from all completed raw rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CORE_BATCH_FILES = ("results.csv", "aggregate_metrics.csv", "metrics_by_regime.csv", "run_metadata.csv")
FIRST_BATCH_FILES = ("validation_sampler_metrics.csv", "selected_sampler_hyperparameters.csv")
VARIANCE_SUMMARY_AGG = {
    "n_units": "mean",
    "r1_noise_var_mean": "mean",
    "r2_noise_var_mean": "mean",
    "reward_noise_var_mean": "mean",
    "reward_noise_sd_mean": "mean",
    "reward_noise_sd_min": "mean",
    "reward_noise_sd_max": "mean",
    "r1_mean_var": "mean",
    "r2_mean_var": "mean",
    "reward_mean_var": "mean",
    "outcome_var": "mean",
    "x2_empirical_var_mean_diag": "mean",
    "transition_conditional_noise_var_diag": "mean",
    "transition_conditional_noise_adjacent_cov": "mean",
    "dtr_priority_mean": "mean",
    "priority_noise_var_corr": "mean",
}

_CHILD: subprocess.Popen[str] | None = None
_INTERRUPTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_dir_name(start_rep: int, reps: int) -> str:
    return f"reps_{start_rep:06d}_{start_rep + reps - 1:06d}"


def batch_complete(batch_dir: Path) -> bool:
    manifest_path = batch_dir / "batch_manifest.json"
    return manifest_path.exists() and all(
        (batch_dir / name).exists() and (batch_dir / name).stat().st_size > 0 for name in CORE_BATCH_FILES
    )


def completed_manifests(batches_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not batches_dir.exists():
        return manifests
    for manifest_path in sorted(batches_dir.glob("*/batch_manifest.json")):
        batch_dir = manifest_path.parent
        if not batch_complete(batch_dir):
            continue
        manifest = load_json(manifest_path)
        manifest["batch_dir"] = str(batch_dir)
        manifests.append(manifest)
    return sorted(manifests, key=lambda row: (int(row["start_rep"]), int(row["reps"])))


def contiguous_completed_reps(manifests: list[dict[str, Any]]) -> int:
    cursor = 0
    for manifest in manifests:
        start = int(manifest["start_rep"])
        reps = int(manifest["reps"])
        if start < cursor:
            cursor = max(cursor, start + reps)
            continue
        if start != cursor:
            break
        cursor += reps
    return cursor


def signal_handler(signum: int, _frame: object) -> None:
    global _CHILD, _INTERRUPTED
    _INTERRUPTED = True
    if _CHILD is not None and _CHILD.poll() is None:
        try:
            os.killpg(_CHILD.pid, signal.SIGTERM)
        except Exception:
            _CHILD.terminate()


def load_optimized_runner(script_dir: Path):
    runner = script_dir / "optimized_annotation_krr_eval.py"
    spec = importlib.util.spec_from_file_location("optimized_annotation_krr_eval", runner)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {runner}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_batch_columns(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    start_rep = int(manifest["start_rep"])
    reps = int(manifest["reps"])
    batch_name = str(manifest["batch_name"])
    frame.insert(0, "mc_batch", batch_name)
    frame.insert(1, "batch_start_rep", start_rep)
    frame.insert(2, "batch_reps", reps)
    if "rep" in frame.columns:
        frame["global_rep"] = frame["rep"].astype(int)
        frame["batch_rep"] = frame["global_rep"] - start_rep
    return frame


def sort_if_present(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    return frame.sort_values(present) if present else frame


def combine_batches(output_dir: Path, manifests: list[dict[str, Any]]) -> None:
    if not manifests:
        return
    script_dir = Path(__file__).resolve().parent
    optimized = load_optimized_runner(script_dir)

    result_frames: list[pd.DataFrame] = []
    variance_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for manifest in manifests:
        batch_dir = Path(manifest["batch_dir"])
        result_frames.append(add_batch_columns(pd.read_csv(batch_dir / "results.csv"), manifest))
        variance_file = batch_dir / "variance_profile.csv"
        if variance_file.exists() and variance_file.stat().st_size > 0:
            variance_frames.append(add_batch_columns(pd.read_csv(variance_file), manifest))
        metadata_file = batch_dir / "run_metadata.csv"
        row: dict[str, Any] = {}
        if metadata_file.exists() and metadata_file.stat().st_size > 0:
            metadata = pd.read_csv(metadata_file)
            row = metadata.iloc[0].to_dict() if not metadata.empty else {}
        row.update(
            {
                "batch_name": manifest["batch_name"],
                "batch_dir": manifest["batch_dir"],
                "start_rep": int(manifest["start_rep"]),
                "reps": int(manifest["reps"]),
                "seed": int(manifest["seed"]),
                "completed_at": manifest.get("completed_at", ""),
            }
        )
        metadata_rows.append(row)

    results = pd.concat(result_frames, ignore_index=True)
    results = sort_if_present(
        results,
        ["global_rep", "sample_size", "budget", "estimator", "sampler", "method", "d1", "d2"],
    )
    results.to_csv(output_dir / "results.csv", index=False)

    overall, by_regime = optimized.aggregate_metrics(results)
    overall.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    by_regime.to_csv(output_dir / "metrics_by_regime.csv", index=False)
    try:
        optimized.plot_rmse(overall, output_dir / "rmse_by_method.png")
    except Exception as exc:  # Plotting should not invalidate completed MC rows.
        (output_dir / "plot_rmse_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    if variance_frames:
        variance_profile = pd.concat(variance_frames, ignore_index=True)
        variance_profile = sort_if_present(variance_profile, ["global_rep", "sample_size", "path"])
        variance_profile.to_csv(output_dir / "variance_profile.csv", index=False)
        agg = {column: reducer for column, reducer in VARIANCE_SUMMARY_AGG.items() if column in variance_profile.columns}
        if agg:
            variance_summary = (
                variance_profile.groupby(["sample_size", "path"], as_index=False)
                .agg(**{column: (column, reducer) for column, reducer in agg.items()})
                .sort_values(["sample_size", "path"])
            )
            variance_summary.to_csv(output_dir / "variance_profile_summary.csv", index=False)

    first_batch_dir = Path(manifests[0]["batch_dir"])
    for name in FIRST_BATCH_FILES:
        source = first_batch_dir / name
        if source.exists() and source.stat().st_size > 0:
            pd.read_csv(source).to_csv(output_dir / name, index=False)

    pd.DataFrame(metadata_rows).sort_values(["start_rep", "reps"]).to_csv(output_dir / "batch_run_metadata.csv", index=False)
    completed_reps = contiguous_completed_reps(manifests)
    pd.DataFrame.from_records(
        [
            {
                "completed_reps": completed_reps,
                "completed_batches": len(manifests),
                "first_rep": min(int(row["start_rep"]) for row in manifests),
                "last_rep_exclusive": max(int(row["start_rep"]) + int(row["reps"]) for row in manifests),
                "combined_at": utc_now(),
            }
        ]
    ).to_csv(output_dir / "incremental_run_metadata.csv", index=False)


def write_summary(output_dir: Path, args: argparse.Namespace, completed_reps: int, target_reps: int, elapsed: float) -> None:
    lines = [
        "# Incremental MC optimized annotation run",
        "",
        f"Status: {'completed' if completed_reps >= target_reps else 'partial'}",
        f"Updated at: {utc_now()}",
        f"Elapsed wrapper seconds: {elapsed:.3f}",
        "",
        "## Incremental design",
        f"- Sample size: `{args.sample_size}`",
        f"- Completed contiguous MC reps: `{completed_reps}`",
        f"- Target MC reps: `{target_reps}`",
        f"- Batch size: `{args.batch_size}`",
        "- Resume rule: rerun with a larger `--target-reps` or use `--add-reps`; completed `batches/reps_*` directories are skipped.",
        "- Replication identity: the base runner receives `--rep-offset`, so the row-level `rep` column is global across batches.",
        "",
        "## Experiment config",
        f"- DGP: `{args.dgp}`",
        f"- Estimators: `{args.estimators}`",
        f"- Budgets: `{args.budgets}`",
        f"- Seed: `{args.seed}`",
        f"- Jobs per batch: `{args.jobs}`",
        f"- Common oracle n: `{args.common_oracle_n}`",
        f"- Heteroskedastic strength: `{args.heteroskedastic_strength}`",
        f"- Behavior min prob: `{args.behavior_min_prob}`",
        f"- Behavior logit scale: `{args.behavior_logit_scale}`",
        "",
        "## Outputs",
        "- `results.csv`: combined raw MC rows with `global_rep`, `mc_batch`, and `batch_rep` columns.",
        "- `aggregate_metrics.csv` and `metrics_by_regime.csv`: recomputed from combined raw rows.",
        "- `incremental_status.json`: heartbeat/status file for monitoring and restarts.",
        "- `batch_run_metadata.csv`: per-batch elapsed wall times and offsets.",
    ]
    (output_dir / "incremental_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_batch_command(args: argparse.Namespace, batch_dir: Path, start_rep: int, reps: int) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "optimized_annotation_krr_eval.py"
    command = [
        sys.executable,
        str(runner),
        "--output-dir",
        str(batch_dir),
        "--sample-sizes",
        str(args.sample_size),
        "--reps",
        str(reps),
        "--rep-offset",
        str(start_rep),
        "--validation-reps",
        str(args.validation_reps),
        "--jobs",
        str(args.jobs),
        "--pilot-n",
        str(args.pilot_n),
        "--pilot-fraction",
        str(args.pilot_fraction),
        "--crossfit-folds",
        str(args.crossfit_folds),
        *(["--stagewise-arm"] if args.stagewise_arm else []),
        "--p",
        str(args.p),
        "--regimes",
        args.regimes,
        "--estimators",
        args.estimators,
        "--common-oracle-n",
        str(args.common_oracle_n),
        "--budgets",
        args.budgets,
        "--floor",
        str(args.floor),
        "--priority-exponent",
        str(args.priority_exponent),
        "--dgp",
        args.dgp,
        "--reward-noise-profile",
        args.reward_noise_profile,
        "--heteroskedastic-strength",
        str(args.heteroskedastic_strength),
        "--behavior-min-prob",
        str(args.behavior_min_prob),
        "--behavior-logit-scale",
        str(args.behavior_logit_scale),
        "--seed",
        str(args.seed),
        "--krr-lambdas",
        args.krr_lambdas,
        "--balancing-rff-features",
        str(args.balancing_rff_features),
        "--balancing-cap-multiplier",
        str(args.balancing_cap_multiplier),
        "--balancing-balance-penalty",
        str(args.balancing_balance_penalty),
        "--balancing-chi2-penalty",
        str(args.balancing_chi2_penalty),
        "--balancing-optimizer-maxiter",
        str(args.balancing_optimizer_maxiter),
    ]
    return command


def run_batch(args: argparse.Namespace, output_dir: Path, status_path: Path, start_rep: int, reps: int, target_reps: int, wrapper_started: float) -> int:
    global _CHILD
    batches_dir = output_dir / "batches"
    batch_name = batch_dir_name(start_rep, reps)
    batch_dir = batches_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "batch.log"
    command = build_batch_command(args, batch_dir, start_rep, reps)
    manifest = {
        "batch_name": batch_name,
        "start_rep": start_rep,
        "reps": reps,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "target_reps_at_launch": target_reps,
        "command": command,
        "launched_at": utc_now(),
    }
    write_json_atomic(batch_dir / "batch_manifest.json", manifest)

    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n## Batch {batch_name} launched at {utc_now()}\n")
        log_file.write("Command: " + " ".join(command) + "\n")
        log_file.flush()
        _CHILD = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        while _CHILD.poll() is None:
            manifests = completed_manifests(output_dir / "batches")
            completed = contiguous_completed_reps(manifests)
            write_json_atomic(
                status_path,
                {
                    "status": "running",
                    "output_dir": str(output_dir),
                    "target_reps": target_reps,
                    "completed_reps": completed,
                    "completed_batches": len(manifests),
                    "current_batch": batch_name,
                    "current_batch_start_rep": start_rep,
                    "current_batch_reps": reps,
                    "current_command": command,
                    "current_log": str(log_path),
                    "wrapper_elapsed_seconds": time.perf_counter() - wrapper_started,
                },
            )
            time.sleep(args.heartbeat_seconds)
    return_code = _CHILD.returncode
    _CHILD = None
    manifest = load_json(batch_dir / "batch_manifest.json")
    manifest["completed_at"] = utc_now()
    manifest["return_code"] = return_code
    write_json_atomic(batch_dir / "batch_manifest.json", manifest)
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--target-reps", type=int, default=100, help="Total contiguous MC reps desired after the run.")
    parser.add_argument("--add-reps", type=int, default=0, help="Add this many reps after the current contiguous completed prefix.")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--budgets", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--dgp", default="paper-two-stage")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--regimes", default="0,0;1,0;0,1;1,1")
    parser.add_argument("--validation-reps", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--pilot-n", type=int, default=0)
    parser.add_argument("--stagewise-arm", action="store_true", help="Add forward-monotone stagewise per-regime designs in the evaluator.")
    parser.add_argument("--crossfit-folds", type=int, default=0, help="K-fold cross-fitting of KRR nuisances in the evaluator; 0 = legacy in-sample.")
    parser.add_argument("--pilot-fraction", type=float, default=0.3, help="Charged in-sample pilot fraction for variance-model labels; 0 = uncharged external pilot.")
    parser.add_argument("--common-oracle-n", type=int, default=1000)
    parser.add_argument("--estimators", default="plugin,drl")
    parser.add_argument("--floor", type=float, default=0.01)
    parser.add_argument("--priority-exponent", type=float, default=0.50)
    parser.add_argument("--reward-noise-profile", default="paper-two-stage")
    parser.add_argument("--heteroskedastic-strength", type=float, default=4.0)
    parser.add_argument("--behavior-min-prob", type=float, default=0.05)
    parser.add_argument("--behavior-logit-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--krr-lambdas", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1")
    parser.add_argument("--balancing-rff-features", type=int, default=30)
    parser.add_argument("--balancing-cap-multiplier", type=float, default=6.0)
    parser.add_argument("--balancing-balance-penalty", type=float, default=25.0)
    parser.add_argument("--balancing-chi2-penalty", type=float, default=0.1)
    parser.add_argument("--balancing-optimizer-maxiter", type=int, default=120)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.target_reps <= 0:
        raise ValueError("--target-reps must be positive.")
    if args.add_reps < 0:
        raise ValueError("--add-reps must be nonnegative.")
    if args.sample_size > args.common_oracle_n > 0:
        raise ValueError("--common-oracle-n must be at least --sample-size when positive.")

    output_dir = args.output_dir
    batches_dir = output_dir / "batches"
    status_path = output_dir / "incremental_status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    wrapper_started = time.perf_counter()
    manifests = completed_manifests(batches_dir)
    completed = contiguous_completed_reps(manifests)
    target_reps = completed + args.add_reps if args.add_reps else args.target_reps
    if target_reps < completed:
        target_reps = completed

    write_json_atomic(
        status_path,
        {
            "status": "running",
            "output_dir": str(output_dir),
            "sample_size": args.sample_size,
            "target_reps": target_reps,
            "completed_reps": completed,
            "completed_batches": len(manifests),
            "batch_size": args.batch_size,
            "budgets": args.budgets,
            "estimators": args.estimators,
            "seed": args.seed,
            "started_at": utc_now(),
        },
    )

    if manifests:
        combine_batches(output_dir, manifests)

    cursor = completed
    while cursor < target_reps:
        reps = min(args.batch_size, target_reps - cursor)
        batch_name = batch_dir_name(cursor, reps)
        batch_dir = batches_dir / batch_name
        if batch_complete(batch_dir):
            manifests = completed_manifests(batches_dir)
            cursor = contiguous_completed_reps(manifests)
            combine_batches(output_dir, manifests)
            continue
        return_code = run_batch(args, output_dir, status_path, cursor, reps, target_reps, wrapper_started)
        if _INTERRUPTED:
            manifests = completed_manifests(batches_dir)
            completed = contiguous_completed_reps(manifests)
            if manifests:
                combine_batches(output_dir, manifests)
            write_json_atomic(
                status_path,
                {
                    "status": "interrupted",
                    "output_dir": str(output_dir),
                    "target_reps": target_reps,
                    "completed_reps": completed,
                    "completed_batches": len(manifests),
                    "wrapper_elapsed_seconds": time.perf_counter() - wrapper_started,
                },
            )
            return 130
        if return_code != 0:
            manifests = completed_manifests(batches_dir)
            completed = contiguous_completed_reps(manifests)
            if manifests:
                combine_batches(output_dir, manifests)
            write_json_atomic(
                status_path,
                {
                    "status": "failed",
                    "output_dir": str(output_dir),
                    "target_reps": target_reps,
                    "completed_reps": completed,
                    "completed_batches": len(manifests),
                    "failed_batch": batch_name,
                    "return_code": return_code,
                    "wrapper_elapsed_seconds": time.perf_counter() - wrapper_started,
                },
            )
            return return_code
        manifests = completed_manifests(batches_dir)
        completed = contiguous_completed_reps(manifests)
        combine_batches(output_dir, manifests)
        write_summary(output_dir, args, completed, target_reps, time.perf_counter() - wrapper_started)
        cursor = completed

    manifests = completed_manifests(batches_dir)
    completed = contiguous_completed_reps(manifests)
    if manifests:
        combine_batches(output_dir, manifests)
    elapsed = time.perf_counter() - wrapper_started
    write_summary(output_dir, args, completed, target_reps, elapsed)
    write_json_atomic(
        status_path,
        {
            "status": "completed",
            "output_dir": str(output_dir),
            "sample_size": args.sample_size,
            "target_reps": target_reps,
            "completed_reps": completed,
            "completed_batches": len(manifests),
            "batch_size": args.batch_size,
            "wrapper_elapsed_seconds": elapsed,
            "finished_at": utc_now(),
        },
    )
    print(f"Completed {completed} contiguous MC reps at n={args.sample_size} in {elapsed:.3f}s")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
