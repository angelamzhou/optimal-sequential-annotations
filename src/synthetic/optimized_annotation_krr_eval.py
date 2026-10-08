#!/usr/bin/env python3
"""Optimized annotation benchmark for plug-in KRR DTR evaluation.

This script is synthetic-only. It asks whether a fixed annotation budget can
improve recursive plug-in KRR value estimates by choosing which reward labels
to annotate. The simulation reuses the Singh/Xu/Gretton binary DTR benchmark
with a fully observed-state ablation: S1, A1, S2, and A2 are always observed,
while R1 and R2 may be selectively annotated. The protocol sampler follows the
passive-annotation note's reward-variance design logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = "results/synthetic/optimized_annotation_krr_eval"
# K-fold cross-fitting of the KRR nuisances (gamma, omega); 0/1 = legacy in-sample fitting.
CROSSFIT_FOLDS = int(os.environ.get("KRR_CROSSFIT_FOLDS", "0"))
# Adds forward-monotone stagewise per-regime designs (separate stage-2 draws, joint T=2 solver).
STAGEWISE_ARM = os.environ.get("KRR_STAGEWISE_ARM", "0") == "1"
DEFAULT_LAMBDAS = "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1"
DEFAULT_REGIMES = "0,0;1,0;0,1;1,1"
MODULE_PATH = Path(__file__).with_name("singh_dtr_benchmark.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from t2_optimal_allocation import solve as solve_t2_allocation  # noqa: E402


def load_dtr_module():
    spec = importlib.util.spec_from_file_location("singh_dtr_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DTR = load_dtr_module()


@dataclass(frozen=True)
class WeightedKernelRidgeModel:
    train_x: np.ndarray
    alpha: np.ndarray
    kernel: object
    lambda_value: float
    gcv: float
    n_train: int
    mean_weight: float

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        k = self.kernel.gram(self.train_x, np.asarray(x_new, dtype=float))
        return k.T @ self.alpha


@dataclass(frozen=True)
class AnnotationDesign:
    name: str
    probabilities: np.ndarray
    priority: np.ndarray
    source: str
    # Charged two-batch designs: pilot reveals are certain (labels already
    # paid for); the second-stage coin applies only to unpiloted units.
    pilot_revealed: np.ndarray | None = None
    second_stage_probabilities: np.ndarray | None = None
    pilot_prob: float = 0.0
    # Stagewise designs: per-stage probabilities for unpiloted units. lambda1 is
    # P(C1=1) and cumulative2 is P(C1=C2=1) <= lambda1 (forward monotone).
    stage1_probabilities: np.ndarray | None = None
    cumulative2_probabilities: np.ndarray | None = None


@dataclass(frozen=True)
class RewardVarianceModel:
    r1_variance_model: object
    r2_variance_model: object
    min_variance: float

    def predict(self, data) -> tuple[np.ndarray, np.ndarray]:
        v1 = self.r1_variance_model.predict(DTR.stage1_features(data))
        v2 = self.r2_variance_model.predict(DTR.stage2_features(data))
        v1 = np.maximum(np.asarray(v1, dtype=float), self.min_variance)
        v2 = np.maximum(np.asarray(v2, dtype=float), self.min_variance)
        return v1, v2


@dataclass(frozen=True)
class BalancingSettings:
    rff_features: int
    cap_multiplier: float
    balance_penalty: float
    chi2_penalty: float
    optimizer_maxiter: int


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_regimes(value: str) -> list[tuple[int, int]]:
    regimes: list[tuple[int, int]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        raw_d1, raw_d2 = chunk.split(",")
        d1 = int(raw_d1)
        d2 = int(raw_d2)
        if d1 not in (0, 1) or d2 not in (0, 1):
            raise ValueError(f"Regime entries must be binary, got {chunk!r}.")
        regimes.append((d1, d2))
    if not regimes:
        raise ValueError("At least one regime is required.")
    return regimes


def normalize_priority(priority: np.ndarray) -> np.ndarray:
    priority = np.asarray(priority, dtype=float)
    priority = np.where(np.isfinite(priority), priority, 0.0)
    priority = np.maximum(priority, 0.0)
    median = float(np.median(priority))
    if median <= 1e-12:
        median = float(np.mean(priority))
    if median <= 1e-12:
        return np.ones_like(priority)
    return np.maximum(priority / median, 1e-6)


def probabilities_from_priority(priority: np.ndarray, budget: float, floor: float) -> np.ndarray:
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0, 1].")
    if floor < 0.0 or floor >= budget:
        raise ValueError("floor must be nonnegative and smaller than budget.")
    scaled = normalize_priority(priority)
    lo = 0.0
    hi = 1.0
    while float(np.mean(np.clip(hi * scaled, floor, 1.0))) < budget:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        probs = np.clip(mid * scaled, floor, 1.0)
        if float(np.mean(probs)) < budget:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * scaled, floor, 1.0)


def observed_stage_overlap_terms(data) -> tuple[np.ndarray, np.ndarray]:
    """Stagewise inverse behavior probabilities for each observed treatment path."""
    g1_obs = np.where(data.d1 == 1, data.p1, 1.0 - data.p1)
    g2_obs = np.where(data.d2 == 1, data.p2, 1.0 - data.p2)
    o1 = 1.0 / np.maximum(g1_obs, 1e-6)
    o2 = 1.0 / np.maximum(g1_obs * g2_obs, 1e-6)
    return o1, o2


def dtr_importance_priority(data, regimes: Sequence[tuple[int, int]], exponent: float) -> np.ndarray:
    """Observable DTR priority based on treatment-history inverse propensities.

    In the real casenote setting these behavior propensities would be estimated from
    always-observed histories. Here the simulator supplies them, so this is an
    oracle-behavior design rather than an oracle-outcome design.
    """
    priority = np.zeros_like(data.y, dtype=float)
    for d1, d2 in regimes:
        g1 = np.where(d1 == 1, data.p1, 1.0 - data.p1)
        g2 = np.where(d2 == 1, data.p2, 1.0 - data.p2)
        matched = ((data.d1 == d1) & (data.d2 == d2)).astype(float)
        priority += matched / np.maximum(g1 * g2, 1e-6)
    return np.power(np.maximum(priority, 1e-6), exponent)


def stagewise_reward_variance_priority(
    data,
    r1_variance: np.ndarray,
    r2_variance: np.ndarray,
) -> np.ndarray:
    """Legacy reward-only priority: power-one behavior overlap inside the sqrt."""
    o1, o2 = observed_stage_overlap_terms(data)
    return np.sqrt(
        np.maximum(r1_variance, 1e-8) * o1
        + np.maximum(r2_variance, 1e-8) * o2
    )


def theorem_stage_overlap_terms(data, regimes: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Regime-summed SQUARED density ratios at the observed treatment path.

    For deterministic regime d, mu_{1,d} = 1{A1=d1}/g1 and
    mu_{1:2,d} = 1{A1=d1,A2=d2}/(g1 g2); a shared design minimizing the
    equal-weighted sum of regime variances uses sum_d mu_{1,d}^2 and
    sum_d mu_{1:2,d}^2 (Theorem thm-twostage-optimal-allocation, bundle case).
    """
    g1_obs = np.where(data.d1 == 1, data.p1, 1.0 - data.p1)
    g2_obs = np.where(data.d2 == 1, data.p2, 1.0 - data.p2)
    w1 = np.zeros_like(g1_obs, dtype=float)
    w2 = np.zeros_like(g1_obs, dtype=float)
    for d1, d2 in regimes:
        match1 = (data.d1 == d1).astype(float)
        match12 = ((data.d1 == d1) & (data.d2 == d2)).astype(float)
        w1 += match1 / np.square(np.maximum(g1_obs, 1e-6))
        w2 += match12 / np.square(np.maximum(g1_obs * g2_obs, 1e-6))
    return w1, w2


def theorem_reward_variance_priority(
    data,
    regimes: Sequence[tuple[int, int]],
    r1_variance: np.ndarray,
    r2_variance: np.ndarray,
) -> np.ndarray:
    """Theorem-form priority: sqrt(sum_d mu_{1,d}^2 v1 + sum_d mu_{1:2,d}^2 v2)."""
    w1, w2 = theorem_stage_overlap_terms(data, regimes)
    return np.sqrt(
        np.maximum(r1_variance, 1e-8) * w1
        + np.maximum(r2_variance, 1e-8) * w2
    )


def oracle_reward_variance_priority(data) -> np.ndarray:
    return stagewise_reward_variance_priority(
        data,
        r1_variance=np.square(data.r1_noise_scale),
        r2_variance=np.square(data.r2_noise_scale),
    )


def estimated_reward_variance_priority(
    data,
    variance_model: RewardVarianceModel,
) -> np.ndarray:
    v1, v2 = variance_model.predict(data)
    return stagewise_reward_variance_priority(
        data,
        r1_variance=v1,
        r2_variance=v2,
    )


def make_annotation_designs(
    data,
    regimes: Sequence[tuple[int, int]],
    budget: float,
    floor: float,
    priority_exponent: float,
    v1_hat: np.ndarray,
    v2_hat: np.ndarray,
    pilot_revealed: np.ndarray | None = None,
    pilot_prob: float = 0.0,
) -> list[AnnotationDesign]:
    n = data.y.shape[0]
    random_priority = np.ones(n, dtype=float)
    dtr_priority = dtr_importance_priority(data, regimes, exponent=priority_exponent)
    estimated_priority = stagewise_reward_variance_priority(
        data, r1_variance=v1_hat, r2_variance=v2_hat
    )
    oracle_priority = oracle_reward_variance_priority(data)
    theorem_estimated_priority = theorem_reward_variance_priority(
        data, regimes, r1_variance=v1_hat, r2_variance=v2_hat
    )
    theorem_oracle_priority = theorem_reward_variance_priority(
        data,
        regimes,
        r1_variance=np.square(data.r1_noise_scale),
        r2_variance=np.square(data.r2_noise_scale),
    )

    def charged_design(name: str, priority: np.ndarray, source: str) -> AnnotationDesign:
        # Designs whose variance signal was learned from paid pilot labels:
        # pilot reveals are certain; residual budget allocated among the rest
        # so total expected cost equals the budget.
        if pilot_revealed is None or pilot_prob <= 0.0:
            return AnnotationDesign(
                name, probabilities_from_priority(priority, budget=budget, floor=floor), priority, source
            )
        residual_mean = (budget - pilot_prob) / max(1.0 - pilot_prob, 1e-8)
        if residual_mean <= floor:
            raise ValueError("Pilot fraction leaves no residual budget above the floor.")
        pi2 = probabilities_from_priority(priority, budget=residual_mean, floor=floor)
        total = pilot_prob + (1.0 - pilot_prob) * pi2
        return AnnotationDesign(
            name,
            total,
            priority,
            source,
            pilot_revealed=pilot_revealed,
            second_stage_probabilities=pi2,
            pilot_prob=float(pilot_prob),
        )

    def stagewise_design(name: str, regime: tuple[int, int]) -> AnnotationDesign:
        # Theorem-form stage increments for one regime: mu_1^2 v1 and mu_{1:2}^2 v2.
        # F1 adds only R1, which does not shift v2 here, so there is one F1 branch.
        w1, w2 = theorem_stage_overlap_terms(data, [regime])
        variance1 = np.maximum(v1_hat, 1e-8) * w1
        variance2 = (np.maximum(v2_hat, 1e-8) * w2)[:, None]
        if pilot_revealed is None or pilot_prob <= 0.0:
            residual_mean = budget
        else:
            residual_mean = (budget - pilot_prob) / max(1.0 - pilot_prob, 1e-8)
        # Cost is per revealed stage: budget fraction b buys 2b stage labels per unit.
        allocation = solve_t2_allocation(
            variance1, variance2, np.ones_like(variance2), 2.0 * residual_mean, floor=floor
        )
        lam1 = allocation.lambda1
        cum2 = allocation.cumulative2[:, 0]
        if pilot_revealed is None or pilot_prob <= 0.0:
            total1, total2 = lam1, cum2
        else:
            total1 = pilot_prob + (1.0 - pilot_prob) * lam1
            total2 = pilot_prob + (1.0 - pilot_prob) * cum2
        return AnnotationDesign(
            name,
            0.5 * (total1 + total2),
            np.sqrt(variance1 + variance2[:, 0]),
            "stagewise_regime_specific",
            pilot_revealed=pilot_revealed,
            pilot_prob=float(pilot_prob) if pilot_revealed is not None else 0.0,
            stage1_probabilities=lam1,
            cumulative2_probabilities=cum2,
        )

    stagewise = (
        [stagewise_design(f"protocol_stagewise_regime_{d1}{d2}_50pct", (d1, d2)) for d1, d2 in regimes]
        if STAGEWISE_ARM
        else []
    )

    return [
        AnnotationDesign("random_50pct", np.full(n, budget, dtype=float), random_priority, "random"),
        # Overlap-only design uses no reward labels, so no pilot charge applies.
        AnnotationDesign(
            "optimized_dtr_priority_50pct",
            probabilities_from_priority(dtr_priority, budget=budget, floor=floor),
            dtr_priority,
            "overlap",
        ),
        charged_design("protocol_estimated_reward_variance_50pct", estimated_priority, "estimated_variance"),
        # Oracle-variance designs are synthetic diagnostics (true noise scales,
        # no labels consumed) and stay uncharged by construction.
        AnnotationDesign(
            "protocol_oracle_reward_variance_50pct",
            probabilities_from_priority(oracle_priority, budget=budget, floor=floor),
            oracle_priority,
            "oracle_variance_diagnostic",
        ),
        charged_design("protocol_theorem_reward_variance_50pct", theorem_estimated_priority, "theorem_estimated_variance"),
        AnnotationDesign(
            "protocol_theorem_oracle_variance_50pct",
            probabilities_from_priority(theorem_oracle_priority, budget=budget, floor=floor),
            theorem_oracle_priority,
            "theorem_oracle_variance_diagnostic",
        ),
        # Theorem-faithful per-regime campaigns: one annotation dataset per
        # target regime, each optimized for that regime's single-policy mu.
        # Evaluations of regime d should read the matching sampler
        # protocol_theorem_regime_{d1}{d2}_50pct.
        *[
            charged_design(
                f"protocol_theorem_regime_{d1}{d2}_50pct",
                theorem_reward_variance_priority(data, [(d1, d2)], r1_variance=v1_hat, r2_variance=v2_hat),
                "theorem_regime_specific",
            )
            for d1, d2 in regimes
        ],
        *stagewise,
    ]


def fit_weighted_kernel_ridge(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    continuous_idx: Sequence[int],
    discrete_idx: Sequence[int],
    lambdas: Sequence[float],
    seed: int,
) -> WeightedKernelRidgeModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    raw_weight = np.asarray(sample_weight, dtype=float)
    raw_weight = np.maximum(raw_weight, 1e-8)
    sample_weight = raw_weight / np.mean(raw_weight)
    n = x.shape[0]
    kernel = DTR.MixedKernel(
        continuous_idx=tuple(continuous_idx),
        discrete_idx=tuple(discrete_idx),
        length_scales=DTR.median_length_scales(x, continuous_idx, seed),
    )
    k = kernel.gram(x, x)
    k = 0.5 * (k + k.T)
    sqrt_weight = np.sqrt(sample_weight)
    weighted_kernel = sqrt_weight[:, None] * k * sqrt_weight[None, :]
    weighted_kernel = 0.5 * (weighted_kernel + weighted_kernel.T)
    evals, evecs = np.linalg.eigh(weighted_kernel)
    evals = np.maximum(evals, 0.0)
    weighted_y_rot = evecs.T @ (sqrt_weight * y)
    best: dict[str, float | np.ndarray] | None = None

    for lambda_value in lambdas:
        denom = evals + n * float(lambda_value)
        beta = evecs @ (weighted_y_rot / denom)
        alpha = sqrt_weight * beta
        y_hat = k @ alpha
        residual = y - y_hat
        trace_h = float(np.sum(evals / denom))
        denom = max(1.0 - trace_h / n, 1e-8)
        gcv = float(np.average(np.square(residual), weights=sample_weight) / (denom * denom))
        if best is None or gcv < float(best["gcv"]):
            best = {"gcv": gcv, "lambda": float(lambda_value), "alpha": alpha}

    assert best is not None
    return WeightedKernelRidgeModel(
        train_x=x,
        alpha=np.asarray(best["alpha"], dtype=float),
        kernel=kernel,
        lambda_value=float(best["lambda"]),
        gcv=float(best["gcv"]),
        n_train=n,
        mean_weight=float(np.mean(raw_weight)),
    )


def fit_reward_variance_model(
    data,
    lambdas: Sequence[float],
    seed: int,
) -> RewardVarianceModel:
    p = data.x1.shape[1]

    continuous1, discrete1 = DTR.stage1_indices(p)
    x1 = DTR.stage1_features(data)
    r1_mean_model = DTR.fit_kernel_ridge(
        x1,
        data.r1,
        continuous1,
        discrete1,
        lambdas=lambdas,
        seed=seed,
    )
    r1_resid_sq = np.square(data.r1 - r1_mean_model.predict(x1))
    r1_variance_model = DTR.fit_kernel_ridge(
        x1,
        r1_resid_sq,
        continuous1,
        discrete1,
        lambdas=lambdas,
        seed=seed + 11,
    )

    continuous2, discrete2 = DTR.stage2_indices(p)
    x2 = DTR.stage2_features(data)
    r2_mean_model = DTR.fit_kernel_ridge(
        x2,
        data.r2,
        continuous2,
        discrete2,
        lambdas=lambdas,
        seed=seed + 23,
    )
    r2_resid_sq = np.square(data.r2 - r2_mean_model.predict(x2))
    r2_variance_model = DTR.fit_kernel_ridge(
        x2,
        r2_resid_sq,
        continuous2,
        discrete2,
        lambdas=lambdas,
        seed=seed + 37,
    )
    min_variance = max(1e-6, 0.01 * float(np.mean(np.concatenate([r1_resid_sq, r2_resid_sq]))))
    return RewardVarianceModel(
        r1_variance_model=r1_variance_model,
        r2_variance_model=r2_variance_model,
        min_variance=min_variance,
    )


def charged_variance_predictions(
    data,
    pilot_mask: np.ndarray,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Variance predictions paid for by the pilot batch, half-split cross-fitted.

    Fits `fit_reward_variance_model` on each half of the pilot and predicts
    every unit from the model whose training half excludes it, so no unit's own
    reward enters its own design score. Falls back to a flat signal when the
    pilot is too small to fit.
    """
    n = data.y.shape[0]
    rng = np.random.default_rng(seed)
    half_assignment = rng.integers(0, 2, size=n)
    v1 = np.ones(n, dtype=float)
    v2 = np.ones(n, dtype=float)
    models = []
    for half in (0, 1):
        train_idx = np.flatnonzero(pilot_mask & (half_assignment == half))
        models.append(
            fit_reward_variance_model(DTR.subset_dtr_data(data, train_idx), lambdas=lambdas, seed=seed + 7 * half)
            if len(train_idx) >= 8
            else None
        )
    if models[0] is None and models[1] is None:
        return v1, v2  # flat: design degrades to overlap-only
    for half in (0, 1):
        predict_idx = np.flatnonzero(half_assignment == half)
        model = models[1 - half] if models[1 - half] is not None else models[half]
        subset = DTR.subset_dtr_data(data, predict_idx)
        v1_half, v2_half = model.predict(subset)
        v1[predict_idx] = v1_half
        v2[predict_idx] = v2_half
    return v1, v2


def variance_model_validation_rows(
    variance_model: RewardVarianceModel,
    n: int,
    p: int,
    budget: float,
    floor: float,
    dgp: str,
    reward_noise_profile: str,
    heteroskedastic_strength: float,
    behavior_min_prob: float,
    behavior_logit_scale: float,
    seed: int,
    validation_reps: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for rep in range(validation_reps):
        data = DTR.simulate_singh_dtr(
            n=n,
            p=p,
            seed=seed + 30_000 + 1_009 * rep + n,
            dgp=dgp,
            reward_noise_profile=reward_noise_profile,
            heteroskedastic_strength=heteroskedastic_strength,
            behavior_min_prob=behavior_min_prob,
            behavior_logit_scale=behavior_logit_scale,
        )
        v1_hat, v2_hat = variance_model.predict(data)
        v1_true = np.square(data.r1_noise_scale)
        v2_true = np.square(data.r2_noise_scale)
        priority = estimated_reward_variance_priority(data, variance_model)
        probs = probabilities_from_priority(priority, budget=budget, floor=floor)
        rows.append(
            {
                "sample_size": n,
                "budget": budget,
                "validation_rep": rep,
                "dgp": dgp,
                "r1_variance_corr": safe_corr(v1_hat, v1_true),
                "r2_variance_corr": safe_corr(v2_hat, v2_true),
                "r1_variance_mae": float(np.mean(np.abs(v1_hat - v1_true))),
                "r2_variance_mae": float(np.mean(np.abs(v2_hat - v2_true))),
                "priority_oracle_corr": safe_corr(
                    priority,
                    oracle_reward_variance_priority(data),
                ),
                "expected_budget": float(np.mean(probs)),
                "expected_cost": float(2.0 * np.mean(probs)),
                "prob_min": float(np.min(probs)),
                "prob_max": float(np.max(probs)),
            }
        )
    return rows


def model_based_ci_diagnostics(
    signal: np.ndarray,
    estimate: float,
    method: str,
) -> dict[str, float | str]:
    signal = np.asarray(signal, dtype=float)
    finite = signal[np.isfinite(signal)]
    if finite.shape[0] < 2 or not np.isfinite(estimate):
        return {
            "model_based_ci_method": method,
            "model_based_n": float(finite.shape[0]),
            "model_based_signal_sd": float("nan"),
            "model_based_se": float("nan"),
            "model_based_ci_lower_95": float("nan"),
            "model_based_ci_upper_95": float("nan"),
            "model_based_ci_width_95": float("nan"),
        }
    signal_sd = float(np.std(finite, ddof=1))
    se = signal_sd / math.sqrt(float(finite.shape[0]))
    half_width = 1.96 * se
    return {
        "model_based_ci_method": method,
        "model_based_n": float(finite.shape[0]),
        "model_based_signal_sd": signal_sd,
        "model_based_se": se,
        "model_based_ci_lower_95": float(estimate - half_width),
        "model_based_ci_upper_95": float(estimate + half_width),
        "model_based_ci_width_95": float(2.0 * half_width),
    }


def annotation_diagnostics(
    c1: np.ndarray,
    c2: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
) -> dict[str, float]:
    idx1 = np.flatnonzero(c1)
    idx2 = np.flatnonzero(c2)
    mean_cost = float(np.mean(c1.astype(float) + c2.astype(float)))
    out = {
        "annotated_n": float(min(idx1.shape[0], idx2.shape[0])),
        "stage1_annotated_n": float(idx1.shape[0]),
        "stage2_annotated_n": float(idx2.shape[0]),
        "stage1_annotation_rate": float(np.mean(c1)),
        "stage2_annotation_rate": float(np.mean(c2)),
        "annotation_rate": mean_cost / 2.0,
        "mean_annotation_cost": mean_cost,
        "annotation_prob_min": float(min(np.min(prob1), np.min(prob2))),
        "annotation_prob_max": float(max(np.max(prob1), np.max(prob2))),
        "monotone_violation_rate": float(np.mean(c2 & ~c1)),
        "stage1_annotation_ess": 0.0,
        "stage2_annotation_ess": 0.0,
        "annotation_ess": 0.0,
    }
    if idx1.shape[0] > 0:
        inverse_prob_weight1 = 1.0 / np.maximum(prob1[idx1], 1e-6)
        normalized_weights1 = inverse_prob_weight1 / np.sum(inverse_prob_weight1)
        out["stage1_annotation_ess"] = float(1.0 / np.sum(np.square(normalized_weights1)))
    if idx2.shape[0] > 0:
        inverse_prob_weight2 = 1.0 / np.maximum(prob2[idx2], 1e-6)
        normalized_weights2 = inverse_prob_weight2 / np.sum(inverse_prob_weight2)
        out["stage2_annotation_ess"] = float(1.0 / np.sum(np.square(normalized_weights2)))
    out["annotation_ess"] = float(min(out["stage1_annotation_ess"], out["stage2_annotation_ess"]))
    return out


def _crossfit_krr_components(data, c1, c2, prob1, prob2, d1, d2, lambdas, seed, n_folds):
    """K-fold cross-fitted version of recursive_reward_annotated_krr_components.

    For each fold k, gamma is fit on annotated stage-2 units outside k, its
    regime-d2 predictions on the training units form the stage-1 pseudo-outcome,
    omega is fit on annotated stage-1 units outside k, and all four nuisance
    predictions for fold-k units come from these out-of-fold models.
    """
    p = data.x1.shape[1]
    n = data.y.shape[0]
    rng = np.random.default_rng(seed + 9001)
    fold = rng.permutation(np.arange(n) % n_folds)
    continuous2, discrete2 = DTR.stage2_indices(p)
    continuous1, discrete1 = DTR.stage1_indices(p)
    stage2_x = DTR.stage2_features(data)
    stage2_x_d2 = DTR.stage2_features(data, d2_override=d2)
    stage1_x = DTR.stage1_features(data)
    stage1_x_d1 = DTR.stage1_features(data, d1_override=d1)
    gamma_obs = np.full(n, np.nan)
    q2_obs = np.full(n, np.nan)
    omega_obs = np.full(n, np.nan)
    omega_at_regime = np.full(n, np.nan)
    lam_g, lam_o, gcv_g, gcv_o = [], [], [], []
    for k in range(n_folds):
        test = fold == k
        train = ~test
        tr2 = np.flatnonzero(train & (c2 > 0))
        tr1 = np.flatnonzero(train & (c1 > 0))
        if tr1.shape[0] < max(20, 2 * p) or tr2.shape[0] < max(20, 2 * p):
            diagnostics = annotation_diagnostics(c1, c2, prob1, prob2)
            diagnostics.update({"gamma_lambda": float("nan"), "omega_lambda": float("nan"), "crossfit_folds": float(n_folds)})
            nan = np.full(n, float("nan"))
            return nan, nan, nan, nan, float("nan"), diagnostics
        gamma_model = fit_weighted_kernel_ridge(
            stage2_x[tr2], data.r2[tr2],
            sample_weight=1.0 / np.maximum(prob2[tr2], 1e-6),
            continuous_idx=continuous2, discrete_idx=discrete2,
            lambdas=lambdas, seed=seed + 101 * k,
        )
        q2_train = gamma_model.predict(stage2_x_d2[tr1])
        omega_model = fit_weighted_kernel_ridge(
            stage1_x[tr1], data.r1[tr1] + q2_train,
            sample_weight=1.0 / np.maximum(prob1[tr1], 1e-6),
            continuous_idx=continuous1, discrete_idx=discrete1,
            lambdas=lambdas, seed=seed + 17 + 101 * k,
        )
        gamma_obs[test] = gamma_model.predict(stage2_x[test])
        q2_obs[test] = gamma_model.predict(stage2_x_d2[test])
        omega_obs[test] = omega_model.predict(stage1_x[test])
        omega_at_regime[test] = omega_model.predict(stage1_x_d1[test])
        lam_g.append(gamma_model.lambda_value); lam_o.append(omega_model.lambda_value)
        gcv_g.append(gamma_model.gcv); gcv_o.append(omega_model.gcv)
    estimate = float(np.mean(omega_at_regime))
    diagnostics = annotation_diagnostics(c1, c2, prob1, prob2)
    diagnostics.update({
        "gamma_lambda": float(np.median(lam_g)),
        "gamma_gcv": float(np.mean(gcv_g)),
        "omega_lambda": float(np.median(lam_o)),
        "omega_gcv": float(np.mean(gcv_o)),
        "crossfit_folds": float(n_folds),
    })
    diagnostics.update(model_based_ci_diagnostics(omega_at_regime, estimate, method="plugin_regime_prediction_sd"))
    return gamma_obs, q2_obs, omega_obs, omega_at_regime, estimate, diagnostics


def recursive_reward_annotated_krr_components(
    data,
    c1: np.ndarray,
    c2: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
    d1: int,
    d2: int,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    p = data.x1.shape[1]
    idx2 = np.flatnonzero(c2)
    idx1 = np.flatnonzero(c1)
    if idx1.shape[0] < max(20, 2 * p) or idx2.shape[0] < max(20, 2 * p):
        diagnostics = annotation_diagnostics(c1, c2, prob1, prob2)
        diagnostics.update(
            {
            "gamma_lambda": float("nan"),
            "omega_lambda": float("nan"),
            }
        )
        nan = np.full(data.y.shape[0], float("nan"))
        return nan, nan, nan, nan, float("nan"), diagnostics

    if CROSSFIT_FOLDS > 1:
        return _crossfit_krr_components(data, c1, c2, prob1, prob2, d1, d2, lambdas, seed, CROSSFIT_FOLDS)

    continuous2, discrete2 = DTR.stage2_indices(p)
    stage2_x = DTR.stage2_features(data)
    inverse_prob_weight2 = 1.0 / np.maximum(prob2[idx2], 1e-6)
    gamma_model = fit_weighted_kernel_ridge(
        stage2_x[idx2],
        data.r2[idx2],
        sample_weight=inverse_prob_weight2,
        continuous_idx=continuous2,
        discrete_idx=discrete2,
        lambdas=lambdas,
        seed=seed,
    )
    gamma_obs = gamma_model.predict(stage2_x)
    q2_obs = gamma_model.predict(DTR.stage2_features(data, d2_override=d2))

    continuous1, discrete1 = DTR.stage1_indices(p)
    inverse_prob_weight1 = 1.0 / np.maximum(prob1[idx1], 1e-6)
    stage1_x = DTR.stage1_features(data)
    omega_model = fit_weighted_kernel_ridge(
        stage1_x[idx1],
        data.r1[idx1] + q2_obs[idx1],
        sample_weight=inverse_prob_weight1,
        continuous_idx=continuous1,
        discrete_idx=discrete1,
        lambdas=lambdas,
        seed=seed + 17,
    )
    omega_obs = omega_model.predict(stage1_x)
    omega_at_regime = omega_model.predict(DTR.stage1_features(data, d1_override=d1))
    estimate = float(np.mean(omega_at_regime))
    diagnostics = annotation_diagnostics(c1, c2, prob1, prob2)
    diagnostics.update({
        "gamma_lambda": gamma_model.lambda_value,
        "gamma_gcv": gamma_model.gcv,
        "omega_lambda": omega_model.lambda_value,
        "omega_gcv": omega_model.gcv,
    })
    diagnostics.update(
        model_based_ci_diagnostics(
            omega_at_regime,
            estimate,
            method="plugin_regime_prediction_sd",
        )
    )
    return gamma_obs, q2_obs, omega_obs, omega_at_regime, estimate, diagnostics


def recursive_reward_annotated_krr_plugin(
    data,
    c1: np.ndarray,
    c2: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
    d1: int,
    d2: int,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    _, _, _, _, estimate, diagnostics = recursive_reward_annotated_krr_components(
        data=data,
        c1=c1,
        c2=c2,
        prob1=prob1,
        prob2=prob2,
        d1=d1,
        d2=d2,
        lambdas=lambdas,
        seed=seed,
    )
    return estimate, diagnostics


def recursive_reward_annotated_oracle_drl(
    data,
    c1: np.ndarray,
    c2: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
    d1: int,
    d2: int,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    gamma_obs, gamma_d2, omega_obs, omega_at_regime, _, diagnostics = recursive_reward_annotated_krr_components(
        data=data,
        c1=c1,
        c2=c2,
        prob1=prob1,
        prob2=prob2,
        d1=d1,
        d2=d2,
        lambdas=lambdas,
        seed=seed,
    )
    if not np.all(np.isfinite(omega_at_regime)):
        return float("nan"), diagnostics

    g1 = np.where(d1 == 1, data.p1, 1.0 - data.p1)
    g2 = np.where(d2 == 1, data.p2, 1.0 - data.p2)
    i1 = (data.d1 == d1).astype(float)
    i2 = (data.d2 == d2).astype(float)
    r1_ipw = c1.astype(float) / np.maximum(prob1, 1e-6) * data.r1
    r2_residual_ipw = c2.astype(float) / np.maximum(prob2, 1e-6) * (data.r2 - gamma_obs)
    signal = (
        omega_at_regime
        + i1 / np.maximum(g1, 1e-8) * (r1_ipw + gamma_d2 - omega_obs)
        + i1 * i2 / np.maximum(g1 * g2, 1e-8) * r2_residual_ipw
    )
    estimate = float(np.mean(signal))
    diagnostics = dict(diagnostics)
    diagnostics["mean_abs_drl_correction"] = float(np.mean(np.abs(signal - omega_at_regime)))
    diagnostics.update(
        model_based_ci_diagnostics(
            signal,
            estimate,
            method="drl_signal_sd",
        )
    )
    return estimate, diagnostics


def recursive_reward_annotated_balancing(
    data,
    c1: np.ndarray,
    c2: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
    d1: int,
    d2: int,
    settings: BalancingSettings,
    seed: int,
) -> tuple[float, dict[str, float | str]]:
    p = data.x1.shape[1]
    n = data.y.shape[0]
    continuous1, discrete1 = DTR.stage1_indices(p)
    source1 = DTR.stage1_features(data)
    target1 = DTR.stage1_features(data, d1_override=d1)
    rff1 = DTR.fit_rff_map(source1, target1, continuous1, discrete1, settings.rff_features, seed)
    phi_source1 = rff1.transform(source1)
    phi_target1 = rff1.transform(target1)
    w1 = DTR.solve_balancing_weights(
        phi_source=phi_source1,
        target_mean=np.mean(phi_target1, axis=0),
        base_weights=np.full(n, 1.0 / n),
        cap_multiplier=settings.cap_multiplier,
        balance_penalty=settings.balance_penalty,
        chi2_penalty=settings.chi2_penalty,
        optimizer_maxiter=settings.optimizer_maxiter,
    )

    continuous2, discrete2 = DTR.stage2_indices(p)
    source2 = DTR.stage2_features(data)
    target2 = DTR.stage2_features(data, d2_override=d2)
    rff2 = DTR.fit_rff_map(source2, target2, continuous2, discrete2, settings.rff_features, seed + 101)
    phi_source2 = rff2.transform(source2)
    phi_target2 = rff2.transform(target2)
    w2 = DTR.solve_balancing_weights(
        phi_source=phi_source2,
        target_mean=w1.weights @ phi_target2,
        base_weights=w1.weights,
        cap_multiplier=settings.cap_multiplier,
        balance_penalty=settings.balance_penalty,
        chi2_penalty=settings.chi2_penalty,
        optimizer_maxiter=settings.optimizer_maxiter,
    )

    annotated_outcome = (
        c1.astype(float) / np.maximum(prob1, 1e-6) * data.r1
        + c2.astype(float) / np.maximum(prob2, 1e-6) * data.r2
    )
    estimate = float(w2.weights @ annotated_outcome)
    diagnostics: dict[str, float | str] = annotation_diagnostics(c1, c2, prob1, prob2)
    diagnostics.update(
        {
            "balancing_stage1_ess": w1.ess,
            "balancing_stage2_ess": w2.ess,
            "balancing_stage1_max_weight": w1.max_weight,
            "balancing_stage2_max_weight": w2.max_weight,
            "balancing_stage1_cap_hit_rate": w1.cap_hit_rate,
            "balancing_stage2_cap_hit_rate": w2.cap_hit_rate,
            "balancing_stage1_imbalance": w1.imbalance,
            "balancing_stage2_imbalance": w2.imbalance,
            "balancing_stage1_solver": f"cvxpy:{w1.solver}",
            "balancing_stage2_solver": f"cvxpy:{w2.solver}",
            "balancing_stage1_status": w1.status,
            "balancing_stage2_status": w2.status,
        }
    )
    return estimate, diagnostics


def make_result_row(
    method: str,
    d1: int,
    d2: int,
    true_value: float,
    estimate: float,
    diagnostics: dict[str, float],
) -> dict[str, float | str | int]:
    error = float(estimate - true_value)
    row: dict[str, float | str | int] = {
        "method": method,
        "d1": d1,
        "d2": d2,
        "true_value": true_value,
        "estimate": estimate,
        "error": error,
        "abs_error": abs(error),
        "squared_error": error * error,
    }
    row.update(diagnostics)
    lower = row.get("model_based_ci_lower_95")
    upper = row.get("model_based_ci_upper_95")
    if isinstance(lower, (float, int)) and isinstance(upper, (float, int)) and np.isfinite(lower) and np.isfinite(upper):
        row["model_based_ci_covers_truth"] = float(lower <= true_value <= upper)
    else:
        row["model_based_ci_covers_truth"] = float("nan")
    return row


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_variance_profile(
    data,
    regimes: Sequence[tuple[int, int]],
    priority_exponent: float,
) -> list[dict[str, float | str | int]]:
    priority = dtr_importance_priority(data, regimes, exponent=priority_exponent)
    r1_noise_var = np.square(data.r1_noise_scale)
    r2_noise_var = np.square(data.r2_noise_scale)
    total_noise_var = r1_noise_var + r2_noise_var
    rows: list[dict[str, float | str | int]] = []
    p = data.x1.shape[1]

    def build_row(path: str, mask: np.ndarray) -> dict[str, float | str | int]:
        if int(np.sum(mask)) == 0:
            return {
                "path": path,
                "n_units": 0,
                "r1_noise_var_mean": float("nan"),
                "r2_noise_var_mean": float("nan"),
                "reward_noise_var_mean": float("nan"),
                "reward_noise_sd_mean": float("nan"),
                "reward_noise_sd_min": float("nan"),
                "reward_noise_sd_max": float("nan"),
                "r1_mean_var": float("nan"),
                "r2_mean_var": float("nan"),
                "reward_mean_var": float("nan"),
                "outcome_var": float("nan"),
                "x2_empirical_var_mean_diag": float("nan"),
                "transition_conditional_noise_var_diag": 0.25,
                "transition_conditional_noise_adjacent_cov": 0.125 if p > 1 else float("nan"),
                "dtr_priority_mean": float("nan"),
                "priority_noise_var_corr": float("nan"),
            }
        x2_var = np.var(data.x2[mask], axis=0, ddof=1) if int(np.sum(mask)) > 1 else np.full(p, np.nan)
        return {
            "path": path,
            "n_units": int(np.sum(mask)),
            "r1_noise_var_mean": float(np.mean(r1_noise_var[mask])),
            "r2_noise_var_mean": float(np.mean(r2_noise_var[mask])),
            "reward_noise_var_mean": float(np.mean(total_noise_var[mask])),
            "reward_noise_sd_mean": float(np.mean(data.reward_noise_scale[mask])),
            "reward_noise_sd_min": float(np.min(data.reward_noise_scale[mask])),
            "reward_noise_sd_max": float(np.max(data.reward_noise_scale[mask])),
            "r1_mean_var": float(np.var(data.r1_mean[mask], ddof=1)) if int(np.sum(mask)) > 1 else float("nan"),
            "r2_mean_var": float(np.var(data.r2_mean[mask], ddof=1)) if int(np.sum(mask)) > 1 else float("nan"),
            "reward_mean_var": float(np.var(data.reward_mean[mask], ddof=1)) if int(np.sum(mask)) > 1 else float("nan"),
            "outcome_var": float(np.var(data.y[mask], ddof=1)) if int(np.sum(mask)) > 1 else float("nan"),
            "x2_empirical_var_mean_diag": float(np.nanmean(x2_var)),
            "transition_conditional_noise_var_diag": 0.25,
            "transition_conditional_noise_adjacent_cov": 0.125 if p > 1 else float("nan"),
            "dtr_priority_mean": float(np.mean(priority[mask])),
            "priority_noise_var_corr": safe_corr(priority[mask], total_noise_var[mask]),
        }

    rows.append(build_row("all", np.ones_like(data.y, dtype=bool)))
    for d1, d2 in regimes:
        rows.append(build_row(f"d1={d1},d2={d2}", (data.d1 == d1) & (data.d2 == d2)))
    return rows


def run_replication(
    n: int,
    p: int,
    regimes: Sequence[tuple[int, int]],
    estimators: Sequence[str],
    budgets: Sequence[float],
    floor: float,
    priority_exponent: float,
    dgp: str,
    reward_noise_profile: str,
    heteroskedastic_strength: float,
    behavior_min_prob: float,
    behavior_logit_scale: float,
    common_oracle_n: int,
    variance_model: RewardVarianceModel,
    balancing_settings: BalancingSettings,
    lambdas: Sequence[float],
    seed: int,
    pilot_fraction: float = 0.0,
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]]]:
    rng = np.random.default_rng(seed + 91)
    if common_oracle_n > 0:
        oracle_data = DTR.simulate_singh_dtr(
            n=common_oracle_n,
            p=p,
            seed=seed,
            dgp=dgp,
            reward_noise_profile=reward_noise_profile,
            heteroskedastic_strength=heteroskedastic_strength,
            behavior_min_prob=behavior_min_prob,
            behavior_logit_scale=behavior_logit_scale,
        )
        permutation = rng.permutation(common_oracle_n)
        data = DTR.subset_dtr_data(oracle_data, permutation[:n])
    else:
        data = DTR.simulate_singh_dtr(
            n=n,
            p=p,
            seed=seed,
            dgp=dgp,
            reward_noise_profile=reward_noise_profile,
            heteroskedastic_strength=heteroskedastic_strength,
            behavior_min_prob=behavior_min_prob,
            behavior_logit_scale=behavior_logit_scale,
        )
    rows: list[dict[str, float | str | int]] = []
    variance_rows = summarize_variance_profile(data, regimes=regimes, priority_exponent=priority_exponent)

    full_annotation = np.ones(n, dtype=bool)
    full_prob = np.ones(n, dtype=float)
    design_uniforms: dict[str, np.ndarray] = {}
    pilot_uniform = rng.uniform(size=n)
    for budget in budgets:
        if pilot_fraction > 0.0:
            pilot_prob = float(pilot_fraction * budget)
            pilot_revealed = pilot_uniform < pilot_prob
            v1_hat, v2_hat = charged_variance_predictions(
                data, pilot_revealed, lambdas=lambdas, seed=seed + 53_000 + int(round(100 * budget))
            )
        else:
            pilot_prob = 0.0
            pilot_revealed = None
            v1_hat, v2_hat = variance_model.predict(data)
        designs = [AnnotationDesign("full_data_krr", full_prob, np.ones(n, dtype=float), "full_data")]
        designs.extend(
            make_annotation_designs(
                data,
                regimes,
                budget,
                floor,
                priority_exponent,
                v1_hat=v1_hat,
                v2_hat=v2_hat,
                pilot_revealed=pilot_revealed,
                pilot_prob=pilot_prob,
            )
        )

        for design in designs:
            prob1 = prob2 = design.probabilities
            if design.name == "full_data_krr":
                c1 = full_annotation
                c2 = full_annotation
            elif design.stage1_probabilities is not None:
                # Stagewise: C1 ~ Bern(lambda1), C2 | C1=1 ~ Bern(cumulative2 / lambda1).
                if design.name not in design_uniforms:
                    design_uniforms[design.name] = rng.uniform(size=n)
                    design_uniforms[design.name + "::stage2"] = rng.uniform(size=n)
                lam1 = design.stage1_probabilities
                cum2 = design.cumulative2_probabilities
                c1 = design_uniforms[design.name] < lam1
                c2 = c1 & (design_uniforms[design.name + "::stage2"] < cum2 / np.maximum(lam1, 1e-12))
                if design.pilot_revealed is not None:
                    c1 = design.pilot_revealed | (c1 & ~design.pilot_revealed)
                    c2 = design.pilot_revealed | (c2 & ~design.pilot_revealed)
                    prob1 = design.pilot_prob + (1.0 - design.pilot_prob) * lam1
                    prob2 = design.pilot_prob + (1.0 - design.pilot_prob) * cum2
                else:
                    prob1, prob2 = lam1, cum2
                if not c2.any():
                    forced = int(rng.integers(0, n))
                    c1[forced] = c2[forced] = True
            else:
                if design.name not in design_uniforms:
                    design_uniforms[design.name] = rng.uniform(size=n)
                if design.pilot_revealed is not None:
                    bundled = design.pilot_revealed | (
                        (design_uniforms[design.name] < design.second_stage_probabilities)
                        & ~design.pilot_revealed
                    )
                else:
                    bundled = design_uniforms[design.name] < design.probabilities
                if not bundled.any():
                    bundled[int(rng.integers(0, n))] = True
                c1 = bundled
                c2 = bundled
            for d1, d2 in regimes:
                for estimator_index, estimator in enumerate(estimators):
                    estimator_seed = seed + 1000 + 13 * d1 + d2 + 100_003 * (1 + estimator_index)
                    if estimator == "plugin":
                        estimate, diagnostics = recursive_reward_annotated_krr_plugin(
                            data,
                            c1=c1,
                            c2=c2,
                            prob1=prob1,
                            prob2=prob2,
                            d1=d1,
                            d2=d2,
                            lambdas=lambdas,
                            seed=estimator_seed,
                        )
                    elif estimator == "drl":
                        estimate, diagnostics = recursive_reward_annotated_oracle_drl(
                            data,
                            c1=c1,
                            c2=c2,
                            prob1=prob1,
                            prob2=prob2,
                            d1=d1,
                            d2=d2,
                            lambdas=lambdas,
                            seed=estimator_seed,
                        )
                    elif estimator == "balancing":
                        estimate, diagnostics = recursive_reward_annotated_balancing(
                            data,
                            c1=c1,
                            c2=c2,
                            prob1=prob1,
                            prob2=prob2,
                            d1=d1,
                            d2=d2,
                            settings=balancing_settings,
                            seed=estimator_seed,
                        )
                    else:
                        raise ValueError(f"Unknown estimator {estimator!r}.")
                    diagnostics = dict(diagnostics)
                    diagnostics["budget"] = float(budget)
                    diagnostics["estimator"] = estimator
                    diagnostics["sampler"] = design.name
                    diagnostics["expected_budget"] = float(np.mean(0.5 * (prob1 + prob2)))
                    diagnostics["expected_cost"] = float(np.mean(prob1 + prob2))
                    diagnostics["pilot_prob"] = float(design.pilot_prob)
                    diagnostics["pilot_revealed_count"] = int(np.sum(design.pilot_revealed)) if design.pilot_revealed is not None else 0
                    diagnostics["priority_mean"] = float(np.mean(design.priority))
                    diagnostics["priority_sd"] = float(np.std(design.priority))
                    diagnostics["sampler_source"] = design.source
                    rows.append(
                        make_result_row(
                            f"{estimator}:{design.name}",
                            d1,
                            d2,
                            DTR.true_regime_value(d1, d2),
                            estimate,
                            diagnostics,
                        )
                    )
    return rows, variance_rows


def run_replication_task(
    task: tuple[
        int,
        int,
        int,
        list[tuple[int, int]],
        list[str],
        list[float],
        float,
        float,
        str,
        str,
        float,
        float,
        float,
        int,
        RewardVarianceModel,
        BalancingSettings,
        list[float],
        int,
        float,
    ],
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]]]:
    (
        n,
        rep,
        p,
        regimes,
        estimators,
        budgets,
        floor,
        priority_exponent,
        dgp,
        reward_noise_profile,
        heteroskedastic_strength,
        behavior_min_prob,
        behavior_logit_scale,
        common_oracle_n,
        variance_model,
        balancing_settings,
        lambdas,
        seed,
        pilot_fraction,
    ) = task
    rep_rows, rep_variance_rows = run_replication(
        n=n,
        p=p,
        regimes=regimes,
        estimators=estimators,
        budgets=budgets,
        floor=floor,
        priority_exponent=priority_exponent,
        dgp=dgp,
        reward_noise_profile=reward_noise_profile,
        heteroskedastic_strength=heteroskedastic_strength,
        behavior_min_prob=behavior_min_prob,
        behavior_logit_scale=behavior_logit_scale,
        common_oracle_n=common_oracle_n,
        variance_model=variance_model,
        balancing_settings=balancing_settings,
        lambdas=lambdas,
        seed=seed,
        pilot_fraction=pilot_fraction,
    )
    for row in rep_rows:
        row["sample_size"] = n
        row["rep"] = rep
        row["p"] = p
        row["common_oracle_n"] = common_oracle_n
        row["dgp"] = dgp
    for row in rep_variance_rows:
        row["sample_size"] = n
        row["rep"] = rep
        row["p"] = p
        row["dgp"] = dgp
    return rep_rows, rep_variance_rows


def aggregate_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["sample_size", "method"]
    if "estimator" in results.columns and "sampler" in results.columns:
        group_cols = ["sample_size", "estimator", "sampler", "method"]
    if "budget" in results.columns:
        group_cols = ["budget"] + group_cols
    overall = (
        results.groupby(group_cols, as_index=False)
        .agg(
            bias=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda x: math.sqrt(float(np.nanmean(x)))),
            sd=("estimate", "std"),
            mean_model_based_se=("model_based_se", "mean"),
            mean_model_based_ci_width_95=("model_based_ci_width_95", "mean"),
            mean_model_based_ci_covers_truth=("model_based_ci_covers_truth", "mean"),
            mean_annotation_rate=("annotation_rate", "mean"),
            mean_stage1_annotation_rate=("stage1_annotation_rate", "mean"),
            mean_stage2_annotation_rate=("stage2_annotation_rate", "mean"),
            mean_annotation_cost=("mean_annotation_cost", "mean"),
            mean_annotation_ess=("annotation_ess", "mean"),
            mean_stage1_annotation_ess=("stage1_annotation_ess", "mean"),
            mean_stage2_annotation_ess=("stage2_annotation_ess", "mean"),
            mean_prob_min=("annotation_prob_min", "mean"),
            mean_prob_max=("annotation_prob_max", "mean"),
            mean_monotone_violation_rate=("monotone_violation_rate", "mean"),
        )
        .sort_values(["sample_size", "rmse", "method"])
    )
    by_regime_group_cols = ["sample_size", "method", "d1", "d2"]
    if "estimator" in results.columns and "sampler" in results.columns:
        by_regime_group_cols = ["sample_size", "estimator", "sampler", "method", "d1", "d2"]
    if "budget" in results.columns:
        by_regime_group_cols = ["budget"] + by_regime_group_cols
    by_regime = (
        results.groupby(by_regime_group_cols, as_index=False)
        .agg(
            bias=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda x: math.sqrt(float(np.nanmean(x)))),
            sd=("estimate", "std"),
            mean_model_based_se=("model_based_se", "mean"),
            mean_model_based_ci_width_95=("model_based_ci_width_95", "mean"),
            mean_model_based_ci_covers_truth=("model_based_ci_covers_truth", "mean"),
            true_value=("true_value", "first"),
            mean_estimate=("estimate", "mean"),
            mean_annotation_rate=("annotation_rate", "mean"),
            mean_annotation_cost=("mean_annotation_cost", "mean"),
        )
        .sort_values(["sample_size", "d1", "d2", "rmse", "method"])
    )
    return overall, by_regime


def plot_rmse(overall: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    if "budget" in overall.columns and overall["budget"].nunique() > 1:
        for (budget, method), sub in overall.groupby(["budget", "method"], sort=True):
            sub = sub.sort_values("sample_size")
            ax.plot(sub["sample_size"], sub["rmse"], marker="o", label=f"b={budget:g} {method}")
    else:
        for method in overall["method"].drop_duplicates():
            sub = overall[overall["method"] == method].sort_values("sample_size")
            ax.plot(sub["sample_size"], sub["rmse"], marker="o", label=method)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("RMSE across regimes and replications")
    ax.set_title("Plug-in KRR evaluation with optimized annotation")
    ax.grid(True, alpha=0.25)
    if overall["sample_size"].nunique() > 1:
        ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_value(value: object) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(markdown_value(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    overall: pd.DataFrame,
    by_regime: pd.DataFrame,
    variance_summary: pd.DataFrame,
    selected_hyperparameters: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    validation_summary = (
        validation_metrics.groupby(
            ["sample_size", "budget"] if "budget" in validation_metrics.columns else ["sample_size"],
            as_index=False,
        )
        .agg(
            r1_variance_corr=("r1_variance_corr", "mean"),
            r2_variance_corr=("r2_variance_corr", "mean"),
            priority_oracle_corr=("priority_oracle_corr", "mean"),
            expected_budget=("expected_budget", "mean"),
            expected_cost=("expected_cost", "mean"),
            prob_min=("prob_min", "mean"),
            prob_max=("prob_max", "mean"),
        )
        if not validation_metrics.empty
        else validation_metrics
    )
    lines = [
        "# Reward-Only Protocol Annotation for Plug-in KRR DTR Evaluation",
        "",
        "Synthetic-only benchmark. No protected cohort or annotation data are read.",
        "",
        "Question: with plug-in KRR accepted as a strong evaluator, can protocol-faithful reward annotation improve value estimates under a 50% reward-label budget?",
        "",
        "Ablation: `S1`, `A1`, `S2`, and `A2` are fully observed. Only `R1` and `R2` are annotated. The implemented monotone design is bundled for this ablation: sampled units reveal both rewards, so `C1 = C2` and monotonicity holds exactly.",
        "",
        "Designs:",
        "",
        "- `full_data_krr`: both rewards annotated for every unit.",
        "- `random_50pct`: random bundled reward annotation with the requested expected budget.",
        "- `optimized_dtr_priority_50pct`: clipped square-root style allocation toward high inverse-propensity DTR histories.",
        "- `protocol_estimated_reward_variance_50pct`: LEGACY clipped allocation with power-one overlap, `sqrt(v1_hat * O1 + v2_hat * O2)`; kept as a sensitivity baseline.",
        "- `protocol_oracle_reward_variance_50pct`: synthetic diagnostic using known reward-noise variances with the legacy power-one overlap; not deployable on real data.",
        "- `protocol_theorem_reward_variance_50pct`: theorem-form allocation `sqrt(v1_hat * sum_d mu_{1,d}^2 + v2_hat * sum_d mu_{1:2,d}^2)` (squared density ratios with regime match indicators; Thm optimal allocation, bundle case).",
        "- `protocol_theorem_oracle_variance_50pct`: theorem-form allocation with known reward-noise variances; synthetic diagnostic.",
        "- `protocol_theorem_regime_{d1}{d2}_50pct`: theorem-faithful per-regime campaigns; evaluate regime d under its own matched design (single-regime mu with match indicators).",
        "",
        "All partial-annotation designs use a positivity floor and train reward KRR nuisances with inverse annotation-propensity weights, matching the missingness-aware nuisance-estimation prescription in `unified_mdp_dtr_passive_annotation.pdf`.",
        "",
        f"Budget fractions: `{getattr(args, 'budget_values', args.budget)}` of full reward annotation. Full reward cost is 2.",
        f"Positivity floor: `{args.floor}`.",
        f"Overlap-only baseline exponent: `{args.priority_exponent}`.",
        f"DGP: `{args.dgp}`.",
        f"Reward noise profile: `{args.reward_noise_profile}`.",
        f"Heteroskedastic strength: `{args.heteroskedastic_strength}`.",
        f"Behavior propensity minimum: `{args.behavior_min_prob}`.",
        f"Behavior logit scale: `{args.behavior_logit_scale}`.",
        f"Estimators: `{args.estimators}`.",
        f"MC rep offset: `{args.rep_offset}`.",
        f"Common oracle subsampling n: `{args.common_oracle_n}` (`0` means independent samples by n).",
        f"Balancing settings: rff `{args.balancing_rff_features}`, cap `{args.balancing_cap_multiplier}`, balance penalty `{args.balancing_balance_penalty}`, chi2 penalty `{args.balancing_chi2_penalty}`, optimizer maxiter `{args.balancing_optimizer_maxiter}`.",
        "",
        "Selected sampler settings:",
        "",
        frame_to_markdown(selected_hyperparameters),
        "",
        f"Monte Carlo jobs: `{args.jobs}` (`0` means automatic worker count).",
        f"Elapsed wall-clock seconds: `{elapsed_seconds:.3f}`.",
        "",
        "Validation-only sampler diagnostics:",
        "",
        frame_to_markdown(validation_summary),
        "",
        "Variance profile:",
        "",
        frame_to_markdown(variance_summary),
        "",
        "Overall held-out metrics:",
        "",
        frame_to_markdown(overall),
        "",
        "Metrics by regime:",
        "",
        frame_to_markdown(by_regime),
        "",
        "Files:",
        "",
        "- `results.csv`: one row per method, replication, sample size, and regime.",
        "- `aggregate_metrics.csv`: pooled regime metrics.",
        "- `metrics_by_regime.csv`: regime-specific metrics.",
        "- `variance_profile.csv`: per-replication reward and transition variance diagnostics.",
        "- `variance_profile_summary.csv`: averaged variance diagnostics.",
        "- `validation_sampler_metrics.csv`: validation-only diagnostics for the estimated reward-variance sampler.",
        "- `selected_sampler_hyperparameters.csv`: fixed sampler settings and pilot sizes.",
        "- `rmse_by_method.png`: RMSE plot.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    if args.synthetic_smoke_test:
        args.sample_sizes = "250"
        args.reps = 2
        args.validation_reps = min(args.validation_reps, 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = parse_int_list(args.sample_sizes)
    regimes = parse_regimes(args.regimes)
    lambdas = parse_float_list(args.krr_lambdas)
    budgets = parse_float_list(args.budgets) if args.budgets else [args.budget]
    if not budgets:
        raise ValueError("At least one budget is required.")
    if any(budget <= 0.0 or budget > 1.0 for budget in budgets):
        raise ValueError("All budgets must be in (0, 1].")
    if args.floor < 0.0 or args.floor >= min(budgets):
        raise ValueError("--floor must be nonnegative and smaller than every budget.")
    if args.behavior_min_prob < 0.0 or args.behavior_min_prob >= 0.5:
        raise ValueError("--behavior-min-prob must be in [0, 0.5).")
    if args.behavior_logit_scale <= 0.0:
        raise ValueError("--behavior-logit-scale must be positive.")
    if args.rep_offset < 0:
        raise ValueError("--rep-offset must be nonnegative.")
    args.budget_values = ",".join(f"{budget:g}" for budget in budgets)
    estimators = parse_str_list(args.estimators)
    allowed_estimators = {"plugin", "drl", "balancing"}
    unknown_estimators = sorted(set(estimators) - allowed_estimators)
    if unknown_estimators:
        raise ValueError(f"Unknown estimators: {unknown_estimators}.")
    if not estimators:
        raise ValueError("At least one estimator is required.")
    if args.common_oracle_n > 0 and max(sample_sizes) > args.common_oracle_n:
        raise ValueError("--common-oracle-n must be at least the largest sample size.")
    balancing_settings = BalancingSettings(
        rff_features=args.balancing_rff_features,
        cap_multiplier=args.balancing_cap_multiplier,
        balance_penalty=args.balancing_balance_penalty,
        chi2_penalty=args.balancing_chi2_penalty,
        optimizer_maxiter=args.balancing_optimizer_maxiter,
    )

    rows: list[dict[str, float | str | int]] = []
    variance_rows: list[dict[str, float | str | int]] = []
    validation_rows: list[dict[str, float | str | int]] = []
    selected_rows: list[dict[str, float | str | int]] = []
    tasks: list[
        tuple[
            int,
            int,
            int,
            list[tuple[int, int]],
            list[str],
            list[float],
            float,
            float,
            str,
            str,
            float,
            float,
            float,
            int,
            RewardVarianceModel,
            BalancingSettings,
            list[float],
            int,
            float,
        ]
    ] = []
    for n in sample_sizes:
        pilot_n = args.pilot_n if args.pilot_n > 0 else n
        pilot_data = DTR.simulate_singh_dtr(
            n=pilot_n,
            p=args.p,
            seed=args.seed + 20_000 + n,
            dgp=args.dgp,
            reward_noise_profile=args.reward_noise_profile,
            heteroskedastic_strength=args.heteroskedastic_strength,
            behavior_min_prob=args.behavior_min_prob,
            behavior_logit_scale=args.behavior_logit_scale,
        )
        variance_model = fit_reward_variance_model(
            pilot_data,
            lambdas=lambdas,
            seed=args.seed + 21_000 + n,
        )
        for budget in budgets:
            selected_rows.append(
                {
                    "sample_size": n,
                    "pilot_n": pilot_n,
                    "validation_reps": args.validation_reps,
                    "budget": budget,
                    "floor": args.floor,
                    "dgp": args.dgp,
                    "overlap_baseline_exponent": args.priority_exponent,
                    "variance_min": variance_model.min_variance,
                    "behavior_min_prob": args.behavior_min_prob,
                    "behavior_logit_scale": args.behavior_logit_scale,
                    "pilot_fraction": args.pilot_fraction,
                    "priority_formula_legacy": "sqrt(v1_hat * O1 + v2_hat * O2), O1=1/g1_obs, O2=1/(g1_obs*g2_obs)",
                    "priority_formula_theorem": "sqrt(v1_hat * sum_d mu_{1,d}^2 + v2_hat * sum_d mu_{1:2,d}^2)",
                    "variance_labels": "charged in-sample pilot (pilot_fraction*budget)" if args.pilot_fraction > 0 else "external pilot (uncharged)",
                    "estimators": ",".join(estimators),
                    "selection_rule": "fixed_from_plan",
                    "test_metrics_used_for_selection": 0,
                }
            )
            validation_rows.extend(
                variance_model_validation_rows(
                    variance_model=variance_model,
                    n=n,
                    p=args.p,
                    budget=budget,
                    floor=args.floor,
                    dgp=args.dgp,
                    reward_noise_profile=args.reward_noise_profile,
                    heteroskedastic_strength=args.heteroskedastic_strength,
                    behavior_min_prob=args.behavior_min_prob,
                    behavior_logit_scale=args.behavior_logit_scale,
                    seed=args.seed,
                    validation_reps=args.validation_reps,
                )
            )
        for local_rep in range(args.reps):
            rep = args.rep_offset + local_rep
            tasks.append(
                (
                    n,
                    rep,
                    args.p,
                    regimes,
                    estimators,
                    budgets,
                    args.floor,
                    args.priority_exponent,
                    args.dgp,
                    args.reward_noise_profile,
                    args.heteroskedastic_strength,
                    args.behavior_min_prob,
                    args.behavior_logit_scale,
                    args.common_oracle_n,
                    variance_model,
                    balancing_settings,
                    lambdas,
                    args.seed + 100_000 + 10_007 * rep + (args.common_oracle_n if args.common_oracle_n > 0 else n),
                    args.pilot_fraction,
                )
            )

    if args.jobs == 1:
        task_outputs = [run_replication_task(task) for task in tasks]
    else:
        jobs = args.jobs if args.jobs > 0 else max(1, min(mp.cpu_count(), len(tasks)))
        with mp.get_context("fork").Pool(processes=jobs) as pool:
            task_outputs = pool.map(run_replication_task, tasks)

    for rep_rows, rep_variance_rows in task_outputs:
        rows.extend(rep_rows)
        variance_rows.extend(rep_variance_rows)

    results = pd.DataFrame.from_records(rows)
    variance_profile = pd.DataFrame.from_records(variance_rows)
    validation_metrics = pd.DataFrame.from_records(validation_rows)
    selected_hyperparameters = pd.DataFrame.from_records(selected_rows)
    variance_summary = (
        variance_profile.groupby(["sample_size", "path"], as_index=False)
        .agg(
            n_units=("n_units", "mean"),
            r1_noise_var_mean=("r1_noise_var_mean", "mean"),
            r2_noise_var_mean=("r2_noise_var_mean", "mean"),
            reward_noise_var_mean=("reward_noise_var_mean", "mean"),
            reward_noise_sd_mean=("reward_noise_sd_mean", "mean"),
            reward_noise_sd_min=("reward_noise_sd_min", "mean"),
            reward_noise_sd_max=("reward_noise_sd_max", "mean"),
            r1_mean_var=("r1_mean_var", "mean"),
            r2_mean_var=("r2_mean_var", "mean"),
            reward_mean_var=("reward_mean_var", "mean"),
            outcome_var=("outcome_var", "mean"),
            x2_empirical_var_mean_diag=("x2_empirical_var_mean_diag", "mean"),
            transition_conditional_noise_var_diag=("transition_conditional_noise_var_diag", "mean"),
            transition_conditional_noise_adjacent_cov=("transition_conditional_noise_adjacent_cov", "mean"),
            dtr_priority_mean=("dtr_priority_mean", "mean"),
            priority_noise_var_corr=("priority_noise_var_corr", "mean"),
        )
        .sort_values(["sample_size", "path"])
    )
    overall, by_regime = aggregate_metrics(results)
    results.to_csv(output_dir / "results.csv", index=False)
    overall.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    by_regime.to_csv(output_dir / "metrics_by_regime.csv", index=False)
    variance_profile.to_csv(output_dir / "variance_profile.csv", index=False)
    variance_summary.to_csv(output_dir / "variance_profile_summary.csv", index=False)
    validation_metrics.to_csv(output_dir / "validation_sampler_metrics.csv", index=False)
    selected_hyperparameters.to_csv(output_dir / "selected_sampler_hyperparameters.csv", index=False)
    plot_rmse(overall, output_dir / "rmse_by_method.png")
    elapsed_seconds = time.perf_counter() - start_time
    pd.DataFrame.from_records(
        [
            {
                "elapsed_wall_seconds": elapsed_seconds,
                "timing_source": "script_perf_counter",
                "rep_offset": args.rep_offset,
            }
        ]
    ).to_csv(output_dir / "run_metadata.csv", index=False)
    write_summary(
        output_dir,
        args,
        overall,
        by_regime,
        variance_summary,
        selected_hyperparameters,
        validation_metrics,
        elapsed_seconds,
    )

    print(f"Wrote {len(results)} result rows to {output_dir}")
    print(f"Elapsed wall-clock seconds: {elapsed_seconds:.3f}")
    print(overall.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", default="500")
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--rep-offset", type=int, default=0, help="Global Monte Carlo replication offset for incremental batch runs.")
    parser.add_argument("--validation-reps", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel Monte Carlo workers; 0 uses an automatic worker count.")
    parser.add_argument("--pilot-n", type=int, default=0, help="External pilot sample size for the variance-model validation diagnostics; 0 uses each test sample size.")
    parser.add_argument(
        "--pilot-fraction",
        type=float,
        default=0.3,
        help="Fraction of each budget spent on a random in-sample pilot batch that pays for the variance-model labels (charged two-batch design). 0 reproduces the uncharged external-pilot behavior.",
    )
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--regimes", default=DEFAULT_REGIMES)
    parser.add_argument("--estimators", default="plugin", help="Comma-separated estimators: plugin,drl,balancing.")
    parser.add_argument("--common-oracle-n", type=int, default=0, help="Draw one dataset of this size per MC rep and use nested subsamples for each n; 0 disables this.")
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--budgets", default="", help="Comma-separated budget fractions to evaluate in one MC run. If set, overrides --budget.")
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--priority-exponent", type=float, default=0.50)
    parser.add_argument("--dgp", choices=["singh", "paper-two-stage"], default="singh")
    parser.add_argument(
        "--reward-noise-profile",
        choices=["homoskedastic", "dtr-overlap", "paper-two-stage"],
        default="homoskedastic",
    )
    parser.add_argument("--heteroskedastic-strength", type=float, default=0.0)
    parser.add_argument("--behavior-min-prob", type=float, default=0.1, help="Lower truncation bound for behavior propensities; default preserves the original DTR DGP.")
    parser.add_argument("--behavior-logit-scale", type=float, default=1.0, help="Multiplier for behavior propensity logits; values >1 create weaker overlap.")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--krr-lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--crossfit-folds", type=int, default=CROSSFIT_FOLDS, help="K-fold cross-fitting of the KRR nuisances; 0/1 keeps legacy in-sample fitting.")
    parser.add_argument("--stagewise-arm", action="store_true", default=STAGEWISE_ARM, help="Add forward-monotone stagewise per-regime designs (protocol_stagewise_regime_*) solved by the joint T=2 allocation.")
    parser.add_argument("--balancing-rff-features", type=int, default=30)
    parser.add_argument("--balancing-cap-multiplier", type=float, default=6.0)
    parser.add_argument("--balancing-balance-penalty", type=float, default=25.0)
    parser.add_argument("--balancing-chi2-penalty", type=float, default=0.1)
    parser.add_argument("--balancing-optimizer-maxiter", type=int, default=120)
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    global CROSSFIT_FOLDS
    CROSSFIT_FOLDS = int(args.crossfit_folds)
    os.environ["KRR_CROSSFIT_FOLDS"] = str(CROSSFIT_FOLDS)
    global STAGEWISE_ARM
    STAGEWISE_ARM = bool(args.stagewise_arm)
    os.environ["KRR_STAGEWISE_ARM"] = "1" if STAGEWISE_ARM else "0"
    run_experiment(args)


if __name__ == "__main__":
    main()
