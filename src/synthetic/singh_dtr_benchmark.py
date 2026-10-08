#!/usr/bin/env python3
"""Dynamic treatment-regime benchmark inspired by Singh, Xu, and Gretton.

This script is synthetic-only. It implements the semiparametric time-varying
treatment-effect simulation from Supplement J.4 of "Sequential Kernel Embedding
for Mediated and Time-Varying Dose Response Curves" and compares recursive
kernel ridge regression with recursive kernel balancing weights.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

try:
    import cvxpy as cp
except ImportError:  # Annotation-only benchmarks import this module for the simulator.
    cp = None


DEFAULT_OUTPUT_DIR = "results/synthetic/singh_dtr_benchmark"
DEFAULT_LAMBDAS = "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1"
DEFAULT_REGIMES = "0,0;1,0;0,1;1,1"


@dataclass(frozen=True)
class DtrData:
    x1: np.ndarray
    d1: np.ndarray
    x2: np.ndarray
    d2: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    y: np.ndarray
    r1_mean: np.ndarray
    r2_mean: np.ndarray
    reward_mean: np.ndarray
    r1_noise_scale: np.ndarray
    r2_noise_scale: np.ndarray
    reward_noise_scale: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class MixedKernel:
    continuous_idx: tuple[int, ...]
    discrete_idx: tuple[int, ...]
    length_scales: np.ndarray

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.continuous_idx:
            idx = np.asarray(self.continuous_idx)
            x_cont = x[:, idx] / self.length_scales
            y_cont = y[:, idx] / self.length_scales
            x_norm = np.sum(np.square(x_cont), axis=1)[:, None]
            y_norm = np.sum(np.square(y_cont), axis=1)[None, :]
            dist2 = np.maximum(x_norm + y_norm - 2.0 * x_cont @ y_cont.T, 0.0)
            out = np.exp(-0.5 * dist2)
        else:
            out = np.ones((x.shape[0], y.shape[0]), dtype=float)
        if self.discrete_idx:
            idx = np.asarray(self.discrete_idx)
            matches = (x[:, None, idx] == y[None, :, idx]).all(axis=2)
            out = out * matches
        return out


@dataclass(frozen=True)
class KernelRidgeModel:
    train_x: np.ndarray
    alpha: np.ndarray
    kernel: MixedKernel
    lambda_value: float
    gcv: float

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        k = self.kernel.gram(self.train_x, np.asarray(x_new, dtype=float))
        return k.T @ self.alpha


@dataclass(frozen=True)
class RffMap:
    continuous_idx: tuple[int, ...]
    discrete_idx: tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    categories: tuple[tuple[int, ...], ...]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if self.continuous_idx:
            idx = np.asarray(self.continuous_idx)
            z = (x[:, idx] - self.mean) / self.scale
            rff = math.sqrt(2.0 / self.omega.shape[1]) * np.cos(z @ self.omega + self.phase)
        else:
            rff = np.ones((n, 1), dtype=float)

        if not self.discrete_idx:
            return rff

        idx = np.asarray(self.discrete_idx)
        disc = np.rint(x[:, idx]).astype(int)
        one_hot = np.zeros((n, len(self.categories)), dtype=float)
        category_index = {cat: j for j, cat in enumerate(self.categories)}
        for i, row in enumerate(disc):
            j = category_index.get(tuple(int(v) for v in row))
            if j is not None:
                one_hot[i, j] = 1.0
        return (one_hot[:, :, None] * rff[:, None, :]).reshape(n, -1)


@dataclass(frozen=True)
class BalancingConfig:
    rff_features: int
    cap_multiplier: float
    balance_penalty: float
    chi2_penalty: float


@dataclass(frozen=True)
class WeightResult:
    weights: np.ndarray
    ess: float
    max_weight: float
    cap_hit_rate: float
    imbalance: float
    solver: str
    status: str


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_regimes(value: str) -> list[tuple[int, int]]:
    regimes: list[tuple[int, int]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        d1_raw, d2_raw = chunk.split(",")
        d1 = int(d1_raw)
        d2 = int(d2_raw)
        if d1 not in (0, 1) or d2 not in (0, 1):
            raise ValueError(f"Regime entries must be binary, got {chunk!r}.")
        regimes.append((d1, d2))
    if not regimes:
        raise ValueError("At least one regime is required.")
    return regimes


def beta_vector(p: int) -> np.ndarray:
    j = np.arange(1, p + 1, dtype=float)
    return np.power(j, -2.0)


def covariance_matrix(p: int) -> np.ndarray:
    sigma = np.eye(p, dtype=float)
    if p > 1:
        off = np.arange(p - 1)
        sigma[off, off + 1] = 0.5
        sigma[off + 1, off] = 0.5
    return sigma


def truncated_logistic(z: np.ndarray, min_prob: float = 0.1) -> np.ndarray:
    if min_prob < 0.0 or min_prob >= 0.5:
        raise ValueError("min_prob must be in [0, 0.5).")
    return min_prob + (1.0 - 2.0 * min_prob) * expit(z)


def reward_noise_scales(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
    profile: str,
    heteroskedastic_strength: float,
) -> np.ndarray:
    if profile == "homoskedastic":
        return np.ones_like(p1, dtype=float)
    if profile != "dtr-overlap":
        raise ValueError(f"Unknown reward noise profile: {profile!r}")
    if heteroskedastic_strength < 0:
        raise ValueError("heteroskedastic_strength must be nonnegative.")

    path_prob = np.where(d1 == 1, p1, 1.0 - p1) * np.where(d2 == 1, p2, 1.0 - p2)
    raw = np.power(1.0 / np.sqrt(np.maximum(path_prob, 1e-8)), heteroskedastic_strength)
    raw = raw / math.sqrt(float(np.mean(np.square(raw))))
    return np.maximum(raw, 1e-6)


def _safe_column(x: np.ndarray, index: int) -> np.ndarray:
    if x.shape[1] <= index:
        return np.zeros(x.shape[0], dtype=float)
    return x[:, index]


def _paper_arm_variance(
    x: np.ndarray,
    action: np.ndarray,
    high_variance_action: int,
    contrast: float,
) -> np.ndarray:
    low = np.maximum(1.3 + 0.4 * np.sin(_safe_column(x, 0)), 0.05)
    high = np.maximum(3.5 + 0.3 * np.cos(_safe_column(x, 2)), 0.05)
    high = high * max(float(contrast), 1.0)
    return np.where(action == float(high_variance_action), high, low)


def simulate_paper_two_stage_dtr(
    n: int,
    p: int,
    seed: int,
    reward_noise_profile: str = "paper-two-stage",
    heteroskedastic_strength: float = 1.0,
    behavior_min_prob: float = 0.1,
    behavior_logit_scale: float = 1.0,
) -> DtrData:
    """Two-stage extension of the paper's single-treatment synthetic ATE DGP.

    The true regime value is kept equal to ``true_regime_value(d1, d2)`` by
    using mean-zero covariate effects and a transition independent of the
    intervened first-stage action. The paper-style arm variance assigns larger
    reward innovation variance to the less common control arm at each stage.
    """
    if reward_noise_profile not in {"homoskedastic", "paper-two-stage"}:
        raise ValueError(
            "paper-two-stage DGP supports reward_noise_profile in "
            "{'homoskedastic', 'paper-two-stage'}."
        )
    if heteroskedastic_strength < 0:
        raise ValueError("heteroskedastic_strength must be nonnegative.")

    rng = np.random.default_rng(seed)
    beta = beta_vector(p)
    x1 = rng.normal(0.0, 1.0, size=(n, p))
    p1_logit = 0.5 - _safe_column(x1, 1) + _safe_column(x1, 2)
    p1 = truncated_logistic(behavior_logit_scale * p1_logit, min_prob=behavior_min_prob)
    d1 = rng.binomial(1, p1).astype(float)

    innovation = rng.normal(0.0, 1.0, size=(n, p))
    x2 = 0.5 * x1 + math.sqrt(0.75) * innovation
    p2_logit = 0.5 - _safe_column(x2, 1) + _safe_column(x2, 2) + 0.35 * d1
    p2 = truncated_logistic(behavior_logit_scale * p2_logit, min_prob=behavior_min_prob)
    d2 = rng.binomial(1, p2).astype(float)

    x1_signal = _safe_column(x1, 0) - 2.0 * _safe_column(x1, 1)
    x2_signal = _safe_column(x2, 0) - 2.0 * _safe_column(x2, 1)
    r1_mean = 0.5 * x1_signal + 1.1 * d1
    r2_mean = 0.5 * x2_signal + 2.2 * d2 + 0.5 * d1 * d2

    if reward_noise_profile == "homoskedastic":
        r1_var = np.ones(n, dtype=float)
        r2_var = np.ones(n, dtype=float)
    else:
        contrast = max(float(heteroskedastic_strength), 1.0)
        r1_var = _paper_arm_variance(x1, d1, high_variance_action=0, contrast=contrast)
        r2_var = _paper_arm_variance(x2, d2, high_variance_action=0, contrast=contrast)

    r1_noise_scale = np.sqrt(np.maximum(r1_var, 1e-8))
    r2_noise_scale = np.sqrt(np.maximum(r2_var, 1e-8))
    eps1 = rng.normal(0.0, r1_noise_scale, size=n)
    eps2 = rng.normal(0.0, r2_noise_scale, size=n)
    r1 = r1_mean + eps1
    r2 = r2_mean + eps2
    reward_mean = r1_mean + r2_mean
    y = r1 + r2
    return DtrData(
        x1=x1,
        d1=d1,
        x2=x2,
        d2=d2,
        r1=r1,
        r2=r2,
        y=y,
        r1_mean=r1_mean,
        r2_mean=r2_mean,
        reward_mean=reward_mean,
        r1_noise_scale=r1_noise_scale,
        r2_noise_scale=r2_noise_scale,
        reward_noise_scale=np.sqrt(np.maximum(r1_var + r2_var, 1e-8)),
        p1=p1,
        p2=p2,
        beta=beta,
    )


def simulate_singh_dtr(
    n: int,
    p: int,
    seed: int,
    dgp: str = "singh",
    reward_noise_profile: str = "homoskedastic",
    heteroskedastic_strength: float = 0.0,
    behavior_min_prob: float = 0.1,
    behavior_logit_scale: float = 1.0,
) -> DtrData:
    if dgp == "paper-two-stage":
        return simulate_paper_two_stage_dtr(
            n=n,
            p=p,
            seed=seed,
            reward_noise_profile=reward_noise_profile,
            heteroskedastic_strength=heteroskedastic_strength,
            behavior_min_prob=behavior_min_prob,
            behavior_logit_scale=behavior_logit_scale,
        )
    if dgp != "singh":
        raise ValueError(f"Unknown DGP: {dgp!r}")

    rng = np.random.default_rng(seed)
    beta = beta_vector(p)
    sigma = covariance_matrix(p)
    w1 = rng.multivariate_normal(np.zeros(p), sigma, size=n)
    w2 = rng.multivariate_normal(np.zeros(p), sigma, size=n)
    x1 = w1
    p1 = truncated_logistic(behavior_logit_scale * (x1 @ beta), min_prob=behavior_min_prob)
    d1 = rng.binomial(1, p1).astype(float)
    x2 = 0.5 * (1.0 - d1[:, None]) * x1 + 0.5 * w2
    p2 = truncated_logistic(
        behavior_logit_scale * (0.5 * (x1 @ beta) + x2 @ beta - 0.2 * d1),
        min_prob=behavior_min_prob,
    )
    d2 = rng.binomial(1, p2).astype(float)
    noise_scale = reward_noise_scales(
        p1=p1,
        d1=d1,
        p2=p2,
        d2=d2,
        profile=reward_noise_profile,
        heteroskedastic_strength=heteroskedastic_strength,
    )
    r1_noise_scale = noise_scale / math.sqrt(2.0)
    r2_noise_scale = noise_scale / math.sqrt(2.0)
    eps1 = rng.normal(0.0, r1_noise_scale, size=n)
    eps2 = rng.normal(0.0, r2_noise_scale, size=n)
    r1_mean = 0.5 * (1.2 * d1 + 1.2 * (x1 @ beta) + np.square(d1) + d1 * x1[:, 0])
    r2_mean = (
        1.2 * d2
        + 1.2 * (x2 @ beta)
        + np.square(d2)
        + d2 * x2[:, 0]
        + 0.5 * d1 * d2
    )
    r1 = r1_mean + eps1
    r2 = r2_mean + eps2
    reward_mean = r1_mean + r2_mean
    y = r1 + r2
    return DtrData(
        x1=x1,
        d1=d1,
        x2=x2,
        d2=d2,
        r1=r1,
        r2=r2,
        y=y,
        r1_mean=r1_mean,
        r2_mean=r2_mean,
        reward_mean=reward_mean,
        r1_noise_scale=r1_noise_scale,
        r2_noise_scale=r2_noise_scale,
        reward_noise_scale=noise_scale,
        p1=p1,
        p2=p2,
        beta=beta,
    )


def subset_dtr_data(data: DtrData, indices: np.ndarray) -> DtrData:
    idx = np.asarray(indices, dtype=int)
    return DtrData(
        x1=data.x1[idx],
        d1=data.d1[idx],
        x2=data.x2[idx],
        d2=data.d2[idx],
        r1=data.r1[idx],
        r2=data.r2[idx],
        y=data.y[idx],
        r1_mean=data.r1_mean[idx],
        r2_mean=data.r2_mean[idx],
        reward_mean=data.reward_mean[idx],
        r1_noise_scale=data.r1_noise_scale[idx],
        r2_noise_scale=data.r2_noise_scale[idx],
        reward_noise_scale=data.reward_noise_scale[idx],
        p1=data.p1[idx],
        p2=data.p2[idx],
        beta=data.beta,
    )


def true_regime_value(d1: int, d2: int) -> float:
    return 1.1 * d1 + 2.2 * d2 + 0.5 * d1 * d2


def stage1_features(data: DtrData, d1_override: int | None = None) -> np.ndarray:
    d1 = data.d1 if d1_override is None else np.full_like(data.d1, float(d1_override))
    return np.column_stack([data.x1, d1])


def stage2_features(data: DtrData, d2_override: int | None = None) -> np.ndarray:
    d2 = data.d2 if d2_override is None else np.full_like(data.d2, float(d2_override))
    return np.column_stack([data.x1, data.d1, data.x2, d2])


def stage1_indices(p: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(range(p)), (p,)


def stage2_indices(p: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    continuous = tuple(range(p)) + tuple(range(p + 1, 2 * p + 1))
    discrete = (p, 2 * p + 1)
    return continuous, discrete


def median_length_scales(x: np.ndarray, continuous_idx: Sequence[int], seed: int) -> np.ndarray:
    if not continuous_idx:
        return np.ones(0, dtype=float)
    rng = np.random.default_rng(seed)
    cont = np.asarray(x[:, continuous_idx], dtype=float)
    n = cont.shape[0]
    pair_count = min(20_000, max(1, n * (n - 1) // 2))
    a = rng.integers(0, n, size=pair_count)
    b = rng.integers(0, n, size=pair_count)
    distances = np.linalg.norm(cont[a] - cont[b], axis=1)
    bandwidth = float(np.median(distances))
    if not np.isfinite(bandwidth) or bandwidth <= 1e-8:
        bandwidth = float(np.sqrt(np.sum(np.var(cont, axis=0))))
    if not np.isfinite(bandwidth) or bandwidth <= 1e-8:
        bandwidth = 1.0
    return np.full(len(continuous_idx), bandwidth, dtype=float)


def fit_kernel_ridge(
    x: np.ndarray,
    y: np.ndarray,
    continuous_idx: Sequence[int],
    discrete_idx: Sequence[int],
    lambdas: Sequence[float],
    seed: int,
) -> KernelRidgeModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    kernel = MixedKernel(
        continuous_idx=tuple(continuous_idx),
        discrete_idx=tuple(discrete_idx),
        length_scales=median_length_scales(x, continuous_idx, seed),
    )
    k = kernel.gram(x, x)
    k = 0.5 * (k + k.T)
    evals, evecs = np.linalg.eigh(k)
    evals = np.maximum(evals, 0.0)
    y_rot = evecs.T @ y

    best: dict[str, float | np.ndarray] | None = None
    for lambda_value in lambdas:
        denom = evals + n * lambda_value
        smoother = evals / denom
        y_hat = evecs @ (smoother * y_rot)
        residual = y - y_hat
        trace_h = float(np.sum(smoother))
        gcv_denom = max(1.0 - trace_h / n, 1e-8)
        gcv = float(np.mean(np.square(residual)) / (gcv_denom * gcv_denom))
        alpha = evecs @ (y_rot / denom)
        if best is None or gcv < float(best["gcv"]):
            best = {"gcv": gcv, "lambda": float(lambda_value), "alpha": alpha}

    assert best is not None
    return KernelRidgeModel(
        train_x=x,
        alpha=np.asarray(best["alpha"], dtype=float),
        kernel=kernel,
        lambda_value=float(best["lambda"]),
        gcv=float(best["gcv"]),
    )


def recursive_krr_components(
    data: DtrData,
    d1: int,
    d2: int,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    p = data.x1.shape[1]
    continuous2, discrete2 = stage2_indices(p)
    gamma_model = fit_kernel_ridge(
        stage2_features(data),
        data.y,
        continuous2,
        discrete2,
        lambdas=lambdas,
        seed=seed,
    )
    q2_obs = gamma_model.predict(stage2_features(data, d2_override=d2))

    continuous1, discrete1 = stage1_indices(p)
    omega_model = fit_kernel_ridge(
        stage1_features(data),
        q2_obs,
        continuous1,
        discrete1,
        lambdas=lambdas,
        seed=seed + 17,
    )
    omega_at_regime = omega_model.predict(stage1_features(data, d1_override=d1))
    estimate = float(np.mean(omega_at_regime))
    diagnostics = {
        "gamma_lambda": gamma_model.lambda_value,
        "gamma_gcv": gamma_model.gcv,
        "omega_lambda": omega_model.lambda_value,
        "omega_gcv": omega_model.gcv,
    }
    return q2_obs, omega_at_regime, estimate, diagnostics


def recursive_krr_plugin(
    data: DtrData,
    d1: int,
    d2: int,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    _, _, estimate, diagnostics = recursive_krr_components(data, d1, d2, lambdas, seed)
    return estimate, diagnostics


def oracle_sequential_ipw(data: DtrData, d1: int, d2: int) -> tuple[float, dict[str, float]]:
    g1 = np.where(d1 == 1, data.p1, 1.0 - data.p1)
    g2 = np.where(d2 == 1, data.p2, 1.0 - data.p2)
    indicator = ((data.d1 == d1) & (data.d2 == d2)).astype(float)
    raw = indicator / np.maximum(g1 * g2, 1e-8)
    weight_sum = float(np.sum(raw))
    if weight_sum <= 0:
        return float("nan"), {"ess": 0.0, "max_weight": 0.0}
    weights = raw / weight_sum
    estimate = float(weights @ data.y)
    ess = float(1.0 / np.sum(np.square(weights)))
    return estimate, {"ess": ess, "max_weight": float(np.max(weights))}


def oracle_propensity_drl(
    data: DtrData,
    d1: int,
    d2: int,
    lambdas: Sequence[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    q2_obs, omega_at_regime, _, diagnostics = recursive_krr_components(data, d1, d2, lambdas, seed)
    g1 = np.where(d1 == 1, data.p1, 1.0 - data.p1)
    g2 = np.where(d2 == 1, data.p2, 1.0 - data.p2)
    i1 = (data.d1 == d1).astype(float)
    i2 = (data.d2 == d2).astype(float)
    signal = (
        omega_at_regime
        + i1 / np.maximum(g1, 1e-8) * (q2_obs - omega_at_regime)
        + i1 * i2 / np.maximum(g1 * g2, 1e-8) * (data.y - q2_obs)
    )
    estimate = float(np.mean(signal))
    diagnostics = dict(diagnostics)
    diagnostics["mean_abs_correction"] = float(np.mean(np.abs(signal - omega_at_regime)))
    return estimate, diagnostics


def fit_rff_map(
    source: np.ndarray,
    target: np.ndarray,
    continuous_idx: Sequence[int],
    discrete_idx: Sequence[int],
    n_features: int,
    seed: int,
) -> RffMap:
    rng = np.random.default_rng(seed)
    continuous_idx = tuple(continuous_idx)
    discrete_idx = tuple(discrete_idx)
    combined = np.vstack([source, target])
    if continuous_idx:
        cont = combined[:, continuous_idx]
        mean = np.mean(cont, axis=0)
        scale = np.std(cont, axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        z = (cont - mean) / scale
        pair_count = min(20_000, max(1, z.shape[0] * (z.shape[0] - 1) // 2))
        a = rng.integers(0, z.shape[0], size=pair_count)
        b = rng.integers(0, z.shape[0], size=pair_count)
        bandwidth = float(np.median(np.linalg.norm(z[a] - z[b], axis=1)))
        bandwidth = bandwidth if bandwidth > 1e-8 else 1.0
        omega = rng.normal(0.0, 1.0 / bandwidth, size=(len(continuous_idx), n_features))
        phase = rng.uniform(0.0, 2.0 * math.pi, size=n_features)
    else:
        mean = np.ones(0, dtype=float)
        scale = np.ones(0, dtype=float)
        omega = np.ones((0, 1), dtype=float)
        phase = np.zeros(1, dtype=float)

    if discrete_idx:
        disc = np.rint(combined[:, discrete_idx]).astype(int)
        categories = tuple(sorted({tuple(int(v) for v in row) for row in disc}))
    else:
        categories = tuple()

    return RffMap(
        continuous_idx=continuous_idx,
        discrete_idx=discrete_idx,
        mean=mean,
        scale=scale,
        omega=omega,
        phase=phase,
        categories=categories,
    )


def normalize_prob(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(weights.shape[0], 1.0 / weights.shape[0])
    return weights / total


def solve_balancing_weights(
    phi_source: np.ndarray,
    target_mean: np.ndarray,
    base_weights: np.ndarray,
    cap_multiplier: float,
    balance_penalty: float,
    chi2_penalty: float,
    optimizer_maxiter: int,
) -> WeightResult:
    if cp is None:
        raise RuntimeError("cvxpy is required for recursive kernel balancing weights.")
    n = phi_source.shape[0]
    base = normalize_prob(base_weights)
    cap = np.maximum(cap_multiplier * base, 1.0 / (n * 1_000_000.0))
    base_safe = np.maximum(base, 1.0 / (n * 1_000_000.0))

    w = cp.Variable(n)
    imbalance_expr = phi_source.T @ w - target_mean
    chi2_expr = cp.sum_squares(cp.multiply(1.0 / np.sqrt(base_safe), w - base))
    objective = cp.Minimize(balance_penalty * cp.sum_squares(imbalance_expr) + 0.5 * chi2_penalty * chi2_expr)
    constraints = [w >= 0.0, cp.sum(w) == 1.0, w <= cap]
    problem = cp.Problem(objective, constraints)

    solver = "OSQP"
    try:
        problem.solve(solver=cp.OSQP, max_iter=optimizer_maxiter, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
    except Exception:
        try:
            solver = "CLARABEL"
            problem.solve(solver=cp.CLARABEL, max_iter=optimizer_maxiter, verbose=False)
        except Exception:
            solver = "fallback"

    if w.value is None or not np.all(np.isfinite(w.value)):
        weights = base
        status = "fallback_base"
    else:
        weights = normalize_prob(np.asarray(w.value).reshape(-1))
        status = str(problem.status)

    imbalance = float(np.linalg.norm(phi_source.T @ weights - target_mean) / math.sqrt(max(phi_source.shape[1], 1)))
    ess = float(1.0 / np.sum(np.square(weights)))
    cap_hit_rate = float(np.mean(weights >= cap - 1e-7))
    return WeightResult(
        weights=weights,
        ess=ess,
        max_weight=float(np.max(weights)),
        cap_hit_rate=cap_hit_rate,
        imbalance=imbalance,
        solver=solver,
        status=status,
    )


def recursive_kernel_balancing(
    data: DtrData,
    d1: int,
    d2: int,
    config: BalancingConfig,
    seed: int,
    optimizer_maxiter: int,
) -> tuple[float, dict[str, float | str]]:
    p = data.x1.shape[1]
    n = data.y.shape[0]
    continuous1, discrete1 = stage1_indices(p)
    source1 = stage1_features(data)
    target1 = stage1_features(data, d1_override=d1)
    rff1 = fit_rff_map(source1, target1, continuous1, discrete1, config.rff_features, seed)
    phi_source1 = rff1.transform(source1)
    phi_target1 = rff1.transform(target1)
    w1 = solve_balancing_weights(
        phi_source=phi_source1,
        target_mean=np.mean(phi_target1, axis=0),
        base_weights=np.full(n, 1.0 / n),
        cap_multiplier=config.cap_multiplier,
        balance_penalty=config.balance_penalty,
        chi2_penalty=config.chi2_penalty,
        optimizer_maxiter=optimizer_maxiter,
    )

    continuous2, discrete2 = stage2_indices(p)
    source2 = stage2_features(data)
    target2 = stage2_features(data, d2_override=d2)
    rff2 = fit_rff_map(source2, target2, continuous2, discrete2, config.rff_features, seed + 101)
    phi_source2 = rff2.transform(source2)
    phi_target2 = rff2.transform(target2)
    w2 = solve_balancing_weights(
        phi_source=phi_source2,
        target_mean=w1.weights @ phi_target2,
        base_weights=w1.weights,
        cap_multiplier=config.cap_multiplier,
        balance_penalty=config.balance_penalty,
        chi2_penalty=config.chi2_penalty,
        optimizer_maxiter=optimizer_maxiter,
    )

    estimate = float(w2.weights @ data.y)
    diagnostics: dict[str, float | str] = {
        "stage1_ess": w1.ess,
        "stage2_ess": w2.ess,
        "stage1_max_weight": w1.max_weight,
        "stage2_max_weight": w2.max_weight,
        "stage1_cap_hit_rate": w1.cap_hit_rate,
        "stage2_cap_hit_rate": w2.cap_hit_rate,
        "stage1_imbalance": w1.imbalance,
        "stage2_imbalance": w2.imbalance,
        "stage1_solver": f"cvxpy:{w1.solver}",
        "stage2_solver": f"cvxpy:{w2.solver}",
        "stage1_status": w1.status,
        "stage2_status": w2.status,
    }
    return estimate, diagnostics


def empty_like_diagnostics() -> dict[str, float | str]:
    return {
        "gamma_lambda": np.nan,
        "omega_lambda": np.nan,
        "ipw_ess": np.nan,
        "stage1_ess": np.nan,
        "stage2_ess": np.nan,
        "stage2_max_weight": np.nan,
    }


def estimate_methods(
    data: DtrData,
    regime: tuple[int, int],
    lambdas: Sequence[float],
    balancing_config: BalancingConfig,
    seed: int,
    optimizer_maxiter: int,
) -> list[dict[str, float | str | int]]:
    d1, d2 = regime
    true_value = true_regime_value(d1, d2)
    rows: list[dict[str, float | str | int]] = []

    plugin_estimate, plugin_diag = recursive_krr_plugin(data, d1, d2, lambdas, seed=seed)
    rows.append(make_result_row("recursive_krr_plugin", d1, d2, true_value, plugin_estimate, plugin_diag))

    drl_estimate, drl_diag = oracle_propensity_drl(data, d1, d2, lambdas, seed=seed + 7)
    rows.append(make_result_row("oracle_propensity_drl", d1, d2, true_value, drl_estimate, drl_diag))

    ipw_estimate, ipw_diag = oracle_sequential_ipw(data, d1, d2)
    rows.append(make_result_row("oracle_sequential_ipw", d1, d2, true_value, ipw_estimate, {"ipw_ess": ipw_diag["ess"], "ipw_max_weight": ipw_diag["max_weight"]}))

    balancing_estimate, balancing_diag = recursive_kernel_balancing(
        data,
        d1,
        d2,
        balancing_config,
        seed=seed + 19,
        optimizer_maxiter=optimizer_maxiter,
    )
    rows.append(make_result_row("recursive_kernel_balancing", d1, d2, true_value, balancing_estimate, balancing_diag))

    return rows


def make_result_row(
    method: str,
    d1: int,
    d2: int,
    true_value: float,
    estimate: float,
    diagnostics: dict[str, float | str],
) -> dict[str, float | str | int]:
    error = float(estimate - true_value)
    row: dict[str, float | str | int] = {
        "method": method,
        "d1": d1,
        "d2": d2,
        "true_value": true_value,
        "estimate": float(estimate),
        "error": error,
        "abs_error": abs(error),
        "squared_error": error * error,
    }
    row.update(diagnostics)
    return row


def tune_balancing_for_n(
    n: int,
    p: int,
    regimes: Sequence[tuple[int, int]],
    configs: Sequence[BalancingConfig],
    validation_reps: int,
    seed: int,
    optimizer_maxiter: int,
    reward_noise_profile: str,
    heteroskedastic_strength: float,
) -> tuple[BalancingConfig, pd.DataFrame]:
    records: list[dict[str, float | int]] = []
    for config_index, config in enumerate(configs):
        errors: list[float] = []
        ess_values: list[float] = []
        imbalance_values: list[float] = []
        for rep in range(validation_reps):
            data = simulate_singh_dtr(
                n=n,
                p=p,
                seed=seed + 10_000 + 101 * rep + config_index,
                reward_noise_profile=reward_noise_profile,
                heteroskedastic_strength=heteroskedastic_strength,
            )
            for d1, d2 in regimes:
                estimate, diag = recursive_kernel_balancing(
                    data,
                    d1,
                    d2,
                    config,
                    seed=seed + 20_000 + 997 * rep + 31 * config_index + 3 * d1 + d2,
                    optimizer_maxiter=optimizer_maxiter,
                )
                err = estimate - true_regime_value(d1, d2)
                errors.append(float(err * err))
                ess_values.append(float(diag["stage2_ess"]))
                imbalance_values.append(float(diag["stage2_imbalance"]))
        records.append(
            {
                "sample_size": n,
                "rff_features": config.rff_features,
                "cap_multiplier": config.cap_multiplier,
                "balance_penalty": config.balance_penalty,
                "chi2_penalty": config.chi2_penalty,
                "validation_rmse": math.sqrt(float(np.mean(errors))),
                "validation_mean_stage2_ess": float(np.mean(ess_values)),
                "validation_mean_stage2_imbalance": float(np.mean(imbalance_values)),
            }
        )

    summary = pd.DataFrame.from_records(records)
    summary = summary.sort_values(
        ["validation_rmse", "validation_mean_stage2_ess"],
        ascending=[True, False],
    ).reset_index(drop=True)
    best_row = summary.iloc[0]
    best = BalancingConfig(
        rff_features=int(best_row["rff_features"]),
        cap_multiplier=float(best_row["cap_multiplier"]),
        balance_penalty=float(best_row["balance_penalty"]),
        chi2_penalty=float(best_row["chi2_penalty"]),
    )
    return best, summary


def aggregate_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        results.groupby(["sample_size", "method"], as_index=False)
        .agg(
            bias=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda x: math.sqrt(float(np.mean(x)))),
            sd=("estimate", "std"),
            mean_estimate=("estimate", "mean"),
        )
        .sort_values(["sample_size", "rmse", "method"])
    )
    by_regime = (
        results.groupby(["sample_size", "method", "d1", "d2"], as_index=False)
        .agg(
            bias=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda x: math.sqrt(float(np.mean(x)))),
            sd=("estimate", "std"),
            true_value=("true_value", "first"),
            mean_estimate=("estimate", "mean"),
        )
        .sort_values(["sample_size", "d1", "d2", "rmse", "method"])
    )
    return overall, by_regime


def plot_rmse(overall: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    methods = list(overall["method"].drop_duplicates())
    for method in methods:
        sub = overall[overall["method"] == method].sort_values("sample_size")
        ax.plot(sub["sample_size"], sub["rmse"], marker="o", label=method)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("RMSE across regimes and replications")
    ax.set_title("Singh dynamic treatment-regime benchmark")
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
    selected: pd.DataFrame,
    overall: pd.DataFrame,
    by_regime: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    lines = [
        "# Singh Dynamic Treatment-Regime Benchmark",
        "",
        "Synthetic-only benchmark. No protected cohort or annotation data are read.",
        "",
        "DGP: Singh, Xu, and Gretton, `Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves`, Supplement J.4.",
        "The data have two binary treatments, time-varying covariates, truncated-logistic propensities, and true regime values:",
        f"Covariate dimension: `p={args.p}` for each of `X1` and `X2`.",
        f"Reward noise profile: `{args.reward_noise_profile}` with heteroskedastic strength `{args.heteroskedastic_strength}`.",
        f"Common oracle subsampling n: `{args.common_oracle_n}` (`0` means independent samples by n).",
        f"Held-out repetitions: `{args.reps}`.",
        f"Elapsed wall-clock seconds: `{elapsed_seconds:.3f}`.",
        "",
        "- `(0, 0)`: 0.0",
        "- `(1, 0)`: 1.1",
        "- `(0, 1)`: 2.2",
        "- `(1, 1)`: 3.8",
        "",
        "Methods:",
        "",
        "- `recursive_krr_plugin`: two-stage kernel ridge g-formula with median-heuristic mixed kernels and GCV ridge selection.",
        "- `oracle_propensity_drl`: longitudinal doubly robust correction using the same KRR nuisances and simulator-known propensities.",
        "- `oracle_sequential_ipw`: Hajek sequential IPW using simulator-known propensities.",
        "- `recursive_kernel_balancing`: stagewise kernel balancing weights with empirical validation tuning for RFF dimension, cap, balance penalty, and chi-square weight regularization.",
        "",
        "Selected balancing hyperparameters:",
        "",
        frame_to_markdown(selected),
        "",
        "Overall held-out metrics:",
        "",
        frame_to_markdown(overall),
        "",
        "Held-out metrics by regime:",
        "",
        frame_to_markdown(by_regime),
        "",
        "Files:",
        "",
        "- `results.csv`: one row per method, replication, sample size, and regime.",
        "- `aggregate_metrics.csv`: pooled regime metrics.",
        "- `metrics_by_regime.csv`: regime-specific metrics.",
        "- `validation_tuning_summary.csv`: validation-only balancing tuning results.",
        "- `run_metadata.csv`: elapsed wall-clock timing metadata.",
        "- `rmse_by_method.png`: RMSE plot.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def config_grid(args: argparse.Namespace) -> list[BalancingConfig]:
    return [
        BalancingConfig(
            rff_features=rff_features,
            cap_multiplier=cap_multiplier,
            balance_penalty=balance_penalty,
            chi2_penalty=chi2_penalty,
        )
        for rff_features, cap_multiplier, balance_penalty, chi2_penalty in itertools.product(
            parse_int_list(args.tune_rff_features),
            parse_float_list(args.tune_cap_multipliers),
            parse_float_list(args.tune_balance_penalties),
            parse_float_list(args.tune_chi2_penalties),
        )
    ]


def run_experiment(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    if args.synthetic_smoke_test:
        args.sample_sizes = "250"
        args.reps = 2
        args.validation_reps = 1
        args.tune_rff_features = "30"
        args.tune_cap_multipliers = "6"
        args.tune_balance_penalties = "25"
        args.tune_chi2_penalties = "0.1"
        args.optimizer_maxiter = min(args.optimizer_maxiter, 120)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = parse_int_list(args.sample_sizes)
    regimes = parse_regimes(args.regimes)
    lambdas = parse_float_list(args.krr_lambdas)
    if args.common_oracle_n > 0 and max(sample_sizes) > args.common_oracle_n:
        raise ValueError("--common-oracle-n must be at least the largest requested sample size.")

    selected_records: list[dict[str, float | int]] = []
    validation_summaries: list[pd.DataFrame] = []
    selected_by_n: dict[int, BalancingConfig] = {}
    grid = config_grid(args)

    for n in sample_sizes:
        best, validation_summary = tune_balancing_for_n(
            n=n,
            p=args.p,
            regimes=regimes,
            configs=grid,
            validation_reps=args.validation_reps,
            seed=args.seed,
            optimizer_maxiter=args.optimizer_maxiter,
            reward_noise_profile=args.reward_noise_profile,
            heteroskedastic_strength=args.heteroskedastic_strength,
        )
        selected_by_n[n] = best
        validation_summaries.append(validation_summary)
        selected_records.append(
            {
                "sample_size": n,
                "rff_features": best.rff_features,
                "cap_multiplier": best.cap_multiplier,
                "balance_penalty": best.balance_penalty,
                "chi2_penalty": best.chi2_penalty,
                "validation_rmse": float(validation_summary.iloc[0]["validation_rmse"]),
            }
        )

    selected_df = pd.DataFrame.from_records(selected_records)
    validation_df = pd.concat(validation_summaries, ignore_index=True)
    selected_df.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    validation_df.to_csv(output_dir / "validation_tuning_summary.csv", index=False)

    rows: list[dict[str, float | str | int]] = []
    if args.common_oracle_n > 0:
        for rep in range(args.reps):
            oracle_data = simulate_singh_dtr(
                n=args.common_oracle_n,
                p=args.p,
                seed=args.seed + 100_000 + 10_007 * rep + args.common_oracle_n,
                reward_noise_profile=args.reward_noise_profile,
                heteroskedastic_strength=args.heteroskedastic_strength,
            )
            rng = np.random.default_rng(args.seed + 300_000 + rep)
            permutation = rng.permutation(args.common_oracle_n)
            for n in sample_sizes:
                data = subset_dtr_data(oracle_data, permutation[:n])
                balancing_config = selected_by_n[n]
                for regime in regimes:
                    method_rows = estimate_methods(
                        data=data,
                        regime=regime,
                        lambdas=lambdas,
                        balancing_config=balancing_config,
                        seed=args.seed + 200_000 + 1_003 * rep + 29 * n + 11 * regime[0] + regime[1],
                        optimizer_maxiter=args.optimizer_maxiter,
                    )
                    for row in method_rows:
                        row["sample_size"] = n
                        row["rep"] = rep
                        row["p"] = args.p
                        row["common_oracle_n"] = args.common_oracle_n
                        row["common_subsample"] = 1
                        rows.append(row)
    else:
        for n in sample_sizes:
            balancing_config = selected_by_n[n]
            for rep in range(args.reps):
                data = simulate_singh_dtr(
                    n=n,
                    p=args.p,
                    seed=args.seed + 100_000 + 37 * rep + n,
                    reward_noise_profile=args.reward_noise_profile,
                    heteroskedastic_strength=args.heteroskedastic_strength,
                )
                for regime in regimes:
                    method_rows = estimate_methods(
                        data=data,
                        regime=regime,
                        lambdas=lambdas,
                        balancing_config=balancing_config,
                        seed=args.seed + 200_000 + 1_003 * rep + 11 * regime[0] + regime[1],
                        optimizer_maxiter=args.optimizer_maxiter,
                    )
                    for row in method_rows:
                        row["sample_size"] = n
                        row["rep"] = rep
                        row["p"] = args.p
                        row["common_oracle_n"] = 0
                        row["common_subsample"] = 0
                        rows.append(row)

    results = pd.DataFrame.from_records(rows)
    overall, by_regime = aggregate_metrics(results)
    results.to_csv(output_dir / "results.csv", index=False)
    overall.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    by_regime.to_csv(output_dir / "metrics_by_regime.csv", index=False)
    plot_rmse(overall, output_dir / "rmse_by_method.png")
    elapsed_seconds = time.perf_counter() - start_time
    pd.DataFrame.from_records(
        [
            {
                "elapsed_wall_seconds": elapsed_seconds,
                "timing_source": "script_perf_counter",
            }
        ]
    ).to_csv(output_dir / "run_metadata.csv", index=False)
    write_summary(output_dir, args, selected_df, overall, by_regime, elapsed_seconds)

    print(f"Wrote {len(results)} result rows to {output_dir}")
    print(f"Elapsed wall-clock seconds: {elapsed_seconds:.3f}")
    print(overall.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", default="500")
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--validation-reps", type=int, default=4)
    parser.add_argument("--p", type=int, default=10, help="Dimension of each Xt covariate vector.")
    parser.add_argument("--regimes", default=DEFAULT_REGIMES)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--krr-lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--common-oracle-n", type=int, default=0, help="Draw one dataset of this size per MC rep and use nested subsamples for each n; 0 disables this.")
    parser.add_argument(
        "--reward-noise-profile",
        choices=["homoskedastic", "dtr-overlap"],
        default="homoskedastic",
    )
    parser.add_argument("--heteroskedastic-strength", type=float, default=0.0)
    parser.add_argument("--tune-rff-features", default="40,80")
    parser.add_argument("--tune-cap-multipliers", default="4,8")
    parser.add_argument("--tune-balance-penalties", default="10,25")
    parser.add_argument("--tune-chi2-penalties", default="0.1,1.0")
    parser.add_argument("--optimizer-maxiter", type=int, default=250)
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
