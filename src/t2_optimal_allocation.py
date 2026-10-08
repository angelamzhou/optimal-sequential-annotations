"""Solve the manuscript's joint T=2 prefix allocation on finite branch laws.

Inputs are full-score increment variances, including discount/density factors
exactly once. Rows represent F0 and columns the conditional F1 branches.
The solver minimizes E[v1/lambda1 + v2/Lambda12] under a single cost budget.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Allocation:
    lambda1: np.ndarray
    cumulative2: np.ndarray
    lambda2: np.ndarray
    multiplier: float
    expected_cost: float
    objective: float


def solve(variance1, variance2, branch_probability, budget, *, weights=None,
          cost1=1., cost2=1., floor=.1):
    """Joint convex solution, not a fixed split of the stage budgets.

    Constraints: floor <= lambda1 <= 1 and
    floor*lambda1 <= cumulative2 <= lambda1. Costs are per revealed stage.
    A common Lagrange multiplier enforces the total expected cost. For each
    multiplier, minimize over cumulative2 analytically, then solve the convex
    one-dimensional reduced problem for each F0 row by derivative bisection.

    cost2 may be a scalar or a length-n vector. A row with cost2 == 0 has no
    stage-2 label to buy (an absorbing state after stage 1): it contributes
    nothing to expected cost, and its floor on cumulative2 is vacuous because
    the stage-2 increment is identically zero there. The scalar default is
    unchanged, so existing callers are unaffected.
    """
    a, b, p = map(lambda x: np.asarray(x, dtype=float), (variance1, variance2, branch_probability))
    if a.ndim != 1 or b.ndim != 2 or b.shape != p.shape or b.shape[0] != len(a) or not len(a):
        raise ValueError('Require v1[n], v2[n,k], and branch_probability[n,k].')
    if any(np.any(~np.isfinite(x)) or np.any(x < 0) for x in (a, b, p)):
        raise ValueError('Variances and branch probabilities must be finite and nonnegative.')
    if not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0):
        raise ValueError('Each conditional branch distribution must sum to one.')
    w = np.full(len(a), 1/len(a)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != a.shape or np.any(~np.isfinite(w)) or np.any(w <= 0) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('Initial-state weights must be positive and sum to one.')
    if not all(np.isfinite(x) for x in (budget, cost1, floor)) or cost1 <= 0 or not 0 <= floor < 1:
        raise ValueError('Require a positive stage-1 cost and 0 <= conditional floor < 1.')
    try:
        c2 = np.array(np.broadcast_to(np.asarray(cost2, dtype=float), a.shape))
    except ValueError:
        raise ValueError('cost2 must be a scalar or a length-n vector.')
    if np.any(~np.isfinite(c2)) or np.any(c2 < 0) or not np.any(c2 > 0):
        raise ValueError('Stage-2 costs must be finite and nonnegative, with at least one positive.')
    # Floor and full-reveal expected costs under the F0 weights; a cost2 == 0 row adds nothing.
    mean_c2 = float(np.sum(w*c2))
    minimum = cost1*floor + mean_c2*floor**2
    maximum = cost1 + mean_c2
    if budget <= 0 or budget < minimum - 1e-12 or budget > maximum + 1e-12:
        raise ValueError('Budget is infeasible for the requested costs/floors.')

    def ratio(num, den):
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def result(lam, cum, kappa):
        value = float(np.sum(w*(a/lam + np.sum(p*ratio(b, cum), axis=1))))
        cost = float(np.sum(w*(cost1*lam + c2*np.sum(p*cum, axis=1))))
        return Allocation(lam, cum, cum/lam[:, None], kappa, cost, value)

    if abs(budget - maximum) < 1e-12:
        return result(np.ones_like(a), np.ones_like(b), 0.)
    if floor > 0 and abs(budget-minimum) < 1e-12:
        return result(np.full_like(a, floor), np.full_like(b, floor**2), float('inf'))
    # Zero-variance branches can make the objective flat before full cost.
    # Spend any remaining budget only along objective-flat directions.
    active2 = (p*b) > 0
    active1 = (a > 0) | np.any(active2, axis=1)
    flat_lam = np.where(active1, 1., max(floor, 1e-12))
    flat_cum = np.where(active2, 1., floor*flat_lam[:, None])
    flat = result(flat_lam, flat_cum, 0.)
    if budget >= flat.expected_cost - 1e-12:
        mix = np.clip((budget-flat.expected_cost)/(maximum-flat.expected_cost), 0., 1.)
        return result(flat_lam+mix*(1-flat_lam), flat_cum+mix*(1-flat_cum), 0.)

    def at_multiplier(kappa):
        kc2 = kappa*c2[:, None]
        # Free stage 2 (cost2 == 0) has unconstrained interior -> cumulative2 = lambda1;
        # the increment variance is zero there, so the choice is objective-neutral.
        interior = np.sqrt(np.divide(b, kc2, out=np.full_like(b, np.inf), where=kc2 > 0))

        def derivative(lam):
            lower = floor*lam[:, None]
            upper = lam[:, None]
            cum = np.minimum(upper, np.maximum(lower, interior))
            slope = np.where(interior < lower, floor, np.where(interior > upper, 1., 0.))
            return -a/lam**2 + kappa*cost1 + np.sum(p*slope*(kc2-ratio(b, cum**2)), axis=1)

        lo = np.full_like(a, max(floor, 1e-12))
        hi = np.ones_like(a)
        for _ in range(60):
            mid = (lo+hi)/2
            negative = derivative(mid) < 0
            lo = np.where(negative, mid, lo)
            hi = np.where(negative, hi, mid)
        lam = (lo+hi)/2
        cum = np.minimum(lam[:, None], np.maximum(floor*lam[:, None], interior))
        return result(lam, cum, kappa)

    lo, hi = 0., 1.
    for _ in range(200):
        current = at_multiplier(hi)
        if current.expected_cost <= budget:
            break
        hi *= 2
    else:
        raise ArithmeticError('Failed to bracket the budget multiplier.')
    for _ in range(75):
        mid = (lo+hi)/2
        current = at_multiplier(mid)
        if current.expected_cost > budget:
            lo = mid
        else:
            hi = mid
    answer = at_multiplier(hi)
    if abs(answer.expected_cost-budget) > 1e-8:
        raise ArithmeticError('Budget residual exceeds tolerance; inspect degenerate coefficients.')
    return answer
