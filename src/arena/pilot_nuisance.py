#!/usr/bin/env python3
"""Nuisance-estimation pilot for the LMArena T=2 design.

Follows main.tex exactly where it matters:
  sigma_t^2(F_{t-1}) = Var[ mu_t (R_t + gamma V_{t+1}) - gamma 1{t<T} mu_{t+1} Q_{t+1} | F_{t-1} ]
  (Prop. variance-decomposition / sec:annotation_prob_derivation)
and feeds those to src/t2_optimal_allocation.solve, whose docstring asks for
"full-score increment variances, including discount/density factors exactly once".

mu_2 is the MARGINAL density ratio, not the second action ratio:
  mu_2 = [P_{pi_e}(S'_2) / P_{pi_b}(S'_2)] * [pi_e(A_2|S'_2) / pi_b(A_2|S'_2)]
with S'_2 = (week, revealed R_1, continued). The occupancy factor is where the
change in *who continues* enters.

Reward is binary so Q/b are conditional means; everything is cross-fit (K=5).
"""
import sys, json
import os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_arena as S
from t2_optimal_allocation import solve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

GAMMA = 1.0
K = 5
RNG = 0

def build(path):
    df = S.load(path)
    orders = df.groupby('evaluation_session_id').evaluation_order.apply(set)
    ok = set(orders[orders.map(min) == 1].index)
    a = df[(df.evaluation_order == 1) & (df.evaluation_session_id.isin(ok))].copy()
    a = a.set_index('evaluation_session_id')
    s2 = df[df.evaluation_order == 2].set_index('evaluation_session_id')
    a['C'] = a.index.isin(s2.index).astype(int)          # continuation (NOT the annotation indicator)
    a['R2'] = s2['R'].reindex(a.index)
    a['pair2'] = s2['pair'].reindex(a.index)
    return df, a

def proxy_features(frame):
    """R~_t : always-observed cheap proxy. All of it is legitimately post-action -
    that is what a proxy reward IS - but none of it is the gold vote."""
    cm = frame.conv_metadata
    ct = frame.category_tag
    def g(d, k, default=0.0):
        try: return float(d.get(k, default))
        except Exception: return default
    def nest(d, k, sub):
        try: return float(sum(d.get(k, {}).values())) if sub is None else float(d.get(k, {}).get(sub, 0))
        except Exception: return 0.0
    def tag(d, grp, key):
        try: return float(bool(d[grp][key]))
        except Exception: return 0.0
    X = pd.DataFrame({
        'user_tok':    [g(d, 'sum_user_tokens') for d in cm],
        'turns':       [g(d, 'turns') for d in cm],
        'a_tok':       [g(d, 'sum_assistant_a_tokens') for d in cm],
        'b_tok':       [g(d, 'sum_assistant_b_tokens') for d in cm],
        'ctx_a':       [g(d, 'context_a_tokens') for d in cm],
        'ctx_b':       [g(d, 'context_b_tokens') for d in cm],
        'hdr_a':       [nest(d, 'header_count_a', None) for d in cm],
        'hdr_b':       [nest(d, 'header_count_b', None) for d in cm],
        'lst_a':       [nest(d, 'list_count_a', None) for d in cm],
        'lst_b':       [nest(d, 'list_count_b', None) for d in cm],
        'bld_a':       [nest(d, 'bold_count_a', '**') for d in cm],
        'bld_b':       [nest(d, 'bold_count_b', '**') for d in cm],
        'math':        [tag(d, 'math_v0.1', 'math') for d in ct],
        'creative':    [tag(d, 'creative_writing_v0.1', 'creative_writing') for d in ct],
        'iff':         [tag(d, 'if_v0.1', 'if') for d in ct],
        'is_code':     frame.is_code.astype(float).values,
        'en':          (frame.language == 'en').astype(float).values,
    }, index=frame.index)
    # asymmetry features: the vote is a comparison, so |a-b| should matter
    X['d_tok'] = (X.a_tok - X.b_tok).abs()
    X['d_hdr'] = (X.hdr_a - X.hdr_b).abs()
    X['d_lst'] = (X.lst_a - X.lst_b).abs()
    X['d_bld'] = (X.bld_a - X.bld_b).abs()
    return X

LEARNER = os.environ.get("NUISANCE_LEARNER", "fixed")   # "fixed" (original) | "reg"

def make_learner():
    if LEARNER == "reg":
        # Regularised: shallow, L2, early stopping. Chosen 2026-09-14 on PPE (disjoint data,
        # ppe_pilot/ppe_tfidf_budget.json): the fixed config has strongly negative
        # cross-fit skill below ~6k labels; this one is >= it at every budget tried.
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=8,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.2,
                                              n_iter_no_change=30, random_state=RNG)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                          max_leaf_nodes=31, random_state=RNG)

def crossfit(X, y, groups_for_strat, label):
    """K-fold cross-fitted out-of-fold predictions, as the CLT's sample-splitting requires."""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RNG)
    for tr, te in skf.split(X, groups_for_strat):
        m = make_learner()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y, oof); br = brier_score_loss(y, oof)
    base = y.mean()
    br0 = brier_score_loss(y, np.full(len(y), base))
    print(f"  {label:<34} n={len(y):>7,}  base={base:.4f}  AUC={auc:.4f}  "
          f"Brier={br:.5f} vs {br0:.5f}  skill={1-br/br0:+.3%}")
    return oof, dict(n=int(len(y)), base=float(base), auc=float(auc),
                     brier=float(br), brier_base=float(br0), skill=float(1 - br / br0))

def main(path):
    out = {}
    df, a = build(path)
    n = len(a)
    print(f"=== units: {n:,} sessions with vote 1 observed | continuation rate {a.C.mean():.4f} ===\n")

    # ---------------- 1. silver / proxy nuisances b_t = E[R_t | F_{t-1}] ----------------
    print("=== 1. NUISANCE QUALITY (cross-fit K=5) — can the cheap proxy predict the gold vote? ===")
    X1 = proxy_features(a)
    b1, d1 = crossfit(X1, a.R.astype(int), a.R.astype(int), "b_1 = E[R_1 | proxy]  (metadata only)")
    out['b1'] = d1
    # continuation model, part of the transition kernel
    cC, dC = crossfit(X1, a.C.astype(int), a.C.astype(int), "P(continue | F_0)")
    out['continuation_model'] = dC
    # stage-2 reward model on continuers, with revealed R_1 in the state
    cont = a[a.C == 1].copy()
    X2 = proxy_features(cont); X2['R1'] = cont.R.values
    b2, d2 = crossfit(X2, cont.R2.astype(int), cont.R2.astype(int), "b_2 = E[R_2 | F_1] (incl. R_1)")
    out['b2'] = d2
    X2n = X2.drop(columns=['R1'])
    _, d2n = crossfit(X2n, cont.R2.astype(int), cont.R2.astype(int), "b_2 WITHOUT R_1 (ablation)")
    out['b2_no_R1'] = d2n
    print(f"  -> value of revealing R_1 for predicting R_2: AUC {d2n['auc']:.4f} -> {d2['auc']:.4f} "
          f"({d2['auc']-d2n['auc']:+.4f})\n")

    # ---------------- 2. policies, ratios, occupancy ----------------
    piv = (df.groupby(['week', 'pair']).size() / df.groupby('week').size()).unstack().fillna(0.)
    cats = list(piv.columns)
    pe = {c: 1.0 / len(cats) for c in cats}            # prespecified target: uniform over categories
    a['mu1'] = [pe[p] / max(piv.loc[w, p], 1e-9) for p, w in zip(a.pair, a.week)]

    # S'_2 = (week, R_1, continued). Occupancy ratio under pi_e vs pi_b.
    cell = a.groupby(['week', 'pair'])
    occ_b, occ_e = {}, {}
    for (w, p), g in cell:
        pb = piv.loc[w, p]
        for r in (0, 1):
            for c in (0, 1):
                q = float(((g.R == r) & (g.C == c)).mean())
                occ_b[(w, r, c)] = occ_b.get((w, r, c), 0.0) + pb * q
                occ_e[(w, r, c)] = occ_e.get((w, r, c), 0.0) + pe[p] * q
    a['occ_ratio'] = [occ_e[(w, r, c)] / max(occ_b[(w, r, c)], 1e-12)
                      for w, r, c in zip(a.week, a.R.astype(int), a.C.astype(int))]
    m = a.C.values == 1
    act2 = np.ones(len(a))
    act2[m] = [pe[p] / max(piv.loc[w, p], 1e-9)
               for p, w in zip(a.pair2.values[m], a.week.values[m])]
    a['mu2'] = a.occ_ratio.values * act2                 # marginal ratio, occupancy x action
    out['ratios'] = {'mu1_mean': float(a.mu1.mean()), 'mu1_max': float(a.mu1.max()),
                     'occ_ratio_mean': float(a.occ_ratio.mean()),
                     'occ_ratio_range': [float(a.occ_ratio.min()), float(a.occ_ratio.max())],
                     'mu2_mean': float(a.mu2.mean()), 'mu2_max': float(a.mu2.max())}
    print("=== 2. DENSITY RATIOS (uniform target) ===")
    print(f"  mu_1 mean {a.mu1.mean():.4f}  max {a.mu1.max():.2f}")
    print(f"  occupancy ratio  mean {a.occ_ratio.mean():.4f}  range "
          f"[{a.occ_ratio.min():.3f}, {a.occ_ratio.max():.3f}]   <- 'who continues' changes here")
    print(f"  mu_2 mean {a.mu2.mean():.4f}  max {a.mu2.max():.2f}\n")

    # ---------------- 3. Q_2, V_2 and the manuscript's sigma_t^2 ----------------
    # Q_2(S'_2, a) = E[R_2 | week, R_1, pair_2]; V_2 = sum_a pi_e(a) Q_2 ; V_3 = 0.
    q2 = (a[a.C == 1].groupby(['R', 'pair2']).R2.mean()).unstack()
    q2 = q2.fillna(a[a.C == 1].R2.mean())
    V2 = {r: float(sum(pe[c] * q2.loc[r, c] for c in q2.columns)) for r in q2.index}
    a['V2'] = [V2[r] * c for r, c in zip(a.R.astype(int), a.C.astype(int))]   # absorbing => 0 if C=0
    a['Q2'] = [q2.loc[r, p] if c == 1 else 0.0
               for r, p, c in zip(a.R.astype(int), a.pair2, a.C.astype(int))]

    # sigma_1^2(F_0) = Var[ mu_1 (R_1 + gamma V_2) - gamma mu_2 Q_2 | F_0 ]
    # sigma_2^2(F_1) = Var[ mu_2 R_2 | F_1 ]            (V_3 = 0)
    a['inc1'] = a.mu1 * (a.R + GAMMA * a.V2) - GAMMA * a.mu2 * a.Q2
    a['inc2'] = np.where(m, a.mu2 * np.nan_to_num(a.R2.values), 0.0)
    return df, a, out, piv, pe, cats, q2

def branch_law(a, out):
    """Rows = F_0 = week bucket (the tabular S'_1). Branches = F_1 = (R_1, continued)."""
    wks = sorted(a.week.unique())
    BR = [(0, 0), (0, 1), (1, 0), (1, 1)]           # (R_1, C)
    v1, v2, bp, wgt = [], [], [], []
    for w in wks:
        g = a[a.week == w]
        wgt.append(len(g))
        v1.append(float(g.inc1.var(ddof=1)))
        vb, pb = [], []
        for (r, c) in BR:
            h = g[(g.R.astype(int) == r) & (g.C == c)]
            pb.append(len(h) / len(g))
            # C=0 is absorbing: no stage-2 increment, hence zero conditional variance
            vb.append(float(h.inc2.var(ddof=1)) if (c == 1 and len(h) > 1) else 0.0)
        v2.append(vb); bp.append(pb)
    v1 = np.array(v1); v2 = np.array(v2); bp = np.array(bp)
    wgt = np.array(wgt, float); wgt /= wgt.sum()
    bp = bp / bp.sum(axis=1, keepdims=True)
    out['branch_law'] = {
        'rows_weeks': len(wks), 'branches': [f"R1={r},C={c}" for r, c in BR],
        'sigma1_sq_mean': float((v1 * wgt).sum()),
        'sigma2_sq_mean_by_branch': {f"R1={r},C={c}": float((v2[:, i] * wgt).sum())
                                     for i, (r, c) in enumerate(BR)},
        'branch_prob_mean': {f"R1={r},C={c}": float((bp[:, i] * wgt).sum())
                             for i, (r, c) in enumerate(BR)},
    }
    return v1, v2, bp, wgt, BR

def compare(v1, v2, bp, wgt, BR, out, floor=0.1):
    print("=== 3. BRANCH LAW (manuscript sigma_t^2, mu factors inside, gamma counted once) ===")
    print(f"  E[sigma_1^2] = {(v1*wgt).sum():.4f}")
    for i, (r, c) in enumerate(BR):
        print(f"  E[sigma_2^2 | R1={r},C={c}] = {(v2[:,i]*wgt).sum():.4f}   "
              f"P(branch) = {(bp[:,i]*wgt).sum():.4f}")
    print()
    print("=== 4. LEARNED vs UNIFORM ALLOCATION AT MATCHED EXPECTED COST ===")
    print("  'uniform' = best NON-ADAPTIVE design: one scalar lambda_1 and one scalar")
    print("  Lambda_{1:2}, split optimally. It is the hard baseline - any gain below is")
    print("  from conditioning on F_0 and F_1, not from choosing the stage split.\n")
    Es1, Es2 = float((v1 * wgt).sum()), float((bp * v2 * wgt[:, None]).sum())
    rows = []
    print(f"{'budget':>7}{'learned obj':>13}{'uniform obj':>13}{'reduction':>11}{'SE ratio':>10}  stage-2 interior?")
    for B in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5):
        try: r = solve(v1, v2, bp, B, weights=wgt, floor=floor)
        except ValueError as e:
            print(f"{B:>7.2f}  infeasible ({e})"); continue
        # best non-adaptive: grid over lambda_1, Lambda = B - lambda_1
        best = np.inf; bl = None
        for lam in np.linspace(max(floor, 1e-6), 1.0, 4000):
            Lam = B - lam
            if Lam < floor * lam - 1e-12 or Lam > lam + 1e-12: continue
            val = Es1 / lam + Es2 / max(Lam, 1e-12)
            if val < best: best, bl = val, (lam, Lam)
        red = (best - r.objective) / best
        at_floor = np.abs(r.cumulative2 - floor * r.lambda1[:, None]) < 1e-9
        live = bp > 1e-9
        interior = "yes" if not (at_floor & live).any() else f"no ({(at_floor&live).sum()} cells at floor)"
        print(f"{B:>7.2f}{r.objective:>13.4f}{best:>13.4f}{red:>10.2%}{np.sqrt(r.objective/best):>10.4f}  {interior}")
        rows.append(dict(budget=B, learned=float(r.objective), uniform=float(best),
                         reduction=float(red), se_ratio=float(np.sqrt(r.objective / best)),
                         uniform_lambda1=float(bl[0]), uniform_Lambda12=float(bl[1]),
                         learned_lambda1_mean=float((r.lambda1 * wgt).sum()),
                         expected_cost=float(r.expected_cost), interior=interior))
    out['allocation_comparison'] = rows
    print("\n  NOTE: these are reductions in the ALLOCATION-DEPENDENT term of Var(Gamma~),")
    print("  i.e. the sum in Prop. variance-decomposition. Var(m_0) is additive on top of")
    print("  both and is NOT included, so the reduction in TOTAL variance is smaller.")

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'arena140k.parquet'
    df, a, out, piv, pe, cats, q2 = main(p)
    v1, v2, bp, wgt, BR = branch_law(a, out)
    compare(v1, v2, bp, wgt, BR, out)
    json.dump(out, open('pilot_nuisance_results.json', 'w'), indent=1, default=str)
    print("\nwrote pilot_nuisance_results.json")
