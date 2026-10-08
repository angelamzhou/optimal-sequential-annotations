"""Charged batch-adaptive annotation experiment on the Arena 140k — the paper's Fig. 4 (casenote) protocol.

Mirrors src/outreach_dtr_annotation_mc.py::run_replication step for step
(ARENA_BATCH_ADAPTIVE_FREEZE.md has the mapping). Per replication: subsample n sessions,
K folds; full-annotation benchmark; one shared uniform vector per sampler; for each budget B:
  pilot  — every session's full trajectory annotated w.p. 0.3*B (paid for);
  fit    — b_1, b_2 (L2 logistic on META+RM+TFIDF), tabular Q/occupancy, TF-IDF text model:
           on PILOT labels only, cross-fitted by fold; text features are nested fold-safe (K x n) and the
           text model of every arm sees CHARGED labels only (v2, 2026-09-18; see label_budget_diagnosis.md);
  design — structural sigma^2 -> joint solve (per-row cost2 = Z1, floor 0.01) on non-pilot rows at the
           residual budget; pooled probability = p_pil + (1-p_pil)*pi_ad (as the casenote driver codes it);
  reveal — pilot fully; adaptive by the design;  uniform arm: q + q^2 P(Z1) = B, no pilot;
  refit  — nuisances refit on ALL annotated rows, IPW-weighted, cross-fitted (casenote-driver post-reveal refit);
  score  — Gamma~ = m0 + C1/L1 (m1 - m0) + C1C2/L12 (Gamma - m1)  [annotation_score, vectorised];
           estimate = mean, model-based SE = sd/sqrt(n); error vs the full-annotation benchmark.
Shared code: t2_optimal_allocation.solve, t2_annotation_score.annotation_score (equality asserted),
arena_full_score.ArenaFullScore (fit / arrays / scores_from_arrays), ppe_tfidf_pilot.fit_text,
pilot_nuisance.proxy_features, when_rm_lands.rm_feats, load_text_hash.

    python run_arena_charged.py --reps 200 --jobs 6
"""
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent  # src/
for p in (ROOT, HERE): sys.path.insert(0, str(p))
import screen_arena as S
from t2_optimal_allocation import solve
from t2_annotation_score import annotation_score
from arena_full_score import ArenaFullScore, CATS
from pilot_nuisance import proxy_features
from when_rm_lands import rm_feats
from load_text_hash import load_text_hash, align
from ppe_tfidf_pilot import fit_text
from arena_paths import DATA, digest, save_once

SEED = 20260915
N_SUB, K, PILOT_FRAC, FLOOR, GAMMA = 21_685, 5, 0.3, 0.01, 1.0
BUDGET_FRACTIONS = [.30, .40, .50, .60, .70, .80]
FIELDS = ("prompt", "response_a", "response_b")


# ----------------------------------------------------------------------------- data (once)
def load_all():
    man = json.load(open(DATA / "arena_session_manifest.json")); summ = json.load(open(DATA / "arena_manifest_summary.json"))
    menu = json.load(open(DATA / "arena_policy_menu.json"))
    pols = {"tilt": menu["menu"][menu["primary_pi_e"]]["pi_e"], "equal": menu["menu"]["uniform over categories"]["pi_e"]}
    df = S.load(DATA / "arena140k.parquet").set_index("id")
    votes = df.join(pd.read_parquet(DATA / "arena_rm_scores.parquet").set_index("id")[["score_a", "score_b"]])
    per_vote = pd.concat([proxy_features(votes).fillna(0.0), rm_feats(votes)], axis=1); per_vote.columns = [str(c) for c in per_vote.columns]
    R_vote = votes["R"].astype(int)
    H = load_text_hash(str(DATA / "arena_text_hash"))
    vote_ids = pd.Index(pd.unique(pd.Series([r["id1"] for r in man] + [r["id2"] for r in man if r["Z1"]])))
    hix = align(H, pd.Series(vote_ids)); fields = [H[f][hix] for f in FIELDS]      # rows aligned to vote_ids
    vpos = pd.Series(np.arange(len(vote_ids)), index=vote_ids)
    return man, summ, pols, per_vote, R_vote, fields, vpos


def make_logit(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


# ----------------------------------------------------------------------------- one replication
def replicate(rep, man, summ, pols, per_vote, R_vote, fields, vpos, reps_total, budgets, audit=False):
    rng = np.random.default_rng([SEED, rep])
    sub = [man[i] for i in np.sort(rng.choice(len(man), N_SUB, replace=False))]
    n = len(sub); z1 = np.array([r["Z1"] for r in sub], bool); mZ = float(z1.mean())
    R1 = np.array([r["reward1"] for r in sub], int); R2 = np.array([r["reward2"] for r in sub], int)
    id1 = [r["id1"] for r in sub]; id2 = [r["id2"] if r["Z1"] else None for r in sub]
    fold = rng.integers(0, K, n)
    u_pilot = rng.random(n); u = {"uniform": rng.random((2, n)), "learned": rng.random((2, n))}
    X1 = per_vote.loc[id1].to_numpy(); X2 = np.full((n, per_vote.shape[1]), np.nan); X2[z1] = per_vote.loc[[i for i in id2 if i]].to_numpy()
    X2f = np.where(np.isnan(X2), 0.0, X2)                                 # logistic cannot take NaN; Z1 flag carries the info
    f1 = vpos.loc[id1].to_numpy(); f2 = np.array([vpos.loc[i] if i else -1 for i in id2])

    def text_scores(mask1, mask2, R1_=None, R2_=None):
        """Per-vote TF-IDF P(R=1|text), trained ONLY on charged labels: R1 of `mask1` sessions and R2 of
        `mask2 & z1` sessions (v1 trained on R2 of every `mask1 & z1` session, i.e. on unpaid second votes
        in the uniform arm; label_budget_diagnosis.md §1). Returns arrays of shape (K, n): row k holds the
        features to use when fitting and predicting outer fold k -- fold-k rows get the prediction of a model
        fit on masked rows outside k; training rows in fold j != k get the prediction of a model fit on
        masked rows outside {k, j}, so nothing fold k contributes reaches fold k's nuisance (diagnosis §3;
        the same nested scheme as ppe_tfidf_pilot.nested_oof, sharing the 10 pairwise fits)."""
        R1_ = R1 if R1_ is None else R1_; R2_ = R2 if R2_ is None else R2_
        tr1 = np.asarray(mask1, bool); tr2 = np.asarray(mask2, bool) & z1

        def fit_pred(excl, pred_mask):
            fit1 = tr1 & ~np.isin(fold, excl); fit2 = tr2 & ~np.isin(fold, excl)
            y = np.concatenate([R1_[fit1], R2_[fit2]])
            if len(y) < 50 or len(np.unique(y)) < 2: return None
            v = np.concatenate([f1[fit1], f2[fit2]]); pr1 = np.flatnonzero(pred_mask); pr2 = np.flatnonzero(pred_mask & z1)
            p = fit_text([f[v] for f in fields], y, [f[np.concatenate([f1[pr1], f2[pr2]])] for f in fields])
            return pr1, p[:len(pr1)], pr2, p[len(pr1):]

        T1, T2 = np.full((K, n), np.nan), np.full((K, n), np.nan)
        for k in range(K):                                                  # outer: fold k's own features
            r = fit_pred([k], fold == k)
            if r: pr1, p1, pr2, p2 = r; T1[k, pr1] = p1; T2[k, pr2] = p2
        for k in range(K):                                                  # pairwise: training-row features
            for j in range(k + 1, K):
                r = fit_pred([k, j], (fold == k) | (fold == j))
                if r is None: continue
                pr1, p1, pr2, p2 = r
                jj = fold[pr1] == j; T1[k, pr1[jj]] = p1[jj]; T1[j, pr1[~jj]] = p1[~jj]
                jj = fold[pr2] == j; T2[k, pr2[jj]] = p2[jj]; T2[j, pr2[~jj]] = p2[~jj]
        ny = int(tr1.sum() + tr2.sum()); fill = float(np.concatenate([R1_[tr1], R2_[tr2]]).mean()) if ny else 0.88
        return np.nan_to_num(T1, nan=fill), np.nan_to_num(T2, nan=fill)

    def fit_nuisances(pe, mask1, mask2, w1, w2, tf1, tf2):
        """Cross-fitted: for fold k, fit b1/b2/tables on rows outside k that are annotated (mask), predict fold k.
        tf1/tf2 are (K, n) from text_scores; row k is used for fold k (training rows' text features never saw fold k)."""
        fs_by_fold, b1, b2 = {}, np.zeros(n), np.zeros((n, 2))
        for k in range(K):
            Xb1 = np.column_stack([X1, tf1[k], z1]); Xb2 = np.column_stack([X1, tf1[k], X2f, tf2[k]])   # fold-k-safe text features
            tr1 = mask1 & (fold != k); tr2 = mask2 & (fold != k) & z1; te = fold == k
            rows = [sub[i] for i in np.flatnonzero(tr1)]; refs = [{"episode_id": sub[i]["episode_id"], "reward1": int(R1[i]), "reward2": int(R2[i]) if mask2[i] else 0} for i in np.flatnonzero(tr1)]
            fs = ArenaFullScore(pe, summ["pi_b_by_week"], gamma=GAMMA).fit(rows, refs, require_train=False, w1=w1[tr1], w2=np.where(mask2[tr1], w2[tr1], 0.0))
            fs_by_fold[k] = fs
            m = make_logit().fit(Xb1[tr1], R1[tr1], logisticregression__sample_weight=w1[tr1]); b1[te] = m.predict_proba(Xb1[te])[:, 1]
            if tr2.sum() >= 30 and len(np.unique(R2[tr2])) == 2:
                m2 = make_logit().fit(np.column_stack([Xb2[tr2], R1[tr2]]), R2[tr2], logisticregression__sample_weight=w2[tr2])
                for r in (0, 1): b2[te, r] = m2.predict_proba(np.column_stack([Xb2[te], np.full(te.sum(), r)]))[:, 1]
            else:
                b2[te, :] = float(np.average(R2[tr2], weights=w2[tr2])) if tr2.sum() else 0.9
        b1 = np.clip(b1, 1e-3, 1 - 1e-3); b2 = np.clip(b2, 1e-3, 1 - 1e-3)
        A = {}
        for k in range(K):
            te = np.flatnonzero(fold == k); Ak = fs_by_fold[k].arrays([sub[i] for i in te])
            for key, v in Ak.items(): A.setdefault(key, np.zeros(n))[te] = v
        A["z1"] = z1
        return ArenaFullScore.scores_from_arrays(A, b1, b2[:, 0], b2[:, 1], GAMMA)

    def estimate(Sc, c1, c2, L1, L12):
        gam = np.where(R1 == 1, np.where(R2 == 1, Sc["Gamma"][1][1], Sc["Gamma"][1][0]), np.where(R2 == 1, Sc["Gamma"][0][1], Sc["Gamma"][0][0]))
        m1 = np.where(R1 == 1, Sc["m1"][1], Sc["m1"][0]); m0 = Sc["m0"]
        contrib = m0 + c1 / L1 * (m1 - m0) + (c1 * c2) / L12 * (gam - m1)
        # the harness formula, checked on a few rows
        for i in np.flatnonzero(c1)[:3]:
            ref = annotation_score(m0[i], m1[i], gam[i] if c2[i] else None, int(c1[i]), int(c2[i]), float(L1[i]), float(L12[i] / L1[i]) if c1[i] else None)
            assert abs(ref - contrib[i]) < 1e-9
        return float(contrib.mean()), float(contrib.std(ddof=1) / np.sqrt(n)), contrib

    out = []
    ones = np.ones(n)
    # ---- full-annotation benchmark per pi_e (text model on all votes, cross-fitted)
    tf1_all, tf2_all = text_scores(np.ones(n, bool), np.ones(n, bool))
    bench = {}
    for name, pe in pols.items():
        Sc = fit_nuisances(pe, np.ones(n, bool), np.ones(n, bool), ones, ones, tf1_all, tf2_all)
        bench[name] = estimate(Sc, ones, ones, ones, ones)[0]
    for B in budgets:
        Breal = B * (1 + mZ); p_pil = PILOT_FRAC * B
        pilot = u_pilot < p_pil
        tf1_p, tf2_p = text_scores(pilot, pilot)                            # pilot-fit text model (pilot = full trajectories)
        # uniform arm: no pilot; q + q^2 mZ = Breal; its text model is fit on its own annotated votes (once, pi_e-free)
        q = (-1 + np.sqrt(1 + 4 * mZ * Breal)) / (2 * mZ)
        cU1 = u["uniform"][0] < q; cU2 = cU1 & (u["uniform"][1] < q) & z1
        LU1, LU12 = np.full(n, q), np.full(n, q * q)
        tf1_u, tf2_u = text_scores(cU1, cU2)                                # charged votes only
        if audit:                                                           # label-access check (diagnosis §4): flipping every
            unpaid = cU1 & z1 & ~cU2                                        # UNPAID second vote must not move the text features
            R2_flip = R2.copy(); R2_flip[unpaid] = 1 - R2_flip[unpaid]
            a1, a2 = text_scores(cU1, cU2, R2_=R2_flip)
            assert np.array_equal(a1, tf1_u) and np.array_equal(a2, tf2_u), "unpaid labels reached the text model"
            print(f"[audit] rep {rep} B={B}: {int(unpaid.sum())} unpaid second votes flipped, text features unchanged", flush=True)
        resid = (Breal - p_pil * (1 + mZ)) / (1 - p_pil)                      # residual real labels per non-pilot session
        for name, pe in pols.items():
            # ---------------- learned: design from pilot-fit nuisances
            Sc_p = fit_nuisances(pe, pilot, pilot, ones, ones, tf1_p, tf2_p)
            # Adaptive design for EVERY unit at the residual budget (casenote driver: probabilities_from_priority over all n);
            # the draw is applied only to non-pilot units, but the pooled inclusion probability
            # p_pil + (1 - p_pil) * pi_ad(i) is the same formula for all units, as in the casenote driver.
            a = np.maximum(Sc_p["sig1"], 1e-9)
            b = np.column_stack([Sc_p["sig2"][0], Sc_p["sig2"][1]]); b[~z1] = 0.; b[z1] = np.maximum(b[z1], 1e-9)
            branch = np.column_stack([1 - Sc_p["b1"], Sc_p["b1"]])
            des = solve(a, b, branch, resid, floor=FLOOR, cost2=z1.astype(float))
            lam1_ad = des.lambda1; L12_ad = des.cumulative2[np.arange(n), R1]        # branch = the R1 that will be revealed
            lam2_ad = L12_ad / np.maximum(lam1_ad, 1e-12)
            cL1 = pilot | (u["learned"][0] < lam1_ad)
            cL2 = (pilot | (cL1 & (u["learned"][1] < lam2_ad))) & z1
            L1 = p_pil + (1 - p_pil) * lam1_ad; L12 = p_pil + (1 - p_pil) * L12_ad       # pooled inclusion probabilities, all units
            # ---------------- post-reveal refit on all annotated rows, IPW-weighted (text model kept pilot-fit)
            Sc_L = fit_nuisances(pe, cL1, cL2, 1 / L1, 1 / np.maximum(L12, 1e-12), tf1_p, tf2_p)
            estL, seL, _ = estimate(Sc_L, cL1.astype(float), cL2.astype(float), L1, np.maximum(L12, 1e-12))
            # ---------------- uniform arm: refit on its annotated rows
            Sc_U = fit_nuisances(pe, cU1, cU2, 1 / LU1, 1 / LU12, tf1_u, tf2_u)
            estU, seU, _ = estimate(Sc_U, cU1.astype(float), cU2.astype(float), LU1, LU12)
            for arm, est, se, c1_, c2_ in (("uniform", estU, seU, cU1, cU2), ("learned", estL, seL, cL1, cL2)):
                out.append(dict(rep=rep, pi_e=name, budget_fraction=B, budget_real=Breal, arm=arm, estimate=est, se=se,
                                benchmark=bench[name], error=est - bench[name], covered=int(abs(est - bench[name]) <= 1.96 * se),
                                real_cost=int(c1_.sum() + c2_.sum()), pilot_sessions=int(pilot.sum()), n=n))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dest", default="charged_v2"); ap.add_argument("--budgets", default=None)
    ap.add_argument("--audit", action="store_true", help="assert unpaid second-vote labels never reach the uniform arm's text model")
    ap.add_argument("--rep-start", type=int, default=0, help="first replication index; reps are seeded (SEED, rep) so shards merge exactly")
    args = ap.parse_args(); DEST = DATA / args.dest; DEST.mkdir(exist_ok=True)
    reps = range(args.rep_start, args.rep_start + args.reps); shard = f"_reps{reps.start}-{reps.stop - 1}"   # every run is a shard; merge_shards.py -> results.csv
    budgets = [float(x) for x in args.budgets.split(",")] if args.budgets else BUDGET_FRACTIONS
    config = dict(seed=SEED, n_sub=N_SUB, K=K, pilot_fraction=PILOT_FRAC, floor=FLOOR, gamma=GAMMA, budgets=budgets, reps=args.reps,
                  proxy="META + RM + TFIDF (text model refit per rep on CHARGED votes only; nested fold-safe features)", version="v2 (2026-09-18: label_budget_diagnosis.md fixes)", learner="L2 logistic C=0.1",
                  variance="structural", allocation="joint solve, per-row cost2=Z1, residual budget, casenote-driver mixture",
                  protocol="ARENA_BATCH_ADAPTIVE_FREEZE.md", source_sha256={p.name: digest(p) for p in (HERE / "run_arena_charged.py", HERE / "arena_full_score.py", ROOT / "t2_optimal_allocation.py")})
    config["rep_start"] = args.rep_start; save_once(DEST / f"freeze{shard}.json", config)   # one freeze per shard (shards share DEST)
    t0 = time.time(); data = load_all(); print(f"[{time.time()-t0:.0f}s] data loaded", flush=True)
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(replicate)(r, *data, args.reps, budgets, args.audit) for r in reps)
    res = pd.DataFrame([x for rr in rows for x in rr]); res.to_csv(DEST / f"results{shard}.csv", index=False)
    g = res.groupby(["pi_e", "budget_fraction", "arm"])
    # error-based statistics: each rep has its own subsample and benchmark, so sd(estimate) mixes target
    # variation with annotation noise. sd here is sd(error) (the casenote driver's convention). See summarize_charged.py.
    summ = g.agg(n_reps=("estimate", "size"), sd=("error", lambda e: e.std(ddof=1)), rmse=("error", lambda e: np.sqrt(np.mean(e ** 2))),
                 bias=("error", "mean"), coverage=("covered", "mean"), mean_se=("se", "mean"), mean_real_cost=("real_cost", "mean")).reset_index()
    piv = summ.pivot_table(index=["pi_e", "budget_fraction"], columns="arm", values=["sd", "coverage", "mean_real_cost"])
    piv["sd_ratio_learned_over_uniform"] = piv[("sd", "learned")] / piv[("sd", "uniform")]
    summ.to_csv(DEST / f"summary{shard}.csv", index=False); piv.to_csv(DEST / f"summary_pivot{shard}.csv")
    print(piv.round(4).to_string()); print(f"\n[{time.time()-t0:.0f}s] wrote {DEST}/results{shard}.csv, summary{shard}.csv, summary_pivot{shard}.csv")


if __name__ == "__main__":
    main()
