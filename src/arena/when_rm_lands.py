#!/usr/bin/env python3
"""Run this the moment arena_rm_scores.parquet has been computed (see README: rebuild the reward-model scores).

    python when_rm_lands.py arena_rm_scores.parquet [--cutoff 2025-07-02] [--label NAME]

PRESPECIFIED (2026-09-14, before any RM score on the 140k was seen):
  protocol   identical to pilot_nuisance.py: HistGradientBoosting(200, 0.08, 31), K=5
             stratified cross-fit, RNG=0; features = proxy_features() (META) plus RM
             features [score_a, score_b, max, min, a-b, |a-b|, a+b].
  floor      META-only b_1 skill on the 140k = +1.17% (pilot_nuisance_results.json).
  expectation from the PPE pilot, 7-8B class: META+RM skill exceeds META by +1.2 to +2.0
             points. (PPE: META 3.26% -> Skywork-8B 4.47%, Eurus-7b 5.25%.)
  decision   increment >= +1.5 pts  -> "helped"; the efficiency story may be argued.
             +0.5 .. +1.5           -> "marginal"; report, do not lean on it.
             <  +0.5                -> "did not help"; drop proxy-quality claims, keep
                                       the allocation claim (which does not need it).
  exclusion  b_2 with TRUE R_1 vs b_2 with IMPUTED b_1(R~_1): if true R_1 adds materially
             over the imputation, the proxy-reward exclusion restriction fails and the
             estimator must keep the b_2(S_2) - b_2(S~_2) term (design.md sec.2).
  contamin.  only if the RM postdates 17 Apr 2025: META+RM skill on votes before vs
             after --cutoff. Large early >> late gap = memorised gold; do not use.
Nothing here is tuned after seeing results. Add rows to the JSON; do not edit the rule.
"""
import sys, os, json, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # src/
sys.path.insert(0, HERE)
from arena_paths import DATA  # noqa: E402
import pilot_nuisance as PN                      # same learner, same folds, same features
from sklearn.metrics import roc_auc_score, brier_score_loss

def rm_feats(fr, sa="score_a", sb="score_b", prefix=""):
    a, b = fr[sa].astype(float), fr[sb].astype(float)
    return pd.DataFrame({f"{prefix}sa": a, f"{prefix}sb": b, f"{prefix}smax": np.maximum(a, b),
                         f"{prefix}smin": np.minimum(a, b), f"{prefix}sdiff": a - b,
                         f"{prefix}sabs": (a - b).abs(), f"{prefix}ssum": a + b}, index=fr.index)

def skill(y, p):
    base = y.mean(); b = brier_score_loss(y, p); b0 = brier_score_loss(y, np.full(len(y), base))
    return dict(n=int(len(y)), base=round(float(base), 4), auc=round(float(roc_auc_score(y, p)), 4),
                brier=round(float(b), 5), skill_pct=round(100 * (1 - b / b0), 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores"); ap.add_argument("--cutoff", default=None,
        help="ISO date; give it only if the RM was released inside the data window")
    ap.add_argument("--label", default="rm")
    ap.add_argument("--learner", default="reg", choices=["reg", "fixed"],
                    help="nuisance learner; 'reg' (default since 2026-09-14, see pilot_nuisance.make_learner)")
    args = ap.parse_args()
    os.environ['NUISANCE_LEARNER'] = args.learner; PN.LEARNER = args.learner
    out = {"scores_file": args.scores, "label": args.label}

    df, a = PN.build(os.path.join(DATA, "arena140k.parquet"))
    sc = pd.read_parquet(args.scores).drop_duplicates("id").set_index("id")
    out["coverage"] = {"score_rows": int(len(sc)), "df_rows": int(len(df)),
                       "matched": int(df.id.isin(sc.index).sum()),
                       "null_scores": int(sc[["score_a", "score_b"]].isna().any(axis=1).sum())}
    print("coverage:", out["coverage"])
    df = df.join(sc[["score_a", "score_b"]], on="id")
    # stage-1 frame `a` is indexed by session; bring scores over by id
    a = a.join(sc[["score_a", "score_b"]], on="id")
    s2 = df[df.evaluation_order == 2].drop_duplicates("evaluation_session_id").set_index("evaluation_session_id")

    # ---------------- b_1 = E[R_1 | F_0]
    y1 = a["R"].astype(int)
    META = PN.proxy_features(a)
    RM = rm_feats(a)
    has = a.score_a.notna() & a.score_b.notna()
    res = {}
    p_meta, r = PN.crossfit(META[has], y1[has], y1[has], "b1  META only (floor)");        res["meta"] = r
    p_rm,   r = PN.crossfit(RM[has],   y1[has], y1[has], "b1  RM only");                  res["rm_only"] = r
    p_both, r = PN.crossfit(pd.concat([META, RM], axis=1)[has], y1[has], y1[has], "b1  META + RM"); res["meta_rm"] = r
    inc = 100 * (res["meta_rm"]["skill"] - res["meta"]["skill"])
    res["increment_pts"] = round(inc, 3)
    res["verdict"] = "helped" if inc >= 1.5 else "marginal" if inc >= 0.5 else "did not help"
    out["b1"] = res
    print(f"\n  b1 increment META->META+RM: {inc:+.3f} pts  ->  {res['verdict'].upper()}")

    # heterogeneity the allocator can use: spread of imputations, residual var by RM decile
    a1 = a[has].copy(); a1["p_both"] = p_both; a1["smax"] = RM.loc[has, "smax"]
    a1["dec"] = pd.qcut(a1.smax.rank(method="first"), 10, labels=False)
    a1["sq"] = (a1.R - a1.p_both) ** 2
    het = a1.groupby("dec").agg(n=("R", "size"), R_rate=("R", "mean"), resid_var=("sq", "mean")).round(4)
    out["b1_heterogeneity"] = {"sd_of_imputation": round(float(np.std(p_both)), 4),
                               "by_rm_max_decile": het.reset_index().to_dict(orient="records")}

    # ---------------- b_2 = E[R_2 | F_1] on paired sessions
    pr = a[(a.C == 1) & a.R2.notna()].copy()
    pr = pr.join(s2[["score_a", "score_b"]].rename(columns={"score_a": "sa2", "score_b": "sb2"}))
    has2 = pr.sa2.notna() & pr.sb2.notna() & pr.score_a.notna()
    pr = pr[has2]
    y2 = pr["R2"].astype(int)
    fr2 = s2.loc[pr.index, ["conv_metadata", "category_tag", "is_code", "language"]]   # stage-2 packet's own proxy
    META2 = PN.proxy_features(fr2)
    RM2 = rm_feats(pr, "sa2", "sb2", "s2_")
    base2 = pd.concat([META2, RM2], axis=1)
    b2 = {}
    _, b2["no_R1"]      = PN.crossfit(base2, y2, y2, "b2  META2+RM2, no R1")
    _, b2["true_R1"]    = PN.crossfit(base2.assign(R1=pr.R.values), y2, y2, "b2  + TRUE R1")
    # imputed R1: the stage-1 META+RM prediction, cross-fit, for these sessions
    imp = pd.Series(p_both, index=a[has].index).reindex(pr.index)
    _, b2["imputed_R1"] = PN.crossfit(base2.assign(R1hat=imp.values), y2, y2, "b2  + IMPUTED R1 = b1(R~1)")
    _, b2["both"]       = PN.crossfit(base2.assign(R1=pr.R.values, R1hat=imp.values), y2, y2, "b2  + true R1 + imputed")
    gap = 100 * (b2["true_R1"]["skill"] - b2["imputed_R1"]["skill"])
    b2["exclusion_gap_pts"] = round(gap, 3)
    b2["exclusion_reading"] = ("true R1 adds materially over its imputation -> restriction FAILS; keep the "
                               "b2(S2)-b2(S~2) term" if gap >= 1.0 else
                               "imputed R1 nearly as good as true R1 -> restriction plausible here")
    out["b2"] = b2
    print(f"\n  exclusion-restriction gap (true R1 vs imputed): {gap:+.3f} pts  -> {b2['exclusion_reading']}")

    # ---------------- contamination (only for an in-window RM)
    if args.cutoff:
        ts = pd.to_datetime(a.loc[has, "timestamp"], utc=True)
        cut = pd.Timestamp(args.cutoff, tz="UTC")
        e, l = (ts < cut).values, (ts >= cut).values
        out["contamination"] = {"cutoff": args.cutoff,
            "early": skill(y1[has][e].values, p_both[e]), "late": skill(y1[has][l].values, p_both[l]),
            "meta_early": skill(y1[has][e].values, p_meta[e]), "meta_late": skill(y1[has][l].values, p_meta[l])}
        d_rm = out["contamination"]["early"]["auc"] - out["contamination"]["late"]["auc"]
        d_me = out["contamination"]["meta_early"]["auc"] - out["contamination"]["meta_late"]["auc"]
        out["contamination"]["rm_gap_minus_meta_gap"] = round(d_rm - d_me, 4)
        out["contamination"]["reading"] = ("RM early>>late beyond what META shows -> contamination; do not use"
                                           if d_rm - d_me > 0.02 else "no early/late asymmetry beyond META -> ok")
        print("\n  contamination:", json.dumps(out["contamination"], indent=1))
    else:
        out["contamination"] = "skipped: RM predates the data window (state its release date in the receipt)"

    fn = os.path.join(DATA, f"rm_landing_{args.label}.json")
    json.dump(out, open(fn, "w"), indent=1, default=str)
    print("\nwrote", fn)

if __name__ == "__main__":
    main()
