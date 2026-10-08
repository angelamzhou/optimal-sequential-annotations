#!/usr/bin/env python3
"""PPE TF-IDF pilot: does a bag-of-words proxy *learned from revealed gold* add to the
metadata floor, and is it complementary to an off-the-shelf reward model?

Companion to ppe_pilot.py: same rows, same reward, same folds (StratifiedKFold(5),
seed 20260914), same learner for the stacked stage (HGB 300/0.05/31), same metrics.

Prespecified before any result was read:
  text arms   P    = TF-IDF(prompt)
              PR   = TF-IDF(prompt) ⊕ TF-IDF(response_1) ⊕ TF-IDF(response_2)   (hstack)
  vectoriser  word 1-2grams, HashingVectorizer(2**18 per field, english stop words,
              alternate_sign=False) -> sublinear tf, idf fit on the TRAINING fold only.
              [Amended 2026-09-14 before any result was read: the first draft fit a
              vocabulary (min_df=5, 50k) per fold; that alone overran the 3-min shell
              window. Hashing is stateless so it can be applied once to all rows with no
              train/test leakage; only the idf is fold-fit.]
  text learner L2 logistic, C=1 (fixed, no tuning), liblinear
  stacking    text arm -> out-of-fold probability, nested (inner 5-fold on the outer
              training fold) so no row's stacked feature was produced by a model that
              saw it; then HGB on META (+ RM) (+ p_text) with the SAME outer folds
  RMs used    Eurus-RM-7b, Skywork-Reward-Llama-3.1-8B (the two 4090 candidates),
              nemotron-4-340b-reward (the ceiling)
  decision    increment of META+TFIDF over META in Brier skill, same thresholds as
              when_rm_lands.py: >= +1.5 pts helped, +0.5..1.5 marginal, < +0.5 no.
Reads: hp_v1/<RM>.json, PPE-Human-Preference-V1/data/test-00000-of-00001.parquet.
Writes: _tfidf_oof.parquet (cached OOF text probabilities), ppe_tfidf_results.json.
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppe_pilot as P

HERE = P.HERE; SEED = P.SEED
RMS = ["Eurus-RM-7b", "Skywork-Reward-Llama-3.1-8B", "nemotron-4-340b-reward"]
TXT = os.path.join(HERE, "PPE-Human-Preference-V1", "data", "test-00000-of-00001.parquet")
OOF = os.path.join(HERE, "_tfidf_oof.parquet")

def load():
    dfs = {}
    for rm in RMS:
        rows = json.load(open(os.path.join(HERE, "hp_v1", rm + ".json")))
        dfs[rm] = P.add_rm(pd.DataFrame([P.flat(r) for r in rows])).set_index("qid")
    base = dfs[RMS[0]].copy()
    base["R"] = (base.winner != "tie (bothbad)").astype(int)
    base["R_dec"] = base.winner.isin(["model_a", "model_b"]).astype(int)
    txt = pd.read_parquet(TXT, columns=["question_id", "prompt", "response_1", "response_2"]).set_index("question_id")
    assert txt.index.is_unique and base.index.is_unique
    txt = txt.loc[base.index]
    return base, dfs, txt

HV = HashingVectorizer(ngram_range=(1, 2), n_features=2**18, stop_words="english",
                       alternate_sign=False, norm=None, dtype=np.float32)

def hashed(texts):
    """Stateless: counts only, no fitted state, so safe to apply to all rows once."""
    return HV.transform(texts).tocsr()

def fit_text(fields_tr, y_tr, fields_te):
    """fields_* are lists of pre-hashed count matrices; idf is fit on the training rows."""
    ts = [TfidfTransformer(sublinear_tf=True).fit(f) for f in fields_tr]
    Xtr = sp.hstack([t.transform(f) for t, f in zip(ts, fields_tr)]).tocsr()
    Xte = sp.hstack([t.transform(f) for t, f in zip(ts, fields_te)]).tocsr()
    m = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(Xtr, y_tr)
    return m.predict_proba(Xte)[:, 1]

def nested_oof(fields, y, arm):
    """Outer OOF probabilities plus, for each outer fold, inner-OOF probabilities on the
    training rows (so the stacked HGB trains on honest text features)."""
    n = len(y); outer = np.zeros(n); inner = np.full((5, n), np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
        t0 = time.time()
        outer[te] = fit_text([f[tr] for f in fields], y[tr], [f[te] for f in fields])
        iskf = StratifiedKFold(5, shuffle=True, random_state=SEED + 1 + k)
        for itr, ite in iskf.split(np.zeros(len(tr)), y[tr]):
            inner[k, tr[ite]] = fit_text([f[tr[itr]] for f in fields], y[tr[itr]], [f[tr[ite]] for f in fields])
        print(f"  {arm} outer fold {k}: {time.time()-t0:.0f}s", flush=True)
    return outer, inner

def stacked(Xbase, y, outer, inner):
    """HGB on Xbase ⊕ p_text with the same outer folds; train rows use inner-OOF p_text."""
    n = len(y); p = np.zeros(n)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
        Xtr = np.column_stack([Xbase[tr], inner[k, tr]]); Xte = np.column_stack([Xbase[te], outer[te]])
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           random_state=SEED).fit(Xtr, y[tr])
        p[te] = m.predict_proba(Xte)[:, 1]
    return p

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    base, dfs, txt = load()
    y = {t: base[t].to_numpy() for t in ("R", "R_dec")}
    t0 = time.time()
    Hp, H1, H2 = hashed(txt.prompt.fillna("")), hashed(txt.response_1.fillna("")), hashed(txt.response_2.fillna(""))
    print(f"hashed 3 fields in {time.time()-t0:.0f}s", flush=True)
    fields = {"P": [Hp], "PR": [Hp, H1, H2]}
    # ---- stage 1: text OOF (cached) ----
    if os.path.exists(OOF) and stage == "all":
        oof = pd.read_parquet(OOF)
        print("loaded cached", OOF, flush=True)
    else:
        oof = pd.DataFrame(index=base.index)
        arm_req, _, tgt_req = stage.partition(":")
        todo = [a for a in ("P", "PR") if arm_req in ("all", "text", a)]
        tgts = [t for t in ("R", "R_dec") if tgt_req in ("", t)]
        if os.path.exists(OOF): oof = pd.read_parquet(OOF)
        for arm in todo:
            for t in tgts:
                o, i = nested_oof(fields[arm], y[t], f"{arm}/{t}")
                oof[f"{arm}_{t}_outer"] = o
                for k in range(5): oof[f"{arm}_{t}_inner{k}"] = i[k]
        oof.to_parquet(OOF); print("wrote", OOF, flush=True)
        if stage != "all": return
    # ---- stage 2: table ----
    out = {"n": int(len(base)), "arms": {}}
    Xm = base[P.META].to_numpy(dtype=float)
    for t in ("R", "R_dec"):
        res = {}
        res["META"] = P.score(y[t], P.crossfit(Xm, y[t]))
        for arm in ("P", "PR"):
            o = oof[f"{arm}_{t}_outer"].to_numpy()
            inner = np.vstack([oof[f"{arm}_{t}_inner{k}"].to_numpy() for k in range(5)])
            res[f"TFIDF_{arm}"] = P.score(y[t], o)
            res[f"META+TFIDF_{arm}"] = P.score(y[t], stacked(Xm, y[t], o, inner))
        for rm in RMS:
            Xmr = dfs[rm][P.META + P.RM].loc[base.index].to_numpy(dtype=float)
            res[f"META+RM[{rm}]"] = P.score(y[t], P.crossfit(Xmr, y[t]))
            o = oof[f"PR_{t}_outer"].to_numpy()
            inner = np.vstack([oof[f"PR_{t}_inner{k}"].to_numpy() for k in range(5)])
            res[f"META+RM[{rm}]+TFIDF_PR"] = P.score(y[t], stacked(Xmr, y[t], o, inner))
        floor = res["META"]["brier_skill_pct"]
        for k_, v in res.items(): v["inc_over_META_pts"] = round(v["brier_skill_pct"] - floor, 2)
        inc = res["META+TFIDF_PR"]["inc_over_META_pts"]
        res["verdict_TFIDF_PR"] = "helped" if inc >= 1.5 else "marginal" if inc >= 0.5 else "did not help"
        out["arms"][t] = res
        print(f"\n== reward {t}", flush=True)
        for k_, v in res.items():
            if isinstance(v, dict):
                print(f"  {k_:44s} AUC {v['auc']:.4f}  skill {v['brier_skill_pct']:+.2f}%  inc {v['inc_over_META_pts']:+.2f}", flush=True)
        print("  verdict:", res["verdict_TFIDF_PR"], flush=True)
    json.dump(out, open(os.path.join(HERE, "ppe_tfidf_results.json"), "w"), indent=1)
    print("\nwrote ppe_tfidf_results.json")

if __name__ == "__main__":
    main()
