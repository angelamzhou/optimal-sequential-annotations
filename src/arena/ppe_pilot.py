#!/usr/bin/env python3
"""PPE pilot: how much does an off-the-shelf reward model add, as a cheap proxy for an
Arena vote, over the metadata we already have?

Inputs (already local): hp_v1/<RM>.json (14 reward models, per-item score_1/score_2 on
the 16,038 PPE-Human-Preference-V1 battles) and hp_v1/<judge>.jsonl (LLM judges, binary
decision). Gold = the human `winner`.

Prespecified before any result was read:
  reward R      = 1{winner != 'tie (bothbad)'}   -- the 140k design's reward, exactly
  reward R_dec  = 1{winner in (model_a, model_b)} -- secondary
  features      = META (conv_metadata, category_tag, flags, lengths, sim)
                  RM   (score_1, score_2, max, min, diff, |diff|, sum)
                  META+RM
  learner       = HistGradientBoosting, 5-fold stratified cross-fit, seed 20260914
  metrics       = AUC, Brier skill vs base rate, log-loss
Also reports the PPE pairwise metric (agreement with the vote among decisive battles)
as a sanity check against the paper's published numbers.
"""
import json, glob, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

SEED = 20260914
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena_paths import DATA  # noqa: E402
HERE = os.path.join(DATA, "ppe")  # PPE-Human-Preference-V1 inputs (public; see README)

def _num(v):
    """Some RMs emit a length-1 list per response; unwrap it."""
    if isinstance(v, (list, tuple)):
        return float(v[0]) if len(v) == 1 else float(np.mean(v))
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan            # e.g. '$ERROR$' rows in nemotron-4-340b

def flat(r):
    cm = r.get("conv_metadata") or {}
    ct = r.get("category_tag") or {}
    cr = ct.get("criteria_v0.1") or {}
    ifv = ct.get("if_v0.1") or {}
    mth = ct.get("math_v0.1") or {}
    return {
        "qid": r["question_id"], "winner": r["winner"],
        "ctx_a": cm.get("context_a_tokens"), "ctx_b": cm.get("context_b_tokens"),
        "tok_a": cm.get("sum_assistant_a_tokens"), "tok_b": cm.get("sum_assistant_b_tokens"),
        "tok_user": cm.get("sum_user_tokens"), "turns": cm.get("turns"),
        **{f"cr_{k}": int(bool(v)) for k, v in cr.items()},
        "if_flag": int(bool(ifv.get("if"))), "if_score": ifv.get("score"),
        "math_flag": int(bool(mth.get("math"))),
        "is_code": int(bool(r.get("is_code"))), "is_refusal": int(bool(r.get("is_refusal"))),
        "lang_en": int(r.get("language") == "English"),
        "length_a": r.get("length_a"), "length_b": r.get("length_b"),
        "token_diff": r.get("token_diff"), "sim": r.get("sim"),
        "hard": int(bool(r.get("hard_prompt"))), "easy": int(bool(r.get("easy_prompt"))),
        "if_prompt": int(bool(r.get("if_prompt"))), "math_prompt": int(bool(r.get("math_prompt"))),
        "s1": _num(r.get("score_1")), "s2": _num(r.get("score_2")),
    }

META = ["ctx_a","ctx_b","tok_a","tok_b","tok_user","turns","cr_complexity","cr_creativity",
        "cr_domain_knowledge","cr_problem_solving","cr_real_world","cr_specificity",
        "cr_technical_accuracy","if_flag","if_score","math_flag","is_code","is_refusal",
        "lang_en","length_a","length_b","token_diff","sim","hard","easy","if_prompt","math_prompt"]
RM = ["s1","s2","smax","smin","sdiff","sabs","ssum"]

def add_rm(df):
    df["smax"] = df[["s1","s2"]].max(axis=1); df["smin"] = df[["s1","s2"]].min(axis=1)
    df["sdiff"] = df.s1 - df.s2; df["sabs"] = df.sdiff.abs(); df["ssum"] = df.s1 + df.s2
    return df

def crossfit(X, y, learner="hgb"):
    X = np.asarray(X, dtype=float)
    if learner != "hgb":
        med = np.nanmedian(X, axis=0); X = np.where(np.isnan(X), med, X)
    p = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        if learner == "hgb":
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, random_state=SEED)
        else:
            m = LogisticRegression(max_iter=2000)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p

def score(y, p):
    base = y.mean()
    b = brier_score_loss(y, p); b0 = brier_score_loss(y, np.full_like(p, base))
    return {"auc": round(roc_auc_score(y, p), 4), "brier": round(b, 5),
            "brier_skill_pct": round(100 * (1 - b / b0), 2),
            "logloss": round(log_loss(y, np.clip(p, 1e-6, 1-1e-6)), 4)}

def main():
    out = {"n": None, "rewards": {}, "meta_only": {}, "reward_models": {}, "judges_pairwise": {}}
    rm_files = sorted(glob.glob(os.path.join(HERE, "hp_v1", "*.json")))
    base_df = None
    for f in rm_files:
        name = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        df = add_rm(pd.DataFrame([flat(r) for r in rows])).set_index("qid")
        df["R"] = (df.winner != "tie (bothbad)").astype(int)
        df["R_dec"] = df.winner.isin(["model_a", "model_b"]).astype(int)
        if base_df is None:
            base_df = df
            out["n"] = int(len(df))
            out["rewards"] = {"R_not_bothbad_rate": round(float(df.R.mean()), 4),
                              "R_decisive_rate": round(float(df.R_dec.mean()), 4),
                              "winner_counts": df.winner.value_counts().to_dict()}
            Xm = df[META].to_numpy(dtype=float)
            for tgt in ("R", "R_dec"):
                out["meta_only"][tgt] = score(df[tgt].to_numpy(), crossfit(Xm, df[tgt].to_numpy()))
            print("META only:", json.dumps(out["meta_only"]), flush=True)
        res = {}
        Xr = df[RM].to_numpy(dtype=float); Xmr = df[META + RM].to_numpy(dtype=float)
        for tgt in ("R", "R_dec"):
            y = df[tgt].to_numpy()
            res[tgt] = {"rm_only": score(y, crossfit(Xr, y)),
                        "rm_max_logit": score(y, crossfit(df[["smax"]].to_numpy(dtype=float), y, "logit")),
                        "meta_plus_rm": score(y, crossfit(Xmr, y))}
        dec = df[df.R_dec == 1]
        agree = ((dec.s1 > dec.s2) == (dec.winner == "model_a")).mean()
        res["ppe_pairwise_accuracy_decisive"] = round(float(agree), 4)
        res["n_missing_scores"] = int(df[["s1","s2"]].isna().any(axis=1).sum())
        out["reward_models"][name] = res
        print(f"{name:34s} R: meta+rm AUC {res['R']['meta_plus_rm']['auc']:.4f} "
              f"skill {res['R']['meta_plus_rm']['brier_skill_pct']:+.2f}%  | rm-only AUC "
              f"{res['R']['rm_only']['auc']:.4f} | pairwise acc {agree:.4f}", flush=True)
    # LLM judges: binary A/B decision, only meaningful for the pairwise task
    for f in sorted(glob.glob(os.path.join(HERE, "hp_v1", "*.jsonl"))):
        name = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f)]
        d = pd.DataFrame({"winner": [r["winner"] for r in rows],
                          "decision": [r.get("decision") for r in rows]})
        dec = d[d.winner.isin(["model_a", "model_b"]) & d.decision.notna()]
        # PPE convention inferred: decision==1 -> model_b? check both mappings, report max & which
        a1 = ((dec.decision == 1.0) == (dec.winner == "model_a")).mean()
        a2 = ((dec.decision == 1.0) == (dec.winner == "model_b")).mean()
        out["judges_pairwise"][name] = {"n_scored": int(len(dec)),
                                        "acc_if_1_means_a": round(float(a1), 4),
                                        "acc_if_1_means_b": round(float(a2), 4)}
    json.dump(out, open(os.path.join(HERE, "ppe_pilot_results.json"), "w"), indent=1)
    print("\nwrote ppe_pilot_results.json")

if __name__ == "__main__":
    main()
