"""Stacked text proxy for the Arena 140k: hashed TF-IDF -> logistic P(R = 1 | text), per vote.

POST-HOC PROXY ARM. ARENA_FREEZE.md sec.6 froze the proxy as META + RM and says adding TF-IDF
is a logged design change; this is that change. Reuses ppe_tfidf_pilot.fit_text
(idf fit on the training rows, liblinear logistic) and load_text_hash for the hashed text-feature bundle.

Honest stacking: votes of TRAIN sessions get 5-fold OUT-OF-FOLD scores, folds by session so a
session's two votes share a fold; dev/test votes get scores from the model fit on ALL train
votes. Target is that vote's R = 1{winner != both_bad}; both stages' votes train the model
(same vote-level task). The projection HGB then never sees a text score produced by a model
that saw that row's label.

Framing for the write-up: unlike the RM (external, saw none of these votes) this is a nuisance
LEARNED ON THIS DATA'S GOLD; in deployment it can only be trained on votes already annotated.

    python build_arena_tfidf_scores.py            # needs $ARENA_DATA/arena_text_hash/ (see README)
Writes arena_tfidf_scores.parquet [id, tfidf_p, source].
"""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena_paths import DATA  # noqa: E402
import screen_arena as S
from load_text_hash import load_text_hash, align
from ppe_tfidf_pilot import fit_text

SEED = 20260914
K = 5
FIELDS = ("prompt", "response_a", "response_b")


def main():
    man = json.load(open(os.path.join(DATA, "arena_session_manifest.json")))
    df = S.load(os.path.join(DATA, "arena140k.parquet")).set_index("id")
    H = load_text_hash(os.path.join(DATA, "arena_text_hash"))

    # every vote in the manifest, with its session, fold and gold R
    votes = []
    for r in man:
        votes.append((r["id1"], r["episode_id"], r["fold"]))
        if r["Z1"]:
            votes.append((r["id2"], r["episode_id"], r["fold"]))
    V = pd.DataFrame(votes, columns=["id", "session", "fold"]).drop_duplicates("id")
    V["R"] = df.loc[V.id, "R"].to_numpy().astype(int)
    ix = align(H, V.id)
    fields = [H[f][ix] for f in FIELDS]
    print(f"votes {len(V):,} | train {int((V.fold=='train').sum()):,} dev {int((V.fold=='dev').sum()):,} test {int((V.fold=='test').sum()):,}", flush=True)

    tr = np.flatnonzero(V.fold.to_numpy() == "train")
    ytr = V.R.to_numpy()[tr]
    # session-grouped folds within train, deterministic
    fold_of = {s: int(hashlib.sha256(f"{SEED}|tfidf|{s}".encode()).hexdigest()[:8], 16) % K for s in V.session.unique()}
    gtr = np.array([fold_of[s] for s in V.session.to_numpy()[tr]])
    p = np.full(len(V), np.nan); src = np.array([""] * len(V), dtype=object)
    for k in range(K):
        t0 = time.time(); fit, te = tr[gtr != k], tr[gtr == k]
        p[te] = fit_text([f[fit] for f in fields], ytr[gtr != k], [f[te] for f in fields]); src[te] = "oof_train"
        print(f"  fold {k}: fit {len(fit):,} -> score {len(te):,} in {time.time()-t0:.0f}s", flush=True)
    rest = np.flatnonzero(V.fold.to_numpy() != "train"); t0 = time.time()
    p[rest] = fit_text([f[tr] for f in fields], ytr, [f[rest] for f in fields]); src[rest] = "fulltrain_pred"
    print(f"  full-train fit -> dev/test {len(rest):,} in {time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame({"id": V.id.to_numpy(), "tfidf_p": p, "source": src})
    assert out.tfidf_p.notna().all()
    out.to_parquet(os.path.join(DATA, "arena_tfidf_scores.parquet"), index=False)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y = V.R.to_numpy(); m = src == "oof_train"
    b0 = brier_score_loss(y[m], np.full(m.sum(), y[m].mean())); b = brier_score_loss(y[m], p[m])
    print(f"train OOF: AUC {roc_auc_score(y[m], p[m]):.4f}  Brier skill {100*(1-b/b0):+.2f}%   (text alone, vote level)")
    print("wrote arena_tfidf_scores.parquet")


if __name__ == "__main__":
    main()
