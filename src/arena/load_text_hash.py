#!/usr/bin/env python3
"""Loader for the hashed text-feature bundle $ARENA_DATA/arena_text_hash/ (see README).

    from load_text_hash import load_text_hash
    H = load_text_hash()                 # dict: ids (Series), prompt/response_a/response_b (CSR)
    Xp = H["prompt"][row_ix]             # rows aligned to H["ids"]

Fit idf + logistic per training fold, exactly as ppe_tfidf_pilot.py
does (fit_text takes pre-hashed count matrices, so it accepts these directly).
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena_paths import DATA  # noqa: E402
DIR = os.path.join(DATA, "arena_text_hash")
FIELDS = ("prompt", "response_a", "response_b")

def load_text_hash(d=DIR):
    meta = json.load(open(os.path.join(d, "meta.json")))
    ids = pd.read_parquet(os.path.join(d, "ids.parquet"))
    assert meta["n_rows"] == len(ids) == 135_634, (meta["n_rows"], len(ids))
    out = {"ids": ids, "meta": meta}
    for f in FIELDS:
        X = sp.load_npz(os.path.join(d, f"{f}.npz")).tocsr()
        assert X.shape[0] == len(ids)
        out[f] = X.astype(np.float32)          # counts -> float for TfidfTransformer
    return out

def align(H, id_series):
    """Row indices into the bundle for a Series of Arena `id`s (raises on any miss)."""
    pos = pd.Series(np.arange(len(H["ids"])), index=H["ids"].id.values)
    ix = pos.reindex(id_series.values)
    assert ix.notna().all(), f"{int(ix.isna().sum())} ids missing from the text bundle"
    return ix.to_numpy(dtype=int)

if __name__ == "__main__":
    H = load_text_hash()
    print({k: (v.shape, v.nnz) for k, v in H.items() if sp.issparse(v)})
