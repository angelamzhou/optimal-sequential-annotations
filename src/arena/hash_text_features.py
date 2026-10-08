#!/usr/bin/env python3
"""Precompute hashed bag-of-words features for the Arena 140k (CPU only) and
produce ONE compact bundle, from which the TF-IDF silver proxy is fit on
the revealed gold with cross-fitting.

Why hashing: HashingVectorizer has no fitted state, so the features computed here are
identical to what any machine would compute from the raw text, with no train/test leakage.
Everything label-dependent (idf, the logistic, the lambda-subsample refits) stays in the
downstream scripts. Nothing here reads `winner`.

Two passes, both resumable:
  pass 1  stream shards (download -> hash -> save per-shard npz -> delete shard), and
          accumulate per-field document frequency over the 2**18 buckets.
  pass 2  keep buckets with document frequency >= MIN_DF (unsupervised pruning; on PPE
          df>=5 keeps 92% of nonzeros), stack shards, write the bundle:
             arena_text_hash/ids.parquet         [id, shard, row]  (135,634 rows)
             arena_text_hash/prompt.npz          CSR uint16 counts, rows aligned to ids
             arena_text_hash/response_a.npz
             arena_text_hash/response_b.npz
             arena_text_hash/meta.json           config, kept-bucket index per field, sizes
Ship the `arena_text_hash/` directory (estimate ~0.8 GB). Per-shard npz stay in
`_hash_parts/` and can be deleted once the bundle is verified.

Text fields (multi-turn conversations are flattened):
  prompt      = all user turns of conversation_a, joined   (identical in _b by construction)
  response_a  = all assistant turns of conversation_a
  response_b  = all assistant turns of conversation_b

Usage:  python3 hash_text_features.py            # both passes, streaming from the Hub
        python3 hash_text_features.py --limit-shards 1   # smoke test
        python3 hash_text_features.py --local-shards <SHARD_DIR> --workdir <WORK_DIR>
                                                          # shards pre-downloaded
Requires: huggingface_hub pyarrow pandas scikit-learn scipy numpy  (no torch).
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

REPO = "lmarena-ai/arena-human-preference-140k"
ROWS_TOTAL = 135_634
FIELDS = ("prompt", "response_a", "response_b")
PARTS = Path("_hash_parts"); OUT = Path("arena_text_hash")
MIN_DF = 5
# Must match ppe_tfidf_pilot.py::HV (same directory)
HV = HashingVectorizer(ngram_range=(1, 2), n_features=2**18, stop_words="english",
                       alternate_sign=False, norm=None, dtype=np.float32)

def messages(conv):
    """conversation_* -> [{role, content}], flattening the nested content parts.
    Copied verbatim from score_reward_model.py so both pipelines read the text identically."""
    out = []
    for turn in (list(conv) if conv is not None else []):
        parts = turn.get("content")
        text = " ".join(str(p.get("text") or "") for p in (list(parts) if parts is not None else []))
        out.append({"role": turn.get("role") or "user", "content": text})
    return [m for m in out if m["content"].strip()]

def fields_of(row):
    ma, mb = messages(row.conversation_a), messages(row.conversation_b)
    return (" \n ".join(m["content"] for m in ma if m["role"] == "user"),
            " \n ".join(m["content"] for m in ma if m["role"] != "user"),
            " \n ".join(m["content"] for m in mb if m["role"] != "user"))

def to_u16(X):
    X = X.tocsr(); X.data = np.minimum(X.data, 65535).astype(np.uint16); return X

def pass1(limit, local_shards=None):
    """local_shards: a directory already holding the 7 parquet shards (e.g. pre-downloaded on
    a transfer node with `huggingface-cli download`); if None, stream from the Hub."""
    PARTS.mkdir(exist_ok=True)
    if local_shards:
        shards = sorted(str(p) for p in Path(local_shards).glob("**/*.parquet"))
        assert shards, f"no parquet files under {local_shards}"
    else:
        from huggingface_hub import HfFileSystem   # only needed when streaming from the Hub
        shards = sorted(HfFileSystem().glob(f"datasets/{REPO}/**/*.parquet"))
    if limit: shards = shards[:limit]
    for path in shards:
        name = path.split("/")[-1]; stem = name.replace(".parquet", "")
        if (PARTS / f"{stem}.ids.parquet").exists():
            print(f"skip {name} (done)", flush=True); continue
        t0 = time.time()
        if local_shards:
            local = path; print(f"--- {name}: reading local copy", flush=True)
        else:
            from huggingface_hub import hf_hub_download
            print(f"--- {name}: downloading", flush=True)
            local = hf_hub_download(REPO, f"data/{name}", repo_type="dataset", local_dir="_shard_tmp")
        df = pq.read_table(local, columns=["id", "conversation_a", "conversation_b"]).to_pandas()
        texts = list(zip(*(fields_of(r) for r in df.itertuples(index=False))))
        for f, tx in zip(FIELDS, texts):
            X = to_u16(HV.transform(tx))
            sp.save_npz(PARTS / f"{stem}.{f}.npz", X, compressed=True)
            np.save(PARTS / f"{stem}.{f}.df.npy", np.bincount(X.indices, minlength=X.shape[1]).astype(np.int32))
        pd.DataFrame({"id": df.id.values, "shard": name, "row": np.arange(len(df))}).to_parquet(PARTS / f"{stem}.ids.parquet")
        if not local_shards: shutil.rmtree("_shard_tmp", ignore_errors=True)
        print(f"    {len(df):,} rows hashed in {time.time()-t0:.0f}s", flush=True)

def pass2():
    OUT.mkdir(exist_ok=True)
    stems = sorted(p.name[:-len(".ids.parquet")] for p in PARTS.glob("*.ids.parquet"))
    ids = pd.concat([pd.read_parquet(PARTS / f"{s}.ids.parquet") for s in stems], ignore_index=True)
    meta = {"repo": REPO, "n_rows": int(len(ids)), "shards": stems, "min_df": MIN_DF,
            "vectorizer": {"ngram_range": [1, 2], "n_features": 2**18, "stop_words": "english",
                           "alternate_sign": False, "norm": None, "counts_dtype": "uint16"},
            "fields": {}}
    for f in FIELDS:
        df_tot = sum(np.load(PARTS / f"{s}.{f}.df.npy") for s in stems)
        keep = np.flatnonzero(df_tot >= MIN_DF)
        X = sp.vstack([sp.load_npz(PARTS / f"{s}.{f}.npz") for s in stems]).tocsr()[:, keep].tocsr()
        sp.save_npz(OUT / f"{f}.npz", X, compressed=True)
        np.save(OUT / f"{f}.kept_buckets.npy", keep.astype(np.int32))
        meta["fields"][f] = {"kept_buckets": int(len(keep)), "nnz": int(X.nnz),
                             "nnz_per_row": round(X.nnz / X.shape[0], 1),
                             "file_mb": round(os.path.getsize(OUT / f"{f}.npz") / 1e6, 1)}
        print(f"{f}: kept {len(keep):,} buckets, nnz/row {X.nnz/X.shape[0]:.0f}, {meta['fields'][f]['file_mb']} MB", flush=True)
    ids.to_parquet(OUT / "ids.parquet")
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    total = sum(os.path.getsize(p) for p in OUT.iterdir()) / 1e6
    print(f"bundle {OUT}/: {len(ids):,} rows (expected {ROWS_TOTAL:,}), {total:.0f} MB total", flush=True)
    if len(ids) != ROWS_TOTAL: print("WARNING: row count differs from expected; do not ship until explained", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--pass", dest="which", choices=["1", "2", "both"], default="both")
    ap.add_argument("--local-shards", default=None, help="directory with pre-downloaded shards")
    ap.add_argument("--workdir", default=None, help="run here instead of cwd")
    a = ap.parse_args()
    if a.workdir: Path(a.workdir).mkdir(parents=True, exist_ok=True); os.chdir(a.workdir)
    if a.which in ("1", "both"): pass1(a.limit_shards, a.local_shards)
    if a.which in ("2", "both"): pass2()
