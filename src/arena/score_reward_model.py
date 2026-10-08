#!/usr/bin/env python3
"""Score every LMArena response with an open-weights reward model. Run on the 4090.

    pip install torch transformers accelerate huggingface_hub pyarrow pandas
    python score_reward_model.py --model Skywork/Skywork-Reward-V2-Llama-3.1-8B

Streams one parquet shard at a time: download -> score -> write scores -> delete the
shard. Peak disk is one shard (~232 MB) plus the model, never the full 1.6 GB release.
Resumable: finished shards are skipped, so it is safe to kill and restart.

Output: arena_rm_scores.parquet with columns [id, score_a, score_b, shard].
That is ~140k rows of floats -- a few MB -- so it is easy to move.

CONTAMINATION NOTE. Skywork-Reward-V2 is trained on SynPref-40M, whose composition is
not disclosed at the level needed to rule out LMArena data. If Arena votes are in the
training mixture the score is partly the gold label and b_1 is optimistically biased.
The arXiv submission date (2507.01352, 2 Jul 2025) falls INSIDE this dataset's window
(17 Apr - 25 Jul 2025), so `--contamination-check` compares AUC before/after a cutoff:
votes after it cannot have been trained on. Run that before trusting the scores.
"""
import argparse, datetime, gc, json, os, socket, subprocess, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "lmarena-ai/arena-human-preference-140k"
OUT = Path("arena_rm_scores.parquet")
STATUS = Path("arena_rm_scores.status.json")
N_SHARDS = 7
ROWS_TOTAL = 135_634


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_status(**kw):
    """Heartbeat JSON. operate-remote-experiments polls this FIRST, ahead of the
    process table and the log tail, so it must always be current and must carry
    error_type / error_message on failure."""
    doc = {"pid": os.getpid(), "host": socket.gethostname(),
           "heartbeat": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "output_path": str(OUT.resolve()), "source_commit": _git_rev()}
    doc.update(kw)
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(STATUS)          # atomic: a watcher never reads a half-written file
    return doc


def messages(conv):
    """conversation_* -> [{role, content}], flattening the nested content parts."""
    out = []
    for turn in (list(conv) if conv is not None else []):
        parts = turn.get("content")
        text = " ".join(str(p.get("text") or "") for p in (list(parts) if parts is not None else []))
        out.append({"role": turn.get("role") or "user", "content": text})
    return [m for m in out if m["content"].strip()]


@torch.no_grad()
def score_batch(model, tok, convs, max_len, device):
    texts = []
    for c in convs:
        m = messages(c)
        if not m:
            texts.append("")
            continue
        texts.append(tok.apply_chat_template(m, tokenize=False))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(device)
    return model(**enc).logits[:, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--contamination-check", metavar="SCORES.parquet",
                    help="skip scoring; run the date-split leakage test on existing scores")
    args = ap.parse_args()

    if args.contamination_check:
        return contamination_check(args.contamination_check)

    write_status(state="starting", model=args.model, shards_done=0,
                 shards_total=N_SHARDS, rows_done=0, rows_total=ROWS_TOTAL)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} model={args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=dtype, device_map=device, num_labels=1).eval()

    done, rows_done = set(), 0
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(prev["shard"].unique()); rows_done = len(prev)
        print(f"resuming: {len(done)} shard(s) already scored, {rows_done:,} rows", flush=True)
    t_start = time.time()

    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{REPO}/**/*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    for path in shards:
        name = path.split("/")[-1]
        if name in done:
            print(f"skip {name} (done)", flush=True)
            continue
        print(f"--- {name}: downloading", flush=True)
        # local_dir= writes a REAL file (not a cache symlink), so deleting it actually
        # frees the space. Without this the blobs accumulate in ~/.cache/huggingface and
        # peak disk becomes the full 1.6 GB, defeating the streaming design.
        local = hf_hub_download(REPO, f"data/{name}", repo_type="dataset",
                                local_dir="_shard_tmp")
        t = pq.read_table(local, columns=["id", "conversation_a", "conversation_b"])
        df = t.to_pandas()
        del t
        n = len(df)
        sa, sb = np.empty(n, "float32"), np.empty(n, "float32")
        import time; t0 = time.time()
        for i in range(0, n, args.batch_size):
            j = min(i + args.batch_size, n)
            sa[i:j] = score_batch(model, tok, df.conversation_a.iloc[i:j], args.max_length, device)
            sb[i:j] = score_batch(model, tok, df.conversation_b.iloc[i:j], args.max_length, device)
            if i and i % (args.batch_size * 50) == 0:
                el = time.time() - t0
                rate = i / el
                remaining = (ROWS_TOTAL - rows_done - i) / max(rate, 1e-9)
                print(f"  {i}/{n}  {rate:.1f} rows/s  eta_shard "
                      f"{(n-i)/rate/60:.1f} min  eta_all {remaining/3600:.1f} h", flush=True)
                write_status(state="running", model=args.model, shard=name,
                             shards_done=len(done), shards_total=N_SHARDS,
                             rows_done=rows_done + i, rows_total=ROWS_TOTAL,
                             rows_per_s=round(rate, 2),
                             eta_seconds=int(remaining),
                             elapsed_seconds=int(time.time() - t_start))
        part = pd.DataFrame({"id": df.id.values, "score_a": sa, "score_b": sb, "shard": name})
        part.to_parquet(OUT) if not OUT.exists() else \
            pd.concat([pd.read_parquet(OUT), part], ignore_index=True).to_parquet(OUT)
        print(f"--- {name}: wrote {n:,} rows in {(time.time()-t0)/60:.1f} min", flush=True)
        import shutil
        shutil.rmtree("_shard_tmp", ignore_errors=True)   # keep peak disk at one shard
        del df, part; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        done.add(name); rows_done += n
        write_status(state="running", model=args.model, shards_done=len(done),
                     shards_total=N_SHARDS, rows_done=rows_done, rows_total=ROWS_TOTAL,
                     elapsed_seconds=int(time.time() - t_start))

    final = pd.read_parquet(OUT)
    receipt = write_status(state="completed", model=args.model,
                           shards_done=len(done), shards_total=N_SHARDS,
                           rows_done=len(final), rows_total=ROWS_TOTAL,
                           rows_complete=bool(len(final) == ROWS_TOTAL),
                           null_scores=int(final[["score_a", "score_b"]].isna().sum().sum()),
                           max_length=args.max_length, batch_size=args.batch_size,
                           elapsed_seconds=int(time.time() - t_start))
    print(json.dumps(receipt, indent=1))
    if len(final) != ROWS_TOTAL:
        print(f"WARNING: {len(final):,} rows written, expected {ROWS_TOTAL:,}. "
              f"Audit before use: successful shards alone do not mean the job is done.")
    print("done ->", OUT)


def contamination_check(scores_path):
    """Leakage test: the RM cannot have trained on votes postdating its release.

    If AUC is materially higher BEFORE the cutoff than after, the score is partly
    reproducing memorised gold labels and must not be used as a cheap proxy.
    """
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(Path(__file__).parent))
    import screen_arena as S
    CUTOFF = "2025-07-02"                          # arXiv 2507.01352 submission date
    df = S.load("arena140k.parquet").set_index("id")
    sc = pd.read_parquet(scores_path).set_index("id")
    d = df.loc[df.index.intersection(sc.index)].join(sc)
    d["best"] = np.maximum(d.score_a, d.score_b)
    d["gap"] = (d.score_a - d.score_b).abs()
    early, late = d[d.ts < CUTOFF], d[d.ts >= CUTOFF]
    print(f"cutoff {CUTOFF}: early n={len(early):,}  late n={len(late):,}\n")
    print(f"{'split':<8}{'n':>9}{'AUC(R~best)':>14}{'AUC(R~gap)':>13}")
    for nm, g in (("early", early), ("late", late), ("all", d)):
        if len(g) < 500 or g.R.nunique() < 2:
            continue
        print(f"{nm:<8}{len(g):>9,}{roc_auc_score(g.R, g.best):>14.4f}"
              f"{roc_auc_score(g.R, -g.gap):>13.4f}")
    print("\nA large early-vs-late gap is evidence of training contamination.")
    print("A small one supports using the score as a genuine cheap proxy.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                     # heartbeat must record the failure
        write_status(state="failed", error_type=type(exc).__name__,
                     error_message=str(exc)[:500])
        raise
