"""PNAS intro table: year-level progress-band shares, human vs Gemma3 base vs Gemma3 July FT.

Reuses the design-weighted estimator in coauthor_model_comparison.py (same directory): all labelers
are scored on the annotated notes of the 1,000-note designed draw (677 human labels), with
design weights over (silver band x consensus tier). The July FT is scored on held-out notes
only (its training split is dropped and weights re-normalized within cells). Adds stratified
bootstrap 95% CIs for every share.

Aggregates only; never echoes note text. Requires the protected casenote data (see README).
Writes aggregates/labeler_comparison/labeler_shares_2019.csv and labeler_share_diffs_2019.csv.

Usage: PROTECTED_DATA_ROOT=<PROTECTED_DATA_ROOT> python src/labeler_comparison/table_labeler_shares.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "aggregates" / "labeler_comparison" / "labeler_shares_2019.csv"
sys.path.insert(0, str(HERE))

import coauthor_model_comparison as cmc  # noqa: E402
from design_weighted_labeler_eval import BANDS, load_generations, wshares  # noqa: E402

N_BOOT = 2000
SEED = 20260930


def main() -> None:
    pp = cmc.REPO / "protected_data" / "llm_generations"
    cmc.GEMMA_FILES = {"gemma_base": pp / "full-gemma3-base_merged_1a_labels.csv",
                       "gemma_june_ft": pp / "full-gemma3-ft_merged_1a_labels.csv"}
    gens = load_generations()
    gens.update(cmc.load_gemma_generations())
    pop = cmc.cohort_population(gens)
    sample = cmc.annotated_sample(gens, pop)
    ann = sample[sample["annotated"] & sample["human_b"].notna()].copy().reset_index(drop=True)
    split = cmc.normalize_id_columns(pd.read_csv(cmc.SPLIT_KEY, dtype=str), [cmc.CONTACT_ID])
    train = set(split.loc[split.split == "train", cmc.CONTACT_ID])
    heldout = ~ann[cmc.CONTACT_ID].isin(train)
    allrows = pd.Series(True, index=ann.index)
    print(f"annotated notes: {len(ann)}; held out from July FT training: {int(heldout.sum())}")

    specs = [("human_b", "Human", allrows), ("gemma_base_b", "Gemma 3 base", allrows),
             ("july_ft_b", "Gemma 3 fine-tuned (held-out)", heldout),
             ("human_b", "Human, same held-out notes", heldout)]
    cells = list(ann.groupby(["silver_band", "tier"]).indices.values())
    rng = np.random.default_rng(SEED)
    rows = []
    for col, name, mask in specs:
        w = cmc.reweight(ann, pop, mask)
        ok = ann[col].notna().to_numpy() & (w > 0)
        point = wshares(ann.loc[ok, col], w[ok])
        boots = []
        for _ in range(N_BOOT):
            take = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in cells])
            b = ann.iloc[take].reset_index(drop=True)
            bw = cmc.reweight(b, pop, pd.Series(mask.to_numpy()[take], index=b.index))
            bok = b[col].notna().to_numpy() & (bw > 0)
            boots.append(wshares(b.loc[bok, col], bw[bok]))
        for band in BANDS:
            v = np.array([bb[band] for bb in boots])
            rows.append(dict(labeler=name, n=int(ok.sum()), band=band, share=point[band],
                             lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Paired differences from the human estimate on the same notes: Base - Human on all
    # annotated notes, Fine-tuned - Human on the held-out notes. Each difference is computed
    # inside the same stratified resample, so the CI accounts for the pairing.
    def shares(frame: pd.DataFrame, col: str, mask: pd.Series) -> dict:
        w = cmc.reweight(frame, pop, mask)
        ok = frame[col].notna().to_numpy() & (w > 0)
        return wshares(frame.loc[ok, col], w[ok])

    pairs = [("Base - Human", "gemma_base_b", allrows), ("Fine-tuned - Human (held-out)", "july_ft_b", heldout)]
    point = {name: (shares(ann, col, m), shares(ann, "human_b", m)) for name, col, m in pairs}
    boot = {name: [] for name, _, _ in pairs}
    rng = np.random.default_rng(SEED + 1)
    for _ in range(N_BOOT):
        take = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in cells])
        b = ann.iloc[take].reset_index(drop=True)
        for name, col, m in pairs:
            bm = pd.Series(m.to_numpy()[take], index=b.index)
            model, human = shares(b, col, bm), shares(b, "human_b", bm)
            boot[name].append({band: model[band] - human[band] for band in BANDS})
    drows = []
    for name, _, _ in pairs:
        model, human = point[name]
        for band in BANDS:
            v = np.array([bb[band] for bb in boot[name]])
            drows.append(dict(contrast=name, band=band, diff=model[band] - human[band],
                              lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    ddf = pd.DataFrame(drows)
    ddf.to_csv(OUT.with_name("labeler_share_diffs_2019.csv"), index=False)
    print(ddf.assign(diff=ddf["diff"] * 100, lo=ddf["lo"] * 100, hi=ddf["hi"] * 100).round(1).to_string(index=False))
    print(df.pivot(index="band", columns="labeler", values="share").reindex(BANDS).mul(100).round(1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
