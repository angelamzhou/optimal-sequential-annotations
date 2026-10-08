"""Table 1: share of 2019 casenotes by coded progress level -- human (design-weighted, %, 95% CI) and
Gemma 3 12B base / fine-tuned (held-out) minus human (percentage points, 95% CI).

Input (shipped): aggregates/labeler_comparison/labeler_shares_2019.csv and labeler_share_diffs_2019.csv,
produced from the protected casenote data by src/labeler_comparison/table_labeler_shares.py.

Usage: python figures/table_llm_judge.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from table_formatting import rhalf, signed  # noqa: E402

AGG = HERE.parent / "aggregates" / "labeler_comparison"
ROWS = [("No progress", "no_progress"), ("Conversation", "conversation_need"),
        (r"Plans \& paperwork", "planning_paperwork"), ("Milestones", "milestones")]
CONTRASTS = ["Base - Human", "Fine-tuned - Human (held-out)"]


def main() -> None:
    shares = pd.read_csv(AGG / "labeler_shares_2019.csv")
    diffs = pd.read_csv(AGG / "labeler_share_diffs_2019.csv")
    human = shares[shares.labeler == "Human"].set_index("band")
    for label, band in ROWS:
        h = human.loc[band] * 1  # shares in [0, 1]
        cells = [f"{rhalf(100 * h.share, 1):.1f} ({rhalf(100 * h.lo, 1):.1f}, {rhalf(100 * h.hi, 1):.1f})"]
        for c in CONTRASTS:
            r = diffs[(diffs.contrast == c) & (diffs.band == band)].iloc[0]
            cells.append(f"{signed(100 * r['diff'], 1)} ({signed(100 * r.lo, 1)}, {signed(100 * r.hi, 1)})")
        print(f"{label:<19} & " + " & ".join(cells) + r" \\")
    n = shares.groupby("labeler").n.first()
    print(f"\n% n: human {n['Human']}, fine-tuned held-out {n['Gemma 3 fine-tuned (held-out)']}")


if __name__ == "__main__":
    main()
