"""Compare regenerated figure PDFs (output/) with the paper's PDFs (paper_figs/) and plot-data CSVs with
the shipped reference CSVs.

Figures are rasterized with poppler's pdftoppm (if installed) at 100 dpi and compared pixel by pixel; the
PDF bytes themselves differ because matplotlib embeds a creation date. Reports, per figure, the share of
pixels that differ and the largest per-channel difference.

Usage: python figures/compare_to_paper.py [--out-dir output]
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_CSV = {  # regenerated plot data -> shipped reference
    "sim_panels_summary.csv": "aggregates/synthetic/sim_panels_summary.csv",
    "fixed_budget_gain_curves_n1000_per_regime_stagewise_plot_data.csv": "aggregates/synthetic/fixed_budget_gain_curves_n1000_per_regime_stagewise_plot_data.csv",
    "drl_coverage_ci_width_n1000_d00_per_regime_stagewise_plot_data.csv": "aggregates/synthetic/drl_coverage_ci_width_n1000_d00_per_regime_stagewise_plot_data.csv",
    "arena_charged_equal_plot_data.csv": "aggregates/arena/arena_charged_equal_plot_data.csv",
    "arena_charged_tilt_plot_data.csv": "aggregates/arena/arena_charged_tilt_plot_data.csv",
}


def raster(pdf: Path, tmp: Path) -> np.ndarray:
    from matplotlib.image import imread
    tmp.mkdir(parents=True, exist_ok=True)
    stem = tmp / pdf.stem
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", str(pdf), str(stem)], check=True)
    return imread(str(stem) + ".png")


def compare_csv(out_dir: Path) -> None:
    for new, ref in REF_CSV.items():
        a = pd.read_csv(ROOT / ref)
        b = pd.read_csv(out_dir / new)
        keys = [c for c in ("panel", "estimator", "d1", "d2", "sampler", "sampler_label", "budget") if c in a.columns]
        a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
        same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
        num = a.select_dtypes("number").columns
        diff = float(np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy()))) if same_shape else float("nan")
        same_labels = same_shape and a.drop(columns=num).equals(b.drop(columns=num))
        print(f"{'OK ' if same_shape and same_labels and diff == 0 else 'DIFF'} {new}: shape {b.shape}, max |diff| = {diff:.3g}")


def compare_pdfs(out_dir: Path) -> None:
    if shutil.which("pdftoppm") is None:
        print("pdftoppm not found (install poppler); skipping the raster comparison")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for ref in sorted((ROOT / "paper_figs").glob("*.pdf")):
            new = out_dir / ref.name
            if not new.exists():
                print(f"--   {ref.name}: not regenerated (schematic or protected)")
                continue
            a, b = raster(ref, tmp / "paper"), raster(new, tmp / "new")
            if a.shape != b.shape:
                print(f"DIFF {ref.name}: raster shape {a.shape} vs {b.shape}")
                continue
            d = np.abs(a.astype(float) - b.astype(float))
            share = float((d.max(axis=-1) > 0).mean())
            print(f"{'OK ' if share == 0 else 'DIFF'} {ref.name}: {100 * share:.3f}% pixels differ, max channel diff {d.max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    compare_csv(args.out_dir)
    compare_pdfs(args.out_dir)


if __name__ == "__main__":
    main()
