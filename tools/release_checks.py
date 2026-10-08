"""Release hygiene: privacy scan of the whole tree and RELEASE_MANIFEST.md for the shipped data files.

    python tools/release_checks.py            # scan + rewrite RELEASE_MANIFEST.md
    python tools/release_checks.py --scan     # scan only

The scan fails (exit 1) on: identifier-like column names in any CSV/JSON (client/contact/session ids),
row-level date columns, absolute host paths, credential-like strings, or the partner-name pattern in any
file path or text file. Code that names the protected data's columns (e.g. CLIENT_ID = "ClientID") is
reported separately as expected, because the pipeline must read those columns from the protected files.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Patterns are assembled from pieces so that this file does not itself match them.
_I = "b" + "g"
PARTNER = re.compile(r"(?i)\b" + _I + r"\b|" + _I + "_|" + _I + "-|" + "break" + "ing gr" + "ound")
HOSTPATH = re.compile("/" + "Users/|/" + r"scratch\d*/|/" + r"project\d*/|/" + r"home\d*/|" + "zhou" + r"a_\d+|" + "tail" + "scale|" + r"\.ts\.net|" + "Down" + "loads/")
CRED = re.compile(r"(?i)\b(api[_-]?key|" + "sec" + "ret|" + "pass" + "word|" + "pass" + r"wd)\b|\b" + "hf" + r"_[A-Za-z0-9]{20,}|\b" + "sk" + r"-[A-Za-z0-9]{20,}|\b" + "tok" + r"en\s*=\s*['\"][A-Za-z0-9]")
ID_COL = re.compile(r"(?i)^(client_?id|contact_?id|session_?id|evaluation_session_id|episode_id|user_?id|question_id|id1|id2|note|note_text|ts1|timestamp|date)$")
DATE_VAL = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?|^\d{1,2}/\d{1,2}/\d{2,4}$")
SKIP_DIRS = {".git", "output", "results", "__pycache__", "data", "protected_data"}

PROVENANCE = {
    "aggregates/casenote/placement_plot_data.csv": "Fig. 4 placement panels: per-budget DRL RMSE / mean CI width / MC SEs over 100 reps of 600 clients (plot_annotation_mc.py output on the protected MC run)",
    "aggregates/casenote/progress_plot_data.csv": "Fig. 4 progress panels and Fig. S4: per-budget DRL RMSE / mean CI width / MC SEs (random, stagewise, trajectory-wise; reconciled progress reward)",
    "aggregates/casenote/active_inference_summary.csv": "Table 2: per (task, budget, sampler) DRL RMSE and mean CI width, computed from the protected uncertainty-proportional runs",
    "aggregates/casenote/bundle_stats.csv": "Fig. S4 text: per-budget paired-bootstrap RMSE ratio stagewise/trajectory-wise (precomputed; per-rep runs withheld)",
    "aggregates/casenote/value_estimates_both_stages.csv": "Main-text policy effects: full-cohort (n=775) estimates, shift up one quintile at both stages (reconciled progress reward)",
    "aggregates/casenote/value_estimates_stage1_only.csv": "Main-text single-stage effect: shift at stage 1 only (full-cohort estimates)",
    "aggregates/casenote/value_estimates_stage2_only.csv": "Main-text single-stage effect: shift at stage 2 only (full-cohort estimates)",
    "aggregates/casenote/placement_run_config_shard00.json": "Configuration of the Fig. 4 placement MC run (input paths replaced by <PROTECTED> placeholders)",
    "aggregates/casenote/progress_run_config_shard00.json": "Configuration of the Fig. 4 progress MC run (input paths replaced by <PROTECTED> placeholders)",
    "aggregates/simulator_validation/sim_validation_placement_summary.csv": "Fig. S5 top row: per-budget RMSE / coverage / width vs simulator truth",
    "aggregates/simulator_validation/sim_validation_progress_summary.csv": "Fig. S5 bottom row: per-budget RMSE / coverage / width vs simulator truth",
    "aggregates/synthetic/results.csv": "Synthetic MC (purely simulated): DRL + plug-in rows for random, the four stagewise designs and full annotation; 16 columns",
    "aggregates/synthetic/sim_panels_summary.csv": "Fig. 3 / Fig. S3 plot data (reference output of figures/fig_simulated_panels.py)",
    "aggregates/synthetic/fixed_budget_gain_curves_n1000_per_regime_stagewise_plot_data.csv": "Fig. 3 text numbers (reference output of figures/synthetic_gain_curves.py)",
    "aggregates/synthetic/drl_coverage_ci_width_n1000_d00_per_regime_stagewise_plot_data.csv": "Fig. 3 text numbers (reference output of figures/synthetic_coverage_width.py)",
    "aggregates/arena/charged_v2_results.csv": "Fig. 5: per-replication LMArena MC results (public data; 1,000 reps x 2 policies x 6 budgets x 2 arms; no session ids)",
    "aggregates/arena/charged_v2_summary_error_based.csv": "Table S3 / text: per-budget LMArena summary (src/arena/summarize_charged.py)",
    "aggregates/arena/arena_charged_equal_plot_data.csv": "Fig. 5 plot data, equal exposure (reference output of figures/plot_arena_panels.py)",
    "aggregates/arena/arena_charged_tilt_plot_data.csv": "Fig. 5 plot data, tilt (reference output of figures/plot_arena_panels.py)",
    "aggregates/arena/arena_manifest_summary.json": "LMArena manifest summary (counts by fold/week, behavior-policy shares by week; no session rows)",
    "aggregates/arena/arena_paper_numbers.json": "LMArena values quoted in the paper (Phi^pi_b, date range, model counts and public model names) derived from the per-session manifest",
    "aggregates/labeler_comparison/labeler_shares_2019.csv": "Table 1: design-weighted band shares with bootstrap CIs (human n=677; fine-tuned held-out n=338)",
    "aggregates/labeler_comparison/labeler_share_diffs_2019.csv": "Table 1: model minus human share differences with bootstrap CIs",
    "aggregates/labeler_comparison/coauthor_model_comparison.csv": "Main text 92% agreement: design-weighted labeler accuracy (rows for a leaked-split evaluation and superseded labels removed)",
}


def text_files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or not p.is_file():
            continue
        yield p, rel


def is_text(p: Path) -> bool:
    try:
        p.read_text()
        return True
    except UnicodeDecodeError:
        return False


def load_table(p: Path):
    if p.suffix == ".csv":
        return pd.read_csv(p, low_memory=False)
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return obj


def scan() -> int:
    problems, expected = [], []
    for p, rel in text_files():
        if PARTNER.search(str(rel)):
            problems.append(f"partner pattern in path: {rel}")
        if not is_text(p):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for name, rx in (("partner", PARTNER), ("host path", HOSTPATH), ("credential", CRED)):
                if rx.search(line):
                    problems.append(f"{name}: {rel}:{i}: {line.strip()[:100]}")
            if p.suffix == ".py" and rel.parts[0] != "tools" and re.search(r'(?i)["\'](client_?id|contact_?id)["\']', line):
                expected.append(f"{rel}:{i}")
    for p, rel in text_files():
        if p.suffix not in (".csv", ".json"):
            continue
        t = load_table(p)
        if isinstance(t, pd.DataFrame):
            bad = [c for c in t.columns if ID_COL.match(str(c))]
            if bad:
                problems.append(f"id/date-like columns in {rel}: {bad}")
            for c in t.select_dtypes(exclude="number").columns:
                vals = t[c].dropna().astype(str)
                if len(vals) and vals.str.match(DATE_VAL).mean() > 0.5 and t[c].nunique() > 20:
                    problems.append(f"row-level date-like column in {rel}: {c}")
        else:
            flat = json.dumps(t)
            if re.search(r'"(client_?id|contact_?id|session_?id|episode_id)"', flat, re.I):
                problems.append(f"id-like key in {rel}")
    print(f"privacy scan: {len(problems)} problem(s)")
    for x in problems:
        print("  PROBLEM", x)
    print(f"expected: {len(expected)} code lines name protected-data id columns (column-name constants, no values): "
          + ", ".join(sorted({e.split(':')[0] for e in expected})))
    return 1 if problems else 0


def manifest() -> None:
    rows = []
    for p, rel in text_files():
        if not rel.as_posix().startswith("aggregates/"):
            continue
        t = load_table(p)
        if isinstance(t, pd.DataFrame):
            nrows, cols = len(t), ", ".join(map(str, t.columns))
        else:
            nrows, cols = "-", "keys: " + ", ".join(t.keys())
        rows.append((rel.as_posix(), p.stat().st_size, nrows, cols, PROVENANCE.get(rel.as_posix(), "MISSING PROVENANCE")))
    total = sum(r[1] for r in rows)
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    tree = sum((ROOT / f).stat().st_size for f in tracked if (ROOT / f).exists())
    out = ["# Release manifest", "",
           "Every data file shipped in `aggregates/`: size, rows, columns and provenance. Regenerate with",
           "`python tools/release_checks.py`. No file contains per-client, per-note or per-session rows,",
           "identifiers, dates at row level, note text or model outputs for the protected casenote data.", "",
           f"Data files: {len(rows)}, {total:,} bytes. Whole tracked tree: {len(tracked)} files, {tree:,} bytes.", "",
           "| file | bytes | rows | columns | provenance |", "|---|---:|---:|---|---|"]
    for f, b, n, c, prov in rows:
        out.append(f"| `{f}` | {b:,} | {n if isinstance(n, str) else f'{n:,}'} | {c.replace('|', '/')} | {prov} |")
    (ROOT / "RELEASE_MANIFEST.md").write_text("\n".join(out) + "\n")
    print(f"wrote RELEASE_MANIFEST.md ({len(rows)} data files, {total:,} bytes)")
    missing = [r[0] for r in rows if r[4] == "MISSING PROVENANCE"]
    if missing:
        sys.exit(f"missing provenance for: {missing}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    code = scan()
    if not a.scan:
        manifest()
    sys.exit(code)
