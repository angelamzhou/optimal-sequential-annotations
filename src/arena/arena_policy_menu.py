"""Prespecified evaluation-policy menu for the Arena 140k arm -- OVERLAP STATISTICS ONLY.

Mirrors redial_full_design_v1/policy_menu_and_stochasticity.py, stricter: Phi is NOT
computed here, so nothing in this file can be read as a gap.

SELECTION RULE, fixed before this runs:
    primary pi_e = the non-behaviour candidate with the smallest maximum per-week
    per-action ratio  max_{week,a} pi_e(a) / pi_b(a | week),
    i.e. ReDial's "per-action ratio <= 1.5" standard applied to the menu. The whole menu
    is reported. The size of any policy contrast plays no role.

Candidates are exactly the two prespecified in screen_arena.check4 plus pi_b itself.
Computed on the TRAIN fold with the train-fitted pi_b from the manifest, so no test-fold
information is used. Behaviour propensities are per-week (the roster drifts; see
design.md sec.2).
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena_paths import DATA  # noqa: E402
import screen_arena as S
CATS = ("closed-collapsed",)  # placeholder overwritten below
CATS = ("closed-closed", "mixed", "open-open")

man = json.load(open(os.path.join(DATA, "arena_session_manifest.json")))
summ = json.load(open(os.path.join(DATA, "arena_manifest_summary.json")))
pi_b = summ["pi_b_by_week"]
train = [p for p in man if p["fold"] == "train"]

base = {c: float(np.mean([pi_b[w][c] for w in pi_b])) for c in CATS}   # pooled mean, for the tilt
tilt = {c: base[c] * (1.5 if c == "open-open" else 1.0) for c in CATS}; z = sum(tilt.values())
MENU = {
    "pi_b (behaviour)":            None,
    "uniform over categories":     {c: 1 / 3 for c in CATS},
    "tilt open-open x1.5":         {c: v / z for c, v in tilt.items()},
}

def ratio(pe, w, a):
    pb = pi_b[w][a]
    return 1.0 if pe is None else (pe[a] / pb if pb > 0 else float("inf"))

out = {"selection_rule": __doc__.split("SELECTION RULE")[1].split("Candidates")[0].strip(),
       "computed_on": "train fold, train-fitted per-week pi_b", "menu": {}}
for name, pe in MENU.items():
    w1 = np.array([ratio(pe, p["week"], p["actions"][0]) for p in train])
    z1 = np.array([p["Z1"] for p in train], bool)
    w2 = np.ones(len(train))
    w2[z1] = [ratio(pe, p["week"], p["actions"][1]) for p in train if p["Z1"]]
    W = w1 * w2
    per_cell = [ratio(pe, w, a) for w in pi_b for a in CATS if pi_b[w][a] > 0]
    ess = lambda v: float(v.sum() ** 2 / (v ** 2).sum() / len(v))
    out["menu"][name] = {
        "pi_e": pe if pe else "per-week pi_b",
        "max_per_action_ratio": round(max(per_cell), 3),
        "max_two_step_product": round(max(per_cell) ** 2, 3),
        "ESS_frac_step1": round(ess(w1), 4),
        "ESS_frac_two_step_on_Z1_sessions": round(ess(W[z1]), 4),
        "share_open_open_at_stage1": None if pe is None else round(pe["open-open"], 4),
    }
cands = {k: v for k, v in out["menu"].items() if not k.startswith("pi_b")}
primary = min(cands, key=lambda k: cands[k]["max_per_action_ratio"])
out["primary_pi_e"] = primary
out["primary_reason"] = ("smallest max per-week per-action ratio among non-behaviour "
                         "candidates; the size of any contrast played no role")
out["warning"] = ("Report the whole menu. Phi is deliberately absent from this file.")
json.dump(out, open(os.path.join(DATA, "arena_policy_menu.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
