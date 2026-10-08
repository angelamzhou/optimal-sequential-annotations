#!/usr/bin/env python3
"""Screen lmarena-ai/arena-human-preference-140k as a T=2 OPE task.

Runs the five prespecified checks in design.md. Read-only; no network.

Getting the data: see DOWNLOAD.md (run it on a machine with open egress).

Then:
    python screen_arena.py arena140k.parquet

Every threshold and category map below is prespecified. Do not tune them after seeing
the outcome columns.
"""
import sys, json
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- configuration
# Prespecified coarsening of the 53 models into action categories. Assignment is by
# provenance only and uses no outcome information. REVIEW THIS LIST against the actual
# model names in the release before running - it is a prefix map, not a verified roster.
# VERIFIED roster partition, fixed 2026-09-13 BEFORE any outcome column was inspected.
# Criterion: a model is "open" iff the weights of THAT EXACT CHECKPOINT were publicly
# released. Verified one name at a time against the HuggingFace Hub model API (200 vs 401
# on the canonical repo), not from memory. The prior prefix map is kept only as a guard.
#
# Three names the prefix map got wrong, all corrected to "closed", all verified:
#   qwen-max-2025-01-25                  Alibaba's proprietary API flagship; no public repo.
#   hunyuan-turbos-20250416              Tencent API model; Hunyuan-A13B is a different model.
#   llama-4-maverick-03-26-experimental  Arena-only variant; only the -17b-128e-instruct
#                                        checkpoint was released (cf. Leaderboard Illusion).
OPEN_WEIGHTS = frozenset({
    "command-a-03-2025",
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    "gemma-3-27b-it",
    "gemma-3n-e4b-it",
    "kimi-k2-0711-preview",
    "llama-3.3-70b-instruct",
    "llama-4-maverick-17b-128e-instruct",
    "llama-4-scout-17b-16e-instruct",
    "minimax-m1",
    "mistral-small-2506",
    "mistral-small-3.1-24b-instruct-2503",
    "qwen3-235b-a22b",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-235b-a22b-no-thinking",
    "qwen3-30b-a3b",
    "qwen3-coder-480b-a35b-instruct",
    "qwq-32b",
})
# Every other name in the 53-model roster is closed-weight. Listed so an unrecognised
# name is an error rather than a silent "closed".
CLOSED_WEIGHTS = frozenset({
    "amazon-nova-experimental-chat-05-14", "amazon.nova-pro-v1:0",
    "chatgpt-4o-latest-20250326", "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219-thinking-32k", "claude-opus-4-20250514",
    "claude-opus-4-20250514-thinking-16k", "claude-sonnet-4-20250514",
    "claude-sonnet-4-20250514-thinking-32k", "gemini-2.0-flash-001",
    "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17-thinking", "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro", "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-preview-05-06",
    "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18", "grok-3-mini-beta", "grok-3-mini-high",
    "grok-3-preview-02-24", "grok-4-0709", "hunyuan-turbos-20250416",
    "llama-4-maverick-03-26-experimental", "magistral-medium-2506",
    "mistral-medium-2505", "o3-2025-04-16", "o3-mini", "o4-mini-2025-04-16",
    "qwen-max-2025-01-25",
})

def model_class(name: str) -> str:
    n = (name or "").lower()
    if n in OPEN_WEIGHTS:
        return "open"
    if n in CLOSED_WEIGHTS:
        return "closed"
    raise KeyError(f"model not in the frozen 53-model roster: {name!r}")

def pair_category(a: str, b: str) -> str:
    """Unordered pair category: the action is a pair, not a response."""
    ca, cb = model_class(a), model_class(b)
    return "open-open" if ca == cb == "open" else \
           "closed-closed" if ca == cb == "closed" else "mixed"

REWARD = lambda w: 0.0 if w == "both_bad" else 1.0          # primary
REWARD2 = lambda w: 1.0 if w in ("model_a", "model_b") else 0.0  # secondary

def load(path):
    df = pd.read_parquet(path, columns=[
        "id", "model_a", "model_b", "winner", "evaluation_session_id",
        "evaluation_order", "language", "is_code", "timestamp", "category_tag",
        "conv_metadata",
    ])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["week"] = df["ts"].dt.to_period("W").astype(str)
    df["pair"] = [pair_category(a, b) for a, b in zip(df.model_a, df.model_b)]
    df["R"] = df.winner.map(REWARD)
    df["R2"] = df.winner.map(REWARD2)
    df = df.sort_values(["evaluation_session_id", "evaluation_order", "ts"])
    # 15 (session, order) cells carry two rows (13 sessions). Keep the earliest by
    # timestamp so the (session, order) key is unique; recorded in check 1.
    n_before = len(df)
    df = df.drop_duplicates(["evaluation_session_id", "evaluation_order"], keep="first")
    df.attrs["n_dropped_dup_order"] = n_before - len(df)
    return df

# ------------------------------------------------------------------ check 0
def check0(df, out):
    """Randomization sanity, and the falsification test for prompt-conditional routing."""
    dec = df[df.winner.isin(["model_a", "model_b"])]
    out["position_balance"] = {
        "model_a_wins": int((dec.winner == "model_a").sum()),
        "model_b_wins": int((dec.winner == "model_b").sum()),
        "share_a": round(float((dec.winner == "model_a").mean()), 4),
        "binom_p_two_sided": float(stats.binomtest(
            int((dec.winner == "model_a").sum()), len(dec), 0.5).pvalue),
    }
    # Does the pair category depend on prompt features, within week?
    # A dependence here means the platform routes on the prompt, which would put those
    # features into S'_t and weaken the unconfounded-by-design claim.
    # design.md requires the routing test on language, is_code, category_tag AND prompt
    # length. Only PRE-treatment prompt features are admissible: sum_user_tokens and the
    # category_tag flags are properties of the user's prompt. conv_metadata's
    # sum_assistant_*/header_*/list_*/bold_* fields describe the MODEL OUTPUTS and are
    # downstream of the action - using them here would be a post-treatment test.
    def _tag(d, group, key):
        try:
            return bool(d[group][key])
        except (KeyError, TypeError):
            return False
    df = df.copy()
    df["f_math"] = [_tag(d, "math_v0.1", "math") for d in df.category_tag]
    df["f_creative"] = [_tag(d, "creative_writing_v0.1", "creative_writing")
                        for d in df.category_tag]
    df["f_if"] = [_tag(d, "if_v0.1", "if") for d in df.category_tag]
    df["f_hard"] = [sum(bool(v) for v in (d or {}).get("criteria_v0.1", {}).values()) >= 4
                    for d in df.category_tag]
    ulen = pd.Series([(d or {}).get("sum_user_tokens", np.nan) for d in df.conv_metadata],
                     index=df.index, dtype="float64")
    df["f_promptlen_q"] = pd.qcut(ulen, 4, labels=False, duplicates="drop")

    feats = ("language_en", "is_code", "f_math", "f_creative", "f_if", "f_hard",
             "f_promptlen_q")
    rows = []
    for feat in feats:
        if feat == "language_en":
            col = (df.language == "en")
        elif feat == "is_code":
            col = df.is_code.astype(bool)
        else:
            col = df[feat]
        per_week = []
        for wk, g in df.groupby("week"):
            if len(g) < 500:
                continue
            c = col.loc[g.index]
            if c.isna().all():
                continue
            tab = pd.crosstab(g["pair"], c)
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            per_week.append({"week": wk, "n": int(len(g)), "p": float(p),
                             "cramers_v": float(np.sqrt(chi2 / (len(g) * (min(tab.shape) - 1))))})
        rows.append({"feature": feat, "weeks_tested": len(per_week),
                     "weeks_p_below_0.01": sum(1 for r in per_week if r["p"] < 0.01),
                     "median_cramers_v": round(float(np.median([r["cramers_v"] for r in per_week])), 4)
                     if per_week else None,
                     "per_week": per_week})
    out["prompt_conditional_routing_test"] = rows
    out["prompt_routing_summary"] = {
        r["feature"]: {"weeks_tested": r["weeks_tested"],
                       "weeks_p_below_0.01": r["weeks_p_below_0.01"],
                       "median_cramers_v": r["median_cramers_v"],
                       "max_cramers_v": round(max((x["cramers_v"] for x in r["per_week"]),
                                                  default=float("nan")), 4)}
        for r in rows}
    out["prompt_routing_reading"] = (
        "Cramer's V well under 0.05 with few significant weeks is consistent with no "
        "prompt-conditional routing. Large or consistent V means the pair depends on the "
        "prompt and those features belong in S'_t.")

# ------------------------------------------------------------------ check 1
def check1(df, out):
    sizes = df.groupby("evaluation_session_id").size()
    out["sessions"] = {
        "n_votes": int(len(df)),
        "n_sessions": int(len(sizes)),
        "votes_per_session_hist": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "sessions_with_ge2": int((sizes >= 2).sum()),
        "share_with_ge2": round(float((sizes >= 2).mean()), 4),
        "mean_votes_per_session": round(float(sizes.mean()), 4),
        "dropped_duplicate_session_order_rows": int(df.attrs.get("n_dropped_dup_order", 0)),
    }
    # --- session integrity: the release is NOT a census of votes ---
    # `sizes >= 2` is not continuation: a session's vote 2 can be absent from the release
    # while a later vote survives, which proves the vote happened and was filtered out.
    # The card says the release is the text-only category, so a mid-session multimodal
    # vote is dropped. That censors C downward.
    orders = df.groupby("evaluation_session_id").evaluation_order.apply(set)
    mins = orders.map(min)
    starts_at_1 = orders[mins == 1]
    n1 = len(starts_at_1)
    has2 = int(sum(1 for v in starts_at_1 if 2 in v))
    cens2 = int(sum(1 for v in starts_at_1 if 2 not in v and max(v) > 2))
    slots = int(sum(max(v) - min(v) + 1 for v in orders))
    observed = int(sum(len(v) for v in orders))
    out["session_integrity"] = {
        "sessions_total": int(len(orders)),
        "sessions_missing_vote_1": int((mins > 1).sum()),
        "share_missing_vote_1": round(float((mins > 1).mean()), 4),
        "sessions_with_order_gaps": int(sum(
            1 for v in orders if sorted(v) != list(range(min(v), min(v) + len(v))))),
        "interior_slots_implied": slots,
        "interior_slots_observed": observed,
        "interior_slots_missing": slots - observed,
        "interior_missingness_rate": round((slots - observed) / slots, 5),
        "among_sessions_with_vote_1": {
            "n": n1,
            "vote_2_observed": has2,
            "vote_2_absent_but_later_vote_seen": cens2,
            "naive_P_C1": round(has2 / n1, 5),
            "lower_bound_P_C1": round((has2 + cens2) / n1, 5),
            "min_relative_undercount": round(cens2 / has2, 4),
        },
        "reading": ("C is right-censored by the release filter, not only by users "
                    "stopping. The lower bound counts only provable censorings (a later "
                    "vote survived), so the true undercount is larger. Under the "
                    "absorbing-state estimand a censored continuation is scored R_2 = 0, "
                    "so Phi is biased DOWNWARD, and differentially if censoring "
                    "correlates with the pair category. This is a real limitation of the "
                    "public release and must be stated, not smoothed over."),
    }

# ------------------------------------------------------------------ check 2
def check2(df, out):
    """Is the stage-2 pair independent of the stage-1 outcome? Dependence is fatal."""
    s1 = df[df.evaluation_order == 1].set_index("evaluation_session_id", verify_integrity=True)
    s2 = df[df.evaluation_order == 2].set_index("evaluation_session_id", verify_integrity=True)
    j = s1[["winner", "pair", "week", "R"]].join(
        s2[["pair"]], how="inner", rsuffix="_2")
    out["stage2_independence"] = {"n_paired": int(len(j))}
    if len(j) < 200:
        out["stage2_independence"]["note"] = "too few paired sessions to test"
        return j
    tab = pd.crosstab(j["winner"], j["pair_2"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    out["stage2_independence"].update({
        "table_stage1winner_x_stage2pair": tab.to_dict(),
        "chi2": round(float(chi2), 3), "dof": int(dof), "p": float(p),
        "cramers_v": round(float(np.sqrt(chi2 / (len(j) * (min(tab.shape) - 1)))), 4),
        "reading": ("Small Cramer's V and a large p support the identification story: the "
                    "stage-2 pair was re-drawn without reference to the stage-1 outcome. "
                    "A clear dependence is fatal."),
    })
    # same test conditioning on week, since both sides drift with the roster
    per_week = []
    for wk, g in j.groupby("week"):
        if len(g) < 300:
            continue
        t = pd.crosstab(g["winner"], g["pair_2"])
        if min(t.shape) < 2:
            continue
        c2, pp, _, _ = stats.chi2_contingency(t)
        per_week.append({"week": wk, "n": int(len(g)), "p": float(pp),
                         "cramers_v": round(float(np.sqrt(c2 / (len(g) * (min(t.shape) - 1)))), 4)})
    out["stage2_independence"]["per_week"] = per_week
    return j

# ------------------------------------------------------------------ check 3
def check3(df, out):
    """Characterise continuation. Dependence is expected and is NOT a problem under the
    absorbing-state estimand - it is part of the transition kernel. Measure it; never
    condition on it."""
    has2 = set(df.loc[df.evaluation_order == 2, "evaluation_session_id"])
    maxo = df.groupby("evaluation_session_id").evaluation_order.max()
    s1 = df[df.evaluation_order == 1].copy()
    # C as observed, and C_lb which also counts provably-censored continuations.
    s1["C"] = s1.evaluation_session_id.isin(has2).astype(int)
    s1["C_lb"] = (s1["C"] | (s1.evaluation_session_id.map(maxo) > 2)).astype(int)
    out["continuation"] = {
        "overall_P_C1": round(float(s1.C.mean()), 4),
        "by_stage1_winner": {k: {"n": int(v["C"].size), "P_C1": round(float(v["C"].mean()), 4)}
                             for k, v in s1.groupby("winner")},
        "by_stage1_pair": {k: {"n": int(v["C"].size), "P_C1": round(float(v["C"].mean()), 4)}
                           for k, v in s1.groupby("pair")},
        "by_stage1_reward": {int(k): {"n": int(v["C"].size), "P_C1": round(float(v["C"].mean()), 4)}
                             for k, v in s1.groupby("R")},
        "by_stage1_pair_censoring_corrected": {
            k: {"n": int(v["C"].size), "P_C1": round(float(v["C"].mean()), 4),
                "P_C1_lower_bound": round(float(v["C_lb"].mean()), 4)}
            for k, v in s1.groupby("pair")},
        "censoring_caveat": ("C here is observed continuation. See session_integrity: the "
                             "release drops votes, so every P(C=1) below is a lower bound "
                             "on true continuation."),
        "reading": ("If P(C=1) depends on the stage-1 reward, revealing R_1 is informative "
                    "about whether stage 2 exists at all, which is exactly what makes the "
                    "forward-monotone reveal structure do real work here."),
    }

# ------------------------------------------------------------------ check 4
def check4(df, j, out):
    """Estimated propensities by week, and overlap under prespecified target policies."""
    pi_b = (df.groupby(["week", "pair"]).size() /
            df.groupby("week").size()).rename("p").reset_index()
    piv = pi_b.pivot(index="week", columns="pair", values="p").fillna(0.0)
    out["estimated_pi_b_by_week"] = {
        "weeks": int(len(piv)),
        "min_category_probability": round(float(piv.replace(0, np.nan).min().min()), 5),
        "mean": {c: round(float(piv[c].mean()), 4) for c in piv.columns},
        "range": {c: [round(float(piv[c].min()), 4), round(float(piv[c].max()), 4)]
                  for c in piv.columns},
    }
    cats = list(piv.columns)
    targets = {
        "uniform_over_categories": {c: 1.0 / len(cats) for c in cats},
        "tilt_to_open_1.5x": None,   # filled below
    }
    base = piv.mean().to_dict()
    tilt = {c: base[c] * (1.5 if c == "open-open" else 1.0) for c in cats}
    z = sum(tilt.values()); targets["tilt_to_open_1.5x"] = {c: v / z for c, v in tilt.items()}

    res = {}
    for name, pe in targets.items():
        pb = piv.stack().rename("pb").reset_index()          # week, pair, pb
        m = df[["week", "pair"]].merge(pb, on=["week", "pair"], how="left")
        w1 = m["pair"].map(pe).to_numpy() / np.clip(m["pb"].to_numpy(), 1e-9, None)
        res[name] = {
            "step1_weight": {"mean": round(float(w1.mean()), 4),
                             "p99": round(float(np.percentile(w1, 99)), 4),
                             "max": round(float(w1.max()), 4)},
            "ESS_step1": round(float(w1.sum() ** 2 / (w1 ** 2).sum()), 1),
            "ESS_fraction_step1": round(float(w1.sum() ** 2 / (w1 ** 2).sum() / len(w1)), 4),
        }
    out["overlap"] = res
    out["overlap_reading"] = ("Two-step products square the step-1 behaviour. An ESS "
                              "fraction below roughly 0.1 at one step means no target "
                              "policy worth evaluating will survive two.")

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: screen_arena.py <arena140k.parquet>")
    df = load(sys.argv[1])
    out = {"n_rows": int(len(df))}
    check0(df, out)
    check1(df, out)
    j = check2(df, out)
    check3(df, out)
    check4(df, j, out)
    json.dump(out, open("arena_screen_results.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("prompt_conditional_routing_test",)}, indent=1, default=str))

if __name__ == "__main__":
    main()
