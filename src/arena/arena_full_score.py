"""Arena real-dynamics arm: full-data T=2 score and feature table (ARENA_FREEZE.md §6-§7).

Adapter to the existing harness. `ArenaFullScore` has the same interface as
`ncte_silver_score.FullScore` (fit / score / value / policy_weight) so the runner can
call it where the NCTE runner calls FullScore. Every nuisance is TABULAR on
(week, R_1, Z_1, pair) and fit on TRAIN rows only:

    Q_2(week, R_1, pair_2) = train mean R_2 on Z_1 = 1;              Q_2(dagger) = 0
    V_2(S'_2)              = sum_a pi_e(a) Q_2(S'_2, a);             V_2(dagger) = 0
    Q_1(week, pair_1)      = train mean of R_1 + gamma V_2(S'_2)
    V_1(week)              = sum_a pi_e(a) Q_1(week, a)
    mu_1                   = pi_e(pair_1) / pi_b(pair_1 | week)
    mu_2                   = [P_pie(R_1,Z_1 | week) / P_pib(R_1,Z_1 | week)] * pi_e(pair_2)/pi_b(pair_2 | week)
    Gamma_2 = V_1 + mu_1 (R_1 + gamma V_2 - Q_1) + gamma mu_2 (R_2 - Q_2)

Sparse-cell rule (prespecified): a (week, ...) cell with fewer than MIN_CELL train
sessions falls back to the same cell pooled over weeks. pi_b comes from the frozen
manifest summary (train fit), never re-estimated here.

Feature table for the projections m_0 / m_1 (HGB, decision 9b): META (pilot_nuisance.
proxy_features) + RM (when_rm_lands.rm_feats) at stage 1 and, where Z_1 = 1, at stage 2;
stage-2 columns are NaN when Z_1 = 0 -- HGB handles NaN natively, so no imputation.
"""
import os, sys, json
from collections import defaultdict
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena_paths import DATA  # noqa: E402
import screen_arena as S
from pilot_nuisance import proxy_features
from when_rm_lands import rm_feats

CATS = ("closed-closed", "mixed", "open-open")
MIN_CELL = 20
GAMMA = 1.0


def _mean_table(keys, values, min_cell, weights=None):
    """dict key -> (weighted) mean, with a pooled fallback keyed on key[1:] when the cell is
    thin. min_cell counts ROWS, not weight, so the sparse-cell rule is unchanged by weighting."""
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    full, pooled = defaultdict(list), defaultdict(list)
    for k, v, w in zip(keys, values, weights):
        full[k].append((v, w)); pooled[k[1:]].append((v, w))
    wm = lambda vw: float(sum(v * w for v, w in vw) / max(sum(w for _, w in vw), 1e-12))
    return ({k: wm(vw) for k, vw in full.items() if len(vw) >= min_cell},
            {k: wm(vw) for k, vw in pooled.items()})


def _lookup(full, pooled, key, default):
    if key in full:
        return full[key]
    return pooled.get(key[1:], default)


class ArenaFullScore:
    def __init__(self, pi_e, pi_b_by_week, gamma=GAMMA, min_cell=MIN_CELL):
        """pi_e is either {category: p} (state-free, both stages) or a SEQUENTIAL policy
        {"stage1": {category: p}, "stage2": {0: {category: p}, 1: {category: p}}} whose stage-2
        serving depends on the revealed R_1. In the latter case mu_2 depends on R_1 through
        pi_e(A_2 | R_1) as well as through the occupancy ratio."""
        if "stage1" in pi_e:
            self.pi_e1 = dict(pi_e["stage1"]); self.pi_e2 = {int(k): dict(v) for k, v in pi_e["stage2"].items()}
        else:
            self.pi_e1 = dict(pi_e); self.pi_e2 = {0: dict(pi_e), 1: dict(pi_e)}
        for d in (self.pi_e1, self.pi_e2[0], self.pi_e2[1]):
            if set(d) != set(CATS) or abs(sum(d.values()) - 1) > 1e-9:
                raise ValueError("each pi_e component must be a distribution over the three pair categories.")
        self.pi_e = self.pi_e1                       # stage-1 policy; kept for back-compatibility
        self.sequential = any(self.pi_e2[0][c] != self.pi_e2[1][c] for c in CATS)
        self.pi_b, self.gamma, self.min_cell = pi_b_by_week, gamma, min_cell

    # ---------------------------------------------------------------- fit (TRAIN only)
    def fit(self, rows, references, require_train=True, w1=None, w2=None):
        """rows/references aligned. require_train=False lets the charged protocol fit on pilot
        or annotated rows of any fold. w1 / w2: optional IPW weights per row for the stage-1
        (R_1, occupancy) and stage-2 (R_2) tables — 1/Lambda_1 and 1/Lambda_{1:2} on annotated
        rows — as the casenote driver's IPW-weighted post-reveal refits."""
        if not rows or (require_train and any(r["fold"] != "train" for r in rows)):
            raise ValueError("Fit accepts training sessions only (pass require_train=False to override).")
        if [r["episode_id"] for r in rows] != [s["episode_id"] for s in references]:
            raise ValueError("Reference alignment failed.")
        self.fit_ids = [r["episode_id"] for r in rows]
        r1 = np.array([s["reward1"] for s in references], int)
        r2 = np.array([s["reward2"] for s in references], int)
        z1 = np.array([r["Z1"] for r in rows], int)
        w1 = np.ones(len(rows)) if w1 is None else np.asarray(w1, float)
        w2 = np.ones(len(rows)) if w2 is None else np.asarray(w2, float)

        # Q_2 on continuing sessions
        cont = [(r, a, b, w) for r, a, b, w in zip(rows, r1, r2, w2) if r["Z1"]]
        self.q2_full, self.q2_pool = _mean_table(
            [(r["week"], int(a), r["actions"][1]) for r, a, _, _ in cont], [b for _, _, b, _ in cont], self.min_cell,
            weights=[w for *_, w in cont])
        self.q2_default = float(np.average([b for _, _, b, _ in cont], weights=[w for *_, w in cont])) if cont else 0.5

        # occupancy: P(R_1, Z_1 | week, a) under the behaviour data
        self.occ_full, self.occ_pool = {}, {}
        cnt = defaultdict(lambda: defaultdict(float)); cnt_pool = defaultdict(lambda: defaultdict(float))
        rows_in = defaultdict(int)
        for r, a, z, w in zip(rows, r1, z1, w1):
            cnt[(r["week"], r["actions"][0])][(int(a), int(z))] += w; rows_in[(r["week"], r["actions"][0])] += 1
            cnt_pool[r["actions"][0]][(int(a), int(z))] += w
        for k, d in cnt.items():
            n = sum(d.values())
            if rows_in[k] >= self.min_cell and n > 0:
                self.occ_full[k] = {o: c / n for o, c in d.items()}
        for k, d in cnt_pool.items():
            n = sum(d.values()); self.occ_pool[k] = {o: c / n for o, c in d.items()}

        # Q_1 = E[R_1 + gamma V_2(S'_2) | week, pair_1]
        tgt = [a + self.gamma * self.value2(r["week"], int(a), int(z)) for r, a, z in zip(rows, r1, z1)]
        self.q1_full, self.q1_pool = _mean_table(
            [(r["week"], r["actions"][0]) for r in rows], tgt, self.min_cell, weights=w1)
        self.q1_default = float(np.average(tgt, weights=w1))
        return self

    # ---------------------------------------------------------------- vectorised evaluation
    def arrays(self, rows):
        """Nuisances for a list of rows as arrays: mu1, V1, Q1, z1 and, for each R_1 branch r,
        V2_r, Q2_r, mu2_r, h_r (h(r) = mu1 (r + gamma V2(r)) - gamma mu2(r) Q2(r), the manuscript's
        sigma_1^2 kernel). Table lookups are cached per unique key."""
        week = [r["week"] for r in rows]; a1 = [r["actions"][0] for r in rows]
        z1 = np.array([r["Z1"] for r in rows], bool); a2 = [r["actions"][1] if r["Z1"] else None for r in rows]
        cV1 = {w: self.value1(w) for w in set(week)}
        cQ1 = {(w, a): self.q1(w, a) for w, a in set(zip(week, a1))}
        cpb = {(w, a): self.pib(w, a) for w, a in set(zip(week, a1)) | {(w, a) for w, a in zip(week, a2) if a}}
        mu1 = np.array([self.pi_e1[a] / cpb[(w, a)] for w, a in zip(week, a1)])
        out = dict(mu1=mu1, V1=np.array([cV1[w] for w in week]), Q1=np.array([cQ1[(w, a)] for w, a in zip(week, a1)]), z1=z1)
        for r in (0, 1):
            cV2 = {(w, z): self.value2(w, r, z) for w, z in set(zip(week, z1))}
            cQ2 = {(w, a): self.q2(w, r, a) for w, a in set(zip(week, a2)) if a}
            cocc = {w: self.occupancy_ratio(w, r, 1) for w in set(week)}
            V2 = np.array([cV2[(w, z)] for w, z in zip(week, z1)])
            Q2 = np.array([cQ2[(w, a)] if a else 0. for w, a in zip(week, a2)])
            mu2 = np.array([cocc[w] * self.pi_e2[r][a] / cpb[(w, a)] if a else 0. for w, a in zip(week, a2)])
            out[f"V2_{r}"], out[f"Q2_{r}"], out[f"mu2_{r}"] = V2, Q2, mu2
            out[f"h_{r}"] = mu1 * (r + self.gamma * V2) - self.gamma * mu2 * Q2
        return out

    @staticmethod
    def scores_from_arrays(A, b1, b2_0, b2_1, gamma=GAMMA):
        """Closed-form projections from the nuisance arrays and reward classifiers:
        Gamma(r1, r2), m1(r1) = E[Gamma | F_1], m0 = E[Gamma | F_0], and the structural sigma^2.
        Returns dict with Gamma[r1][r2] (n,), m1[r1] (n,), m0 (n,), sig1 (n,), sig2[r1] (n,)."""
        V1, mu1, Q1 = A["V1"], A["mu1"], A["Q1"]
        G, m1, s2 = {}, {}, {}
        for r1, b2 in ((0, b2_0), (1, b2_1)):
            base = V1 + mu1 * (r1 + gamma * A[f"V2_{r1}"] - Q1)
            mu2, Q2 = A[f"mu2_{r1}"], A[f"Q2_{r1}"]
            G[r1] = {r2: base + gamma * mu2 * (r2 - Q2) for r2 in (0, 1)}
            m1[r1] = base + gamma * mu2 * (b2 - Q2)           # = Gamma exactly when mu2 == 0 (Z1 = 0)
            s2[r1] = (gamma * mu2) ** 2 * b2 * (1 - b2)
        m0 = (1 - b1) * m1[0] + b1 * m1[1]
        sig1 = b1 * (1 - b1) * (A["h_1"] - A["h_0"]) ** 2
        return dict(Gamma=G, m1=m1, m0=m0, sig1=sig1, sig2=s2, b1=b1, b2=(b2_0, b2_1))

    # ---------------------------------------------------------------- nuisances
    def pib(self, week, a):
        return self.pi_b[week][a]

    def q2(self, week, r1, a2):
        return _lookup(self.q2_full, self.q2_pool, (week, int(r1), a2), self.q2_default)

    def value2(self, week, r1, z1):
        if not z1:
            return 0.0
        return float(sum(self.pi_e2[int(r1)][a] * self.q2(week, r1, a) for a in CATS))

    def q1(self, week, a1):
        return _lookup(self.q1_full, self.q1_pool, (week, a1), self.q1_default)

    def value1(self, week):
        return float(sum(self.pi_e1[a] * self.q1(week, a) for a in CATS))

    def p_outcome(self, week, a, r1, z1):
        d = self.occ_full.get((week, a)) or self.occ_pool[a]
        return d.get((int(r1), int(z1)), 0.0)

    def occupancy_ratio(self, week, r1, z1):
        num = sum(self.pi_e1[a] * self.p_outcome(week, a, r1, z1) for a in CATS)
        den = sum(self.pib(week, a) * self.p_outcome(week, a, r1, z1) for a in CATS)
        return num / den if den > 0 else 0.0

    def mu1(self, row):
        return self.pi_e1[row["actions"][0]] / self.pib(row["week"], row["actions"][0])

    def mu2(self, row, r1):
        if not row["Z1"]:
            return 0.0
        a2 = row["actions"][1]
        return self.occupancy_ratio(row["week"], r1, 1) * self.pi_e2[int(r1)][a2] / self.pib(row["week"], a2)

    # ---------------------------------------------------------------- FullScore interface
    def policy_weight(self, row, stage, reward1=None):
        return self.mu1(row) if stage == 1 else self.mu2(row, reward1)

    def value(self, row, stage, reward1=None):
        return self.value1(row["week"]) if stage == 1 else self.value2(row["week"], reward1, row["Z1"])

    def score(self, row, reward1, reward2):
        """Gamma_2 with both rewards revealed. For Z_1 = 0 the stage-2 term is identically 0
        and reward2 must be 0 by absorption."""
        if reward1 not in (0, 1) or reward2 not in (0, 1):
            raise ValueError("Full score requires binary revealed rewards.")
        if not row["Z1"] and reward2 != 0:
            raise ValueError("Absorbing session cannot carry a nonzero stage-2 reward.")
        w, a1 = row["week"], row["actions"][0]
        g = self.value1(w) + self.mu1(row) * (reward1 + self.gamma * self.value2(w, reward1, row["Z1"]) - self.q1(w, a1))
        if row["Z1"]:
            g += self.gamma * self.mu2(row, reward1) * (reward2 - self.q2(w, reward1, row["actions"][1]))
        return float(g)

    def stage1_exact_if_absorbing(self, row, reward1):
        """m_1 := Gamma_2 exactly when Z_1 = 0 (Gamma_2 is F_1-measurable there); else None."""
        return None if row["Z1"] else self.score(row, reward1, 0)


# -------------------------------------------------------------------- feature table
def build_feature_table(manifest_rows, parquet=os.path.join(DATA, "arena140k.parquet"),
                        rm_scores=os.path.join(DATA, "arena_rm_scores.parquet"), tfidf_scores=None):
    """One row per session, indexed by episode_id. Stage-2 columns NaN when Z_1 = 0.
    tfidf_scores: optional parquet [id, tfidf_p] from build_arena_tfidf_scores.py (post-hoc proxy arm)."""
    df = S.load(parquet).set_index("id")
    sc = pd.read_parquet(rm_scores).set_index("id")[["score_a", "score_b"]]
    votes = df.join(sc, how="left")
    meta = proxy_features(votes)
    rm = rm_feats(votes)
    parts = [meta, rm]
    if tfidf_scores is not None:
        tf = pd.read_parquet(tfidf_scores).set_index("id")[["tfidf_p"]].reindex(votes.index)
        parts.append(tf)
    per_vote = pd.concat(parts, axis=1)
    per_vote.columns = [str(c) for c in per_vote.columns]

    ids1 = pd.Index([r["id1"] for r in manifest_rows])
    ids2 = pd.Index([r["id2"] if r["Z1"] else None for r in manifest_rows])
    f1 = per_vote.reindex(ids1); f1.columns = [f"s1_{c}" for c in f1.columns]
    f2 = per_vote.reindex(ids2); f2.columns = [f"s2_{c}" for c in f2.columns]
    f1.index = f2.index = [r["episode_id"] for r in manifest_rows]

    weeks = sorted({r["week"] for r in manifest_rows})
    tab = pd.DataFrame(index=f1.index)
    tab["Z1"] = [r["Z1"] for r in manifest_rows]
    for w in weeks:
        tab[f"week_{w}"] = [float(r["week"] == w) for r in manifest_rows]
    for c in CATS:
        tab[f"pair1_{c}"] = [float(r["actions"][0] == c) for r in manifest_rows]
        tab[f"pair2_{c}"] = [float(r["actions"][1] == c) if r["Z1"] else np.nan for r in manifest_rows]
    out = pd.concat([tab, f1, f2], axis=1).astype("float64")
    return out


if __name__ == "__main__":
    # self-check on the frozen manifest: fit on train, report nuisance coverage and mu means.
    man = json.load(open(os.path.join(DATA, "arena_session_manifest.json")))
    summ = json.load(open(os.path.join(DATA, "arena_manifest_summary.json")))
    menu = json.load(open(os.path.join(DATA, "arena_policy_menu.json")))
    pe = menu["menu"][menu["primary_pi_e"]]["pi_e"]
    tr = [r for r in man if r["fold"] == "train"]
    refs = [{"episode_id": r["episode_id"], "reward1": r["reward1"], "reward2": r["reward2"]} for r in tr]
    fs = ArenaFullScore(pe, summ["pi_b_by_week"]).fit(tr, refs)
    print(f"fit on {len(tr):,} train sessions | pi_e = {menu['primary_pi_e']}")
    print(f"Q2 cells kept {len(fs.q2_full)} (pooled fallbacks {len(fs.q2_pool)}) | Q1 cells kept {len(fs.q1_full)} | occupancy cells kept {len(fs.occ_full)}")
    m1 = np.array([fs.mu1(r) for r in tr]); m2 = np.array([fs.mu2(r, r["reward1"]) for r in tr if r["Z1"]])
    occ = np.array([fs.occupancy_ratio(r["week"], r["reward1"], 1) for r in tr if r["Z1"]])
    print(f"mu_1: mean {m1.mean():.4f} max {m1.max():.2f} | mu_2 (Z1=1): mean {m2.mean():.4f} max {m2.max():.2f} | occupancy: mean {occ.mean():.4f} range [{occ.min():.3f}, {occ.max():.3f}]")
    g0 = fs.score(tr[0], tr[0]["reward1"], tr[0]["reward2"]); print(f"score(row0) = {g0:.4f}  (finite: {np.isfinite(g0)})")
    ex = fs.stage1_exact_if_absorbing(next(r for r in tr if not r["Z1"]), 1); print(f"absorbing m_1 exact: {ex:.4f}")
    F = build_feature_table(man)
    print(f"feature table: {F.shape[0]:,} sessions x {F.shape[1]} cols | stage-2 NaN share {F.filter(like='s2_').isna().mean().mean():.3f} (expect ~{1-np.mean([r['Z1'] for r in man]):.3f})")
    print("cols:", list(F.columns[:6]), "...", list(F.columns[-4:]))
