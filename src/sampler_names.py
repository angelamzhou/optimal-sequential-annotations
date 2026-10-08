"""Canonical annotation-design (sampler) names for the annotation experiments.

History (renamed 2026-09-27): the experiment driver used to call the
trajectory-wise ("bundle") design ``stagewise_adaptive`` and the paper's
forward-monotone stagewise design ``prefix_adaptive``. The paper calls these
the trajectory-wise and stagewise designs, so the old name
``stagewise_adaptive`` pointed at the *wrong* design. Worse, some derived
figure directories (e.g. results/casenote_progress_qwen3ft/fig_stagewise and
results/mdp_simulator_validation/*_stagewise_fig) were hand-relabelled so that
``stagewise_adaptive`` rows hold the stagewise (prefix) design. The string
``stagewise_adaptive`` is therefore ambiguous in saved CSVs and must never be
guessed:

* adaptive_stagewise  -- forward-monotone stagewise allocation of the T=2
                         theorem (annotation may stop after stage 1).
                         Legacy name: prefix_adaptive.
* adaptive_trajectory -- trajectory-wise (bundle) design: one priority per
                         client, all stages annotated together.
                         Legacy names: stagewise_adaptive (raw driver output),
                         bundle_adaptive (fig_three).

For terminal-only targets (a single annotated stage) the two designs coincide.

* uncertainty_proportional -- active-inference baseline (Zrnic and Candes):
                         one probability per client proportional to the
                         square root of the summed stage variances, ignoring
                         the density ratios. Same variance signal, budget and
                         estimator as adaptive_trajectory.
"""
from __future__ import annotations

import warnings

import pandas as pd

ADAPTIVE_STAGEWISE = "adaptive_stagewise"
ADAPTIVE_TRAJECTORY = "adaptive_trajectory"
UNCERTAINTY_PROPORTIONAL = "uncertainty_proportional"
CANONICAL = ("random", ADAPTIVE_STAGEWISE, ADAPTIVE_TRAJECTORY, UNCERTAINTY_PROPORTIONAL)

# Names whose meaning never changed.
LEGACY_UNAMBIGUOUS = {"prefix_adaptive": ADAPTIVE_STAGEWISE, "bundle_adaptive": ADAPTIVE_TRAJECTORY}
# Driver CLI aliases: on the command line 'stagewise_adaptive' always meant the bundle design.
CLI_ALIASES = {"stagewise_adaptive": ADAPTIVE_TRAJECTORY, **LEGACY_UNAMBIGUOUS}
TWO_STAGE_TARGETS = frozenset({"progress_total"})


def canonical_cli_sampler(name: str) -> str:
    if name in CANONICAL:
        return name
    if name in CLI_ALIASES:
        warnings.warn(f"sampler {name!r} is deprecated; use {CLI_ALIASES[name]!r}", stacklevel=2)
        return CLI_ALIASES[name]
    raise ValueError(f"Unknown sampler {name!r}; choose from {', '.join(CANONICAL)}.")


def canonicalize_raw_driver_output(frame: pd.DataFrame) -> pd.DataFrame:
    """For raw (never hand-relabelled) outreach_dtr_annotation_mc.py outputs only."""
    out = frame.copy()
    out["sampler"] = out["sampler"].replace({**CLI_ALIASES})
    return out


def canonicalize_saved(frame: pd.DataFrame, legacy_stagewise_adaptive: str | None) -> pd.DataFrame:
    """Map saved sampler names to canonical names.

    ``legacy_stagewise_adaptive`` states what 'stagewise_adaptive' rows mean in this
    file: 'trajectory' (raw driver output) or 'stagewise' (hand-relabelled figure
    directories). It is only required when such rows exist for a two-stage target.
    """
    out = frame.copy()
    already_stagewise = out["sampler"].isin(["prefix_adaptive", ADAPTIVE_STAGEWISE]).any()
    out["sampler"] = out["sampler"].replace(LEGACY_UNAMBIGUOUS)
    legacy = out["sampler"].eq("stagewise_adaptive")
    if legacy.any() and legacy_stagewise_adaptive == "stagewise" and already_stagewise:
        raise ValueError(
            "This file already contains the stagewise design (prefix_adaptive / adaptive_stagewise), so its "
            "'stagewise_adaptive' rows are raw driver output (the trajectory-wise design); use "
            "--legacy-stagewise-adaptive trajectory."
        )
    if legacy.any():
        two_stage = legacy & out["target"].isin(TWO_STAGE_TARGETS) if "target" in out.columns else legacy
        if two_stage.any() and legacy_stagewise_adaptive not in {"trajectory", "stagewise"}:
            raise ValueError(
                "Rows labelled 'stagewise_adaptive' for a two-stage target are ambiguous (see sampler_names.py). "
                "Pass --legacy-stagewise-adaptive trajectory for raw driver output, or stagewise for "
                "hand-relabelled figure directories such as fig_stagewise / *_stagewise_fig."
            )
        mapped = ADAPTIVE_TRAJECTORY if legacy_stagewise_adaptive == "trajectory" else ADAPTIVE_STAGEWISE
        # Terminal-only targets: the designs coincide; report as the paper's stagewise design.
        out.loc[legacy & ~two_stage, "sampler"] = ADAPTIVE_STAGEWISE
        out.loc[two_stage, "sampler"] = mapped
    return out
