"""Two-stage annotation correction with explicitly revealed score projections."""
import math


def annotation_score(m0, m1, full_score, c1, c2, lambda1, lambda2):
    """Return the prefix correction without accessing unrevealed score values.

    m1 is the stage-1 joint-score projection, not a separately imputed reward.
    lambda2 is conditional on obtaining stage 1. Probabilities come from design.
    """
    if c1 not in (0, 1) or c2 not in (0, 1) or c2 > c1:
        raise ValueError('Annotation indicators must be binary and forward monotone.')
    if not 0 < lambda1 <= 1:
        raise ValueError('Stage-1 annotation probability must lie in (0, 1].')
    if c1 and (lambda2 is None or not 0 < lambda2 <= 1):
        raise ValueError('An annotated prefix requires a stage-2 probability in (0, 1].')
    if not math.isfinite(m0):
        raise ValueError('Initial score projection must be finite.')
    value = m0
    if c1:
        if m1 is None or not math.isfinite(m1):
            raise ValueError('Stage 1 requires its revealed score projection.')
        value += (m1-m0)/lambda1
    if c2:
        if full_score is None or not math.isfinite(full_score):
            raise ValueError('Stage 2 requires the full score.')
        value += (full_score-m1)/(lambda1*lambda2)
    return value
