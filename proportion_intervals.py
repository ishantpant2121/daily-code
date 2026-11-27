"""Wilson score intervals for binomial proportions.

The score interval avoids the zero-width endpoint pathology of the
normal/Wald approximation. It is an approximate frequentist interval, not
an exact interval and not a guarantee that every realized interval covers.
"""
from __future__ import annotations

import math
from numbers import Real
from statistics import NormalDist


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion.

    Args:
        successes: Number of observed successes, from zero through trials.
        trials: Positive number of independent Bernoulli trials.
        confidence_level: Central confidence level strictly between zero and one.

    Returns:
        Inclusive lower and upper bounds, each in [0, 1].

    Notes:
        This is the uncorrected Wilson score interval. It assumes a binomial
        sampling model with independent, identically distributed trials.
        It does not account for clustering, survey weights, sequential
        monitoring, or multiple comparisons.
    """
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise ValueError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise ValueError("confidence_level must be strictly between zero and one")
    confidence_level = float(confidence_level)
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")

    observed = successes / trials
    # Form the upper-tail probability directly. 0.5 + confidence / 2
    # can round to exactly 1.0 for the largest representable confidence
    # levels below one, producing an infinite quantile.
    upper_probability = 1.0 - (1.0 - confidence_level) / 2.0
    z = NormalDist().inv_cdf(upper_probability)
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (observed + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z * math.sqrt(
            observed * (1.0 - observed) / trials
            + z_squared / (4.0 * trials * trials)
        ) / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)
