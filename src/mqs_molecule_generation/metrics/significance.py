"""Scenario-comparison significance, Z0 (paper §2.3, Eq. 5).

    Z0^m = Delta^m / sqrt((sigma_1^m)^2 + (sigma_2^m)^2)
    <Z0> = (1/N) * sum_m Z0^m

Each run is trained with 5 different random seeds; mu/sigma for a metric are
the mean/std of that metric's value ACROSS those 5 runs -- a different axis
of variation from, say, PropertyReport's per-molecule property lists (which
vary across molecules within one run, not across seeds). This module is
deliberately independent of PropertyReport: its inputs are plain (mean, std)
summary statistics per metric per scenario, which is what Eq. 5 operates on
and what a future 5-seed training sweep will produce.

Sign convention: positive Z0^m means the "sample" scenario is better than the
"reference" scenario for that metric; negative means worse. Direction
(maximize vs minimize) determines how Delta is signed -- see
SIGNIFICANCE_METRICS.

Reconstructing Z0^m from the paper's own published Table 14 (mean+-std,
rounded to 2-3 decimals) does not reproduce the table exactly: when the
combined sigma is small, rounding on mu/sigma is amplified by the division,
occasionally by 10-20% on an individual metric (confirmed empirically against
Table 14: eps_u and IntDiv, both cases with a sigma displayed as "0.000",
show the largest reconstruction error). The aggregate <Z0> is far more
robust -- errors across 10 metrics partially cancel -- which is why <Z0> is
the tight gate and individual Z0^m values are checked loosely.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Direction = Literal["max", "min"]


@dataclass(frozen=True)
class MetricStat:
    """Mean and standard deviation of one metric across repeated runs."""

    mean: float
    std: float

    @classmethod
    def from_samples(cls, values: list[float], ddof: int = 1) -> MetricStat:
        """Build from raw per-seed values (paper: 5 seeds).

        ddof=1 (sample standard deviation, Bessel's correction) is assumed to
        match the paper's "mean and standard deviation over these five runs"
        -- the standard convention for summarising a small number of
        independent replicate runs, and pandas' default. Not verified against
        the paper directly (it doesn't state ddof), since Table 14 supplies
        already-rounded mean+-std rather than raw per-seed values -- there is
        no way to check this choice from the published table alone.
        """
        if len(values) < 2:
            raise ValueError(f"Need at least 2 samples for std(ddof={ddof}), got {len(values)}")
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - ddof)
        return cls(mean=mean, std=math.sqrt(variance))


#: Paper §2.3 (scenario 3, "Classical and quantum GAN performance comparison")
#: and Eq. 5's own enumeration: the 10 metrics used for scenario-comparison
#: significance, and whether each is intended to be maximized or minimized.
#: Confirmed by reconstructing all 10 rows of Table 14 (see module docstring).
SIGNIFICANCE_METRICS: dict[str, Direction] = {
    "eps_d": "max",
    "eps_v": "max",
    "eps_u": "max",
    "novelty": "max",
    "intdiv": "max",
    "filters": "max",
    "eps_logp": "max",
    "sa": "min",
    "qed": "max",
    "weight": "min",
}


def per_metric_significance(
    reference: MetricStat, sample: MetricStat, direction: Direction
) -> float:
    """Z0^m for one metric (Eq. 5).

    Args:
        reference: The benchmark/reference scenario's (mean, std) -- e.g. the
            classical GAN, per the paper's convention.
        sample: The scenario under study's (mean, std) -- e.g. a QGAN variant.
        direction: "max" if higher values of this metric are better
            (Delta = sample.mean - reference.mean), "min" if lower is better
            (Delta = reference.mean - sample.mean). Positive Z0^m always means
            "sample is better than reference" regardless of direction.

    Returns:
        Delta / sqrt(reference.std**2 + sample.std**2). If the combined
        variance is exactly zero (both stds are zero), returns +inf, -inf, or
        0.0 matching the sign of Delta -- the mathematical limiting behaviour
        of a z-score with vanishing denominator, not a special case to hide.
    """
    delta = (sample.mean - reference.mean) if direction == "max" else (reference.mean - sample.mean)
    denom = math.sqrt(reference.std**2 + sample.std**2)
    if denom == 0.0:
        return math.copysign(float("inf"), delta) if delta != 0.0 else 0.0
    return delta / denom


def average_significance(
    reference: Mapping[str, MetricStat],
    sample: Mapping[str, MetricStat],
    metrics: Mapping[str, Direction] = SIGNIFICANCE_METRICS,
) -> tuple[dict[str, float], float]:
    """Per-metric Z0^m and the aggregate <Z0> (Eq. 5, averaged over `metrics`).

    Args:
        reference: metric name -> MetricStat for the reference scenario.
            Must contain every key in `metrics`.
        sample: metric name -> MetricStat for the scenario under study. Must
            contain every key in `metrics`.
        metrics: metric name -> direction map. Defaults to the paper's full
            10-metric set (SIGNIFICANCE_METRICS); pass a subset to reproduce
            a different comparison (e.g. Table 3's VAE-vs-GAN scenario, which
            paper §2.4.2 says uses only 7 fractions + 3 means, matching the
            same 10 by construction -- kept as an explicit parameter rather
            than hardcoded so future scenario comparisons aren't locked in).

    Returns:
        (per-metric Z0^m dict in `metrics`' order, <Z0> = mean of those values).

    Raises:
        KeyError: If `reference` or `sample` is missing a metric in `metrics`.
    """
    per_metric: dict[str, float] = {}
    for name, direction in metrics.items():
        if name not in reference:
            raise KeyError(f"reference is missing metric {name!r}")
        if name not in sample:
            raise KeyError(f"sample is missing metric {name!r}")
        per_metric[name] = per_metric_significance(reference[name], sample[name], direction)
    aggregate = sum(per_metric.values()) / len(per_metric)
    return per_metric, aggregate