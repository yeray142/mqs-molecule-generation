"""Wasserstein (Earth-Mover) distance between property distributions.

Paper §2.3: "the last four metrics are reported ... as the Wasserstein
distance (denoted W) between the test and generated dataset". Used here as
W(LogP), W(SA), W(QED), W(Weight) between any two samples' property values
(e.g. train vs test for the Table 7 gate; test vs generated for Table 3
onwards).
"""

from __future__ import annotations

from scipy.stats import wasserstein_distance


def property_wasserstein_distance(
    reference_values: list[float], sample_values: list[float]
) -> float:
    """1-D Wasserstein distance between two samples of the same property.

    A thin, explicitly-named wrapper around ``scipy.stats.wasserstein_distance``
    -- kept as a separate function (rather than calling scipy directly at each
    call site) so the metric has one place to document the paper's usage and
    one place to change if a weighted variant is ever needed.

    Args:
        reference_values: Property values from the reference sample (e.g. test set).
        sample_values: Property values from the sample being scored (e.g. train,
            or a generative model's output).

    Returns:
        The Wasserstein-1 distance. Always >= 0; 0.0 for two empty inputs by
        convention (scipy raises on empty input, which is unhelpful for a
        metrics report that should degrade gracefully on a bad batch).
    """
    if not reference_values or not sample_values:
        return 0.0
    return float(wasserstein_distance(reference_values, sample_values))