"""Tests for Z0 significance (paper §2.3, Eq. 5)."""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from mqs_molecule_generation.metrics.significance import (
    SIGNIFICANCE_METRICS,
    MetricStat,
    average_significance,
    per_metric_significance,
)


class TestPerMetricSignificanceExactArithmetic:
    """Clean, hand-picked numbers with no rounding ambiguity -- must be exact."""

    def test_maximize_positive_improvement(self) -> None:
        # Delta = 0.6 - 0.5 = 0.1; sigma_comb = sqrt(0.1^2 + 0.1^2) = sqrt(0.02)
        ref = MetricStat(mean=0.5, std=0.1)
        sample = MetricStat(mean=0.6, std=0.1)
        expected = 0.1 / math.sqrt(0.02)
        assert per_metric_significance(ref, sample, "max") == pytest.approx(expected, rel=1e-12)

    def test_maximize_degradation_is_negative(self) -> None:
        ref = MetricStat(mean=0.6, std=0.1)
        sample = MetricStat(mean=0.5, std=0.1)
        result = per_metric_significance(ref, sample, "max")
        assert result < 0
        assert result == pytest.approx(-0.1 / math.sqrt(0.02), rel=1e-12)

    def test_minimize_flips_delta_sign(self) -> None:
        # For a minimized metric (e.g. SA, Weight), a LOWER sample mean is an
        # IMPROVEMENT and must give a POSITIVE Z0^m.
        ref = MetricStat(mean=10.0, std=1.0)
        sample = MetricStat(mean=8.0, std=1.0)  # sample is lower = better
        expected = (10.0 - 8.0) / math.sqrt(2.0)
        result = per_metric_significance(ref, sample, "min")
        assert result == pytest.approx(expected, rel=1e-12)
        assert result > 0

    def test_minimize_worse_is_negative(self) -> None:
        ref = MetricStat(mean=10.0, std=1.0)
        sample = MetricStat(mean=12.0, std=1.0)  # sample is higher = worse
        result = per_metric_significance(ref, sample, "min")
        assert result < 0

    def test_identical_scenarios_give_zero(self) -> None:
        stat = MetricStat(mean=0.5, std=0.05)
        assert per_metric_significance(stat, stat, "max") == 0.0
        assert per_metric_significance(stat, stat, "min") == 0.0

    def test_symmetric_swap_negates_result(self) -> None:
        # Swapping which scenario is "reference" must exactly negate Z0^m.
        a = MetricStat(mean=0.3, std=0.02)
        b = MetricStat(mean=0.4, std=0.03)
        assert per_metric_significance(a, b, "max") == pytest.approx(
            -per_metric_significance(b, a, "max"), rel=1e-12
        )

    def test_zero_combined_variance_positive_delta_is_inf(self) -> None:
        ref = MetricStat(mean=0.5, std=0.0)
        sample = MetricStat(mean=0.6, std=0.0)
        assert per_metric_significance(ref, sample, "max") == float("inf")

    def test_zero_combined_variance_negative_delta_is_neg_inf(self) -> None:
        ref = MetricStat(mean=0.6, std=0.0)
        sample = MetricStat(mean=0.5, std=0.0)
        assert per_metric_significance(ref, sample, "max") == float("-inf")

    def test_zero_combined_variance_no_delta_is_zero(self) -> None:
        stat = MetricStat(mean=0.5, std=0.0)
        assert per_metric_significance(stat, stat, "max") == 0.0


class TestMetricStatFromSamples:
    def test_matches_hand_computed_sample_std(self) -> None:
        # values = [1, 2, 3, 4, 5]; mean=3; sample variance (ddof=1) = 2.5
        stat = MetricStat.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stat.mean == pytest.approx(3.0)
        assert stat.std == pytest.approx(math.sqrt(2.5))

    def test_requires_at_least_two_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            MetricStat.from_samples([1.0])


class TestAverageSignificance:
    def test_two_metrics_hand_computed(self) -> None:
        reference = {
            "a": MetricStat(mean=0.5, std=0.1),
            "b": MetricStat(mean=10.0, std=1.0),
        }
        sample = {
            "a": MetricStat(mean=0.6, std=0.1),  # max: (0.6-0.5)/sqrt(0.02)
            "b": MetricStat(mean=8.0, std=1.0),  # min: (10-8)/sqrt(2)
        }
        metrics: dict[str, str] = {"a": "max", "b": "min"}
        per_metric, aggregate = average_significance(reference, sample, metrics)  # type: ignore[arg-type]
        expected_a = 0.1 / math.sqrt(0.02)
        expected_b = 2.0 / math.sqrt(2.0)
        assert per_metric["a"] == pytest.approx(expected_a, rel=1e-12)
        assert per_metric["b"] == pytest.approx(expected_b, rel=1e-12)
        assert aggregate == pytest.approx((expected_a + expected_b) / 2, rel=1e-12)

    def test_identical_scenarios_give_zero_aggregate(self) -> None:
        stats = {name: MetricStat(mean=1.0, std=0.1) for name in SIGNIFICANCE_METRICS}
        _, aggregate = average_significance(stats, stats)
        assert aggregate == 0.0

    def test_missing_reference_metric_raises(self) -> None:
        with pytest.raises(KeyError, match="reference"):
            average_significance({}, {"eps_d": MetricStat(1, 0.1)}, {"eps_d": "max"})

    def test_missing_sample_metric_raises(self) -> None:
        with pytest.raises(KeyError, match="sample"):
            average_significance({"eps_d": MetricStat(1, 0.1)}, {}, {"eps_d": "max"})


class TestAgainstPaperTable14:
    """Reconstructs Table 14 from the paper's published mean+-std values.

    Individual Z0^m values carry real reconstruction noise from the table's
    2-3 decimal rounding (confirmed by hand: up to ~0.8 absolute error when a
    std displays as "0.000", since that hides anything from 0 to ~0.0005) --
    so per-metric checks use a loose tolerance. The aggregate <Z0> is the
    real gate: reconstruction error was empirically ~0.01-0.06 absolute.
    """

    # name -> (mu1, sigma1, mu2, sigma2, direction, Z0_paper)
    SIMPLE_QGAN_ROWS: ClassVar[dict[str, tuple[float, float, float, float, str, float]]] = {
        "eps_d": (0.481, 0.031, 0.529, 0.003, "max", 1.56),
        "eps_v": (0.917, 0.012, 0.831, 0.002, "max", -7.21),
        "eps_u": (0.986, 0.003, 0.994, 0.000, "max", 3.50),
        "novelty": (0.536, 0.015, 0.572, 0.003, "max", 2.40),
        "intdiv": (0.898, 0.004, 0.882, 0.000, "max", -4.60),
        "filters": (0.721, 0.011, 0.737, 0.003, "max", 1.43),
        "eps_logp": (0.898, 0.007, 0.883, 0.001, "max", -2.35),
        "sa": (2.391, 0.080, 2.575, 0.007, "min", -2.31),
        "qed": (0.599, 0.007, 0.641, 0.001, "max", 6.17),
        "weight": (203.9, 10.6, 294.8, 0.8, "min", -8.56),
    }
    BEL_QGAN_ROWS: ClassVar[dict[str, tuple[float, float, float, float, str, float]]] = {
        "eps_d": (0.481, 0.031, 0.523, 0.003, "max", 1.36),
        "eps_v": (0.917, 0.012, 0.875, 0.003, "max", -3.43),
        "eps_u": (0.986, 0.003, 0.993, 0.001, "max", 2.90),
        "novelty": (0.536, 0.015, 0.548, 0.003, "max", 0.82),
        "intdiv": (0.898, 0.004, 0.883, 0.000, "max", -4.21),
        "filters": (0.721, 0.011, 0.719, 0.002, "max", -0.17),
        "eps_logp": (0.898, 0.007, 0.896, 0.002, "max", -0.23),
        "sa": (2.391, 0.080, 2.448, 0.007, "min", -0.71),
        "qed": (0.599, 0.007, 0.643, 0.001, "max", 6.55),
        "weight": (203.9, 10.6, 261.4, 0.8, "min", -5.42),
    }

    @staticmethod
    def _split(rows: dict[str, tuple[float, float, float, float, str, float]]):
        reference = {k: MetricStat(v[0], v[1]) for k, v in rows.items()}
        sample = {k: MetricStat(v[2], v[3]) for k, v in rows.items()}
        directions = {k: v[4] for k, v in rows.items()}
        paper_z0 = {k: v[5] for k, v in rows.items()}
        return reference, sample, directions, paper_z0

    def test_simple_qgan_per_metric_within_loose_tolerance(self) -> None:
        reference, sample, directions, paper_z0 = self._split(self.SIMPLE_QGAN_ROWS)
        per_metric, _ = average_significance(reference, sample, directions)  # type: ignore[arg-type]
        for name, paper_val in paper_z0.items():
            assert per_metric[name] == pytest.approx(paper_val, abs=1.0), name

    def test_simple_qgan_aggregate_matches_paper(self) -> None:
        reference, sample, directions, _ = self._split(self.SIMPLE_QGAN_ROWS)
        _, aggregate = average_significance(reference, sample, directions)  # type: ignore[arg-type]
        assert aggregate == pytest.approx(-1.00, abs=0.1)

    def test_bel_qgan_per_metric_within_loose_tolerance(self) -> None:
        reference, sample, directions, paper_z0 = self._split(self.BEL_QGAN_ROWS)
        per_metric, _ = average_significance(reference, sample, directions)  # type: ignore[arg-type]
        for name, paper_val in paper_z0.items():
            assert per_metric[name] == pytest.approx(paper_val, abs=1.0), name

    def test_bel_qgan_aggregate_matches_paper(self) -> None:
        reference, sample, directions, _ = self._split(self.BEL_QGAN_ROWS)
        _, aggregate = average_significance(reference, sample, directions)  # type: ignore[arg-type]
        assert aggregate == pytest.approx(-0.26, abs=0.1)

    def test_row_keys_match_significance_metrics(self) -> None:
        # The 10 metrics hardcoded in this test fixture must be exactly the
        # module's canonical set -- if someone edits one without the other,
        # this fails loudly instead of silently testing a different set.
        assert set(self.SIMPLE_QGAN_ROWS) == set(SIGNIFICANCE_METRICS)
        assert set(self.BEL_QGAN_ROWS) == set(SIGNIFICANCE_METRICS)