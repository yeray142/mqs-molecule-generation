"""Bundles the individual metrics into one report per sample (paper Tables 2/3/7).

This module does not implement anything new; it composes
``validity``, ``diversity``, ``properties``, ``fractions``, and ``wasserstein``
into a single call so callers don't have to remember which function belongs
to which stage of Eq. 6's pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from mqs_molecule_generation.data.filters import pass_rate as filters_pass_rate
from mqs_molecule_generation.metrics.diversity import compute_diversity
from mqs_molecule_generation.metrics.fractions import (
    distinct_fraction,
    novelty_fraction,
    unique_fraction,
)
from mqs_molecule_generation.metrics.properties import (
    compute_logps_batch,
    compute_molecular_weights_batch,
    compute_qeds_batch,
    logp_in_range_fraction,
)
from mqs_molecule_generation.metrics.sa_score import compute_sa_scores_batch
from mqs_molecule_generation.metrics.validity import compute_validity, is_valid_smiles
from mqs_molecule_generation.metrics.wasserstein import property_wasserstein_distance


@dataclass(frozen=True)
class PropertyReport:
    """Everything computable from a single sample without a reference set.

    Fields map directly onto the paper's Table 2/3 rows for a given column
    (e.g. "Train set", "Tuned VAE"): validity/IntDiv/means, plus the raw
    per-molecule property lists so a caller can feed two reports' lists into
    ``wasserstein_report`` for the W(X) comparisons.
    """

    n_input: int
    n_valid: int
    validity: float
    int_div: float
    logp: list[float]
    sa: list[float]
    qed: list[float]
    weight: list[float]
    logp_fraction: float  # epsilon_LogP

    @property
    def mean_logp(self) -> float:
        return sum(self.logp) / len(self.logp) if self.logp else float("nan")

    @property
    def mean_sa(self) -> float:
        return sum(self.sa) / len(self.sa) if self.sa else float("nan")

    @property
    def mean_qed(self) -> float:
        return sum(self.qed) / len(self.qed) if self.qed else float("nan")

    @property
    def mean_weight(self) -> float:
        return sum(self.weight) / len(self.weight) if self.weight else float("nan")


@dataclass(frozen=True)
class WassersteinReport:
    """The four W(X) distances between two ``PropertyReport``s (paper §2.3)."""

    logp: float
    sa: float
    qed: float
    weight: float


def compute_property_report(smiles_list: list[str]) -> PropertyReport:
    """Compute everything in ``PropertyReport`` for a single sample.

    Args:
        smiles_list: Raw SMILES (validity is checked internally; invalid
            entries are excluded from IntDiv and every property list).
    """
    n_valid, n_input, validity = compute_validity(smiles_list)
    valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
    return PropertyReport(
        n_input=n_input,
        n_valid=n_valid,
        validity=validity,
        int_div=compute_diversity(valid_smiles),
        logp=compute_logps_batch(valid_smiles),
        sa=compute_sa_scores_batch(valid_smiles),
        qed=compute_qeds_batch(valid_smiles),
        weight=compute_molecular_weights_batch(valid_smiles),
        logp_fraction=logp_in_range_fraction(valid_smiles),
    )


def compute_wasserstein_report(
    reference: PropertyReport, sample: PropertyReport
) -> WassersteinReport:
    """The four W(X) distances between two reports (paper §2.3).

    Args:
        reference: E.g. the test set (paper's convention: W is between the
            test set and the sample being scored).
        sample: E.g. train set, VAE output, or GAN output.
    """
    return WassersteinReport(
        logp=property_wasserstein_distance(reference.logp, sample.logp),
        sa=property_wasserstein_distance(reference.sa, sample.sa),
        qed=property_wasserstein_distance(reference.qed, sample.qed),
        weight=property_wasserstein_distance(reference.weight, sample.weight),
    )


def compute_significance_metrics(
    generated_smiles: list[str], reference_smiles: list[str]
) -> dict[str, float]:
    """The 10 metrics Eq. 5's significance comparison uses (paper §2.3, Table 14).

    Keys match ``metrics.significance.SIGNIFICANCE_METRICS`` exactly, so the
    result can be wrapped directly into ``MetricStat``s (one per seed, then
    aggregated with ``MetricStat.from_samples``) and fed to
    ``average_significance``. Shared by the VAE-reference evaluation and the
    GAN evaluation scripts so both sides of the Z0 comparison run through the
    identical computation -- important for the comparison to be meaningful,
    not just convenient.

    Args:
        generated_smiles: Raw SMILES from the scenario being scored (VAE
            prior-sampled, or GAN-generated-then-VAE-decoded).
        reference_smiles: The training set, for the novelty check and the
            distinct/unique fractions' denominators. NOT a second scenario
            to compare against -- that comparison happens later, in
            average_significance, using summary stats across seeds.

    Returns:
        dict with keys eps_d, eps_v, eps_u, novelty, intdiv, filters,
        eps_logp, sa, qed, weight -- means/fractions, matching
        SIGNIFICANCE_METRICS's maximize/minimize directions.
    """
    _, _, eps_v = compute_validity(generated_smiles)
    valid_smiles = [s for s in generated_smiles if is_valid_smiles(s)]

    report = compute_property_report(generated_smiles)

    return {
        "eps_d": distinct_fraction(generated_smiles),
        "eps_v": eps_v,
        "eps_u": unique_fraction(valid_smiles),
        "novelty": novelty_fraction(valid_smiles, reference_smiles),
        "intdiv": report.int_div,
        "filters": filters_pass_rate(valid_smiles, "moses_native"),
        "eps_logp": report.logp_fraction,
        "sa": report.mean_sa,
        "qed": report.mean_qed,
        "weight": report.mean_weight,
    }