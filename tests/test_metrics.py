"""Tests for molecular metrics (paper §2.3)."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import Crippen

from mqs_molecule_generation.metrics.diversity import compute_diversity
from mqs_molecule_generation.metrics.fractions import (
    distinct_fraction,
    novelty_fraction,
    unique_fraction,
)
from mqs_molecule_generation.metrics.moses_metrics import (
    compute_property_report,
    compute_significance_metrics,
    compute_wasserstein_report,
)
from mqs_molecule_generation.metrics.properties import (
    LOGP_RANGE,
    compute_logp,
    compute_logps_batch,
    compute_molecular_weight,
    compute_qed,
    logp_in_range_fraction,
)
from mqs_molecule_generation.metrics.sa_score import compute_sa_score, compute_sa_scores_batch
from mqs_molecule_generation.metrics.validity import compute_validity, is_valid_smiles
from mqs_molecule_generation.metrics.wasserstein import property_wasserstein_distance

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class TestValidity:
    def test_valid_smiles(self) -> None:
        assert is_valid_smiles(ETHANOL)
        assert is_valid_smiles(BENZENE)

    def test_invalid_smiles(self) -> None:
        assert not is_valid_smiles("INVALID")
        assert not is_valid_smiles("")

    def test_compute_validity(self) -> None:
        valid, total, frac = compute_validity([ETHANOL, BENZENE, "INVALID", ""])
        assert (valid, total) == (2, 4)
        assert frac == pytest.approx(0.5)

    def test_compute_validity_empty(self) -> None:
        assert compute_validity([]) == (0, 0, 0.0)


class TestSAScore:
    def test_single_molecule(self) -> None:
        # Benzene is about as easy to synthesise as it gets; RDKit's own
        # scale floors at 1.0, confirmed against the installed contrib script.
        assert compute_sa_score(BENZENE) == pytest.approx(1.0, abs=0.05)

    def test_range(self) -> None:
        for smi in (ETHANOL, BENZENE, ASPIRIN):
            score = compute_sa_score(smi)
            assert 1.0 <= score <= 10.0

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            compute_sa_score("not_a_molecule")

    def test_batch_returns_list_not_dict(self) -> None:
        scores = compute_sa_scores_batch([ETHANOL, BENZENE])
        assert isinstance(scores, list)

    def test_batch_preserves_duplicates(self) -> None:
        # A dict keyed by SMILES would silently collapse this to one entry;
        # a generated sample's duplicates should count toward the mean.
        scores = compute_sa_scores_batch([ETHANOL, ETHANOL, ETHANOL])
        assert len(scores) == 3
        assert scores[0] == scores[1] == scores[2]

    def test_batch_skips_invalid(self) -> None:
        scores = compute_sa_scores_batch([ETHANOL, "garbage", BENZENE])
        assert len(scores) == 2


class TestDiversity:
    def test_empty(self) -> None:
        assert compute_diversity([]) == 0.0

    def test_single_molecule(self) -> None:
        assert compute_diversity([ETHANOL]) == 0.0

    def test_identical_molecules_give_zero(self) -> None:
        assert compute_diversity([ETHANOL, ETHANOL, ETHANOL]) == pytest.approx(0.0, abs=1e-9)

    def test_different_molecules_give_positive(self) -> None:
        assert compute_diversity([ETHANOL, BENZENE]) > 0.0

    def test_matches_analytic_two_molecule_formula(self) -> None:
        # For n=2, IntDiv_1 = 1 - mean(full 2x2 similarity matrix)
        #                   = 1 - (1 + T + T + 1) / 4 = (1 - T) / 2
        # where T is the pairwise Tanimoto similarity. This checks the
        # implementation against the literal formula, independent of any
        # paper-specific target number.
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp_a = gen.GetFingerprint(Chem.MolFromSmiles(ETHANOL))
        fp_b = gen.GetFingerprint(Chem.MolFromSmiles(ASPIRIN))
        from rdkit import DataStructs

        t = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        expected = (1.0 - t) / 2.0
        assert compute_diversity([ETHANOL, ASPIRIN]) == pytest.approx(expected, abs=1e-9)

    def test_default_bit_width_matches_moses(self) -> None:
        # moses/metrics/utils.py::fingerprint has morgan__n=1024 as its
        # default, not the more commonly-seen 2048 -- confirmed against the
        # real MOSES source, not assumed. Pinned here so a future "helpful"
        # bump to 2048 doesn't silently drift from the paper's stated metric
        # definitions again.
        import inspect

        default_n_bits = inspect.signature(compute_diversity).parameters["n_bits"].default
        assert default_n_bits == 1024

    def test_invalid_entries_are_skipped_not_counted(self) -> None:
        with_garbage = compute_diversity([ETHANOL, BENZENE, "garbage"])
        without = compute_diversity([ETHANOL, BENZENE])
        assert with_garbage == pytest.approx(without)


class TestProperties:
    def test_logp_ethanol_is_small(self) -> None:
        # Ethanol is polar/hydrophilic; RDKit's Crippen LogP should be small
        # and negative-ish. Loose bound -- this checks wiring, not RDKit's model.
        assert compute_logp(ETHANOL) < 1.0

    def test_qed_in_unit_interval(self) -> None:
        for smi in (ETHANOL, BENZENE, ASPIRIN):
            assert 0.0 <= compute_qed(smi) <= 1.0

    def test_molecular_weight_ethanol(self) -> None:
        # C2H6O: 2*12.011 + 6*1.008 + 15.999 = 46.069
        assert compute_molecular_weight(ETHANOL) == pytest.approx(46.07, abs=0.05)

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            compute_logp("garbage")

    def test_batch_preserves_duplicates_and_order(self) -> None:
        values = compute_logps_batch([ETHANOL, BENZENE, ETHANOL])
        assert len(values) == 3
        assert values[0] == values[2]

    def test_logp_in_range_fraction_self_consistent(self) -> None:
        smiles = [ETHANOL, BENZENE, ASPIRIN]
        low, high = LOGP_RANGE
        expected_hits = sum(
            1 for s in smiles if low < Crippen.MolLogP(Chem.MolFromSmiles(s)) < high
        )
        assert logp_in_range_fraction(smiles) == pytest.approx(expected_hits / len(smiles))

    def test_logp_in_range_fraction_empty(self) -> None:
        import math

        assert math.isnan(logp_in_range_fraction([]))


class TestFractions:
    def test_distinct_fraction_no_duplicates(self) -> None:
        assert distinct_fraction([ETHANOL, BENZENE, ASPIRIN]) == pytest.approx(1.0)

    def test_distinct_fraction_with_duplicates(self) -> None:
        # 2 unique strings out of 4 entries.
        assert distinct_fraction([ETHANOL, ETHANOL, BENZENE, BENZENE]) == pytest.approx(0.5)

    def test_distinct_fraction_operates_on_raw_strings(self) -> None:
        # "CCO" and "OCC" are the same molecule but different raw strings;
        # epsilon_d must NOT canonicalise -- that's epsilon_u's job.
        assert distinct_fraction(["CCO", "OCC"]) == pytest.approx(1.0)

    def test_unique_fraction_collapses_equivalent_representations(self) -> None:
        # "CCO" and "OCC" canonicalise to the same molecule.
        assert unique_fraction(["CCO", "OCC", BENZENE]) == pytest.approx(2 / 3)

    def test_unique_fraction_all_unique(self) -> None:
        assert unique_fraction([ETHANOL, BENZENE, ASPIRIN]) == pytest.approx(1.0)

    def test_novelty_all_seen(self) -> None:
        assert novelty_fraction([ETHANOL, BENZENE], reference_smiles=[ETHANOL, BENZENE]) == 0.0

    def test_novelty_all_new(self) -> None:
        assert novelty_fraction([ASPIRIN], reference_smiles=[ETHANOL, BENZENE]) == 1.0

    def test_novelty_mixed(self) -> None:
        result = novelty_fraction([ETHANOL, ASPIRIN], reference_smiles=[ETHANOL, BENZENE])
        assert result == pytest.approx(0.5)

    def test_novelty_matches_across_representations(self) -> None:
        # "OCC" is ethanol under a different string; must still count as seen.
        assert novelty_fraction(["OCC"], reference_smiles=[ETHANOL]) == 0.0


class TestWasserstein:
    def test_identical_distributions_give_zero(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert property_wasserstein_distance(values, values) == pytest.approx(0.0)

    def test_shifted_distribution_is_positive(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert property_wasserstein_distance(a, b) == pytest.approx(3.0)

    def test_empty_input_returns_zero_not_raise(self) -> None:
        assert property_wasserstein_distance([], [1.0, 2.0]) == 0.0
        assert property_wasserstein_distance([1.0], []) == 0.0


class TestMosesMetricsIntegration:
    def test_property_report_shapes(self) -> None:
        report = compute_property_report([ETHANOL, BENZENE, ASPIRIN, "garbage"])
        assert report.n_input == 4
        assert report.n_valid == 3
        assert report.validity == pytest.approx(0.75)
        assert len(report.logp) == len(report.sa) == len(report.qed) == len(report.weight) == 3

    def test_property_report_mean_properties(self) -> None:
        report = compute_property_report([ETHANOL, BENZENE])
        assert report.mean_logp == pytest.approx(sum(report.logp) / 2)

    def test_wasserstein_report_self_comparison_is_zero(self) -> None:
        report = compute_property_report([ETHANOL, BENZENE, ASPIRIN])
        w = compute_wasserstein_report(report, report)
        assert (w.logp, w.sa, w.qed, w.weight) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-9)

    def test_wasserstein_report_differing_samples_nonnegative(self) -> None:
        a = compute_property_report([ETHANOL, ETHANOL, BENZENE])
        b = compute_property_report([ASPIRIN, ASPIRIN, ASPIRIN])
        w = compute_wasserstein_report(a, b)
        assert w.logp >= 0 and w.sa >= 0 and w.qed >= 0 and w.weight >= 0


class TestComputeSignificanceMetrics:
    """The 10-metric dict feeding Eq. 5 / Table 14 significance comparisons."""

    def test_returns_all_ten_significance_keys(self) -> None:
        from mqs_molecule_generation.metrics.significance import SIGNIFICANCE_METRICS

        result = compute_significance_metrics(
            generated_smiles=[ETHANOL, BENZENE, ASPIRIN],
            reference_smiles=[ETHANOL, BENZENE],
        )
        assert set(result.keys()) == set(SIGNIFICANCE_METRICS.keys())

    def test_values_are_finite_floats(self) -> None:
        result = compute_significance_metrics(
            generated_smiles=[ETHANOL, BENZENE, ASPIRIN, ETHANOL],
            reference_smiles=[ETHANOL, BENZENE],
        )
        for name, value in result.items():
            assert isinstance(value, float), name
            assert value == value, name  # NaN check (NaN != NaN)

    def test_fractions_are_bounded_zero_one(self) -> None:
        result = compute_significance_metrics(
            generated_smiles=[ETHANOL, BENZENE, ASPIRIN, "garbage", ETHANOL],
            reference_smiles=[ETHANOL, BENZENE],
        )
        for name in ("eps_d", "eps_v", "eps_u", "novelty", "intdiv", "filters", "eps_logp"):
            assert 0.0 <= result[name] <= 1.0, name

    def test_novelty_all_seen_is_zero(self) -> None:
        result = compute_significance_metrics(
            generated_smiles=[ETHANOL, BENZENE],
            reference_smiles=[ETHANOL, BENZENE, ASPIRIN],
        )
        assert result["novelty"] == pytest.approx(0.0)

    def test_novelty_all_new_is_one(self) -> None:
        result = compute_significance_metrics(
            generated_smiles=[ASPIRIN],
            reference_smiles=[ETHANOL, BENZENE],
        )
        assert result["novelty"] == pytest.approx(1.0)

    def test_matches_independently_computed_values(self) -> None:
        # Cross-check against the individual functions this composes, called
        # directly -- not just "runs without crashing".
        generated = [ETHANOL, BENZENE, ASPIRIN, ETHANOL]
        reference = [ETHANOL, BENZENE]
        result = compute_significance_metrics(generated, reference)

        _, _, expected_eps_v = compute_validity(generated)
        assert result["eps_v"] == pytest.approx(expected_eps_v)
        assert result["eps_d"] == pytest.approx(distinct_fraction(generated))

        valid = [s for s in generated if is_valid_smiles(s)]
        assert result["eps_u"] == pytest.approx(unique_fraction(valid))
        assert result["novelty"] == pytest.approx(novelty_fraction(valid, reference))
        assert result["intdiv"] == pytest.approx(compute_diversity(valid))

    def test_weight_and_sa_are_minimized_scale_not_fractions(self) -> None:
        # Sanity check these are raw property means, not accidentally routed
        # through a 0-1 fraction helper.
        result = compute_significance_metrics(
            generated_smiles=[ASPIRIN, ASPIRIN],
            reference_smiles=[ETHANOL],
        )
        assert result["weight"] > 1.0  # molecular weight, not a fraction
        assert result["sa"] > 0.5  # SA score is roughly in [1, 10]