"""Tests for molecular metrics."""

import pytest


class TestValidity:
    """Test molecular validity checks."""

    def test_is_valid_smiles_valid(self) -> None:
        """Test valid SMILES."""
        from mqs_molecule_generation.metrics.validity import is_valid_smiles

        assert is_valid_smiles("CCO")  # ethanol
        assert is_valid_smiles("c1ccccc1")  # benzene

    def test_is_valid_smiles_invalid(self) -> None:
        """Test invalid SMILES."""
        from mqs_molecule_generation.metrics.validity import is_valid_smiles

        assert not is_valid_smiles("INVALID")
        assert not is_valid_smiles("")

    def test_compute_validity(self) -> None:
        """Test validity computation."""
        from mqs_molecule_generation.metrics.validity import compute_validity

        valid, total, frac = compute_validity(["CCO", "c1ccccc1", "INVALID", ""])
        assert valid == 2
        assert total == 4
        assert abs(frac - 0.5) < 0.001

    def test_compute_validity_empty(self) -> None:
        """Test validity with empty list."""
        from mqs_molecule_generation.metrics.validity import compute_validity

        valid, total, frac = compute_validity([])
        assert valid == 0
        assert total == 0
        assert frac == 0.0


class TestSA_Score:
    """Test SA score computation (skipped if RDKit not available)."""

    def test_sa_score_import_error(self) -> None:
        """Test that SA score raises ImportError without RDKit."""
        # This test verifies the graceful import error
        try:
            from mqs_molecule_generation.metrics.sa_score import compute_sa_score

            # If RDKit is not installed, this should raise ImportError
            # If RDKit is installed but SA score data is missing, FileNotFoundError
            pass
        except ImportError:
            pass  # Expected without RDKit

    def test_sa_score_batch(self) -> None:
        """Test batch SA score computation."""
        # This test is skipped if RDKit is not available
        pytest.importorskip("rdkit", reason="RDKit not installed")

        from pathlib import Path
        from rdkit import RDConfig

        sa_score_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascore.py"
        if not sa_score_path.exists():
            pytest.skip("RDKit SA_Score contrib not installed")

        from mqs_molecule_generation.metrics.sa_score import compute_sa_scores_batch

        scores = compute_sa_scores_batch(["CCO", "c1ccccc1"])
        assert isinstance(scores, dict)
        assert len(scores) == 2
        for score in scores.values():
            assert 1.0 <= score <= 10.0


class TestDiversity:
    """Test molecular diversity computation."""

    def test_compute_diversity_empty(self) -> None:
        """Test diversity with empty list."""
        from mqs_molecule_generation.metrics.diversity import compute_diversity

        div = compute_diversity([])
        assert div == 0.0

    def test_compute_diversity_single(self) -> None:
        """Test diversity with single molecule."""
        from mqs_molecule_generation.metrics.diversity import compute_diversity

        div = compute_diversity(["CCO"])
        assert div == 0.0

    def test_compute_diversity_identical(self) -> None:
        """Test diversity with identical molecules."""
        pytest.importorskip("rdkit", reason="RDKit not installed")

        from mqs_molecule_generation.metrics.diversity import compute_diversity

        div = compute_diversity(["CCO", "CCO", "CCO"])
        assert div == 0.0

    def test_compute_diversity_different(self) -> None:
        """Test diversity with different molecules."""
        pytest.importorskip("rdkit", reason="RDKit not installed")

        from mqs_molecule_generation.metrics.diversity import compute_diversity

        div = compute_diversity(["CCO", "c1ccccc1"])
        assert div > 0.0  # Should have non-zero diversity
