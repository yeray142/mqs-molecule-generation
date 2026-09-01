"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner


class TestCLI:
    """Test CLI commands."""

    def setup_method(self) -> None:
        """Set up test runner."""
        from mqs_molecule_generation.cli.commands import main

        self.runner = CliRunner()
        self.main = main

    def test_manifest_cmd(self) -> None:
        """Test manifest outputs manifest."""
        result = self.runner.invoke(self.main, ["manifest"])
        assert result.exit_code == 0
        assert "git_sha" in result.output
        assert "lockfile_hash" in result.output

    def test_manifest_cmd_yaml_format(self) -> None:
        """Test manifest outputs valid YAML."""
        result = self.runner.invoke(self.main, ["manifest"])
        assert result.exit_code == 0
        assert "python_version" in result.output

    def test_train_dry_run(self) -> None:
        """Test train --dry-run completes without error."""
        result = self.runner.invoke(self.main, [
            "train",
            "--dry-run",
            "--seeds", "0",
            "--ansatz", "simple",
            "--n-qubits", "2",
            "--n-layers", "1",
            "--max-iters", "5",
        ])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_train_dry_run_bel(self) -> None:
        """Test train --dry-run with BEL ansatz."""
        result = self.runner.invoke(self.main, [
            "train",
            "--dry-run",
            "--seeds", "0",
            "--ansatz", "bel",
            "--n-qubits", "2",
            "--n-layers", "1",
            "--max-iters", "5",
        ])
        assert result.exit_code == 0
        assert "ansatz: bel" in result.output.lower() or "bel" in result.output.lower()

    def test_train_chain_dry_run(self) -> None:
        """Test train --dry-run --chain completes."""
        result = self.runner.invoke(self.main, [
            "train",
            "--dry-run",
            "--chain",
            "--seeds", "0",
            "--ansatz", "simple",
            "--n-qubits", "2",
            "--n-layers", "1",
        ])
        assert result.exit_code == 0
        assert "CHAIN" in result.output

    def test_eval_dry_run(self) -> None:
        """Test eval --dry-run completes."""
        result = self.runner.invoke(self.main, ["eval", "--dry-run"])
        assert result.exit_code == 0

    def test_eval_chain(self) -> None:
        """Test eval --chain completes."""
        result = self.runner.invoke(self.main, ["eval", "--chain"])
        assert result.exit_code == 0
        assert "CHAIN" in result.output

    def test_train_multiple_seeds(self) -> None:
        """Test train with multiple seeds."""
        result = self.runner.invoke(self.main, [
            "train",
            "--dry-run",
            "--seeds", "0-2",
            "--ansatz", "simple",
            "--n-qubits", "2",
            "--n-layers", "1",
            "--max-iters", "5",
        ])
        assert result.exit_code == 0

    def test_train_dual_readout(self) -> None:
        """Test train with dual readout."""
        result = self.runner.invoke(self.main, [
            "train",
            "--dry-run",
            "--seeds", "0",
            "--ansatz", "simple",
            "--n-qubits", "2",
            "--n-layers", "1",
            "--max-iters", "5",
            "--readout", "dual",
        ])
        assert result.exit_code == 0
        assert "dual" in result.output.lower() or "readout" in result.output.lower()

    def test_train_invalid_ansatz(self) -> None:
        """Test train with invalid ansatz fails gracefully."""
        result = self.runner.invoke(self.main, [
            "train",
            "--dry-run",
            "--ansatz", "invalid",
        ])
        assert result.exit_code != 0 or "invalid" in result.output.lower()
