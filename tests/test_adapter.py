"""Tests for LatentStylePennyLaneAdapter with mocked PennyLane."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from qpubench.schemas.primitives import CircuitFormat, ComputingModel, JobStatus
from qpubench.schemas.circuit import CircuitSpec
from qpubench.schemas.execution import ExecutionOptions


class TestLatentStylePennyLaneAdapter:
    """Test LatentStylePennyLaneAdapter."""

    def test_spec_property(self) -> None:
        """Test spec property returns BackendSpec."""
        from mqs_molecule_generation.adapters.pennylane_latent_style_adapter import (
            LatentStylePennyLaneAdapter,
        )

        adapter = LatentStylePennyLaneAdapter(n_qubits=4)
        assert adapter.spec.name == "lightning.qubit"
        assert adapter.spec.provider == "lightning"

    def test_validate_warnings_on_wrong_model(self) -> None:
        """Test validate returns warnings for wrong computing model."""
        from mqs_molecule_generation.adapters.pennylane_latent_style_adapter import (
            LatentStylePennyLaneAdapter,
        )

        adapter = LatentStylePennyLaneAdapter()
        circuit = CircuitSpec(
            num_qubits=2,
            format=CircuitFormat.QASM3,
            serialized="OPENQASM 3.0;",
        )
        circuit = circuit.model_copy(update={"computing_model": ComputingModel.MBQC})
        warnings = adapter.validate(circuit)
        assert any("GATE_BASED" in w for w in warnings)

    def test_validate_warnings_on_unbound_params(self) -> None:
        """Test validate returns warnings for unbound parameters."""
        from mqs_molecule_generation.adapters.pennylane_latent_style_adapter import (
            LatentStylePennyLaneAdapter,
        )

        adapter = LatentStylePennyLaneAdapter()
        circuit = CircuitSpec(
            num_qubits=2,
            format=CircuitFormat.QASM3,
            serialized="OPENQASM 3.0;",
            parameters=["theta_0", "theta_1"],
            # No parameter_bindings - unbound
        )
        warnings = adapter.validate(circuit)
        assert any("unbound" in w.lower() for w in warnings)

    def test_run_returns_failed_when_pennylane_not_installed(self) -> None:
        """Core install (no PennyLane) should return FAILED, not raise."""
        # Simulate PennyLane not installed
        original_pennylane = sys.modules.get("pennylane")
        sys.modules["pennylane"] = None  # type: ignore

        try:
            # Need to reimport to pick up the patched state
            import importlib
            import mqs_molecule_generation.adapters.pennylane_latent_style_adapter as mod

            importlib.reload(mod)

            adapter = mod.LatentStylePennyLaneAdapter()
            circuit = CircuitSpec(
                num_qubits=2,
                format=CircuitFormat.QASM3,
                serialized="OPENQASM 3.0;",
            )
            result = adapter.run(circuit, ExecutionOptions())
            assert result.status == JobStatus.FAILED
            assert "not installed" in result.error_message
        finally:
            if original_pennylane is not None:
                sys.modules["pennylane"] = original_pennylane
            else:
                del sys.modules["pennylane"]

    def test_run_returns_failed_when_qiskit_not_installed(self) -> None:
        """Test FAILED when qiskit is not installed (but PennyLane is)."""
        # First restore pennylane
        original_pennylane = sys.modules.get("pennylane")
        original_qiskit = sys.modules.get("qiskit")

        pennylane_mock = MagicMock()
        sys.modules["pennylane"] = pennylane_mock
        sys.modules["pennylane_qiskit"] = MagicMock()
        sys.modules["qiskit"] = None

        try:
            import importlib
            import mqs_molecule_generation.adapters.pennylane_latent_style_adapter as mod

            importlib.reload(mod)

            adapter = mod.LatentStylePennyLaneAdapter()
            circuit = CircuitSpec(
                num_qubits=2,
                format=CircuitFormat.QASM3,
                serialized="OPENQASM 3.0;",
            )
            result = adapter.run(circuit, ExecutionOptions())
            assert result.status == JobStatus.FAILED
            assert "qiskit" in result.error_message.lower()
        finally:
            sys.modules["pennylane"] = original_pennylane
            sys.modules["qiskit"] = original_qiskit


class TestAdapterWithMocks:
    """Full adapter path with mocked backends."""

    def test_validate_accepts_qasm3(self) -> None:
        """Test validate accepts QASM3 format."""
        from mqs_molecule_generation.adapters.pennylane_latent_style_adapter import (
            LatentStylePennyLaneAdapter,
        )

        adapter = LatentStylePennyLaneAdapter()
        circuit = CircuitSpec(
            num_qubits=4,
            format=CircuitFormat.QASM3,
            serialized="OPENQASM 3.0;",
        )
        warnings = adapter.validate(circuit)
        assert not any("QASM3" in w for w in warnings)

    def test_adapter_initialization(self) -> None:
        """Test adapter initialization with all parameters."""
        from mqs_molecule_generation.adapters.pennylane_latent_style_adapter import (
            LatentStylePennyLaneAdapter,
        )

        adapter = LatentStylePennyLaneAdapter(
            n_qubits=8,
            ansatz_type="bel",
            n_layers=3,
        )
        assert adapter._n_qubits == 8
        assert adapter._ansatz_type == "bel"
        assert adapter._n_layers == 3
