"""Shared test fixtures for mqs-molecule-generation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """Provide a temporary directory that is cleaned up."""
    import shutil

    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Mock BackendAdapter that returns a fixed QuantumResult."""
    from qpubench.schemas.primitives import ComputingModel, JobStatus
    from qpubench.schemas.result import QuantumResult

    mock = MagicMock()
    mock.run.return_value = QuantumResult(
        computing_model=ComputingModel.GATE_BASED,
        status=JobStatus.SUCCEEDED,
    )
    return mock


@pytest.fixture
def pennylane_mock() -> MagicMock:
    """Mock PennyLane module for adapter tests."""
    mock = MagicMock()
    mock.device.return_value = MagicMock()
    mock.qnode.return_value = MagicMock()
    mock.expval = MagicMock()
    mock.var = MagicMock()
    return mock


# Import helper for tests
def make_simple_circuit_spec(num_qubits: int = 4) -> Any:
    """Helper to create a simple CircuitSpec for testing."""
    from qpubench.schemas.circuit import CircuitSpec
    from qpubench.schemas.primitives import CircuitFormat

    from mqs_molecule_generation.circuits.ansatz import build_simple_ansatz_qasm

    qasm = build_simple_ansatz_qasm(num_qubits, n_l=2)
    return CircuitSpec(
        num_qubits=num_qubits,
        format=CircuitFormat.QASM3,
        serialized=qasm,
    )


def make_bel_circuit_spec(num_qubits: int = 4) -> Any:
    """Helper to create a BEL CircuitSpec for testing."""
    from qpubench.schemas.circuit import CircuitSpec
    from qpubench.schemas.primitives import CircuitFormat

    from mqs_molecule_generation.circuits.ansatz import build_bel_ansatz_qasm

    qasm = build_bel_ansatz_qasm(num_qubits, n_l=2)
    return CircuitSpec(
        num_qubits=num_qubits,
        format=CircuitFormat.QASM3,
        serialized=qasm,
    )
