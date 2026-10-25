"""Readout observables (gate 11: single -> D_l = n_qb, dual -> 2*n_qb)."""

from __future__ import annotations

import pennylane as qml
import pytest

from mqs_molecule_generation.circuits.ansatz import AnsatzParams
from mqs_molecule_generation.circuits.readout import (
    readout_dim,
    readout_observables,
    readout_pauli_strings,
    readout_paulis,
)


@pytest.mark.parametrize("n_qb", [1, 2, 5, 10, 15])
class TestDimensions:
    def test_single_is_n_qb(self, n_qb) -> None:
        assert readout_dim(n_qb, "single") == n_qb
        assert len(readout_paulis(n_qb, "single")) == n_qb

    def test_dual_is_two_n_qb(self, n_qb) -> None:
        assert readout_dim(n_qb, "dual") == 2 * n_qb
        assert len(readout_paulis(n_qb, "dual")) == 2 * n_qb

    def test_agrees_with_ansatz_params(self, n_qb) -> None:
        ap = AnsatzParams(n_qb, 1, "simple")
        assert ap.readout_dim_single == readout_dim(n_qb, "single")
        assert ap.readout_dim_dual == readout_dim(n_qb, "dual")


class TestObservables:
    def test_dual_is_z_block_then_x_block(self) -> None:
        assert readout_pauli_strings(3, "dual") == ["Z0", "Z1", "Z2", "X0", "X1", "X2"]
        assert readout_pauli_strings(3, "single") == ["Z0", "Z1", "Z2"]

    def test_pennylane_observables_match_the_strings(self) -> None:
        obs = readout_observables(2, "dual")
        assert [type(o).__name__ for o in obs] == ["PauliZ", "PauliZ", "PauliX", "PauliX"]
        assert [o.wires.tolist() for o in obs] == [[0], [1], [0], [1]]
        assert all(isinstance(o, qml.operation.Operator) for o in obs)

    def test_invalid_readout_rejected(self) -> None:
        for fn in (readout_dim, readout_paulis, readout_pauli_strings):
            with pytest.raises(ValueError, match="readout must be"):
                fn(3, "triple")