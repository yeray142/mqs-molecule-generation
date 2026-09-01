"""Tests for ansatz parameter derivation and invariants."""

import pytest

from mqs_molecule_generation.circuits.ansatz import (
    PAPER_N_PARAMS_TABLE,
    AnsatzParams,
    build_bel_ansatz_qasm,
    build_simple_ansatz_qasm,
    make_ansatz_circuit,
)


class TestAnsatzParams:
    """Test AnsatzParams invariants."""

    @pytest.mark.parametrize("n_qb,n_l", [(2, 1), (4, 2), (6, 3), (8, 1), (10, 2)])
    def test_simple_n_angles(self, n_qb: int, n_l: int) -> None:
        """Test simple ansatz n_angles = n_qb * n_l."""
        ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type="simple")
        assert ap.n_angles == n_qb * n_l

    @pytest.mark.parametrize("n_qb,n_l", [(2, 1), (4, 2), (6, 3), (8, 1), (10, 2)])
    def test_bel_n_angles(self, n_qb: int, n_l: int) -> None:
        """Test BEL ansatz n_angles = n_qb * (5*n_l + 1)."""
        ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type="bel")
        assert ap.n_angles == n_qb * (5 * n_l + 1)

    @pytest.mark.parametrize("n_qb,n_l", [(2, 1), (4, 2), (6, 3), (8, 1), (10, 2)])
    def test_simple_n_params(self, n_qb: int, n_l: int) -> None:
        """Test simple ansatz n_params = 2 * n_angles."""
        ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type="simple")
        assert ap.n_params == 2 * ap.n_angles

    @pytest.mark.parametrize("n_qb,n_l", [(2, 1), (4, 2), (6, 3), (8, 1), (10, 2)])
    def test_bel_n_params(self, n_qb: int, n_l: int) -> None:
        """Test BEL ansatz n_params = 2 * n_angles."""
        ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type="bel")
        assert ap.n_params == 2 * ap.n_angles

    def test_readout_dims_simple(self) -> None:
        """Test readout dimensions for simple ansatz."""
        ap = AnsatzParams(n_qb=4, n_l=2, ansatz_type="simple")
        assert ap.readout_dim_single == 4
        assert ap.readout_dim_dual == 8

    def test_readout_dims_bel(self) -> None:
        """Test readout dimensions for BEL ansatz."""
        ap = AnsatzParams(n_qb=4, n_l=2, ansatz_type="bel")
        assert ap.readout_dim_single == 4
        assert ap.readout_dim_dual == 8

    def test_expected_z0_simple(self) -> None:
        """Test ⟨Z₀⟩ = -1.00 for simple ansatz (Table 14)."""
        ap = AnsatzParams(n_qb=4, n_l=2, ansatz_type="simple")
        assert ap.expected_z0 == -1.0

    def test_expected_z0_bel(self) -> None:
        """Test ⟨Z₀⟩ = -0.26 for BEL ansatz (Table 14)."""
        ap = AnsatzParams(n_qb=4, n_l=2, ansatz_type="bel")
        assert ap.expected_z0 == -0.26

    def test_invalid_ansatz_type(self) -> None:
        """Test ValueError for invalid ansatz type."""
        with pytest.raises(ValueError, match="Unknown ansatz type"):
            AnsatzParams(n_qb=4, n_l=2, ansatz_type="invalid")


class TestPaperNParamsTable:
    """Test all 11 rows from the paper's N_params table."""

    @pytest.mark.parametrize("n_qb,n_l,ansatz", PAPER_N_PARAMS_TABLE)
    def test_n_params_table_row(self, n_qb: int, n_l: int, ansatz: str) -> None:
        """Test each row of the paper's N_params table."""
        ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type=ansatz)

        # Verify n_angles
        if ansatz == "simple":
            expected_angles = n_qb * n_l
        else:
            expected_angles = n_qb * (5 * n_l + 1)

        assert ap.n_angles == expected_angles, (
            f"n_angles mismatch for ({n_qb}, {n_l}, {ansatz}): "
            f"expected {expected_angles}, got {ap.n_angles}"
        )

        # Verify n_params = 2 * n_angles
        assert ap.n_params == 2 * ap.n_angles

        # Verify readout dims
        assert ap.readout_dim_single == n_qb
        assert ap.readout_dim_dual == 2 * n_qb


class TestSimpleAnsatzQASM:
    """Test simple ansatz QASM generation."""

    def test_simple_qasm_parameter_count(self) -> None:
        """Test simple ansatz has correct number of parameters."""
        n_qb, n_l = 4, 2
        qasm = build_simple_ansatz_qasm(n_qb, n_l)
        # 4*2 = 8 angles
        for i in range(8):
            assert f"angle_{i}" in qasm

    def test_simple_qasm_format(self) -> None:
        """Test simple ansatz QASM format."""
        qasm = build_simple_ansatz_qasm(n_qb=2, n_l=1)
        assert qasm.startswith("OPENQASM 3.0")
        assert "qubit[2] q" in qasm
        assert "ry angle_0 q[0]" in qasm

    def test_simple_qasm_no_entanglement_last_layer(self) -> None:
        """Test simple ansatz has no CX after last layer."""
        qasm = build_simple_ansatz_qasm(n_qb=2, n_l=1)
        lines = qasm.split("\n")
        # Count CX gates - should be 0 for n_l=1
        cx_count = sum(1 for line in lines if line.strip().startswith("cx "))
        assert cx_count == 0

    def test_simple_qasm_entanglement_between_layers(self) -> None:
        """Test simple ansatz has CX gates between layers."""
        qasm = build_simple_ansatz_qasm(n_qb=2, n_l=2)
        lines = qasm.split("\n")
        # Count CX gates - should be 1 for n_l=2 (between layer 0 and 1)
        cx_count = sum(1 for line in lines if line.strip().startswith("cx "))
        assert cx_count == 1


class TestBELAnsatzQASM:
    """Test BEL ansatz QASM generation."""

    def test_bel_qasm_parameter_count(self) -> None:
        """Test BEL ansatz has correct number of parameters."""
        n_qb, n_l = 4, 2
        qasm = build_bel_ansatz_qasm(n_qb, n_l)
        # 4*(5*2+1) = 4*11 = 44 angles
        for i in range(44):
            assert f"bel_angle_{i}" in qasm

    def test_bel_qasm_format(self) -> None:
        """Test BEL ansatz QASM format."""
        qasm = build_bel_ansatz_qasm(n_qb=2, n_l=1)
        assert qasm.startswith("OPENQASM 3.0")
        assert "qubit[2] q" in qasm
        assert "ry bel_angle_0 q[0]" in qasm

    def test_bel_qasm_has_final_ry(self) -> None:
        """Test BEL ansatz has final RY per qubit."""
        n_qb, n_l = 2, 1
        qasm = build_bel_ansatz_qasm(n_qb, n_l)
        lines = qasm.split("\n")
        # Final RY indices: for n_qb=2, n_l=1: bel_angle_10 and bel_angle_11
        assert "ry bel_angle_10 q[0]" in qasm
        assert "ry bel_angle_11 q[1]" in qasm


class TestMakeAnsatzCircuit:
    """Test make_ansatz_circuit function."""

    def test_make_simple_circuit(self) -> None:
        """Test make_ansatz_circuit for simple ansatz."""
        qasm, ap = make_ansatz_circuit(n_qb=4, n_l=2, ansatz_type="simple")
        assert ap.n_angles == 8
        assert ap.n_params == 16
        assert ap.ansatz_type == "simple"

    def test_make_bel_circuit(self) -> None:
        """Test make_ansatz_circuit for BEL ansatz."""
        qasm, ap = make_ansatz_circuit(n_qb=4, n_l=2, ansatz_type="bel")
        assert ap.n_angles == 44
        assert ap.n_params == 88
        assert ap.ansatz_type == "bel"

    def test_make_ansatz_with_params(self) -> None:
        """Test make_ansatz_circuit with parameter values."""
        qasm, ap = make_ansatz_circuit(
            n_qb=2, n_l=1, ansatz_type="simple", param_values=[0.1] * 4
        )
        assert ap.n_params == 4

    def test_make_ansatz_wrong_param_count(self) -> None:
        """Test ValueError for wrong parameter count."""
        with pytest.raises(ValueError, match="Expected 4 params"):
            make_ansatz_circuit(
                n_qb=2, n_l=1, ansatz_type="simple", param_values=[0.1] * 10
            )

    def test_make_ansatz_invalid_type(self) -> None:
        """Test ValueError for invalid ansatz type."""
        with pytest.raises(ValueError, match="Unknown ansatz"):
            make_ansatz_circuit(n_qb=2, n_l=1, ansatz_type="invalid")
