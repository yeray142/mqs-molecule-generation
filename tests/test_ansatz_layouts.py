"""Ansatz layouts: counts, single-source-of-truth consistency, QASM validity (no torch needed)."""

from __future__ import annotations

import re

import pytest

from mqs_molecule_generation.circuits.ansatz import (
    AnsatzParams,
    build_bel_ansatz_qasm,
    build_simple_ansatz_qasm,
    make_ansatz_circuit,
)
from mqs_molecule_generation.circuits.paper_configs import (
    CONFIGS_NAMED_IN_PAPER_TEXT,
    PAPER_STATED_N_PARAMS,
)

GRID = [
    (kind, n_qb, n_l)
    for kind in ("simple", "bel")
    for n_qb in (2, 3, 5, 10, 15)
    for n_l in (1, 2, 4, 6)
]
PAPER_GRID = [
    (kind, c.n_qb, c.n_l) for c in CONFIGS_NAMED_IN_PAPER_TEXT for kind in ("simple", "bel")
]


@pytest.mark.parametrize(("kind", "n_qb", "n_l"), GRID)
class TestCounts:
    def test_n_angles_matches_paper_formula(self, kind, n_qb, n_l) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        expected = n_qb * n_l if kind == "simple" else n_qb * (5 * n_l + 1)
        assert ap.n_angles == expected

    def test_params_are_two_per_angle(self, kind, n_qb, n_l) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        assert ap.n_params == 2 * ap.n_angles

    def test_every_angle_index_used_exactly_once(self, kind, n_qb, n_l) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        used = [g.param for g in ap.layout if g.param is not None]
        assert sorted(used) == list(range(ap.n_angles))

    def test_each_qubit_owns_the_expected_number_of_angles(self, kind, n_qb, n_l) -> None:
        # Eq. 4 feeds angle (q, l, k) with xi_q: every qubit should own n_l angles
        # (simple) or 5*n_l + 1 angles (BEL, incl. the final RY).
        ap = AnsatzParams(n_qb, n_l, kind)
        owners = ap.noise_qubit_per_angle
        per_qubit = [owners.count(q) for q in range(n_qb)]
        assert per_qubit == [n_l if kind == "simple" else 5 * n_l + 1] * n_qb

    def test_qasm_parses_and_declares_every_parameter_once(self, kind, n_qb, n_l) -> None:
        openqasm3 = pytest.importorskip("openqasm3")
        build = build_simple_ansatz_qasm if kind == "simple" else build_bel_ansatz_qasm
        qasm = build(n_qb, n_l)
        openqasm3.parse(qasm)
        ap = AnsatzParams(n_qb, n_l, kind)
        assert len(re.findall(r"^input float\[32\]", qasm, flags=re.M)) == ap.n_angles
        for name in ap.param_names:
            # declared once, used in exactly one gate call
            assert len(re.findall(rf"\({name}\)", qasm)) == 1


class TestBelStructure:
    @pytest.mark.parametrize(("n_qb", "n_l"), [(2, 1), (5, 2), (10, 4)])
    def test_gate_counts(self, n_qb, n_l) -> None:
        names = [g.name for g in AnsatzParams(n_qb, n_l, "bel").layout]
        assert names.count("RY") == 2 * n_l * n_qb + n_qb  # two per layer + final
        assert names.count("RZ") == 2 * n_l * n_qb
        assert names.count("CRY") == n_l * n_qb  # the paper: controlled rotations carry angles
        assert "CX" not in names

    def test_controlled_rotations_form_a_ring_each_layer(self) -> None:
        n_qb = 5
        cry = [g.wires for g in AnsatzParams(n_qb, 1, "bel").layout if g.name == "CRY"]
        assert cry == [(q, (q + 1) % n_qb) for q in range(n_qb)]

    def test_needs_two_qubits(self) -> None:
        with pytest.raises(ValueError, match="n_qb >= 2"):
            _ = AnsatzParams(1, 1, "bel").layout


class TestPaperAnchors:
    def test_nominal_bel_has_the_110_parameters_the_paper_states(self) -> None:
        (stated,) = [
            v
            for (kind, nq, nl), v in PAPER_STATED_N_PARAMS.items()
            if (kind, nq, nl) == ("bel", 5, 2)
        ]
        assert stated == 110
        assert AnsatzParams(5, 2, "bel").n_params == 110

    @pytest.mark.parametrize(("kind", "n_qb", "n_l"), PAPER_GRID)
    def test_paper_named_configs_give_latent_dims_10_20_30(self, kind, n_qb, n_l) -> None:
        config = next(c for c in CONFIGS_NAMED_IN_PAPER_TEXT if (c.n_qb, c.n_l) == (n_qb, n_l))
        assert config.latent_dim in (10, 20, 30)


class TestRegressionsAgainstScaffold:
    def test_expected_z0_values_preserved_for_backwards_compatibility(self) -> None:
        assert AnsatzParams(5, 2, "simple").expected_z0 == -1.0
        assert AnsatzParams(5, 2, "bel").expected_z0 == -0.26

    def test_unknown_ansatz_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown ansatz"):
            AnsatzParams(5, 2, "strongly")

    def test_make_ansatz_circuit_checks_param_values_length(self) -> None:
        with pytest.raises(ValueError, match="Expected 20 params"):
            make_ansatz_circuit(5, 2, "simple", param_values=[0.0] * 19)
        qasm, ap = make_ansatz_circuit(5, 2, "simple", param_values=[0.0] * 20)
        assert ap.n_params == 20
        assert qasm.startswith("OPENQASM 3.0;")

    def test_scaffold_rotation_syntax_is_rejected_by_the_reference_parser(self) -> None:
        # Documents WHY the QASM builders changed: `ry angle_0 q[0];` (no parentheses).
        openqasm3 = pytest.importorskip("openqasm3")
        header = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\n'
        with pytest.raises(openqasm3.parser.QASM3ParsingError):
            openqasm3.parse(header + "ry angle_0 q[0];")