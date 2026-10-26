"""The QNode: shapes, physics sanity, and 1:1 correspondence with the gate list."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from mqs_molecule_generation.circuits.ansatz import AnsatzParams
from mqs_molecule_generation.circuits.pennylane_circuits import (
    apply_ansatz,
    build_qnode,
    stack_expvals,
)
from mqs_molecule_generation.circuits.readout import readout_dim

CONFIGS = [
    (k, nq, nl, ro)
    for k in ("simple", "bel")
    for (nq, nl, ro) in [(2, 1, "single"), (3, 2, "dual"), (5, 2, "dual"), (6, 3, "single")]
]


@pytest.mark.parametrize(("kind", "n_qb", "n_l", "readout"), CONFIGS)
class TestQNode:
    def test_output_dimension_is_the_readout_dimension(self, kind, n_qb, n_l, readout) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        qn = build_qnode(ap, readout, interface="autograd")
        out = stack_expvals(qn(pnp.zeros(ap.n_angles)))
        assert out.shape == (readout_dim(n_qb, readout),)

    def test_batch_broadcasting_equals_per_sample_evaluation(
        self, kind, n_qb, n_l, readout
    ) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        qn = build_qnode(ap, readout, interface="autograd")
        angles = pnp.array(np.random.default_rng(0).uniform(-6, 6, (4, ap.n_angles)))
        batched = stack_expvals(qn(angles))
        assert batched.shape == (4, readout_dim(n_qb, readout))
        for i in range(4):
            np.testing.assert_allclose(batched[i], stack_expvals(qn(angles[i])), atol=1e-10)

    def test_expectation_values_are_bounded(self, kind, n_qb, n_l, readout) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        qn = build_qnode(ap, readout, interface="autograd")
        angles = pnp.array(np.random.default_rng(1).uniform(-7, 7, (16, ap.n_angles)))
        out = np.asarray(stack_expvals(qn(angles)))
        assert np.all(np.abs(out) <= 1 + 1e-9)

    def test_zero_angles_give_the_state_the_fixed_gates_alone_produce(
        self, kind, n_qb, n_l, readout
    ) -> None:
        """All angles 0 makes every rotation the identity, leaving only the fixed gates.

        simple: CX on |0...0> is still |0...0>, so <Z> = +1 and <X> = 0.
        BEL: its leading Hadamard layer (Fig. 3) gives |+>^n, so <Z> = 0 and <X> = +1.
        """
        ap = AnsatzParams(n_qb, n_l, kind)
        qn = build_qnode(ap, readout, interface="autograd")
        out = np.asarray(stack_expvals(qn(pnp.zeros(ap.n_angles))))
        z_expected, x_expected = (0.0, 1.0) if kind == "bel" else (1.0, 0.0)
        np.testing.assert_allclose(out[:n_qb], z_expected, atol=1e-12)
        np.testing.assert_allclose(out[n_qb:], x_expected, atol=1e-12)

    def test_legacy_layout_still_starts_from_all_zero(
        self, kind, n_qb, n_l, readout
    ) -> None:
        """figure_layout=False has no Hadamard layer, so |0...0> survives zero angles."""
        ap = AnsatzParams(n_qb, n_l, kind, figure_layout=False)
        qn = build_qnode(ap, readout, interface="autograd")
        out = np.asarray(stack_expvals(qn(pnp.zeros(ap.n_angles))))
        np.testing.assert_allclose(out[:n_qb], 1.0, atol=1e-12)
        np.testing.assert_allclose(out[n_qb:], 0.0, atol=1e-12)

    def test_gradients_reach_every_angle(self, kind, n_qb, n_l, readout) -> None:
        ap = AnsatzParams(n_qb, n_l, kind)
        qn = build_qnode(ap, readout, interface="autograd")
        angles = pnp.array(
            np.random.default_rng(2).uniform(0.3, 2.8, ap.n_angles), requires_grad=True
        )
        jac = np.asarray(qml.jacobian(lambda a: stack_expvals(qn(a)))(angles))
        assert np.isfinite(jac).all()
        assert (np.abs(jac).max(axis=0) > 1e-9).all(), "some angle has no influence on any output"


@pytest.mark.parametrize("kind", ["simple", "bel"])
def test_queued_gates_match_the_gate_list_one_to_one(kind) -> None:
    ap = AnsatzParams(4, 2, kind)
    angles = np.arange(1, ap.n_angles + 1, dtype=float) / 10.0
    ops = qml.tape.make_qscript(apply_ansatz)(ap.layout, angles).operations
    to_ir = {"RY": "RY", "RZ": "RZ", "CRY": "CRY", "CNOT": "CX", "Hadamard": "H"}
    assert [to_ir[o.name] for o in ops] == [g.name for g in ap.layout]
    for op, gate in zip(ops, ap.layout, strict=True):
        assert list(op.wires) == list(gate.wires)
        if gate.param is not None:
            assert float(op.data[0]) == pytest.approx(angles[gate.param])