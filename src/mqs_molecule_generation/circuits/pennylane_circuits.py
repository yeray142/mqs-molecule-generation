"""PennyLane realisation of the ansätze: gate list -> QNode.

Framework-neutral on purpose (no torch import): ``models.quantum_generator``
wraps the QNode built here in an ``nn.Module``, while tests and
``scripts/print_circuits.py`` use the very same QNode with the autograd
interface, so what gets printed is what gets trained.
"""

from __future__ import annotations

import pennylane as qml

from mqs_molecule_generation.circuits.ansatz import AnsatzParams, Gate
from mqs_molecule_generation.circuits.readout import readout_observables


def apply_ansatz(layout: tuple[Gate, ...], angles) -> None:
    """Queue the gates of ``layout``. ``angles`` has shape (n_angles,) or (batch, n_angles);
    a batch dimension becomes PennyLane parameter broadcasting.
    """
    for g in layout:
        if g.name == "RY":
            qml.RY(angles[..., g.param], wires=g.wires[0])
        elif g.name == "RZ":
            qml.RZ(angles[..., g.param], wires=g.wires[0])
        elif g.name == "CRY":
            qml.CRY(angles[..., g.param], wires=list(g.wires))
        elif g.name == "CX":
            qml.CNOT(wires=list(g.wires))
        else:
            raise ValueError(f"Unknown gate {g.name!r}")


def build_qnode(
    ansatz: AnsatzParams,
    readout: str,
    *,
    interface: str = "torch",
    diff_method: str = "backprop",
    device: str = "default.qubit",
):
    """QNode: angles (n_angles,) or (batch, n_angles) -> readout expectation values."""
    dev = qml.device(device, wires=ansatz.n_qb)
    layout = ansatz.layout
    observables = readout_observables(ansatz.n_qb, readout)

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(angles):
        apply_ansatz(layout, angles)
        return [qml.expval(o) for o in observables]

    return circuit


def stack_expvals(result):
    """QNode output (tuple of per-observable arrays) -> array of shape (..., D_l)."""
    if isinstance(result, (tuple, list)):
        return qml.math.stack(result, axis=-1)
    return result[..., None]