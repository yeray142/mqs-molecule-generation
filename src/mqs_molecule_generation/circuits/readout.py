"""Readout observables: how the circuit's output becomes the D_l-dimensional latent vector.

    single:  <Z_q> for each qubit            -> D_l = n_qb
    dual:    <Z_q> and <X_q> for each qubit  -> D_l = 2 * n_qb   (Z block first, then X block)

D_l is the generator's OUTPUT dimension (the VAE latent size), so this is the
generator's readout stage. The paper's discriminator is classical (the same
[512, 256, 1] network as in the classical runs); the scaffold's
``circuits/discriminator.py`` is misnamed in that respect.

Provenance: the dimension rule (n_qb / 2*n_qb) is from the paper. The choice of
Z for single and Z+X for dual is inferred from the authors' earlier style-based
qGAN papers, which read out <sigma_x> and <sigma_z> per qubit; it has not been
confirmed against this paper's text. Any fixed ordering of the D_l outputs is
equivalent for training, since the GAN learns against whatever order is used.
"""

from __future__ import annotations

READOUTS = ("single", "dual")

READOUT_STATUS = (
    "CONFIRMED against the paper (Sec. 2.2.2). Single readout: 'one Pauli operator (usually "
    "chosen as the Z operator) is measured for each qubit, so that a quantum generator with "
    "n_qb qubits produces a latent vector of dimension n_qb'. Dual readout: 'we exploit the "
    "non-commutativity of Pauli X and Pauli Z operators to use the same quantum circuit but "
    "performing two independent measurements for each qubit', giving dimension 2*n_qb. The "
    "ORDER of the 2*n_qb components (all Z, then all X) is this project's choice; the paper "
    "does not state one, and any fixed order is equivalent up to a permutation of the latent "
    "coordinates, which the VAE decoder sees as a relabelling."
)


def _check(readout: str) -> None:
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")


def readout_dim(n_qubits: int, readout: str) -> int:
    """D_l: n_qb for single readout, 2*n_qb for dual."""
    _check(readout)
    return n_qubits if readout == "single" else 2 * n_qubits


def readout_paulis(n_qubits: int, readout: str) -> tuple[tuple[str, int], ...]:
    """(pauli, qubit) pairs in output order."""
    _check(readout)
    zs = tuple(("Z", q) for q in range(n_qubits))
    if readout == "single":
        return zs
    return zs + tuple(("X", q) for q in range(n_qubits))


def readout_pauli_strings(n_qubits: int, readout: str) -> list[str]:
    """Pauli strings like ["Z0", ..., "X0", ...], the format ``make_circuit_spec`` takes."""
    return [f"{p}{q}" for p, q in readout_paulis(n_qubits, readout)]


def readout_observables(n_qubits: int, readout: str) -> list:
    """PennyLane observables in output order."""
    import pennylane as qml

    ops = {"Z": qml.PauliZ, "X": qml.PauliX}
    return [ops[p](q) for p, q in readout_paulis(n_qubits, readout)]