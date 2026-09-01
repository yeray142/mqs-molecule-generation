"""Discriminator circuit builders for LatentStyleGAN.

Readout dimensions (from arXiv:2603.22399):
  single: D_l = n_qb
  dual:   D_l = 2 * n_qb
"""

from __future__ import annotations

from mqs_molecule_generation.circuits.ansatz import AnsatzParams


def build_discriminator_circuit(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str,
    readout: str = "single",
) -> str:
    """Build discriminator circuit QASM.

    The discriminator takes the quantum state from the generator ansatz
    and measures it to produce a D_l-dimensional readout.

    Args:
        n_qubits: Number of qubits (n_qb).
        n_layers: Number of ansatz layers (n_l).
        ansatz_type: "simple" or "bel".
        readout: "single" (D_l = n_qb) or "dual" (D_l = 2*n_qb).

    Returns:
        OpenQASM 3.0 string for the discriminator circuit.

    Raises:
        ValueError: If readout is not "single" or "dual".
    """
    if readout not in ("single", "dual"):
        raise ValueError(f"Readout must be 'single' or 'dual', got {readout!r}")

    ap = AnsatzParams(n_qb=n_qubits, n_l=n_layers, ansatz_type=ansatz_type)

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{n_qubits}] q;",
        f"bit[{ap.readout_dim_single}] c;",  # single readout uses n_qb bits
    ]

    # For dual readout, we measure twice (hence 2*n_qb bits total)
    # The dual readout measures Z on all qubits, then again on all qubits

    if readout == "single":
        # Single readout: measure all n_qb qubits in Z basis
        # This gives D_l = n_qb outcomes
        for q in range(n_qubits):
            lines.append(f"measure q[{q}] -> c[{q}];")
    else:
        # Dual readout: measure all n_qb qubits twice
        # First set of measurements
        for q in range(n_qubits):
            lines.append(f"measure q[{q}] -> c[{q}];")
        # Second set of measurements (into additional bits)
        for q in range(n_qubits):
            lines.append(f"measure q[{q}] -> c[{n_qubits + q}];")

    return "\n".join(lines)


def get_discriminator_readout_dim(ansatz_params: AnsatzParams, readout: str) -> int:
    """Get the discriminator readout dimension.

    Args:
        ansatz_params: AnsatzParams instance.
        readout: "single" or "dual".

    Returns:
        D_l = n_qb (single) or D_l = 2*n_qb (dual).
    """
    if readout == "single":
        return ansatz_params.readout_dim_single
    elif readout == "dual":
        return ansatz_params.readout_dim_dual
    else:
        raise ValueError(f"Readout must be 'single' or 'dual', got {readout!r}")
