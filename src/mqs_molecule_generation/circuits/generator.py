"""Generator circuit builder for LatentStyleGAN."""

from __future__ import annotations

from mqs_molecule_generation.circuits.ansatz import (
    AnsatzParams,
    build_bel_ansatz_qasm,
    build_simple_ansatz_qasm,
)


def build_generator_circuit(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str,
) -> tuple[str, AnsatzParams]:
    """Build generator circuit QASM using the specified ansatz.

    The generator uses the ansatz (simple or BEL) to produce a quantum state
    that represents a molecular configuration. The latent style input
    is encoded in the circuit parameters.

    Args:
        n_qubits: Number of qubits (n_qb).
        n_layers: Number of ansatz layers (n_l).
        ansatz_type: "simple" or "bel".

    Returns:
        Tuple of (OpenQASM 3.0 string, AnsatzParams).
    """
    if ansatz_type == "simple":
        qasm = build_simple_ansatz_qasm(n_qubits, n_layers)
    elif ansatz_type == "bel":
        qasm = build_bel_ansatz_qasm(n_qubits, n_layers)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type!r}")

    ap = AnsatzParams(n_qb=n_qubits, n_l=n_layers, ansatz_type=ansatz_type)
    return qasm, ap


def build_full_generator_discriminator_circuit(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str,
    readout: str = "single",
) -> tuple[str, str, AnsatzParams]:
    """Build both generator and discriminator circuits.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of layers.
        ansatz_type: "simple" or "bel".
        readout: "single" or "dual".

    Returns:
        Tuple of (generator_qasm, discriminator_qasm, ansatz_params).
    """
    from mqs_molecule_generation.circuits.discriminator import (
        build_discriminator_circuit,
    )

    gen_qasm, ap = build_generator_circuit(n_qubits, n_layers, ansatz_type)
    disc_qasm = build_discriminator_circuit(n_qubits, n_layers, ansatz_type, readout)
    return gen_qasm, disc_qasm, ap
