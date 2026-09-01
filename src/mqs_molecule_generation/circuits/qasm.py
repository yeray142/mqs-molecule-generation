"""QASM serialization helpers for LatentStyleGAN circuits."""

from __future__ import annotations

from qpubench.schemas.circuit import CircuitSpec
from qpubench.schemas.primitives import CircuitFormat

from mqs_molecule_generation.circuits.ansatz import AnsatzParams


def make_circuit_spec(
    qasm: str,
    n_qubits: int,
    ansatz_params: AnsatzParams,
    observables: list[str] | None = None,
) -> CircuitSpec:
    """Create a CircuitSpec from QASM and ansatz parameters.

    Args:
        qasm: OpenQASM 3.0 string.
        n_qubits: Number of qubits.
        ansatz_params: AnsatzParams with verified counts.
        observables: Optional list of Pauli observables (e.g. ["Z0", "Z0 Z1"]).

    Returns:
        CircuitSpec ready for QPUBench.
    """
    from qpubench.schemas.observable import Pauli, SparsePauliObservable

    # Build parameter names for binding
    if ansatz_params.ansatz_type == "simple":
        param_names = [f"angle_{i}" for i in range(ansatz_params.n_angles)]
    else:
        param_names = [f"bel_angle_{i}" for i in range(ansatz_params.n_angles)]

    # Build observable specs if provided
    obs_list: list[SparsePauliObservable] = []
    if observables:
        for i, obs_str in enumerate(observables):
            pauli = Pauli(obs_str)
            sparse_obs = SparsePauliObservable(
                observable_index=i,
                pauli_terms=[pauli],
            )
            obs_list.append(sparse_obs)

    return CircuitSpec(
        num_qubits=n_qubits,
        format=CircuitFormat.QASM3,
        serialized=qasm,
        parameters=param_names,
        observables=obs_list,
    )
