"""PennyLane BackendAdapter for LatentStyleGAN circuits.

SDK imports are INSIDE methods (deferred imports pattern).
Recoverable failures return QuantumResult(status=FAILED, error_message=...).
Unit-testable: PennyLane mocked out via pytest-mock.
"""

from __future__ import annotations

from typing import Any

from qpubench.schemas.backend import BackendSpec
from qpubench.schemas.circuit import CircuitSpec
from qpubench.schemas.execution import ExecutionOptions
from qpubench.schemas.primitives import CircuitFormat, ComputingModel, JobStatus
from qpubench.schemas.result import ExpectationResult, QuantumResult, ShotResult


class LatentStylePennyLaneAdapter:
    """PennyLane lightning.qubit adapter for LatentStyleGAN circuits.

    Implements BackendAdapter (spec / validate / run).

    Deferred imports: pennylane and pennylane-qiskit are imported INSIDE run()
    so the package can be installed without them (core install).

    Recoverable errors: returns QuantumResult(status=FAILED, error_message=...)
    rather than raising.
    """

    def __init__(
        self,
        n_qubits: int | None = None,
        ansatz_type: str = "simple",
        n_layers: int = 2,
    ) -> None:
        self._n_qubits = n_qubits
        self._ansatz_type = ansatz_type
        self._n_layers = n_layers
        self._spec = BackendSpec(
            name="lightning.qubit",
            provider="lightning",
            description="PennyLane lightning.qubit for LatentStyleGAN",
            num_qubits=n_qubits,
        )

    @property
    def spec(self) -> BackendSpec:
        return self._spec

    def validate(self, circuit: CircuitSpec) -> list[str]:
        """Validate the circuit specification.

        Returns warnings for soft errors. Raises ValueError for hard errors.
        """
        warnings: list[str] = []
        if circuit.computing_model != ComputingModel.GATE_BASED:
            warnings.append(
                f"LatentStylePennyLaneAdapter expects GATE_BASED; "
                f"got {circuit.computing_model}"
            )
        if circuit.format not in (CircuitFormat.QASM3, CircuitFormat.QASM2):
            warnings.append(
                f"LatentStylePennyLaneAdapter supports QASM3 or QASM2; "
                f"got {circuit.format}"
            )
        if circuit.is_parametric() and not circuit.is_bound():
            warnings.append(
                "Circuit has unbound parameters; bind before running"
            )
        return warnings

    def run(
        self,
        circuit: CircuitSpec,
        options: ExecutionOptions,
    ) -> QuantumResult:
        """Execute LatentStyleGAN circuit on lightning.qubit.

        Returns:
            QuantumResult with expectation_values (estimator path) or
            shots (sampler path).

        Recoverable errors return FAILED status.
        """
        # Deferred import - core install does not require PennyLane
        try:
            import pennylane as qml
        except ImportError:
            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                status=JobStatus.FAILED,
                error_message=(
                    "PennyLane not installed. "
                    "Install with: uv sync --extra pennylane"
                ),
            )

        try:
            import importlib.util

            try:
                spec = importlib.util.find_spec("pennylane_qiskit")
            except ValueError:
                spec = None
            if spec is None or spec.name is None:
                raise ImportError("pennylane_qiskit not found")
        except ImportError:
            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                status=JobStatus.FAILED,
                error_message="pennylane-qiskit not installed for QASM import",
            )

        try:
            from qiskit import QuantumCircuit as QiskitQC
        except ImportError:
            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                status=JobStatus.FAILED,
                error_message="qiskit not installed",
            )

        # Parse QASM to Qiskit circuit
        try:
            if circuit.format == CircuitFormat.QASM3:
                qiskit_circuit = QiskitQC.from_qasm_str(circuit.serialized or "")
            else:
                qiskit_circuit = QiskitQC.from_qasm_str(circuit.serialized or "")
        except Exception as exc:
            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                status=JobStatus.FAILED,
                error_message=f"Failed to parse QASM: {exc}",
            )

        # Convert to PennyLane quantum function
        try:
            quantum_fn = qml.from_qiskit(qiskit_circuit)
        except Exception as exc:
            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                status=JobStatus.FAILED,
                error_message=f"Failed to convert to PennyLane: {exc}",
            )

        dev = qml.device("lightning.qubit", wires=circuit.num_qubits)

        if circuit.observables:
            # ---- Estimator path ----
            try:
                observables = [
                    obs.to_pennylane_observable() for obs in circuit.observables
                ]
            except Exception as exc:
                return QuantumResult(
                    computing_model=ComputingModel.GATE_BASED,
                    status=JobStatus.FAILED,
                    error_message=f"Failed to convert observables: {exc}",
                )

            @qml.qnode(dev)
            def estimator_qnode() -> tuple[list[Any], list[Any]]:
                quantum_fn(wires=range(circuit.num_qubits))
                return (
                    [qml.expval(obs) for obs in observables],
                    [qml.var(obs) for obs in observables],
                )

            raw_values, raw_variances = estimator_qnode()

            shots = options.shots or 1
            evs = [
                ExpectationResult(
                    observable_index=i,
                    value=float(value),
                    std_error=(
                        float((variance / shots) ** 0.5) if shots > 0 else 0.0
                    ),
                    num_shots=shots,
                )
                for i, (value, variance) in enumerate(zip(raw_values, raw_variances, strict=True))
            ]

            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                expectation_values=evs,
                status=JobStatus.SUCCEEDED,
            )

        else:
            # ---- Sampler path ----
            try:
                shots = options.require_shots("LatentStylePennyLaneAdapter")
            except ValueError as exc:
                return QuantumResult(
                    computing_model=ComputingModel.GATE_BASED,
                    status=JobStatus.FAILED,
                    error_message=str(exc),
                )

            @qml.qnode(dev)
            def counts_fn() -> Any:
                quantum_fn(wires=range(circuit.num_qubits))
                return qml.counts()

            counts = counts_fn()
            counts_dict = {str(k): int(v) for k, v in counts.items()}

            return QuantumResult(
                computing_model=ComputingModel.GATE_BASED,
                shots=ShotResult(
                    num_qubits=circuit.num_qubits,
                    num_shots=shots,
                    counts=counts_dict,
                ),
                status=JobStatus.SUCCEEDED,
            )
