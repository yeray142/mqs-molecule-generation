from qpubench import BenchmarkRunner, CircuitSpec, ExecutionOptions, Pauli

# What to benchmark: a 3-qubit GHZ circuit with TWO observables
ghz_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];"""

zzz = Pauli("Z0 Z1 Z2")
xxx = Pauli("X0 X1 X2")

circuit = CircuitSpec(num_qubits=3, serialized=ghz_qasm, observables=[zzz, xxx])

# How to run it, and where results go.  A plain string path becomes an
# append-only NDJSONStore; no pathlib, no store import needed.
runner = BenchmarkRunner(store="results/ghz.ndjson")
runner.register(name="stub", seed=42)

# An explicit ExecutionOptions exposes settings runner.run(..., shots=) hides:
# a reproducibility seed and the transpiler optimization tier (0–3).
options = ExecutionOptions(shots=4096, seed=7, optimization_level=2)

record = runner.run(
    circuit, "stub", options,
    tags=["tutorial", "ghz"],          # queryable labels on the record
    notes="first GHZ benchmark",       # free-text provenance
)

for ev in record.result.expectation_values:
    label = "".join(p.name for p in circuit.observables[ev.observable_index].terms[0].pauli_ops)
    print(f"<{label}> = {ev.value:.4f} ± {ev.std_error:.4f}")
print("created at:", record.timestamp)          # UTC timestamp, auto-stamped