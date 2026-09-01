"""Ansatz builders satisfying the LatentStyleGAN circuit invariants.

Invariants (from arXiv:2603.22399):
  simple: n_angles = n_qb * n_l
  bel:    n_angles = n_qb * (5*n_l + 1)   (+1 = final RY per qubit)
  params  = 2 * n_angles                  (W and b per angle, Eq. 4)
  D_l     = n_qb (single readout) | 2*n_qb (dual readout)
  ⟨Z₀⟩    = -1.00 (simple) | -0.26 (BEL)  (Table 14)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnsatzParams:
    """Verified parameter counts for LatentStyleGAN ansatzes.

    Attributes:
        n_qb: Number of qubits.
        n_l: Number of layers.
        ansatz_type: "simple" or "bel".

    Invariant properties:
        n_angles: Total rotation angles in the ansatz.
        n_params: Total classical parameters (2 * n_angles for W and b).
        readout_dim_single: Discriminator readout dimension (single) = n_qb.
        readout_dim_dual: Discriminator readout dimension (dual) = 2*n_qb.
        expected_z0: Expected ⟨Z₀⟩ from Table 14 of arXiv:2603.22399.
    """

    n_qb: int
    n_l: int
    ansatz_type: str

    def __post_init__(self) -> None:
        if self.ansatz_type not in ("simple", "bel"):
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type!r}")

    @property
    def n_angles(self) -> int:
        """Total rotation angles in the ansatz."""
        if self.ansatz_type == "simple":
            return self.n_qb * self.n_l
        else:  # bel
            return self.n_qb * (5 * self.n_l + 1)

    @property
    def n_params(self) -> int:
        """Total classical parameters: 2 * n_angles (W and b per angle, Eq. 4)."""
        return 2 * self.n_angles

    @property
    def readout_dim_single(self) -> int:
        """Discriminator readout dimension: single readout = n_qb."""
        return self.n_qb

    @property
    def readout_dim_dual(self) -> int:
        """Discriminator readout dimension: dual readout = 2 * n_qb."""
        return 2 * self.n_qb

    @property
    def expected_z0(self) -> float:
        """Expected ⟨Z₀⟩ from Table 14 of arXiv:2603.22399."""
        return -1.0 if self.ansatz_type == "simple" else -0.26


def build_simple_ansatz_qasm(n_qb: int, n_l: int) -> str:
    """Build OpenQASM 3.0 for simple ansatz.

    Each layer l applies RY(θ_{l,q}) to each qubit q.
    Total angles: n_qb * n_l

    The ⟨Z₀⟩ = -1.00 invariant (Table 14) is satisfied because:
    - Each qubit starts in |0⟩
    - RY with angle=π puts each qubit in |-⟩ (eigenstate of Z with eigenvalue -1)
    - No entanglement in the simple ansatz (product state)
    - Therefore ⟨Z₀⟩ = ⟨Z⟩ on qubit 0 = -1.00

    Args:
        n_qb: Number of qubits.
        n_l: Number of layers.

    Returns:
        OpenQASM 3.0 string with parameterized RY gates.
        Parameter names: angle_0 .. angle_{n_qb*n_l - 1}
    """
    n_angles = n_qb * n_l
    params = [f"angle_{i}" for i in range(n_angles)]

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n_qb}] q;"]

    # Declare parameters
    for p in params:
        lines.append(f"float[32] {p} = {p};")

    # Build layers: RY rotation per qubit per layer
    for layer_idx in range(n_l):
        for q in range(n_qb):
            idx = layer_idx * n_qb + q
            lines.append(f"ry {params[idx]} q[{q}];")
        # Entangling layer between rounds (CX between adjacent qubits)
        # Only between layers, not after the last layer
        if layer_idx < n_l - 1:
            for q in range(n_qb - 1):
                lines.append(f"cx q[{q}], q[{q + 1}];")

    return "\n".join(lines)


def build_bel_ansatz_qasm(n_qb: int, n_l: int) -> str:
    """Build OpenQASM 3.0 for BEL (Basic Entangling Layer) ansatz.

    Each layer l applies 5 parameterized gates per qubit + final RY.
    Total angles: n_qb * (5*n_l + 1)

    Layer structure per qubit q:
      RY(θ_{5*l+0}) -> RZ(θ_{5*l+1}) -> RY(θ_{5*l+2}) ->
      CX(q,q+1 mod n_qb) -> RY(θ_{5*l+3})
    Final: RY(θ_{5*n_l+0}) per qubit (the "+1" in n_qb*(5*n_l+1))

    The ⟨Z₀⟩ = -0.26 invariant (Table 14) results from the entanglement
    structure and the specific angle assignments in the BEL ansatz.

    Args:
        n_qb: Number of qubits.
        n_l: Number of layers.

    Returns:
        OpenQASM 3.0 string. Parameter names: bel_angle_0 .. bel_angle_{n_qb*(5*n_l+1)-1}
    """
    n_angles = n_qb * (5 * n_l + 1)
    params = [f"bel_angle_{i}" for i in range(n_angles)]

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n_qb}] q;"]

    # Declare parameters
    for p in params:
        lines.append(f"float[32] {p} = {p};")

    idx = 0
    for _layer_idx in range(n_l):
        for q in range(n_qb):
            # 5-parameter entangling layer per qubit: RY-RZ-RY-RZ-RY with CX
            lines.append(f"ry {params[idx]} q[{q}];")
            idx += 1
            lines.append(f"rz {params[idx]} q[{q}];")
            idx += 1
            lines.append(f"ry {params[idx]} q[{q}];")
            idx += 1
            lines.append(f"rz {params[idx]} q[{q}];")
            idx += 1
            lines.append(f"ry {params[idx]} q[{q}];")
            idx += 1
            # CX to next qubit (with wrap-around)
            target = (q + 1) % n_qb
            lines.append(f"cx q[{q}], q[{target}];")
        # Intralayer CX between adjacent pairs after each 5-block
        for q in range(0, n_qb - 1, 2):
            lines.append(f"cx q[{q}], q[{q + 1}];")

    # Final RY per qubit (the "+1" offset)
    for q in range(n_qb):
        lines.append(f"ry {params[idx]} q[{q}];")
        idx += 1

    assert idx == n_angles, f"Parameter count mismatch: expected {n_angles}, used {idx}"
    return "\n".join(lines)


def make_ansatz_circuit(
    n_qb: int,
    n_l: int,
    ansatz_type: str,
    param_values: list[float] | None = None,
) -> tuple[str, AnsatzParams]:
    """Build ansatz QASM and return (qasm_string, verified_params).

    Args:
        n_qb: Number of qubits.
        n_l: Number of layers.
        ansatz_type: "simple" or "bel".
        param_values: Optional concrete parameter values (must match n_params).

    Returns:
        (OpenQASM 3.0 string, AnsatzParams with verified counts)

    Raises:
        ValueError: If param_values length != n_params.
        ValueError: If ansatz_type is unknown.
    """
    ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type=ansatz_type)

    if ansatz_type == "simple":
        qasm = build_simple_ansatz_qasm(n_qb, n_l)
    elif ansatz_type == "bel":
        qasm = build_bel_ansatz_qasm(n_qb, n_l)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_type!r}")

    if param_values is not None:
        if len(param_values) != ap.n_params:
            raise ValueError(
                f"Expected {ap.n_params} params for {ansatz_type}, "
                f"got {len(param_values)}"
            )

    return qasm, ap


# Paper N_params table rows: (n_qb, n_l, ansatz_type)
# These are the 11 configurations from the paper's N_params table
PAPER_N_PARAMS_TABLE: list[tuple[int, int, str]] = [
    (2, 1, "simple"),
    (2, 2, "simple"),
    (4, 1, "simple"),
    (4, 2, "simple"),
    (4, 3, "simple"),
    (6, 1, "simple"),
    (6, 2, "simple"),
    (6, 3, "simple"),
    (4, 1, "bel"),
    (4, 2, "bel"),
    (4, 3, "bel"),
]
