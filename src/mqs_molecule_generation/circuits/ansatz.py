"""Ansatz definitions for the LatentStyleGAN quantum generator.

Each ansatz is defined ONCE, as a gate list (``Gate`` records). The PennyLane
QNode (``pennylane_circuits``), the OpenQASM export below, the parameter
counts and the printed circuit (``scripts/print_circuits.py``) are all derived
from that single definition, so they cannot drift apart.

Invariants (arXiv:2603.22399):
  simple: n_angles = n_qb * n_l
  bel:    n_angles = n_qb * (5*n_l + 1)   (+1 = a final RY per qubit)
  params  = 2 * n_angles                  (a scalar W and b per angle, Eq. 4)
  noise   = n_qb components xi_q          (Eq. 4 indexes the noise by qubit only)
  D_l     = n_qb (single readout) | 2*n_qb (dual readout)
  check   : nominal BEL, n_qb=5, n_l=2 -> 2*5*(5*2+1) = 110 trainable parameters,
            the figure the paper quotes against the classical GAN's 705,162.

NOT an invariant of the circuit: the "<Z_0> = -1.00 (simple) / -0.26 (BEL)"
of Table 14. That <Z_0> is the paper's average *significance score* (Eq. 5,
built from per-metric Z_0 scores comparing a QGAN against the classical
reference). It is a result of training plus evaluation; it has nothing to do
with the Pauli-Z expectation value on qubit 0, and no circuit property makes
it come out that way.

Layout provenance -- read ``LAYOUT_STATUS`` before trusting a gate order. The
paper defines the two circuits in its Figures 2-3, which are images; only the
angle counts above are confirmed from its text.
"""

from __future__ import annotations

from dataclasses import dataclass

LAYOUT_STATUS: dict[str, str] = {
    "simple": (
        "PARTIALLY VERIFIED against Fig. 2 (read directly from the PDF). Fig. 2 draws, per "
        "qubit for its single layer: RY RZ RY RZ, then a CNOT staircase CX(0,1)...CX(n-2,n-1). "
        "The staircase placement is now reproduced (after every layer). The FOUR rotations per "
        "qubit per layer are NOT reproduced and cannot be: with Eq. 4's two trainable numbers "
        "(W and b) per angle they would give 8*n_qb*n_l parameters, i.e. 80 for the nominal "
        "n_qb=5, n_l=2, whereas every parameter count the paper prints for the simple ansatz "
        "(20, 40, 40, 80, 40, 60 across Tables 4, 5 and the supplementary) equals 2*n_qb*n_l, "
        "which admits exactly ONE parameterised rotation per qubit per layer. Figure 2 and the "
        "paper's own parameter counts are mutually inconsistent; this implementation follows "
        "the counts, since those are what the 'far fewer trainable parameters' claim rests on."
    ),
    "bel": (
        "PARTIALLY VERIFIED against Fig. 3 (read directly from the PDF). Fig. 3 draws: H on "
        "every qubit; then per layer RY RZ, an entangling block, RY RZ, a second entangling "
        "block; plus (per the caption) a final RY per qubit. The leading Hadamard layer is now "
        "reproduced. Two differences remain, both forced by the parameter counts: the figure's "
        "entanglers are drawn as unparameterised controlled-Z (filled dot at BOTH ends, no "
        "angle label), in a ladder (0,1),(1,2),(2,3),(3,4) and then a skip pattern "
        "(0,2),(1,3),(2,4); this implementation instead uses ONE ring of n_qb parameterised "
        "controlled-RY per layer. With unparameterised entanglers the figure gives "
        "n_qb*(4*n_l+1) angles = 90 parameters for the nominal n_qb=5, n_l=2, but every count "
        "the paper prints for BEL (110, 210, 220, 420, 620, 330) equals 2*n_qb*(5*n_l+1), "
        "requiring a fifth parameterised angle per qubit per layer. A ring of n_qb controlled "
        "rotations supplies exactly that, and the paper's text does say the controlled rotation "
        "gates are parameterised, so this is the closest structure that satisfies both."
    ),
}


@dataclass(frozen=True)
class Gate:
    """One gate of an ansatz.

    Attributes:
        name: "RY", "RZ", "CRY" (parameterised) or "CX", "H" (fixed).
        wires: Qubits acted on; for CRY/CX it is (control, target).
        param: Index into the angle vector, or None for fixed gates.
        noise_qubit: The q of xi_q in Eq. 4 that feeds this angle (the qubit
            "owning" the gate; the control qubit for CRY), or None for fixed gates.
    """

    name: str
    wires: tuple[int, ...]
    param: int | None = None
    noise_qubit: int | None = None


def simple_layout(
    n_qb: int, n_l: int, *, entangle_after_last_layer: bool = True
) -> tuple[Gate, ...]:
    """RY(theta_{l,q}) on every qubit each layer, then a CX staircase.

    Figure 2 (five qubits, ONE layer) draws the CNOT staircase
    CX(0,1), CX(1,2), ..., CX(n-2,n-1) AFTER the rotations of that single layer, so a
    one-layer simple circuit is entangled. ``entangle_after_last_layer=False`` restores
    the earlier behaviour (staircase only BETWEEN layers), which leaves n_l=1 a product
    state -- kept only to reproduce runs made before this was corrected.

    Figure 2 also draws FOUR rotations per qubit per layer (RY RZ RY RZ), which is
    irreconcilable with the paper's own parameter counts; see LAYOUT_STATUS.
    """
    gates: list[Gate] = []
    for layer in range(n_l):
        for q in range(n_qb):
            gates.append(Gate("RY", (q,), param=layer * n_qb + q, noise_qubit=q))
        if entangle_after_last_layer or layer < n_l - 1:
            gates.extend(Gate("CX", (q, q + 1)) for q in range(n_qb - 1))
    return tuple(gates)


def bel_layout(n_qb: int, n_l: int, *, initial_hadamard: bool = True) -> tuple[Gate, ...]:
    """BEL: H on every qubit, then [RY RZ RY RZ per qubit, a CRY ring] x n_l, final RY.

    The leading Hadamard layer is drawn explicitly in Figure 3 (one H on every qubit,
    before any rotation). It carries no parameters, so it does not affect N_params, but
    it does change the state the rotations act on. ``initial_hadamard=False`` restores
    the earlier behaviour, kept only to reproduce runs made before this was corrected.

    Angle index = (layer*n_qb + q)*5 + k for slot k in 0..4 of the layered part
    (k=4 is the controlled rotation), and 5*n_l*n_qb + q for the final RY.
    """
    if n_qb < 2:
        raise ValueError("BEL needs n_qb >= 2 (its controlled rotations form a ring)")

    def idx(layer: int, q: int, k: int) -> int:
        return (layer * n_qb + q) * 5 + k

    gates: list[Gate] = []
    if initial_hadamard:
        gates.extend(Gate("H", (q,)) for q in range(n_qb))
    for layer in range(n_l):
        for q in range(n_qb):
            gates.append(Gate("RY", (q,), idx(layer, q, 0), q))
            gates.append(Gate("RZ", (q,), idx(layer, q, 1), q))
            gates.append(Gate("RY", (q,), idx(layer, q, 2), q))
            gates.append(Gate("RZ", (q,), idx(layer, q, 3), q))
        for q in range(n_qb):
            gates.append(Gate("CRY", (q, (q + 1) % n_qb), idx(layer, q, 4), q))
    final_base = 5 * n_l * n_qb
    gates.extend(Gate("RY", (q,), final_base + q, q) for q in range(n_qb))
    return tuple(gates)


@dataclass(frozen=True)
class AnsatzParams:
    """Verified counts and the gate list for one ansatz configuration.

    Attributes:
        n_qb: Number of qubits.
        n_l: Number of layers.
        ansatz_type: "simple" or "bel".
    """

    n_qb: int
    n_l: int
    ansatz_type: str
    #: Figure-faithful structure (see simple_layout/bel_layout). False reproduces the
    #: pre-audit layouts: no leading H on BEL, no CX after the simple ansatz's last layer.
    figure_layout: bool = True

    def __post_init__(self) -> None:
        if self.ansatz_type not in ("simple", "bel"):
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type!r}")
        if self.n_qb < 1 or self.n_l < 1:
            raise ValueError(f"n_qb and n_l must be >= 1, got {self.n_qb}, {self.n_l}")

    @property
    def n_angles(self) -> int:
        """Total rotation angles in the ansatz."""
        if self.ansatz_type == "simple":
            return self.n_qb * self.n_l
        return self.n_qb * (5 * self.n_l + 1)

    @property
    def n_params(self) -> int:
        """Total trainable parameters: 2 * n_angles (scalar W and b per angle, Eq. 4)."""
        return 2 * self.n_angles

    @property
    def noise_dim(self) -> int:
        """Noise components the generator consumes: one xi_q per qubit (Eq. 4)."""
        return self.n_qb

    @property
    def readout_dim_single(self) -> int:
        """Generator output dimension D_l with single readout = n_qb."""
        return self.n_qb

    @property
    def readout_dim_dual(self) -> int:
        """Generator output dimension D_l with dual readout = 2 * n_qb."""
        return 2 * self.n_qb

    @property
    def expected_z0(self) -> float:
        """The paper's reported average significance <Z_0> (Table 14), for comparison only.

        A training/evaluation RESULT, not a property of the circuit -- see the
        module docstring. Nothing here derives or enforces it.
        """
        return -1.0 if self.ansatz_type == "simple" else -0.26

    @property
    def layout_status(self) -> str:
        return LAYOUT_STATUS[self.ansatz_type]

    @property
    def layout(self) -> tuple[Gate, ...]:
        """The gate list, checked on every call against the paper's angle count."""
        gates = (
            simple_layout(
                self.n_qb, self.n_l, entangle_after_last_layer=self.figure_layout
            )
            if self.ansatz_type == "simple"
            else bel_layout(self.n_qb, self.n_l, initial_hadamard=self.figure_layout)
        )
        used = sorted(g.param for g in gates if g.param is not None)
        if used != list(range(self.n_angles)):
            raise AssertionError(
                f"{self.ansatz_type} layout uses angle indices {used[:5]}... "
                f"but the paper's count is {self.n_angles} (0..{self.n_angles - 1}, each once)"
            )
        return gates

    @property
    def noise_qubit_per_angle(self) -> tuple[int, ...]:
        """For each angle index, which noise component xi_q feeds it (Eq. 4's q)."""
        owner = [-1] * self.n_angles
        for g in self.layout:
            if g.param is not None:
                owner[g.param] = g.noise_qubit
        return tuple(owner)

    @property
    def param_names(self) -> list[str]:
        """Parameter names used for QASM / QPUBench binding (unchanged from the scaffold)."""
        prefix = "angle_" if self.ansatz_type == "simple" else "bel_angle_"
        return [f"{prefix}{i}" for i in range(self.n_angles)]


def layout_to_qasm(n_qb: int, layout: tuple[Gate, ...], param_names: list[str]) -> str:
    """OpenQASM 3.0 for a gate list.

    Angles are declared ``input float[32]`` (the OpenQASM 3 form for runtime-bound
    parameters) and passed to gates in parentheses -- ``ry(angle_0) q[0];``. The
    scaffold wrote ``ry angle_0 q[0];``, which the reference OpenQASM 3 parser
    rejects (checked), so none of its rotation lines were valid.
    """
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n_qb}] q;"]
    lines += [f"input float[32] {name};" for name in param_names]
    for g in layout:
        if g.name in ("RY", "RZ"):
            lines.append(f"{g.name.lower()}({param_names[g.param]}) q[{g.wires[0]}];")
        elif g.name == "CRY":
            lines.append(f"cry({param_names[g.param]}) q[{g.wires[0]}], q[{g.wires[1]}];")
        elif g.name == "CX":
            lines.append(f"cx q[{g.wires[0]}], q[{g.wires[1]}];")
        elif g.name == "H":
            lines.append(f"h q[{g.wires[0]}];")
        else:
            raise ValueError(f"Unknown gate {g.name!r}")
    return "\n".join(lines)


def build_simple_ansatz_qasm(n_qb: int, n_l: int) -> str:
    """OpenQASM 3.0 for the simple ansatz. Parameters: angle_0 .. angle_{n_qb*n_l - 1}."""
    ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type="simple")
    return layout_to_qasm(n_qb, ap.layout, ap.param_names)


def build_bel_ansatz_qasm(n_qb: int, n_l: int) -> str:
    """OpenQASM 3.0 for the BEL ansatz. Parameters: bel_angle_0 .. bel_angle_{n_angles - 1}."""
    ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type="bel")
    return layout_to_qasm(n_qb, ap.layout, ap.param_names)


def make_ansatz_circuit(
    n_qb: int,
    n_l: int,
    ansatz_type: str,
    param_values: list[float] | None = None,
) -> tuple[str, AnsatzParams]:
    """Build ansatz QASM and return (qasm_string, verified_params).

    Raises:
        ValueError: If ansatz_type is unknown, or param_values length != n_params.
    """
    ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type=ansatz_type)
    qasm = layout_to_qasm(n_qb, ap.layout, ap.param_names)
    if param_values is not None and len(param_values) != ap.n_params:
        raise ValueError(
            f"Expected {ap.n_params} params for {ansatz_type}, got {len(param_values)}"
        )
    return qasm, ap


# Rows carried over from the original scaffold. These are NOT the paper's grid
# (the paper's text names n_qb in {5, 10, 15}, never 2/4/6) and are kept only so
# existing imports keep working. The paper's configurations live in
# ``paper_configs.py``.
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