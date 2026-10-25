#!/usr/bin/env python
"""Print the quantum circuit(s) the QGAN generator uses.

    uv run python scripts/print_circuits.py            # simple + BEL, nominal 5q/2l/dual
    uv run python scripts/print_circuits.py --ansatz bel --n-qubits 10 --n-layers 4 --readout single
    uv run python scripts/print_circuits.py --paper-configs   # counts per config the paper names
    uv run python scripts/print_circuits.py --qasm            # also print the OpenQASM 3 export

The diagram is drawn from the SAME QNode that ``models.quantum_generator`` trains
(``pennylane_circuits.build_qnode``), not from a separate description, so what is
printed is what runs. Each circuit is printed with its provenance: which parts are
confirmed by the paper's text and which gate orders are still unverified against
the paper's figures. Needs PennyLane only (no torch).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pennylane as qml

from mqs_molecule_generation.circuits.angle_map import ANGLE_MODES
from mqs_molecule_generation.circuits.ansatz import AnsatzParams, layout_to_qasm
from mqs_molecule_generation.circuits.paper_configs import (
    CONFIGS_NAMED_IN_PAPER_TEXT,
    N_PARAMS_TABLE,
    N_PARAMS_TABLE_EXPECTED_ROWS,
    PAPER_STATED_N_PARAMS,
)
from mqs_molecule_generation.circuits.pennylane_circuits import build_qnode
from mqs_molecule_generation.circuits.readout import (
    READOUT_STATUS,
    readout_dim,
    readout_pauli_strings,
)

ANGLE_MAP_TEXT = {
    "literal": "theta = 2*pi*tanh(xi_q*W + b)    Eq. 4 as printed; range (-2pi, 2pi)",
    "shifted": "theta = pi*(1 + tanh(xi_q*W + b))  inside [0, 2pi]",
    "wrapped": "theta = (2*pi*tanh(xi_q*W + b)) mod 2pi  inside [0, 2pi]",
}


def describe(
    ansatz_type: str, n_qb: int, n_l: int, readout: str, angle_mode: str, qasm: bool
) -> None:
    ap = AnsatzParams(n_qb=n_qb, n_l=n_l, ansatz_type=ansatz_type)
    counts = Counter(g.name for g in ap.layout)
    gate_text = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
    obs = readout_pauli_strings(n_qb, readout)

    title = f"{ansatz_type.upper()} ansatz | n_qb={n_qb}, n_l={n_l} | readout={readout}"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(f"angles           : {ap.n_angles}")
    print(f"trainable params : {ap.n_params}   (a scalar W and b per angle, Eq. 4)")
    print(f"noise input      : {ap.noise_dim} components (xi_q, one per qubit)")
    print(f"output D_l       : {readout_dim(n_qb, readout)}   ({', '.join(obs)})")
    print(f"angle map        : {ANGLE_MAP_TEXT[angle_mode]}")
    print(f"gates            : {gate_text}  ({len(ap.layout)} total)")
    print(f"layout status    : {ap.layout_status}")
    print(f"readout status   : {READOUT_STATUS}")
    stated = PAPER_STATED_N_PARAMS.get((ansatz_type, n_qb, n_l))
    if stated is not None:
        verdict = "matches" if stated == ap.n_params else "DOES NOT MATCH"
        print(f"paper check      : paper states {stated} trainable parameters -> {verdict}")

    print("\ncircuit (as executed by the training QNode; angle values omitted):")
    circuit = build_qnode(ap, readout, interface="autograd")
    print(
        qml.draw(circuit, decimals=None, show_all_wires=True, max_length=110)(
            qml.numpy.zeros(ap.n_angles)
        )
    )

    if qasm:
        print("\nOpenQASM 3.0 export:")
        print(layout_to_qasm(n_qb, ap.layout, ap.param_names))
    print()


def paper_config_table() -> None:
    print("Configurations the paper's text names (see circuits/paper_configs.py for quotes)\n")
    header = (
        f"{'n_qb':>4} {'n_l':>3} {'readout':>7} {'D_l':>4} {'noise':>5} | "
        f"{'simple angles/params':>21} | {'bel angles/params':>18}"
    )
    print(header)
    print("-" * len(header))
    for c in CONFIGS_NAMED_IN_PAPER_TEXT:
        s = AnsatzParams(c.n_qb, c.n_l, "simple")
        b = AnsatzParams(c.n_qb, c.n_l, "bel")
        print(
            f"{c.n_qb:>4} {c.n_l:>3} {c.readout:>7} {c.latent_dim:>4} {c.n_qb:>5} | "
            f"{s.n_angles:>10}/{s.n_params:<10} | {b.n_angles:>8}/{b.n_params:<9}"
        )
    print(
        f"\nPaper's printed N_params table: {len(N_PARAMS_TABLE)}/"
        f"{N_PARAMS_TABLE_EXPECTED_ROWS} rows entered",
        end="",
    )
    print(
        " -- these counts come from the formulas, not yet checked against it."
        if not N_PARAMS_TABLE
        else "."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ansatz", choices=["simple", "bel", "both"], default="both")
    parser.add_argument("--n-qubits", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--readout", choices=["single", "dual"], default="dual")
    parser.add_argument("--angle-mode", choices=list(ANGLE_MODES), default="literal")
    parser.add_argument("--qasm", action="store_true", help="also print the OpenQASM 3 export")
    parser.add_argument(
        "--paper-configs", action="store_true", help="print the count table instead"
    )
    args = parser.parse_args(argv)

    if args.paper_configs:
        paper_config_table()
        return 0

    kinds = ["simple", "bel"] if args.ansatz == "both" else [args.ansatz]
    for kind in kinds:
        describe(kind, args.n_qubits, args.n_layers, args.readout, args.angle_mode, args.qasm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())