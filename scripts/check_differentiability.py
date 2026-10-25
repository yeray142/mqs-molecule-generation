#!/usr/bin/env python
"""Phase 3 step 13: print a differentiability report for the quantum generator.

    uv run python scripts/check_differentiability.py
    uv run python scripts/check_differentiability.py --paper-configs
    uv run python scripts/check_differentiability.py --ansatz bel --n-qubits 5 --n-layers 2

For each configuration this reports, in float64:
  * n_dead      parameters with |grad| < 1e-10 (should be 0)
  * |grad|      mean / min / max of the probe-loss gradient over the 2*n_angles parameters
  * bp-vs-ps    max |backprop - parameter-shift| / max|grad|
  * bp-vs-fd    max |backprop - central finite difference| / max|grad|
  * batch       max |batched - per-sample loop| for outputs and gradients

Exit code 1 if any configuration fails the gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mqs_molecule_generation.circuits.angle_map import ANGLE_MODES
from mqs_molecule_generation.circuits.paper_configs import CONFIGS_NAMED_IN_PAPER_TEXT
from mqs_molecule_generation.models.grad_check import (
    batch_consistency,
    compare_gradients,
    gradient_report,
)
from mqs_molecule_generation.models.quantum_generator import QuantumGenerator

TOL_SHIFT = 1e-7
TOL_FD = 1e-5
TOL_BATCH = 1e-10


def check_one(
    ansatz: str,
    n_qb: int,
    n_l: int,
    readout: str,
    mode: str,
    seed: int,
    batch: int,
    exact_max_qubits: int,
):
    """The shift / finite-difference references cost O(n_params) extra simulations each, so
    above ``exact_max_qubits`` they are skipped (``cmp`` is None) and only the finite/dead
    and batch-consistency checks are applied.
    """
    torch.manual_seed(seed)
    gen = QuantumGenerator(n_qb, n_l, ansatz, readout=readout, angle_mode=mode).double()
    xi = torch.rand(batch, gen.noise_dim, dtype=torch.float64) * 2 - 1
    rep = gradient_report(gen, xi)
    cmp = compare_gradients(gen, xi) if n_qb <= exact_max_qubits else None
    d_out, d_grad = batch_consistency(gen, xi)
    ok = (
        rep.all_finite
        and rep.n_dead == 0
        and d_out < TOL_BATCH
        and d_grad < TOL_BATCH
        and (
            cmp is None
            or (cmp.rel_backprop_vs_shift < TOL_SHIFT and cmp.rel_backprop_vs_fd < TOL_FD)
        )
    )
    return gen, rep, cmp, d_out, d_grad, ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ansatz", choices=["simple", "bel", "both"], default="both")
    p.add_argument("--n-qubits", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--readout", choices=["single", "dual", "both"], default="both")
    p.add_argument("--angle-mode", choices=list(ANGLE_MODES), default="literal")
    p.add_argument(
        "--paper-configs", action="store_true", help="the configs named in the paper text"
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exact-max-qubits",
        type=int,
        default=6,
        help="run the parameter-shift / finite-difference references only up to this many "
        "qubits (they cost O(n_params) extra simulations); larger circuits get the "
        "finite/dead + batch-consistency checks only",
    )
    args = p.parse_args(argv)

    if args.paper_configs:
        cases = [
            (a, c.n_qb, c.n_l, c.readout)
            for c in CONFIGS_NAMED_IN_PAPER_TEXT
            for a in ("simple", "bel")
        ]
    else:
        ansatze = ["simple", "bel"] if args.ansatz == "both" else [args.ansatz]
        readouts = ["single", "dual"] if args.readout == "both" else [args.readout]
        cases = [(a, args.n_qubits, args.n_layers, r) for a in ansatze for r in readouts]

    header = (
        f"{'ansatz':<7}{'nq':>3}{'nl':>3} {'readout':<7}{'params':>7}{'n_dead':>7}"
        f"{'mean|g|':>10}{'min|g|':>10}{'max|g|':>10}{'bp-vs-ps':>10}{'bp-vs-fd':>10}"
        f"{'batch':>10}  gate"
    )
    print(f"angle_mode={args.angle_mode}  batch={args.batch}  seed={args.seed}  (float64)")
    print(header)
    print("-" * len(header))
    all_ok = True
    for ansatz, n_qb, n_l, readout in cases:
        _, rep, cmp, d_out, d_grad, ok = check_one(
            ansatz, n_qb, n_l, readout, args.angle_mode, args.seed, args.batch,
            args.exact_max_qubits,
        )
        ps = f"{cmp.rel_backprop_vs_shift:>10.1e}" if cmp else f"{'skipped':>10}"
        fd = f"{cmp.rel_backprop_vs_fd:>10.1e}" if cmp else f"{'skipped':>10}"
        all_ok &= ok
        print(
            f"{ansatz:<7}{n_qb:>3}{n_l:>3} {readout:<7}{rep.n_params:>7}{rep.n_dead:>7}"
            f"{rep.abs_mean:>10.2e}{rep.abs_min:>10.2e}{rep.abs_max:>10.2e}"
            f"{ps}{fd}"
            f"{max(d_out, d_grad):>10.1e}  {'PASS' if ok else 'FAIL'}"
        )
        if rep.n_dead:
            print(f"    dead parameters: {', '.join(rep.dead_names)}")
    print("\nGATE:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())