#!/usr/bin/env python
"""Check the Table 8 gate: N_ep=800/1000 should agree, 100/250 should diverge.

    uv run python scripts/train_vae.py --n-epochs 100 --seed 0
    uv run python scripts/train_vae.py --n-epochs 250 --seed 0
    uv run python scripts/train_vae.py --n-epochs 800 --seed 0
    uv run python scripts/train_vae.py --n-epochs 1000 --seed 0
    uv run python scripts/check_table8.py

Reads the four results/vae_runs/n_ep_<N>_seed_<S>.json files written by
train_vae.py and checks two things:

1. Do your 800 and 1000 results agree with each other (small relative
   difference), and do 100/250 clearly differ from that plateau? This is
   the actual paper-stated gate and doesn't require matching the paper's
   absolute numbers -- only the qualitative convergence shape.
2. For reference, how do your numbers compare to the paper's own Table 8?
   Given the known upstream data discrepancy documented in Phase 0
   (docs/paper_mapping.md), exact numeric agreement isn't expected here
   either -- this is a sanity check, not a second gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Paper Table 8 (supplementary material section 2).
PAPER_TABLE_8: dict[str, dict[int, float]] = {
    "IntDiv": {100: 0.892, 250: 0.919, 800: 0.896, 1000: 0.896},
    "W(LogP)": {100: 0.190, 250: 0.030, 800: 0.048, 1000: 0.048},
    "W(SA)": {100: 0.258, 250: 0.2646, 800: 0.2796, 1000: 0.2751},
    "W(QED)": {100: 0.084, 250: 0.090, 800: 0.096, 1000: 0.097},
    "W(Weight)": {100: 67.6, 250: 51.2, 800: 43.8, 1000: 43.9},
}

N_EP_POINTS = (100, 250, 800, 1000)

#: Relative-difference threshold for the qualitative gate, calibrated against
#: the paper's OWN Table 8 numbers (computed by hand, not guessed): across
#: all 5 metrics, every 800-vs-1000 gap is <=1.61% and every max(100,250)-
#: vs-800 gap is >=2.50%, so any threshold in (1.61%, 2.50%) separates both
#: checks correctly on the paper's real data. 2% is used. Note this margin is
#: thin for IntDiv specifically (2.50% divergence against a 2% cutoff) --
#: IntDiv is naturally tightly clustered (paper's own values span only
#: 0.892-0.919), so its gate is more fragile than the other four metrics.
#: Your own run, on a different 12k-molecule dataset (see Phase 0 findings
#: in docs/paper_mapping.md), may not reproduce this exact margin even if
#: the qualitative trend is correct.
AGREE_THRESHOLD = 0.02


def _load_run(output_dir: Path, n_epochs: int, seed: int) -> dict | None:
    path = output_dir / f"n_ep_{n_epochs}_seed_{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def check_convergence_pattern(ours: dict[int, dict[str, float]]) -> bool:
    """The actual gate: 800~1000 agree, 100/250 visibly diverge from that plateau."""
    print("Convergence pattern (your runs only):\n")
    all_ok = True
    metric_names = ["IntDiv", "W(LogP)", "W(SA)", "W(QED)", "W(Weight)"]
    header = f"{'metric':<10} " + "".join(f"{n:>12}" for n in N_EP_POINTS)
    print(header)
    print("-" * len(header))
    for metric in metric_names:
        row = ours.get(metric, {})
        print(f"{metric:<10} " + "".join(f"{row.get(n, float('nan')):>12.4f}" for n in N_EP_POINTS))

    print()
    for metric in metric_names:
        row = ours.get(metric, {})
        if not all(n in row for n in N_EP_POINTS):
            print(f"{metric}: incomplete (missing a run), skipping gate check")
            all_ok = False
            continue
        agree_800_1000 = _rel_diff(row[800], row[1000])
        diverge_100 = _rel_diff(row[100], row[800])
        diverge_250 = _rel_diff(row[250], row[800])

        agrees = agree_800_1000 <= AGREE_THRESHOLD
        diverges = max(diverge_100, diverge_250) > AGREE_THRESHOLD
        ok = agrees and diverges
        all_ok &= ok
        flag = "ok" if ok else "MISMATCH"
        print(
            f"{metric}: 800~1000 rel.diff={agree_800_1000:.1%} "
            f"({'agree' if agrees else 'DO NOT AGREE'}), "
            f"max(100,250) vs 800 rel.diff={max(diverge_100, diverge_250):.1%} "
            f"({'diverges' if diverges else 'DOES NOT DIVERGE'})  [{flag}]"
        )
    return all_ok


def print_paper_comparison(ours: dict[int, dict[str, float]]) -> None:
    print(
        "\nReference comparison against the paper's Table 8 (not a gate -- see module docstring):\n"
    )
    metric_names = ["IntDiv", "W(LogP)", "W(SA)", "W(QED)", "W(Weight)"]
    for metric in metric_names:
        print(f"{metric}:")
        for n in N_EP_POINTS:
            our_val = ours.get(metric, {}).get(n)
            paper_val = PAPER_TABLE_8[metric][n]
            our_str = f"{our_val:.4f}" if our_val is not None else "  n/a "
            print(f"  N_ep={n:<5} ours={our_str}  paper={paper_val:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vae_runs"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ours: dict[str, dict[int, float]] = {
        m: {} for m in ("IntDiv", "W(LogP)", "W(SA)", "W(QED)", "W(Weight)")
    }
    missing = []
    for n_epochs in N_EP_POINTS:
        run = _load_run(args.output_dir, n_epochs, args.seed)
        if run is None:
            missing.append(n_epochs)
            continue
        for metric_name, value in run["metrics"].items():
            if metric_name in ours:
                ours[metric_name][n_epochs] = value

    if missing:
        print(
            f"Missing run(s) for N_ep={missing} (seed={args.seed}). "
            "Run scripts/train_vae.py for each before checking the gate.",
            file=sys.stderr,
        )

    gate_ok = check_convergence_pattern(ours)
    print_paper_comparison(ours)

    print()
    if missing:
        print("Cannot fully evaluate the gate: missing run(s) above.")
        return 1
    print("Table 8 convergence gate: PASS" if gate_ok else "Table 8 convergence gate: FAIL")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())