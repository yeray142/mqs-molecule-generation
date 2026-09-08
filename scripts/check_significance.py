#!/usr/bin/env python
"""Check the Z0 significance implementation against paper Table 14.

    uv run python scripts/03_check_significance.py

Pure arithmetic, no real data needed: feeds Table 14's published mean+-std
values (classical GAN vs simple/BEL QGAN, n_qb=5, n_l=2, dual readout)
through Eq. 5 and compares against the paper's own Z0 columns and <Z0>.

Individual Z0^m values carry real reconstruction noise from the table's 2-3
decimal rounding -- confirmed by hand, up to ~0.8 absolute error when a std
displays as "0.000" (hides anything from 0 to ~0.0005). The aggregate <Z0>
is the tight, meaningful gate: reconstruction error was ~0.01-0.06 absolute
when this was worked out by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.metrics.significance import MetricStat, average_significance

# name -> (mu_reference, sigma_reference, mu_sample, sigma_sample, direction, Z0_paper)
Row = tuple[float, float, float, float, str, float]

# Table 14: classical (tuned) GAN vs styled simple QGAN, n_qb=5, n_l=2, dual readout.
SIMPLE_QGAN: dict[str, Row] = {
    "eps_d": (0.481, 0.031, 0.529, 0.003, "max", 1.56),
    "eps_v": (0.917, 0.012, 0.831, 0.002, "max", -7.21),
    "eps_u": (0.986, 0.003, 0.994, 0.000, "max", 3.50),
    "novelty": (0.536, 0.015, 0.572, 0.003, "max", 2.40),
    "intdiv": (0.898, 0.004, 0.882, 0.000, "max", -4.60),
    "filters": (0.721, 0.011, 0.737, 0.003, "max", 1.43),
    "eps_logp": (0.898, 0.007, 0.883, 0.001, "max", -2.35),
    "sa": (2.391, 0.080, 2.575, 0.007, "min", -2.31),
    "qed": (0.599, 0.007, 0.641, 0.001, "max", 6.17),
    "weight": (203.9, 10.6, 294.8, 0.8, "min", -8.56),
}
SIMPLE_QGAN_AGGREGATE_PAPER = -1.00

# Table 14: classical (tuned) GAN vs styled BEL QGAN, n_qb=5, n_l=2, dual readout.
BEL_QGAN: dict[str, Row] = {
    "eps_d": (0.481, 0.031, 0.523, 0.003, "max", 1.36),
    "eps_v": (0.917, 0.012, 0.875, 0.003, "max", -3.43),
    "eps_u": (0.986, 0.003, 0.993, 0.001, "max", 2.90),
    "novelty": (0.536, 0.015, 0.548, 0.003, "max", 0.82),
    "intdiv": (0.898, 0.004, 0.883, 0.000, "max", -4.21),
    "filters": (0.721, 0.011, 0.719, 0.002, "max", -0.17),
    "eps_logp": (0.898, 0.007, 0.896, 0.002, "max", -0.23),
    "sa": (2.391, 0.080, 2.448, 0.007, "min", -0.71),
    "qed": (0.599, 0.007, 0.643, 0.001, "max", 6.55),
    "weight": (203.9, 10.6, 261.4, 0.8, "min", -5.42),
}
BEL_QGAN_AGGREGATE_PAPER = -0.26

PER_METRIC_TOLERANCE = 1.0  # absolute; see module docstring
AGGREGATE_TOLERANCE = 0.1  # absolute; see module docstring


def _check_scenario(name: str, rows: dict[str, Row], paper_aggregate: float) -> bool:
    reference = {k: MetricStat(v[0], v[1]) for k, v in rows.items()}
    sample = {k: MetricStat(v[2], v[3]) for k, v in rows.items()}
    directions = {k: v[4] for k, v in rows.items()}
    paper_z0 = {k: v[5] for k, v in rows.items()}

    per_metric, aggregate = average_significance(reference, sample, directions)  # type: ignore[arg-type]

    print(f"\n{name}")
    width = max(len(k) for k in rows)
    print(f"{'metric':<{width}}  {'ours':>8}  {'paper':>8}  {'|diff|':>8}")
    print("-" * (width + 30))
    all_ok = True
    for metric_name, ours in per_metric.items():
        paper_val = paper_z0[metric_name]
        diff = abs(ours - paper_val)
        ok = diff <= PER_METRIC_TOLERANCE
        all_ok &= ok
        flag = "" if ok else "  MISMATCH"
        print(f"{metric_name:<{width}}  {ours:>8.2f}  {paper_val:>8.2f}  {diff:>8.3f}{flag}")

    agg_diff = abs(aggregate - paper_aggregate)
    agg_ok = agg_diff <= AGGREGATE_TOLERANCE
    all_ok &= agg_ok
    flag = "" if agg_ok else "  MISMATCH"
    print("-" * (width + 30))
    print(f"{'<Z0>':<{width}}  {aggregate:>8.4f}  {paper_aggregate:>8.4f}  {agg_diff:>8.4f}{flag}")
    return all_ok


def main() -> int:
    ok_simple = _check_scenario(
        "Simple QGAN vs classical (tuned) GAN", SIMPLE_QGAN, SIMPLE_QGAN_AGGREGATE_PAPER
    )
    ok_bel = _check_scenario(
        "BEL QGAN vs classical (tuned) GAN", BEL_QGAN, BEL_QGAN_AGGREGATE_PAPER
    )

    print()
    if ok_simple and ok_bel:
        print("Both scenarios within tolerance. Phase 0 step 3 gate: PASS.")
        return 0
    print("One or more scenarios outside tolerance.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())