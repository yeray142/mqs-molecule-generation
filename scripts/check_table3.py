#!/usr/bin/env python
"""Check the Table 3 gate: nominal GAN Z0=+1.45, tuned GAN Z0=+0.86 vs tuned VAE.

    uv run python scripts/eval_vae_reference.py --vae-checkpoint <tuned VAE> --seed 0..4
    uv run python scripts/train_gan.py --vae-checkpoint <tuned VAE> \
        --gan-scenario nominal --seed 0..4
    uv run python scripts/train_gan.py --vae-checkpoint <tuned VAE> \
        --gan-scenario tuned --seed 0..4
    uv run python scripts/check_table3.py

Reads the 5-seed JSON runs written by eval_vae_reference.py and train_gan.py,
aggregates each of the 10 significance metrics into a MetricStat (mean/std
across seeds), and runs metrics.significance.average_significance -- the
same Eq. 5 implementation already verified against Table 14 in Phase 0.

This is the reference scenario every quantum GAN result in this project is
ultimately measured against, so getting the sign and rough magnitude of
these two numbers right matters more than any other single gate so far.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.metrics.significance import (
    SIGNIFICANCE_METRICS,
    MetricStat,
    average_significance,
)

#: Paper Table 3.
NOMINAL_TARGET = 1.45
TUNED_TARGET = 0.86
#: Absolute tolerance for the gate. Loose, deliberately: this is the
#: culmination of the VAE, the encoding step, and 5 real GAN training runs
#: each -- there is no realistic path to exact agreement with the paper's
#: own (differently-seeded, and per Phase 0's findings, differently-sourced)
#: numbers. The sign and rough scale are what matter.
TOLERANCE = 1.0


def load_seed_runs(output_dir: Path, prefix: str, seeds: list[int]) -> list[dict]:
    runs = []
    missing = []
    for seed in seeds:
        path = output_dir / f"{prefix}_seed_{seed}.json"
        if not path.exists():
            missing.append(seed)
            continue
        runs.append(json.loads(path.read_text()))
    if missing:
        print(f"Missing {prefix} run(s) for seed(s) {missing}", file=sys.stderr)
    return runs


def aggregate_metric_stats(runs: list[dict]) -> dict[str, MetricStat]:
    stats: dict[str, MetricStat] = {}
    for name in SIGNIFICANCE_METRICS:
        values = [run["metrics"][name] for run in runs]
        stats[name] = (
            MetricStat.from_samples(values) if len(values) >= 2 else MetricStat(values[0], 0.0)
        )
    return stats


def print_scenario_report(
    label: str, per_metric: dict[str, float], aggregate: float, target: float
) -> bool:
    print(f"\n{label}")
    width = max(len(k) for k in per_metric)
    print(f"{'metric':<{width}}  {'Z0':>8}")
    print("-" * (width + 12))
    for name, z0 in per_metric.items():
        print(f"{name:<{width}}  {z0:>8.2f}")
    diff = abs(aggregate - target)
    ok = diff <= TOLERANCE
    flag = "PASS" if ok else "MISMATCH"
    print("-" * (width + 12))
    print(
        f"{'<Z0>':<{width}}  {aggregate:>8.4f}   "
        f"(paper: {target:+.2f}, |diff|={diff:.2f})  [{flag}]"
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/significance_runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)

    reference_runs = load_seed_runs(args.output_dir, "reference", args.seeds)
    nominal_runs = load_seed_runs(args.output_dir, "nominal_gan", args.seeds)
    tuned_runs = load_seed_runs(args.output_dir, "tuned_gan", args.seeds)

    if len(reference_runs) < 2:
        print(
            f"Need at least 2 reference seed runs (have {len(reference_runs)}); "
            "run scripts/eval_vae_reference.py for more seeds first.",
            file=sys.stderr,
        )
        return 1

    reference_stats = aggregate_metric_stats(reference_runs)

    all_ok = True
    if len(nominal_runs) >= 2:
        nominal_stats = aggregate_metric_stats(nominal_runs)
        per_metric, aggregate = average_significance(reference_stats, nominal_stats)
        all_ok &= print_scenario_report(
            "Nominal GAN vs tuned VAE reference", per_metric, aggregate, NOMINAL_TARGET
        )
    else:
        print(
            f"\nSkipping nominal GAN (only {len(nominal_runs)} seed run(s) found)", file=sys.stderr
        )
        all_ok = False

    if len(tuned_runs) >= 2:
        tuned_stats = aggregate_metric_stats(tuned_runs)
        per_metric, aggregate = average_significance(reference_stats, tuned_stats)
        all_ok &= print_scenario_report(
            "Tuned GAN vs tuned VAE reference", per_metric, aggregate, TUNED_TARGET
        )
    else:
        print(f"\nSkipping tuned GAN (only {len(tuned_runs)} seed run(s) found)", file=sys.stderr)
        all_ok = False

    print()
    print("Table 3 gate: PASS" if all_ok else "Table 3 gate: INCOMPLETE or FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())