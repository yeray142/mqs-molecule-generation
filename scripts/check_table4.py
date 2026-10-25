#!/usr/bin/env python
"""Table 4 gate: quantum GAN <Z0> against the tuned-VAE reference (Phase 3, step 14).

    uv run python scripts/check_table4.py
    uv run python scripts/check_table4.py --vs-classical     # also QGAN vs classical GAN

Reads every ``{scenario}_qgan_{ansatz}_q{n}_l{l}_{readout}_seed_{s}.json`` written by
train_qgan.py, groups the runs by configuration, aggregates each of the 10 significance
metrics across seeds and computes Eq. 5 against the reference runs
(``reference_seed_{s}.json`` from eval_vae_reference.py) -- the same code path as
check_table3.py, so classical and quantum numbers are directly comparable.

TABLE4_TARGETS is EMPTY on purpose: the paper's Table 4 values were not available in the
text this project could read, and filling it with guesses would make "PASS" meaningless.
Paste the paper's numbers in; until then each configuration is reported against the
paper's qualitative criterion only (|<Z0>| < 1 = comparable to the reference).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.metrics.significance import (
    SIGNIFICANCE_METRICS,
    MetricStat,
    average_significance,
)

#: (gan_scenario, ansatz, n_qb, n_l, readout) -> paper's <Z0>. Paste Table 4 here, e.g.
#: ("nominal", "bel", 5, 2, "dual"): <value>,
TABLE4_TARGETS: dict[tuple[str, str, int, int, str], float] = {}

#: Same absolute tolerance as check_table3.py.
TOLERANCE = 1.0
#: Paper's "comparable performance" criterion.
COMPARABLE = 1.0

RUN_RE = re.compile(
    r"^(?P<scenario>nominal|tuned)_qgan_(?P<ansatz>simple|bel)_q(?P<nq>\d+)_l(?P<nl>\d+)_"
    r"(?P<readout>single|dual)_seed_(?P<seed>\d+)\.json$"
)


def aggregate(runs: list[dict]) -> dict[str, MetricStat]:
    stats = {}
    for name in SIGNIFICANCE_METRICS:
        vals = [r["metrics"][name] for r in runs]
        stats[name] = MetricStat.from_samples(vals) if len(vals) >= 2 else MetricStat(vals[0], 0.0)
    return stats


def load(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for p in paths]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("results/significance_runs"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--vs-classical", action="store_true", help="also report QGAN vs classical GAN")
    ap.add_argument("--per-metric", action="store_true", help="print the 10 per-metric Z0 values")
    args = ap.parse_args(argv)

    ref_paths = [args.output_dir / f"reference_seed_{s}.json" for s in args.seeds]
    ref_runs = load([q for q in ref_paths if q.exists()])
    if len(ref_runs) < 2:
        print(f"Need >= 2 reference runs, found {len(ref_runs)}", file=sys.stderr)
        return 1
    ref = aggregate(ref_runs)

    groups: dict[tuple, list[Path]] = defaultdict(list)
    for path in sorted(args.output_dir.glob("*_qgan_*_seed_*.json")):
        m = RUN_RE.match(path.name)
        if m and int(m["seed"]) in args.seeds:
            key = (m["scenario"], m["ansatz"], int(m["nq"]), int(m["nl"]), m["readout"])
            groups[key].append(path)
    if not groups:
        print(f"No QGAN runs found in {args.output_dir}", file=sys.stderr)
        return 1

    header = (
        f"{'scenario':<8} {'ansatz':<6} {'nq':>3} {'nl':>3} {'readout':<7} {'seeds':>5} "
        f"{'<Z0>':>8} {'paper':>7} {'|diff|':>7}  verdict"
    )
    if args.vs_classical:
        header += "   <Z0> vs classical GAN"
    print(header)
    print("-" * len(header))

    all_ok = True
    for key, paths in sorted(groups.items()):
        runs = load(paths)
        scenario, ansatz, nq, nl, readout = key
        if len(runs) < 2:
            print(f"{scenario:<8} {ansatz:<6} {nq:>3} {nl:>3} {readout:<7} {len(runs):>5}  "
                  "need >= 2 seeds")
            all_ok = False
            continue
        stats = aggregate(runs)
        per_metric, z0 = average_significance(ref, stats)
        target = TABLE4_TARGETS.get(key)
        if target is None:
            paper, diff = f"{'—':>7}", f"{'—':>7}"
            verdict = "comparable" if abs(z0) < COMPARABLE else "NOT comparable"
            verdict += " (no Table 4 value entered)"
        else:
            d = abs(z0 - target)
            ok = d <= TOLERANCE
            all_ok &= ok
            paper, diff = f"{target:>+7.2f}", f"{d:>7.2f}"
            verdict = "PASS" if ok else "MISMATCH"
        line = (
            f"{scenario:<8} {ansatz:<6} {nq:>3} {nl:>3} {readout:<7} {len(runs):>5} "
            f"{z0:>+8.3f} {paper} {diff}  {verdict}"
        )
        if args.vs_classical:
            cls_paths = [args.output_dir / f"{scenario}_gan_seed_{s}.json" for s in args.seeds]
            cls_runs = load([q for q in cls_paths if q.exists()])
            if len(cls_runs) >= 2 and runs[0]["latent_dim"] == cls_runs[0]["latent_dim"]:
                _, zc = average_significance(aggregate(cls_runs), stats)
                line += f"   {zc:+.3f}"
            else:
                line += "   (no matching classical runs)"
        print(line)
        if args.per_metric:
            for name, v in per_metric.items():
                print(f"      {name:<10} {v:+.2f}")

    if not TABLE4_TARGETS:
        print("\nTABLE4_TARGETS is empty: verdicts use the |<Z0>| < 1 criterion only.")
    print("Table 4 gate:", "PASS" if all_ok and TABLE4_TARGETS else "INCOMPLETE")
    return 0 if all_ok and TABLE4_TARGETS else 1


if __name__ == "__main__":
    raise SystemExit(main())