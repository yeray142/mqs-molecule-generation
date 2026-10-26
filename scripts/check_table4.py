#!/usr/bin/env python
"""Tables 4 and 5 gate: QGAN <Z0> against the TUNED CLASSICAL GAN (Phase 3, step 14).

    uv run python scripts/check_table4.py
    uv run python scripts/check_table4.py --per-metric

Reference scenario: the paper is explicit that the quantum results are scored against the
classical GAN, not the VAE -- "The ground truth in this calculation is the classical GAN
result taken as the benchmark reference point" (Sec. 2.3), and Table 4's <Z0> column reads
"Reference" on the *Tuned GAN* row, with its caption "All models use tuned VAE and GAN
training parameters". Table 3's +1.45/+0.86 are a different comparison (GANs vs the tuned
VAE); that one is checked by scripts/check_table3.py. An earlier version of this script
used the VAE as the reference here, which silently scored the quantum runs against the
wrong baseline.

Targets come from paper_configs.TABLE5_Z0 (Table 5, whose nominal n_qb=5/n_l=2/dual
entries are also printed in Table 4). --reference lets the VAE or the nominal GAN be used
instead, for cross-checking only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.circuits.paper_configs import TABLE5_Z0
from mqs_molecule_generation.metrics.significance import (
    SIGNIFICANCE_METRICS,
    MetricStat,
    average_significance,
)

#: Same absolute tolerance as check_table3.py: the sign and rough scale are what matter,
#: since the reference distribution itself differs from the paper's (Phase 0 finding).
TOLERANCE = 1.0
#: The paper's own "comparable performance" criterion (Sec. 2.3).
COMPARABLE = 1.0

REFERENCE_PREFIX = {"tuned-gan": "tuned_gan", "nominal-gan": "nominal_gan", "vae": "reference"}

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
    return [json.loads(p.read_text()) for p in paths if p.exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("results/significance_runs"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--reference",
        choices=sorted(REFERENCE_PREFIX),
        default="tuned-gan",
        help="scenario the QGANs are scored against (paper: tuned-gan)",
    )
    ap.add_argument(
        "--per-metric",
        action="store_true",
        help="print each metric's Z0 with the reference and QGAN means behind it",
    )
    ap.add_argument(
        "--vs-classical", action="store_true", help=argparse.SUPPRESS
    )  # deprecated: comparing against the tuned classical GAN is now the default
    args = ap.parse_args(argv)

    prefix = REFERENCE_PREFIX[args.reference]
    ref_runs = load([args.output_dir / f"{prefix}_seed_{s}.json" for s in args.seeds])
    if len(ref_runs) < 2:
        print(
            f"Need >= 2 '{prefix}' reference runs in {args.output_dir}, found {len(ref_runs)}. "
            f"Run scripts/train_gan.py --gan-scenario tuned for seeds {args.seeds} first.",
            file=sys.stderr,
        )
        return 1
    ref = aggregate(ref_runs)
    ref_dim = ref_runs[0].get("latent_dim")

    groups: dict[tuple, list[Path]] = defaultdict(list)
    for path in sorted(args.output_dir.glob("*_qgan_*_seed_*.json")):
        m = RUN_RE.match(path.name)
        if m and int(m["seed"]) in args.seeds:
            run = json.loads(path.read_text())
            # Pre-audit runs carry no "figure_layout" key: they used the old circuits.
            layout = "figure" if run.get("figure_layout", False) else "legacy"
            scaling = (run.get("latent_scaling") or {}).get("mode", "?")
            key = (m["scenario"], m["ansatz"], int(m["nq"]), int(m["nl"]), m["readout"])
            groups[(*key, layout, scaling)].append(path)
    if not groups:
        print(f"No QGAN runs found in {args.output_dir}", file=sys.stderr)
        return 1

    print(f"Reference: {args.reference} ({len(ref_runs)} seeds, D_l={ref_dim})")
    header = (
        f"{'scenario':<8} {'ansatz':<6} {'nq':>3} {'nl':>3} {'readout':<7} {'D_l':>4} "
        f"{'layout':<6} {'scale':<6} {'seeds':>5} {'npar':>5} {'<Z0>':>8} {'paper':>7} "
        f"{'|diff|':>7}  verdict"
    )
    print(header)
    print("-" * len(header))

    all_ok, n_checked = True, 0
    for key, paths in sorted(groups.items()):
        runs = load(paths)
        scenario, ansatz, nq, nl, readout, layout, scaling = key
        d_l = nq if readout == "single" else 2 * nq
        cfg = (
            f"{scenario:<8} {ansatz:<6} {nq:>3} {nl:>3} {readout:<7} {d_l:>4} "
            f"{layout:<6} {scaling:<6}"
        )
        if len(runs) < 2:
            print(f"{cfg} {len(runs):>5}  need >= 2 seeds")
            all_ok = False
            continue
        note = ""
        if ref_dim is not None and d_l != ref_dim:
            note = f"reference is D_l={ref_dim}; the paper scores each D_l against its own GAN"
        stats = aggregate(runs)
        per_metric, z0 = average_significance(ref, stats)
        npar = runs[0].get("n_generator_params", "-")
        # Tables 4 and 5 are for the TUNED QGAN only ("All models use tuned VAE and GAN
        # training parameters"); a nominal run has no published target to match.
        target = TABLE5_Z0.get((ansatz, nq, nl, readout)) if scenario == "tuned" else None
        if target is None:
            paper, diff = f"{'—':>7}", f"{'—':>7}"
            verdict = "comparable" if abs(z0) < COMPARABLE else "NOT comparable"
            why = "nominal QGAN: paper's targets are for tuned" if scenario == "nominal" else (
                "not in Table 5"
            )
            verdict += f" ({why})"
        else:
            d = abs(z0 - target)
            ok = d <= TOLERANCE
            all_ok &= ok
            n_checked += 1
            paper, diff = f"{target:>+7.2f}", f"{d:>7.2f}"
            verdict = "PASS" if ok else "MISMATCH"
        if note:
            verdict += f"  [{note}]"
        print(f"{cfg} {len(runs):>5} {npar:>5} {z0:>+8.3f} {paper} {diff}  {verdict}")
        if args.per_metric:
            print(f"      {'metric':<10} {'Z0':>6}   {'reference':>18}   {'QGAN':>18}")
            for name, v in per_metric.items():
                r, q = ref[name], stats[name]
                print(
                    f"      {name:<10} {v:+6.2f}   {r.mean:>9.4f} ± {r.std:<6.4f}   "
                    f"{q.mean:>9.4f} ± {q.std:<6.4f}"
                )

    print(
        f"\nTable 4/5 gate: {'PASS' if all_ok and n_checked else 'INCOMPLETE'} "
        f"({n_checked} configuration(s) checked against the paper)"
    )
    if any(k[5] == "legacy" for k in groups):
        print(
            "NOTE: 'legacy' rows were trained before the paper audit (no BEL Hadamard layer, "
            "no last-layer CX on simple). Re-run them for the corrected circuits."
        )
    if args.reference != "tuned-gan":
        print("NOTE: --reference is not the paper's (tuned-gan); the targets will not apply.")
    return 0 if all_ok and n_checked else 1


if __name__ == "__main__":
    raise SystemExit(main())