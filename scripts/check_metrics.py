#!/usr/bin/env python
"""Check the frozen train/valid splits against paper Table 7.

    uv run python scripts/01_check_metrics.py

Reads data/processed/{train,valid}.smi (written by scripts/00_prepare_data.py
prepare) and reports IntDiv for each, plus the four W(X) distances between
them, next to the paper's target numbers:

    IntDiv(train) = 0.898   IntDiv(test) = 0.889
    W(LogP) = 0.091   W(SA) = 0.268   W(QED) = 0.027   W(Weight) = 42.8

This does NOT depend on the filter-set discrepancy investigated in Phase 0
step 1 -- these metrics are computed directly from RDKit descriptors and a
Wasserstein distance, independent of any structural-alert filter definition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.data.moses import load_splits
from mqs_molecule_generation.metrics.moses_metrics import (
    PropertyReport,
    compute_property_report,
    compute_wasserstein_report,
)

#: Paper Table 7 (supplementary material).
TARGETS = {
    "IntDiv(train)": 0.898,
    "IntDiv(test)": 0.889,
    "W(LogP)": 0.091,
    "W(SA)": 0.268,
    "W(QED)": 0.027,
    "W(Weight)": 42.8,
}


def _print_distribution_summary(train: PropertyReport, valid: PropertyReport) -> None:
    """Print mean/std/min/max per property for both samples.

    A single Wasserstein scalar can't distinguish "these two populations are
    genuinely similar" from "something's compressing the computed values" --
    seeing the actual spread (especially std/min/max) for both sides at once
    makes that distinguishable at a glance.
    """
    print("\nPer-property distribution summary:")
    header = (
        f"{'property':<8}  {'sample':<6}  {'n':>6}  {'mean':>10}  "
        f"{'std':>10}  {'min':>10}  {'max':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, train_vals, valid_vals in (
        ("LogP", train.logp, valid.logp),
        ("SA", train.sa, valid.sa),
        ("QED", train.qed, valid.qed),
        ("Weight", train.weight, valid.weight),
    ):
        for sample_name, vals in (("train", train_vals), ("valid", valid_vals)):
            arr = np.asarray(vals, dtype=float)
            print(
                f"{label:<8}  {sample_name:<6}  {len(arr):>6}  {arr.mean():>10.4f}  "
                f"{arr.std():>10.4f}  {arr.min():>10.4f}  {arr.max():>10.4f}"
            )


def _print_overlap_check(train: list[str], valid: list[str]) -> None:
    """Report exact-string overlap between the two frozen splits.

    Cheap sanity check: if train and valid share molecules (they shouldn't,
    by MOSES's own split design), every metric here becomes uninterpretable.
    """
    overlap = set(train) & set(valid)
    print(
        f"\nTrain/valid exact-string overlap: {len(overlap)} molecule(s) "
        f"({'ok, disjoint' if not overlap else 'UNEXPECTED -- splits should not overlap'})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="relative tolerance for the pass/fail flag (default 2%%)",
    )
    args = parser.parse_args(argv)

    train_path = args.processed_dir / "train.smi"
    if not train_path.exists():
        print(
            f"{train_path} not found. Run scripts/00_prepare_data.py prepare "
            "--config configs/data/moses_subset.yaml first.",
            file=sys.stderr,
        )
        return 1

    train, valid = load_splits(args.processed_dir)
    print(f"Loaded {len(train)} train / {len(valid)} valid molecules from {args.processed_dir}\n")
    _print_overlap_check(train, valid)

    print("Computing properties + IntDiv for train set...")
    train_report = compute_property_report(train)
    print("Computing properties + IntDiv for valid (test) set...")
    valid_report = compute_property_report(valid)

    _print_distribution_summary(train_report, valid_report)

    # Paper convention: W(X) is the distance between the test set and the
    # sample being scored (here, the train set).
    w = compute_wasserstein_report(reference=valid_report, sample=train_report)

    results = {
        "IntDiv(train)": train_report.int_div,
        "IntDiv(test)": valid_report.int_div,
        "W(LogP)": w.logp,
        "W(SA)": w.sa,
        "W(QED)": w.qed,
        "W(Weight)": w.weight,
    }

    width = max(len(k) for k in TARGETS)
    print(f"\n{'metric':<{width}}  {'ours':>10}  {'paper':>10}  {'rel. Δ':>8}")
    print("-" * (width + 36))
    all_ok = True
    for name, target in TARGETS.items():
        ours = results[name]
        rel_delta = abs(ours - target) / target if target != 0 else float("inf")
        ok = rel_delta <= args.tolerance
        all_ok &= ok
        flag = "  ok" if ok else "  MISMATCH"
        print(f"{name:<{width}}  {ours:>10.4f}  {target:>10.4f}  {rel_delta:>7.1%}{flag}")

    print()
    if all_ok:
        print("All Table 7 metrics within tolerance. Phase 0 step 2 gate: PASS.")
    else:
        print(
            "One or more metrics outside tolerance. This is the real litmus test "
            "for the data/metrics pipeline (unlike the Filters discrepancy from "
            "step 1, these numbers don't depend on any filter definition)."
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())