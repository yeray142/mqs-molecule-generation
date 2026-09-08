#!/usr/bin/env python
"""Diagnose whether the weight/QED bias comes from the pool or the subsample.

    uv run python scripts/02_diagnose_pool.py

Draws a large, fresh random sample directly from the FULL MOSES train pool
(bypassing the frozen 12,000-molecule subsample entirely) and reports
mean/std for Weight, QED, SA -- compared against both the paper's Table 3
"Train set" column and our own frozen 12k subsample.

If the full-pool numbers already match the paper, the bias is introduced
somewhere in our subsampling step. If the full-pool numbers already look like
our frozen subsample (not the paper), the difference predates subsampling --
either a real data-vintage difference from what the paper's authors used, or
a wrong split/column being read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.data.moses import canonicalize, download, load_raw, load_splits
from mqs_molecule_generation.metrics.properties import (
    compute_molecular_weights_batch,
    compute_qeds_batch,
)
from mqs_molecule_generation.metrics.sa_score import compute_sa_scores_batch

#: Paper Table 3, "Train set" column.
PAPER_TRAIN = {"mean_weight": 261.5, "mean_qed": 0.596, "mean_sa": 2.561}


def _summarize(label: str, values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    print(f"  {label:<20} n={len(arr):>7}  mean={arr.mean():>10.4f}  std={arr.std():>10.4f}")
    return float(arr.mean()), float(arr.std())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--pool-sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-verify", action="store_true", help="skip the source checksum")
    args = parser.parse_args(argv)

    raw_path = download(args.raw_dir / "dataset_v1.csv", verify=not args.no_verify)
    frame = load_raw(raw_path)
    pool_smiles = frame.loc[frame["SPLIT"] == args.train_split, "SMILES"].tolist()
    print(f"Full '{args.train_split}' split: {len(pool_smiles)} rows in dataset_v1.csv")

    rng = np.random.default_rng(args.seed)
    n = min(args.pool_sample_size, len(pool_smiles))
    idx = rng.choice(len(pool_smiles), size=n, replace=False)
    raw_sample = [pool_smiles[int(i)] for i in idx]
    canonical = [c for c in (canonicalize(s) for s in raw_sample) if c is not None]
    print(
        f"Fresh random draw from full pool: {len(canonical)} valid molecules (seed={args.seed})\n"
    )

    print("Full-pool fresh sample:")
    pool_weight_mean, _ = _summarize("Weight", compute_molecular_weights_batch(canonical))
    pool_qed_mean, _ = _summarize("QED", compute_qeds_batch(canonical))
    pool_sa_mean, _ = _summarize("SA", compute_sa_scores_batch(canonical))

    frozen_train, _ = load_splits(args.processed_dir)
    print(f"\nFrozen data/processed/train.smi ({len(frozen_train)} molecules):")
    frozen_weight_mean, _ = _summarize("Weight", compute_molecular_weights_batch(frozen_train))
    frozen_qed_mean, _ = _summarize("QED", compute_qeds_batch(frozen_train))
    frozen_sa_mean, _ = _summarize("SA", compute_sa_scores_batch(frozen_train))

    print(f"\n{'metric':<10}  {'paper':>10}  {'full pool':>10}  {'frozen 12k':>10}")
    print("-" * 46)
    for label, paper_key, pool_val, frozen_val in (
        ("Weight", "mean_weight", pool_weight_mean, frozen_weight_mean),
        ("QED", "mean_qed", pool_qed_mean, frozen_qed_mean),
        ("SA", "mean_sa", pool_sa_mean, frozen_sa_mean),
    ):
        paper_val = PAPER_TRAIN[paper_key]
        print(f"{label:<10}  {paper_val:>10.4f}  {pool_val:>10.4f}  {frozen_val:>10.4f}")

    print()
    pool_close_to_paper = abs(pool_weight_mean - PAPER_TRAIN["mean_weight"]) < 10
    if pool_close_to_paper:
        print(
            "Full pool matches the paper reasonably well -> the bias is most "
            "likely introduced during subsampling (data/moses.py::subsample or "
            "the dedup step feeding it), not in which split/rows are read."
        )
    else:
        print(
            "Full pool ALREADY differs from the paper, before any subsampling "
            "happens -> the difference predates our subsampling step. Check "
            "the raw split sizes/composition, or whether this is a different "
            "vintage of dataset_v1.csv than the paper's authors used."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())