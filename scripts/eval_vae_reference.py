#!/usr/bin/env python
"""Evaluate a trained VAE's prior-sampled generation (paper Table 3/14 reference).

    uv run python scripts/eval_vae_reference.py \\
        --vae-checkpoint results/vae_runs/tuned_n_ep_1000_seed_0.pt \\
        --scenario tuned --seed 0

The paper's Table 3 significance comparison (⟨Z0⟩) is always measured
against the TUNED VAE as the reference scenario -- this script produces that
reference's metrics: sample from the VAE's prior (z ~ N(0,1), NOT from a
GAN), decode, and compute the same 10-metric suite (eps_d, eps_v, eps_u,
novelty, IntDiv, Filters, eps_LogP, SA, QED, Weight) that
scripts/train_gan.py computes for the GAN scenarios, via the same shared
function (metrics.moses_metrics.compute_significance_metrics) -- so both
sides of the eventual comparison run through identical code.

Needs to be run once per seed (the paper uses 5). Requires a VAE checkpoint
already trained for the FULL N_epochs=1000 (paper's production training
length) -- NOT one of the shorter Table-8-study checkpoints, which use fewer
epochs specifically to study under-convergence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mqs_molecule_generation.data.moses import load_splits
from mqs_molecule_generation.metrics.moses_metrics import compute_significance_metrics
from mqs_molecule_generation.models.checkpoint_io import (
    load_tokenizer_for_checkpoint,
    load_vae_checkpoint,
)


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-vocab", type=Path, default=None)
    parser.add_argument("--scenario", choices=["nominal", "tuned"], default="tuned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=30_000, help="paper's default gen. batch")
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/significance_runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)

    train_smiles, _ = load_splits(args.processed_dir)
    print(f"Loaded {len(train_smiles)} train molecules")

    tokenizer = load_tokenizer_for_checkpoint(
        args.vae_checkpoint, args.vae_vocab, args.processed_dir
    )
    model = load_vae_checkpoint(args.vae_checkpoint, tokenizer, args.scenario, device)

    print(f"Sampling {args.n_samples} molecules from the prior (z ~ N(0,1))...")
    generated = model.sample(args.n_samples, max_len=args.max_len, temp=args.temp)

    print("Computing significance metrics...")
    metrics = compute_significance_metrics(generated, train_smiles)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    results = {
        "role": "reference",
        "scenario": args.scenario,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "vae_checkpoint": str(args.vae_checkpoint),
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"reference_seed_{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())