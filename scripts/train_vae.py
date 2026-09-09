#!/usr/bin/env python
"""Train the VAE for the Table 8 epoch-count study (paper Supp. §2).

    uv run python scripts/train_vae.py --n-epochs 100 --seed 0
    uv run python scripts/train_vae.py --n-epochs 250 --seed 0
    uv run python scripts/train_vae.py --n-epochs 800 --seed 0
    uv run python scripts/train_vae.py --n-epochs 1000 --seed 0

Requires torch and the frozen train/valid splits from
scripts/prepare_data.py. Uses the NOMINAL VAE scenario throughout (constant
lr=3e-4) -- the paper's own Table 8 study is explicitly done on the nominal
VAE ("Figure 13 ... for the nominal VAE"), not the tuned one.

This is real training: expect it to take a non-trivial amount of wall-clock
time (the paper reports 9m-53m on an A100 GPU for 100-1000 epochs; CPU-only
will be much slower). Writes:

  - results/vae_runs/n_ep_<N>_seed_<S>.json    -- metrics for check_table8.py
  - results/vae_runs/n_ep_<N>_seed_<S>.pt      -- model checkpoint
  - results/vae_runs/n_ep_<N>_seed_<S>.vocab.json -- exact tokenizer vocabulary,
    needed to correctly reload the checkpoint later (scripts/reconstruct.py)
  - results/vae_runs/n_ep_<N>_seed_<S>.history.csv -- per-epoch loss/kl_loss/
    recon_loss/kl_weight/lr (train and, since a val split is always passed
    here, eval rows too), for plotting convergence after the fact

By default every epoch's metrics are also printed live during training (via
the tqdm progress bar's postfix) and as a permanent line to stdout -- use
--log-every N to print less often on a long run, or --log-every 0 to print
nothing and rely on the CSV file only.

A single scalar Wasserstein distance can't distinguish "the decoder
genuinely reconstructs the test distribution well" from "the decoder has
collapsed onto a narrow band of generic molecules that happens to sit near
the test set's center" -- both would show up as a suspiciously small W(X).
This script also prints and saves per-property distribution summaries
(mean/std/min/max for both the test set and the generated set) and a sample
of actual generated SMILES, so that distinction is checkable directly rather
than inferred from the aggregate number alone.

Use --checkpoint-in to re-run this diagnostic on an already-trained model
without paying the training cost again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqs_molecule_generation.data.moses import load_splits
from mqs_molecule_generation.data.tokenizer import SmilesTokenizer
from mqs_molecule_generation.metrics.moses_metrics import (
    compute_property_report,
    compute_wasserstein_report,
)
from mqs_molecule_generation.models.vae import VAE, VAEConfig
from mqs_molecule_generation.training.trainer_config import TrainerConfig
from mqs_molecule_generation.training.vae_trainer import VAETrainer, build_collate_fn


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distribution_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def print_distribution_comparison(valid_report, generated_report) -> dict[str, dict]:
    """Print (and return) mean/std/min/max for each property, test vs generated."""
    print("\nPer-property distribution: test set vs generated sample")
    header = (
        f"{'property':<8} {'sample':<10} {'n':>6} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
    )
    print(header)
    print("-" * len(header))

    summaries: dict[str, dict] = {}
    for label, test_vals, gen_vals in (
        ("LogP", valid_report.logp, generated_report.logp),
        ("SA", valid_report.sa, generated_report.sa),
        ("QED", valid_report.qed, generated_report.qed),
        ("Weight", valid_report.weight, generated_report.weight),
    ):
        test_summary = _distribution_summary(test_vals)
        gen_summary = _distribution_summary(gen_vals)
        summaries[label] = {"test": test_summary, "generated": gen_summary}
        for sample_name, s in (("test", test_summary), ("generated", gen_summary)):
            print(
                f"{label:<8} {sample_name:<10} {s['n']:>6} {s['mean']:>10.4f} "
                f"{s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}"
            )
    return summaries


def check_uniqueness(generated: list[str]) -> dict[str, float]:
    """Fraction of raw strings and canonical molecules that are unique.

    Low uniqueness (many repeated strings, or many strings collapsing to a
    handful of canonical molecules) is the classic mode-collapse signature
    that a small Wasserstein distance alone would not reveal.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    n = len(generated)
    distinct_strings = len(set(generated))
    canonical = [Chem.MolToSmiles(m) for m in (Chem.MolFromSmiles(s) for s in generated) if m]
    distinct_canonical = len(set(canonical))

    from collections import Counter

    most_common = Counter(canonical).most_common(5) if canonical else []

    return {
        "n_generated": n,
        "distinct_raw_strings": distinct_strings,
        "distinct_raw_fraction": distinct_strings / n if n else float("nan"),
        "n_valid_canonical": len(canonical),
        "distinct_canonical": distinct_canonical,
        "distinct_canonical_fraction": distinct_canonical / len(canonical)
        if canonical
        else float("nan"),
        "most_common_canonical": most_common,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-epochs", type=int, required=True, choices=[100, 250, 800, 1000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/vae_runs"))
    parser.add_argument("--n-eval-samples", type=int, default=5000)
    parser.add_argument("--n-example-smiles", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        default=None,
        help="load an already-trained checkpoint instead of training (re-run diagnostics only)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="print full metrics every N epochs (default: every epoch); 0 disables printed lines",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)

    train_smiles, valid_smiles = load_splits(args.processed_dir)
    print(f"Loaded {len(train_smiles)} train / {len(valid_smiles)} valid molecules")

    tokenizer = SmilesTokenizer.from_data(train_smiles)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    config = VAEConfig.nominal(d_z=args.latent_dim)
    model = VAE(tokenizer, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameter count: {n_params:,}")

    elapsed = 0.0
    history = None
    trainer_config = TrainerConfig.nominal(n_epochs=args.n_epochs)

    if args.checkpoint_in is not None:
        print(f"Loading checkpoint from {args.checkpoint_in} (skipping training)")
        model.load_state_dict(torch.load(args.checkpoint_in, map_location=device))
    else:
        collate = build_collate_fn(tokenizer, device=device)
        train_loader = DataLoader(train_smiles, batch_size=64, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(valid_smiles, batch_size=64, shuffle=False, collate_fn=collate)

        trainer = VAETrainer(trainer_config)
        print(
            f"Training for {trainer_config.n_epoch} epochs "
            f"(nominal, lr={trainer_config.lr_start})..."
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"n_ep_{args.n_epochs}_seed_{args.seed}.history.csv"
        start = time.time()
        history = trainer.fit(
            model, train_loader, val_loader, log_every=args.log_every, csv_path=csv_path
        )
        elapsed = time.time() - start
        print(f"Training finished in {elapsed / 60:.1f} min")
        print(f"Wrote per-epoch metrics to {csv_path}")

    print(f"Sampling {args.n_eval_samples} molecules for evaluation...")
    model.eval()
    generated = model.sample(args.n_eval_samples, max_len=args.max_len)

    valid_report = compute_property_report(valid_smiles)
    generated_report = compute_property_report(generated)
    wasserstein = compute_wasserstein_report(reference=valid_report, sample=generated_report)

    distribution_summary = print_distribution_comparison(valid_report, generated_report)
    uniqueness = check_uniqueness(generated)

    print("\nUniqueness check (mode-collapse signature):")
    print(f"  distinct raw strings:      {uniqueness['distinct_raw_fraction']:.1%}")
    print(f"  distinct canonical mols:   {uniqueness['distinct_canonical_fraction']:.1%}")
    if uniqueness["most_common_canonical"]:
        print("  most common generated molecules:")
        for smi, count in uniqueness["most_common_canonical"]:
            print(f"    {count:>5}x  {smi}")

    rng = np.random.default_rng(args.seed)
    example_idx = rng.choice(
        len(generated), size=min(args.n_example_smiles, len(generated)), replace=False
    )
    examples = [generated[i] for i in example_idx]
    print(f"\n{len(examples)} example generated SMILES:")
    for smi in examples:
        print(f"  {smi!r}")

    results = {
        "n_epochs_requested": args.n_epochs,
        "n_epochs_actual": trainer_config.n_epoch,
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "train_size": len(train_smiles),
        "valid_size": len(valid_smiles),
        "n_eval_samples": args.n_eval_samples,
        "elapsed_seconds": elapsed,
        "final_train_loss": history[-2].loss if history is not None else None,
        "final_val_loss": history[-1].loss if history is not None else None,
        "metrics": {
            "IntDiv": generated_report.int_div,
            "W(LogP)": wasserstein.logp,
            "W(SA)": wasserstein.sa,
            "W(QED)": wasserstein.qed,
            "W(Weight)": wasserstein.weight,
            "validity": generated_report.validity,
        },
        "distribution_summary": distribution_summary,
        "uniqueness": uniqueness,
        "example_smiles": examples,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"n_ep_{args.n_epochs}_seed_{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    if args.checkpoint_in is None:
        ckpt_path = args.output_dir / f"n_ep_{args.n_epochs}_seed_{args.seed}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Wrote {ckpt_path}")

        vocab_path = args.output_dir / f"n_ep_{args.n_epochs}_seed_{args.seed}.vocab.json"
        tokenizer.save(vocab_path)
        print(f"Wrote {vocab_path}")

    print()
    print(json.dumps(results["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())