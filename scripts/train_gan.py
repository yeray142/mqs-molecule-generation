#!/usr/bin/env python
"""Train the classical WGAN-GP baseline on a VAE's latent space (paper §2.4.2).

    uv run python scripts/train_gan.py \\
        --vae-checkpoint results/vae_runs/tuned_n_ep_1000_seed_0.pt \\
        --gan-scenario nominal --seed 0

    uv run python scripts/train_gan.py \\
        --vae-checkpoint results/vae_runs/tuned_n_ep_1000_seed_0.pt \\
        --gan-scenario tuned --seed 0

Pipeline: SMILES -> (already-trained) VAE encoder -> real latent vectors ->
train Generator/Discriminator via WGAN-GP on those latents -> sample fake
latents from the trained Generator -> (same) VAE decoder -> generated
SMILES -> the same 10-metric significance suite eval_vae_reference.py uses.

Needs a VAE checkpoint from scripts/train_vae.py, ideally the TUNED scenario
at N_epochs=1000 (the paper's production training length -- the paper's own
Table 3 always references the tuned VAE, not nominal). --vae-scenario tells
this script how to reconstruct that checkpoint's architecture; get it wrong
and loading fails loudly (a state_dict shape mismatch), not silently.

The paper's own scenario runs use 5 seeds and average; run this once per
seed (--seed 0..4) and aggregate with scripts/check_table3.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mqs_molecule_generation.data.moses import load_splits
from mqs_molecule_generation.metrics.moses_metrics import compute_significance_metrics
from mqs_molecule_generation.models.checkpoint_io import (
    load_tokenizer_for_checkpoint,
    load_vae_checkpoint,
)
from mqs_molecule_generation.models.classical_gan import Discriminator, Generator
from mqs_molecule_generation.models.vae import VAE
from mqs_molecule_generation.training.gan_trainer import GANTrainer
from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_all(
    model: VAE, smiles_list: list[str], device: torch.device, batch_size: int = 1024
) -> torch.Tensor:
    """Encode a large SMILES list to VAE latents (mu) in chunks, avoiding a
    single very large pack_sequence call.
    """
    chunks = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            x = [
                torch.tensor(
                    model.tokenizer.encode(s, add_bos=True, add_eos=True),
                    dtype=torch.long,
                    device=device,
                )
                for s in batch
            ]
            chunks.append(model.encode_mu(x))
    return torch.cat(chunks, dim=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-vocab", type=Path, default=None)
    parser.add_argument(
        "--vae-scenario",
        choices=["nominal", "tuned"],
        default="tuned",
        help="how the VAE checkpoint was trained (default: tuned, matching the paper's reference)",
    )
    parser.add_argument("--gan-scenario", choices=["nominal", "tuned"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-eval-samples", type=int, default=30_000, help="paper's default gen. batch"
    )
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/significance_runs"))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)

    train_smiles, _ = load_splits(args.processed_dir)
    print(f"Loaded {len(train_smiles)} train molecules")

    tokenizer = load_tokenizer_for_checkpoint(
        args.vae_checkpoint, args.vae_vocab, args.processed_dir
    )
    vae = load_vae_checkpoint(args.vae_checkpoint, tokenizer, args.vae_scenario, device)
    latent_dim = vae.q_mu.out_features

    print(f"Encoding {len(train_smiles)} real molecules to latent space (D_l={latent_dim})...")
    real_latents = encode_all(vae, train_smiles, device)
    print(f"Real latents: {tuple(real_latents.shape)}")

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)
    gan_config = (
        GANTrainerConfig.nominal(latent_dim=latent_dim)
        if args.gan_scenario == "nominal"
        else GANTrainerConfig.tuned(latent_dim=latent_dim)
    )
    print(f"GAN scenario: {args.gan_scenario} (n_critic={gan_config.n_critic})")

    trainer = GANTrainer(gan_config)
    run_name = f"{args.gan_scenario}_gan_seed_{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{run_name}.history.csv"

    start = time.time()
    history = trainer.fit(
        generator,
        discriminator,
        real_latents,
        log_every=args.log_every,
        csv_path=csv_path,
        seed=args.seed,
    )
    elapsed = time.time() - start
    print(f"GAN training finished in {elapsed / 60:.1f} min")

    print(f"Sampling {args.n_eval_samples} fake latents and decoding via the VAE...")
    generator.eval()
    with torch.no_grad():
        noise = torch.rand(args.n_eval_samples, latent_dim, device=device) * 2 - 1
        fake_latents = generator(noise)
    generated = vae.sample(args.n_eval_samples, max_len=args.max_len, z=fake_latents)

    print("Computing significance metrics...")
    metrics = compute_significance_metrics(generated, train_smiles)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    results = {
        "role": "gan",
        "gan_scenario": args.gan_scenario,
        "vae_scenario": args.vae_scenario,
        "seed": args.seed,
        "latent_dim": latent_dim,
        "n_critic": gan_config.n_critic,
        "n_epochs": gan_config.n_epochs,
        "n_eval_samples": args.n_eval_samples,
        "elapsed_seconds": elapsed,
        "vae_checkpoint": str(args.vae_checkpoint),
        "final_d_loss": history[-1].d_loss,
        "final_g_loss": history[-1].g_loss,
        "metrics": metrics,
    }

    out_path = args.output_dir / f"{run_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")

    gen_path = args.output_dir / f"{run_name}.generator.pt"
    disc_path = args.output_dir / f"{run_name}.discriminator.pt"
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    print(f"Wrote {gen_path}")
    print(f"Wrote {disc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())