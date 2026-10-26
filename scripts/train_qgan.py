#!/usr/bin/env python
"""Train the latent style-based quantum WGAN-GP on a VAE's latent space (Phase 3, step 14).

    uv run python scripts/train_qgan.py \\
        --vae-checkpoint results/vae_runs/tuned_n_ep_1000_seed_0.epoch0200.pt \\
        --gan-scenario nominal --ansatz bel --n-qubits 5 --n-layers 2 --readout dual --seed 0

Same pipeline as scripts/train_gan.py, with only the generator swapped:

    SMILES -> VAE encoder -> real latents -> LatentScaler (-> [-1, 1])
    noise (batch, n_qb) ~ U[-1, 1] -> QuantumGenerator -> fake (batch, D_l) in [-1, 1]
    WGAN-GP vs the SAME classical Discriminator as the baseline
    sampling: QuantumGenerator -> LatentScaler.inverse -> VAE decoder -> SMILES -> metrics

Choices this script makes that the paper text available here does not pin down (all are
flags, all are written into the run's JSON):
  * --latent-scaling (default none, i.e. the generator's [-1, 1] output is fed to the
    decoder unchanged): the paper never mentions rescaling, and two things support that
    reading. Its own decoder study (supplementary Sec. 5, Fig. 18) shows the real latent
    vectors are close to Normal(0, 1), so [-1, 1] already covers the bulk of the latent
    space; and its BEL QGAN reports <Weight> = 261.4 against a train-set 261.5, which is
    what you get when the generator covers the CENTRE of the latent space rather than its
    full range. minmax/zscore remain available (see training/latent_scaling.py).
  * --angle-mode (default literal): Eq. 4 as printed, theta = 2*pi*tanh(.) in (-2pi, 2pi).
    The paper's prose says the angles lie in [0, 2pi], which its own Eq. 4 does not
    produce; the formula is taken as authoritative.
  * Optimiser and schedule: the classical GAN's. This is now confirmed -- Sec. 3.1 says
    the default scenario uses "latent dimension 10, tuned VAE, and classical GAN
    hyperparameters" and that "the quantum GANs are trained with 100 epochs".
    Table 4 and Table 5 use the TUNED GAN settings (--gan-scenario tuned, n_critic=1).
  * W ~ N(0, --weight-init-std), b = 0.

The run's D_l (n_qb for single readout, 2*n_qb for dual) must equal the VAE checkpoint's
d_z; e.g. the nominal setting (n_qb=5, dual) needs a D_l=10 VAE.

Seeds 0..4 per configuration, then aggregate with scripts/check_table4.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mqs_molecule_generation.circuits.angle_map import ANGLE_MODES
from mqs_molecule_generation.data.moses import load_splits
from mqs_molecule_generation.metrics.moses_metrics import compute_significance_metrics
from mqs_molecule_generation.models.checkpoint_io import (
    load_tokenizer_for_checkpoint,
    load_vae_checkpoint,
)
from mqs_molecule_generation.models.classical_gan import Discriminator
from mqs_molecule_generation.models.quantum_generator import QuantumGenerator
from mqs_molecule_generation.training.gan_trainer import GANTrainer, sample_noise
from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig
from mqs_molecule_generation.training.latent_scaling import SCALING_MODES, LatentScaler


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_all(vae, smiles_list: list[str], device: torch.device, batch_size: int = 1024):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            x = [
                torch.tensor(
                    vae.tokenizer.encode(s, add_bos=True, add_eos=True),
                    dtype=torch.long,
                    device=device,
                )
                for s in batch
            ]
            chunks.append(vae.encode_mu(x))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def sample_generator(
    generator: QuantumGenerator, n: int, device: torch.device, chunk: int = 1024
) -> torch.Tensor:
    """n generated samples in the generator's [-1, 1] space, chunked to bound memory."""
    out = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        out.append(generator(sample_noise(m, generator.noise_dim, device)))
    return torch.cat(out, dim=0)


def run_name(args: argparse.Namespace) -> str:
    return (
        f"{args.gan_scenario}_qgan_{args.ansatz}_q{args.n_qubits}_l{args.n_layers}_"
        f"{args.readout}_seed_{args.seed}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--vae-checkpoint", type=Path, required=True)
    p.add_argument("--vae-vocab", type=Path, default=None)
    p.add_argument("--vae-scenario", choices=["nominal", "tuned"], default="tuned")
    p.add_argument("--gan-scenario", choices=["nominal", "tuned"], required=True)
    p.add_argument("--ansatz", choices=["simple", "bel"], default="bel")
    p.add_argument("--n-qubits", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--readout", choices=["single", "dual"], default="dual")
    p.add_argument("--angle-mode", choices=list(ANGLE_MODES), default="literal")
    p.add_argument("--latent-scaling", choices=list(SCALING_MODES), default="none")
    p.add_argument("--scaling-bound", type=float, default=1.0, help="minmax target range")
    p.add_argument("--weight-init-std", type=float, default=1.0)
    p.add_argument(
        "--legacy-layout",
        action="store_true",
        help="reproduce the pre-audit circuits: no leading Hadamard layer on BEL, and no "
        "CX staircase after the simple ansatz's last layer (see circuits/ansatz.py)",
    )
    p.add_argument("--qdevice", default="default.qubit", help="PennyLane device name")
    p.add_argument("--lr", type=float, default=None, help="override the baseline's 2e-4")
    p.add_argument("--n-epochs", type=int, default=None, help="override the baseline's 100")
    p.add_argument("--max-real", type=int, default=None, help="debug: subsample real latents")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval-samples", type=int, default=30_000)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("results/significance_runs"))
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", default="cpu", help="torch device for VAE + critic")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)

    train_smiles, _ = load_splits(args.processed_dir)
    print(f"Loaded {len(train_smiles)} train molecules")

    tokenizer = load_tokenizer_for_checkpoint(
        args.vae_checkpoint, args.vae_vocab, args.processed_dir
    )
    vae = load_vae_checkpoint(args.vae_checkpoint, tokenizer, args.vae_scenario, device)
    vae_dim = vae.q_mu.out_features

    generator = QuantumGenerator(
        args.n_qubits,
        args.n_layers,
        args.ansatz,
        readout=args.readout,
        angle_mode=args.angle_mode,
        weight_init_std=args.weight_init_std,
        figure_layout=not args.legacy_layout,
        device=args.qdevice,
    ).to(device)
    if generator.latent_dim != vae_dim:
        p.error(
            f"QGAN D_l = {generator.latent_dim} ({args.readout} readout, n_qb={args.n_qubits}) "
            f"but the VAE checkpoint has d_z = {vae_dim}"
        )
    n_params = sum(q.numel() for q in generator.parameters())
    print(f"Quantum generator: {generator.extra_repr()}  (trainable params: {n_params})")

    print(f"Encoding {len(train_smiles)} real molecules (D_l={vae_dim})...")
    real_latents = encode_all(vae, train_smiles, device)
    if args.max_real is not None:
        real_latents = real_latents[torch.randperm(len(real_latents))[: args.max_real]]
    scaler = LatentScaler.fit(real_latents, args.latent_scaling, bound=args.scaling_bound)
    real_scaled = scaler.transform(real_latents)
    # 1e-6 slack: with minmax, the extreme latents land on +-1 up to float32 rounding.
    inside = ((real_scaled.abs() <= 1 + 1e-6).all(dim=1)).float().mean().item()
    print(
        f"Latent scaling: {args.latent_scaling}; real latents in [-1,1]^D_l after scaling: "
        f"{inside:.1%}"
    )

    discriminator = Discriminator(vae_dim).to(device)
    base = (
        GANTrainerConfig.nominal(latent_dim=vae_dim)
        if args.gan_scenario == "nominal"
        else GANTrainerConfig.tuned(latent_dim=vae_dim)
    )
    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.n_epochs is not None:
        overrides["n_epochs"] = args.n_epochs
    config = GANTrainerConfig(**{**base.__dict__, **overrides})
    print(
        f"GAN scenario: {args.gan_scenario} (n_critic={config.n_critic}, lr={config.lr}, "
        f"epochs={config.n_epochs}, noise_dim={generator.noise_dim})"
    )

    name = run_name(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    history = GANTrainer(config).fit(
        generator,
        discriminator,
        real_scaled,
        log_every=args.log_every,
        csv_path=args.output_dir / f"{name}.history.csv",
        seed=args.seed,
    )
    elapsed = time.time() - start
    print(f"QGAN training finished in {elapsed / 60:.1f} min")

    print(f"Sampling {args.n_eval_samples} latents and decoding via the VAE...")
    generator.eval()
    fake_scaled = sample_generator(generator, args.n_eval_samples, device)
    fake_latents = scaler.inverse(fake_scaled)
    generated = vae.sample(args.n_eval_samples, max_len=args.max_len, z=fake_latents)

    print("Computing significance metrics...")
    metrics = compute_significance_metrics(generated, train_smiles)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    results = {
        "role": "qgan",
        "gan_scenario": args.gan_scenario,
        "vae_scenario": args.vae_scenario,
        "seed": args.seed,
        "ansatz": args.ansatz,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "readout": args.readout,
        "angle_mode": args.angle_mode,
        "latent_scaling": scaler.state_dict(),
        "weight_init_std": args.weight_init_std,
        "figure_layout": not args.legacy_layout,
        "latent_dim": vae_dim,
        "noise_dim": generator.noise_dim,
        "n_generator_params": n_params,
        "n_critic": config.n_critic,
        "lr": config.lr,
        "n_epochs": config.n_epochs,
        "n_real": len(real_latents),
        "n_eval_samples": args.n_eval_samples,
        "elapsed_seconds": elapsed,
        "vae_checkpoint": str(args.vae_checkpoint),
        "final_d_loss": history[-1].d_loss,
        "final_g_loss": history[-1].g_loss,
        "metrics": metrics,
    }
    out = args.output_dir / f"{name}.json"
    out.write_text(json.dumps(results, indent=2))
    torch.save(
        {"generator": generator.state_dict(), "latent_scaling": scaler.state_dict()},
        args.output_dir / f"{name}.generator.pt",
    )
    torch.save(discriminator.state_dict(), args.output_dir / f"{name}.discriminator.pt")
    print(f"Wrote {out} (+ .generator.pt, .discriminator.pt, .history.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())