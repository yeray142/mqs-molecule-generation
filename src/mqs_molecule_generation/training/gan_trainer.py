"""Classical WGAN-GP training loop (paper §2.2.2/§2.4.2), ported from
MOSES's moses/latentgan/trainer.py.

Operates on the VAE's LATENT SPACE. MOSES's own generic trainer sources its
"real" latent vectors from a separate, pretrained "heteroencoder" pipeline
(a different model entirely, not a VAE) -- this project instead uses the
paper's own VAE (``models.vae.VAE.encode_mu``) to encode the real training
set once, up front, outside this trainer. That's a deliberate substitution,
not an incidental difference: the paper's pipeline is SMILES -> VAE encoder
-> latent z -> GAN -> VAE decoder -> SMILES, with no separate heteroencoder
anywhere. Encoding happens once, in the calling script (see
scripts/train_gan.py), not inside fit() -- this trainer just takes a tensor
of real latent vectors.

Two things ported exactly from source rather than reconstructed from the
paper's prose, both confirmed against MOSES's own trainer.py:

- Noise is drawn from Uniform[-1, 1] (``models.classical_gan`` expects this;
  NOT Gaussian, a common assumption that would be wrong here).
- The discriminator trains on EVERY batch; the generator trains only every
  n_critic-th batch (``if batch_idx % n_critic == 0``), with a FRESH sample
  of fake latents (not reusing the discriminator step's fakes). n_critic=5
  for "nominal", 1 for "tuned" -- see gan_trainer_config.py.

One deliberate (not quirk-preserving) deviation from MOSES's literal code:
fake latents are ``.detach()``ed before the discriminator's own loss/backward
pass. MOSES's source doesn't show this explicitly, which would compute (and
immediately discard, since only the discriminator's optimizer steps) wasted
gradients through the generator during the critic's own update. This changes
nothing about the VALUE of any loss or the resulting training dynamics --
only avoids unnecessary and potentially confusing autograd bookkeeping -- so
it's treated as a safe cleanup, not a paper-fidelity concern (unlike the
BatchNorm eps=0.8 quirk in classical_gan.py, which IS preserved exactly
because it changes actual numbers).
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from mqs_molecule_generation.models.classical_gan import Discriminator, Generator
from mqs_molecule_generation.training.gan_trainer_config import GANEpochMetrics, GANTrainerConfig
from mqs_molecule_generation.training.wgan_gp import critic_loss, generator_loss, gradient_penalty

__all__ = ["GANEpochMetrics", "GANTrainer", "GANTrainerConfig", "sample_noise"]


def sample_noise(n_batch: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Uniform[-1, 1] noise, matching MOSES's Sampler.sample exactly (not Gaussian)."""
    return torch.rand(n_batch, latent_dim, device=device) * 2 - 1


class GANTrainer:
    """Trains a classical Generator/Discriminator pair on VAE-encoded latents."""

    def __init__(self, config: GANTrainerConfig) -> None:
        self.config = config

    def fit(
        self,
        generator: Generator,
        discriminator: Discriminator,
        real_latents: torch.Tensor,
        log_every: int = 1,
        csv_path: Path | None = None,
        seed: int | None = None,
    ) -> list[GANEpochMetrics]:
        """Train for config.n_epochs epochs on `real_latents` (n_real, D_l).

        Args:
            generator: The classical Generator (models.classical_gan.Generator).
            discriminator: The classical Discriminator.
            real_latents: (n_real, D_l) tensor of VAE-encoded real molecules
                (e.g. from VAE.encode_mu on the training set).
            log_every: Print a full metrics line via tqdm.write every this
                many epochs (default: every epoch). 0 disables printed lines.
            csv_path: If given, write every epoch's metrics as CSV, flushed
                after every row.
            seed: If given, used only to seed torch's RNG before training
                (noise sampling, batch shuffling) -- for reproducibility
                across the paper's 5-seed scenario runs.

        Returns:
            One GANEpochMetrics per epoch.
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = next(generator.parameters()).device
        real_latents = real_latents.to(device)
        n_real = real_latents.size(0)

        optimizer_d = optim.Adam(
            discriminator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2)
        )
        optimizer_g = optim.Adam(
            generator.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2)
        )
        scheduler_d = optim.lr_scheduler.StepLR(
            optimizer_d, self.config.step_size, self.config.gamma
        )
        scheduler_g = optim.lr_scheduler.StepLR(
            optimizer_g, self.config.step_size, self.config.gamma
        )

        csv_file = None
        csv_writer = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="")
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=["epoch", "d_loss", "g_loss", "gradient_penalty", "lr"]
            )
            csv_writer.writeheader()

        history: list[GANEpochMetrics] = []
        try:
            pbar = tqdm(range(self.config.n_epochs))
            for epoch in pbar:
                scheduler_d.step()
                scheduler_g.step()

                perm = torch.randperm(n_real, device=device)
                d_losses: list[float] = []
                g_losses: list[float] = []
                gp_values: list[float] = []
                lr = optimizer_d.param_groups[0]["lr"]

                for batch_idx, start in enumerate(range(0, n_real, self.config.batch_size)):
                    real_batch = real_latents[perm[start : start + self.config.batch_size]]
                    batch_size = real_batch.size(0)
                    if batch_size < 2:
                        continue  # BatchNorm1d in the generator requires batch_size > 1

                    # ---- Discriminator step (every batch) ----
                    optimizer_d.zero_grad()
                    noise = sample_noise(batch_size, self.config.latent_dim, device)
                    fake_batch = generator(noise).detach()

                    real_validity = discriminator(real_batch)
                    fake_validity = discriminator(fake_batch)
                    gp = gradient_penalty(discriminator, real_batch, fake_batch)
                    d_loss = critic_loss(real_validity, fake_validity, gp, self.config.lambda_gp)

                    d_loss.backward()
                    optimizer_d.step()

                    d_losses.append(d_loss.item())
                    gp_values.append(gp.item())

                    # ---- Generator step (every n_critic-th batch) ----
                    if batch_idx % self.config.n_critic == 0:
                        optimizer_g.zero_grad()
                        noise = sample_noise(batch_size, self.config.latent_dim, device)
                        fake_batch = generator(noise)
                        fake_validity = discriminator(fake_batch)
                        g_loss = generator_loss(fake_validity)

                        g_loss.backward()
                        optimizer_g.step()
                        g_losses.append(g_loss.item())

                mean_d_loss = sum(d_losses) / max(len(d_losses), 1)
                mean_g_loss = sum(g_losses) / max(len(g_losses), 1)
                mean_gp = sum(gp_values) / max(len(gp_values), 1)
                metrics = GANEpochMetrics(
                    epoch=epoch,
                    d_loss=mean_d_loss,
                    g_loss=mean_g_loss,
                    gradient_penalty=mean_gp,
                    lr=lr,
                )
                history.append(metrics)

                pbar.set_postfix(
                    {
                        "d_loss": f"{mean_d_loss:.4f}",
                        "g_loss": f"{mean_g_loss:.4f}",
                        "gp": f"{mean_gp:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                if log_every > 0 and epoch % log_every == 0:
                    tqdm.write(
                        f"epoch {epoch:>5d}/{self.config.n_epochs}  "
                        f"d_loss={mean_d_loss:.4f}  g_loss={mean_g_loss:.4f}  "
                        f"gp={mean_gp:.4f}  lr={lr:.2e}"
                    )
                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "epoch": metrics.epoch,
                            "d_loss": metrics.d_loss,
                            "g_loss": metrics.g_loss,
                            "gradient_penalty": metrics.gradient_penalty,
                            "lr": metrics.lr,
                        }
                    )
                    csv_file.flush()
        finally:
            if csv_file is not None:
                csv_file.close()

        return history