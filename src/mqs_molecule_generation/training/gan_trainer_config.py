"""Torch-free pieces of the classical GAN training config (paper §2.2.2/§2.4.2).

Split out from gan_trainer.py for the same reason as the VAE's
trainer_config.py: keeps this importable without torch installed.

Hyperparameters ported from MOSES's moses/latentgan/config.py defaults,
with the paper's explicit overrides applied and documented -- MOSES's own
default b2 is 0.999; the paper states beta=(0.5, 0.9), so b2=0.9 here is a
deliberate override, not a MOSES default carried over by accident.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GANTrainerConfig:
    """Classical WGAN-GP training hyperparameters (paper §2.4.2, Table 1).

    lr/b1/lambda_gp/n_epochs/batch_size match both "nominal" and "tuned"
    scenarios; n_critic is the one value that differs between them (5 vs 1)
    -- see nominal()/tuned().
    """

    lr: float = 2e-4  # paper: Adam lr = 2e-4 (matches MOSES's own default)
    b1: float = 0.5  # matches MOSES's own default
    b2: float = 0.9  # paper's override of MOSES's own default (0.999)
    lambda_gp: float = 10.0  # paper: gradient penalty coefficient = 10
    n_critic: int = 5  # nominal: 5, tuned: 1 -- see presets below
    n_epochs: int = 100  # paper's override of MOSES's own default (2000)
    batch_size: int = 64  # matches MOSES's own default
    latent_dim: int = 10  # D_l -- must match the VAE checkpoint's d_z

    # StepLR(step_size, gamma), called once per epoch. MOSES's own default
    # gamma=1 makes this a no-op (constant lr) -- matching the paper, which
    # states a single flat lr with no decay schedule for the GAN.
    step_size: int = 10
    gamma: float = 1.0

    @classmethod
    def nominal(cls, latent_dim: int = 10) -> GANTrainerConfig:
        """Paper's "nominal" GAN scenario: n_critic=5 (Table 1)."""
        return cls(n_critic=5, latent_dim=latent_dim)

    @classmethod
    def tuned(cls, latent_dim: int = 10) -> GANTrainerConfig:
        """Paper's "tuned" GAN scenario: n_critic=1 (Table 1)."""
        return cls(n_critic=1, latent_dim=latent_dim)


@dataclass
class GANEpochMetrics:
    """Summary of one epoch's GAN training pass."""

    epoch: int
    d_loss: float
    g_loss: float
    gradient_penalty: float
    lr: float