"""Classical WGAN-GP discriminator and generator (paper §2.2.2), ported from
MOSES's moses/latentgan/model.py.

This operates on the VAE's LATENT SPACE, not on SMILES directly: the
discriminator's "real" inputs are latent vectors of real training molecules
(this project's ``VAE.encode_mu``, not MOSES's own separate "heteroencoder"
pipeline -- see the module docstring in ``training/gan_trainer.py`` for why
that substitution is the right one for this paper, and why it's a deliberate
choice rather than an assumption). Both discriminator and generator's
input/output dimension is ``D_l``, the VAE's latent dimension (10/20/30 in
the paper's scenarios) -- confirmed from source: MOSES constructs
``Generator(data_shape=(1, D_l))`` with no separate noise dimension, so the
generator's noise input is D_l-dimensional too, the same size as what it
outputs.

One faithfully-preserved quirk: ``nn.BatchNorm1d(out_feat, 0.8)`` in the
generator passes 0.8 as the SECOND positional argument, which in PyTorch's
signature (``num_features, eps, momentum, ...``) is ``eps``, not
``momentum``. This is a well-known copy-paste artifact from the
eriklindernoren/PyTorch-GAN template that MOSES's own code inherited -- the
evident intent was momentum=0.8, but the actual effect is an unusually large
eps (default is 1e-5), which measurably weakens the normalization when the
true activation variance is small relative to 0.8. Kept exactly as-is:
matching the paper's actual training dynamics matters more than correcting
what looks like an upstream mistake. Does not affect parameter counts either
way (see gan_param_count.py).
"""

from __future__ import annotations

from torch import nn


class Discriminator(nn.Module):
    """WGAN-GP critic: D_l -> 512 -> 256 -> 1. No batch norm (standard
    WGAN-GP practice -- batch norm couples samples within a batch, which
    invalidates the per-sample gradient penalty). No output activation: a
    WGAN critic outputs a raw (unbounded) score, not a probability.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        return self.model(z)


class Generator(nn.Module):
    """Classical generator: D_l -> 128 -> 256 -> 512 -> 1024 -> D_l.

    First block (D_l -> 128) and the output layer (1024 -> D_l) have no
    batch norm; the three middle blocks do.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        def block(in_feat: int, out_feat: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # eps=0.8, see module docstring
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, z):
        return self.model(z)