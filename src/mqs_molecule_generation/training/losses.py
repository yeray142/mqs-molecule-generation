"""WGAN loss functions for LatentStyleGAN (stubbed)."""

from __future__ import annotations


def wasserstein_loss(
    discriminator_output_real: float,
    discriminator_output_fake: float,
) -> float:
    """Wasserstein GAN loss function.

    L = -E[D(real)] + E[D(fake)]

    Args:
        discriminator_output_real: D(real samples).
        discriminator_output_fake: D(fake samples).

    Returns:
        Wasserstein loss value.
    """
    raise NotImplementedError(
        "TODO(paper §4): Wasserstein loss computation. "
        "See arXiv:2603.22399 §4 for the discriminator objective."
    )


def gradient_penalty(
    discriminator_fn: object,
    real_samples: object,
    fake_samples: object,
    lambda_gp: float = 10.0,
) -> float:
    """Compute gradient penalty for WGAN-GP.

    E[||grad D(lambda)|| - 1]^2 where lambda = a*real + (1-a)*fake, a ~ Uniform[0,1]

    Args:
        discriminator_fn: Discriminator function.
        real_samples: Real samples.
        fake_samples: Fake samples.
        lambda_gp: Gradient penalty coefficient.

    Returns:
        Gradient penalty value.
    """
    raise NotImplementedError(
        "TODO(paper §4.2): Gradient penalty computation. "
        "See arXiv:2603.22399 §4.2 for the WGAN-GP formulation."
    )


def generator_loss(discriminator_output_fake: float) -> float:
    """Generator loss for WGAN.

    L_G = -E[D(fake)]  (maximize discriminator output on fake samples)

    Args:
        discriminator_output_fake: D(fake samples).

    Returns:
        Generator loss value.
    """
    raise NotImplementedError(
        "TODO(paper §4): Generator loss computation. "
        "See arXiv:2603.22399 §4 for the generator objective."
    )
