"""WGAN-GP losses (paper §2.2.2/§2.4.2), ported from MOSES's
moses/latentgan/model.py::LatentGAN.compute_gradient_penalty and
moses/latentgan/trainer.py's loss expressions.

Three pieces, matching MOSES's actual training loop exactly (fetched from
source, not reconstructed from the paper's prose alone):

    gradient_penalty(critic, real, fake)
        E[(||grad_interp D(interp)||_2 - 1)^2], interp = alpha*real +
        (1-alpha)*fake, alpha ~ Uniform[0,1] per-sample (not per-batch).

    critic_loss(real_validity, fake_validity, gradient_penalty, lambda_gp)
        -mean(D(real)) + mean(D(fake)) + lambda_gp * gradient_penalty
        (MOSES's own d_loss expression, lambda_gp = 10 in the paper)

    generator_loss(fake_validity)
        -mean(D(G(noise)))

The discriminator trains on EVERY batch; the generator trains only every
n_critic-th batch (paper: n_critic=5 for "nominal", 1 for "tuned") -- that
ratio lives in the trainer (gan_trainer.py), not here, since it's a training-
loop concern, not a loss-function one.
"""

from __future__ import annotations

import torch
from torch import autograd, nn


def gradient_penalty(
    critic: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor
) -> torch.Tensor:
    """E[(||grad_interp D(interp)||_2 - 1)^2] (paper's WGAN-GP penalty term).

    Interpolates real and fake samples along the line between them with a
    fresh Uniform[0,1] weight PER SAMPLE (not one shared weight for the whole
    batch), matching MOSES's ``alpha = Tensor(np.random.random((batch, 1)))``.

    Args:
        critic: The discriminator/critic network.
        real_samples: (batch, D_l) real latent vectors.
        fake_samples: (batch, D_l) generated latent vectors. Must have the
            same batch size as real_samples.

    Returns:
        Scalar gradient penalty.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device, dtype=real_samples.dtype)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)

    grad_outputs = torch.ones_like(d_interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def critic_loss(
    real_validity: torch.Tensor,
    fake_validity: torch.Tensor,
    gp: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """-E[D(real)] + E[D(fake)] + lambda_gp * gradient_penalty (MOSES's d_loss)."""
    return -real_validity.mean() + fake_validity.mean() + lambda_gp * gp


def generator_loss(fake_validity: torch.Tensor) -> torch.Tensor:
    """-E[D(G(noise))] (paper's generator objective: maximise critic score on fakes)."""
    return -fake_validity.mean()