"""Exact classical GAN parameter-count formula (paper §2.2.2, N_params gate).

Pure Python, no torch dependency -- lets the gate (705,162 / 716,692 / 728,222
params at D_l = 10 / 20 / 30) be checked without needing torch importable,
and gives an independent cross-check against the real ``torch.nn.Module`` in
``classical_gan.py`` (see ``test_classical_gan.py``).

Ported from MOSES's actual ``moses/latentgan/model.py`` (fetched from
source, not assumed) rather than reconstructed from the paper's prose alone.
Confirmed against all three targets: the paper's reported N_params is the
GENERATOR alone -- the discriminator's own param count (which also varies
with D_l via its first layer) is a different, unreported number.
"""

from __future__ import annotations


def linear_param_count(in_features: int, out_features: int) -> int:
    """Exact parameter count for torch.nn.Linear (weight + bias)."""
    return in_features * out_features + out_features


def batchnorm1d_param_count(num_features: int) -> int:
    """Exact parameter count for torch.nn.BatchNorm1d (weight + bias only).

    running_mean/running_var are buffers, not parameters -- not counted by
    sum(p.numel() for p in model.parameters()), so not counted here either.
    """
    return 2 * num_features


def discriminator_param_count(d_l: int) -> int:
    """Discriminator: Linear(D_l, 512) -> LeakyReLU -> Linear(512, 256) ->
    LeakyReLU -> Linear(256, 1). No batch norm (standard WGAN-GP practice --
    batch norm couples samples within a batch, which invalidates the
    per-sample gradient penalty). Not the metric the paper's N_params
    reports, but implemented for completeness and its own tests.
    """
    return linear_param_count(d_l, 512) + linear_param_count(512, 256) + linear_param_count(256, 1)


def generator_param_count(d_l: int) -> int:
    """Generator: D_l -> 128 -> 256 -> 512 -> 1024 -> D_l.

    Layer 1 (D_l -> 128) and the output layer (1024 -> D_l) have no batch
    norm; the three middle blocks (128->256, 256->512, 512->1024) each do.
    This is the metric the paper's Table 2 N_params column reports.
    """
    total = linear_param_count(d_l, 128)
    total += linear_param_count(128, 256) + batchnorm1d_param_count(256)
    total += linear_param_count(256, 512) + batchnorm1d_param_count(512)
    total += linear_param_count(512, 1024) + batchnorm1d_param_count(1024)
    total += linear_param_count(1024, d_l)
    return total