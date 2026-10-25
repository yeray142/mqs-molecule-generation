"""Step 13: differentiability of the quantum generator.

Gate (defined by this project, not quoted from the paper -- the paper text I could
access does not state one for this step):
  * every W and b receives a finite gradient (no NaN/Inf);
  * backprop == parameter-shift == central finite difference (float64);
  * parameter broadcasting over the batch == a per-sample loop;
  * the gradient reaches W and b through a classical critic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from torch import nn  # noqa: E402

from mqs_molecule_generation.circuits.angle_map import ANGLE_MODES  # noqa: E402
from mqs_molecule_generation.models.grad_check import (  # noqa: E402
    batch_consistency,
    compare_gradients,
    critic_gradient,
    gradient_report,
)
from mqs_molecule_generation.models.quantum_generator import QuantumGenerator  # noqa: E402

CASES = [
    ("simple", 3, 2, "single"),
    ("simple", 3, 2, "dual"),
    ("bel", 3, 1, "single"),
    ("bel", 3, 2, "dual"),
    ("bel", 5, 2, "dual"),
]


def _make(ansatz, n_qb, n_l, readout, mode="literal", seed=0):
    torch.manual_seed(seed)
    gen = QuantumGenerator(n_qb, n_l, ansatz, readout=readout, angle_mode=mode).double()
    xi = (torch.rand(6, gen.noise_dim, dtype=torch.float64) * 2) - 1
    return gen, xi


@pytest.mark.parametrize(("ansatz", "n_qb", "n_l", "readout"), CASES)
def test_all_gradients_finite_and_alive(ansatz, n_qb, n_l, readout):
    gen, xi = _make(ansatz, n_qb, n_l, readout)
    rep = gradient_report(gen, xi)
    assert rep.n_params == gen.ansatz.n_params
    assert rep.all_finite
    assert rep.n_dead == 0, f"dead parameters: {rep.dead_names}"


@pytest.mark.parametrize("mode", ANGLE_MODES)
def test_gradients_finite_for_every_angle_mode(mode):
    gen, xi = _make("bel", 3, 2, "dual", mode=mode)
    rep = gradient_report(gen, xi)
    assert rep.all_finite
    assert rep.n_dead == 0, rep.dead_names


@pytest.mark.parametrize(("ansatz", "n_qb", "n_l", "readout"), CASES)
def test_backprop_matches_parameter_shift_and_finite_difference(ansatz, n_qb, n_l, readout):
    gen, xi = _make(ansatz, n_qb, n_l, readout)
    cmp = compare_gradients(gen, xi)
    assert cmp.grad_scale > 0
    assert cmp.rel_backprop_vs_shift < 1e-7, cmp
    assert cmp.rel_backprop_vs_fd < 1e-5, cmp


@pytest.mark.parametrize(("ansatz", "n_qb", "n_l", "readout"), CASES[:4])
def test_broadcast_batch_matches_per_sample_loop(ansatz, n_qb, n_l, readout):
    gen, xi = _make(ansatz, n_qb, n_l, readout)
    d_out, d_grad = batch_consistency(gen, xi)
    assert d_out < 1e-12
    assert d_grad < 1e-10


@pytest.mark.parametrize(("ansatz", "n_qb", "n_l", "readout"), CASES)
def test_gradient_reaches_parameters_through_critic(ansatz, n_qb, n_l, readout):
    gen, xi = _make(ansatz, n_qb, n_l, readout)
    torch.manual_seed(1)
    critic = nn.Sequential(nn.Linear(gen.latent_dim, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    critic = critic.double()
    rep = critic_gradient(gen, critic, xi)
    assert rep.all_finite
    assert rep.n_dead == 0, rep.dead_names


def test_optimizer_step_changes_loss():
    """A few Adam steps on a fixed critic loss must actually lower it (end-to-end sanity)."""
    gen, xi = _make("bel", 3, 2, "dual")
    torch.manual_seed(2)
    critic = nn.Linear(gen.latent_dim, 1).double()
    opt = torch.optim.Adam(gen.parameters(), lr=0.05)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = -critic(gen(xi)).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 1e-3, losses