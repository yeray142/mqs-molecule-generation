"""Differentiability checks for the quantum generator (Phase 3, step 13).

The generator is trained by back-propagating a WGAN loss through
``xi --Eq. 4--> theta --QNode--> <O>``. These helpers answer, numerically, whether
that chain is trustworthy before any training run relies on it:

  1. ``gradient_report``      every one of the 2 * n_angles parameters (W, b) receives a
                              finite gradient, and how many are (numerically) dead.
  2. ``compare_gradients``    the training gradient (PennyLane ``backprop``) agrees with
                              (a) the hardware-compatible ``parameter-shift`` rule and
                              (b) a central finite difference, in float64.
  3. ``batch_consistency``    PennyLane parameter broadcasting over the batch gives the
                              same outputs and gradients as a per-sample Python loop.
  4. ``critic_gradient``      the gradient actually reaches W and b through a classical
                              critic (the QGAN generator step), not only through a
                              hand-picked probe loss.

Everything is run in float64 on purpose: the finite-difference and parameter-shift
comparisons are meaningless at float32 resolution.

The probe loss is ``sum(out * c)`` for a fixed random vector ``c``. A plain ``mean(out)``
would let the per-dimension gradients cancel and could hide a genuinely dead parameter;
distinct random weights per output dimension avoid that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mqs_molecule_generation.circuits.pennylane_circuits import stack_expvals
from mqs_molecule_generation.models.quantum_generator import QuantumGenerator

__all__ = [
    "GradComparison",
    "GradReport",
    "batch_consistency",
    "compare_gradients",
    "critic_gradient",
    "forward_unbroadcast",
    "gradient_report",
    "make_probe_loss",
]

#: |grad| below this counts as "dead" (numerically zero) in ``gradient_report``.
DEAD_TOL = 1e-10


@dataclass(frozen=True)
class GradReport:
    n_params: int
    all_finite: bool
    n_dead: int  # parameters whose |grad| < DEAD_TOL
    abs_mean: float
    abs_min: float
    abs_max: float
    dead_names: tuple[str, ...]  # e.g. ("W[3]", "b[3]")


@dataclass(frozen=True)
class GradComparison:
    max_abs_backprop_vs_shift: float
    max_abs_backprop_vs_fd: float
    grad_scale: float  # max |grad| of the backprop gradient, for relative judgement

    @property
    def rel_backprop_vs_shift(self) -> float:
        return self.max_abs_backprop_vs_shift / max(self.grad_scale, 1e-12)

    @property
    def rel_backprop_vs_fd(self) -> float:
        return self.max_abs_backprop_vs_fd / max(self.grad_scale, 1e-12)


def make_probe_loss(latent_dim: int, seed: int = 0) -> torch.Tensor:
    """Fixed random weights, one per output dimension (float64, entries in [0.5, 1.5] * sign)."""
    g = torch.Generator().manual_seed(seed)
    mag = 0.5 + torch.rand(latent_dim, generator=g, dtype=torch.float64)
    sign = torch.where(torch.rand(latent_dim, generator=g) < 0.5, -1.0, 1.0).double()
    return mag * sign


def forward_unbroadcast(gen: QuantumGenerator, xi: torch.Tensor) -> torch.Tensor:
    """Same map as ``gen(xi)`` but one QNode call per sample on 1-D angles (no broadcasting).

    Needed for ``parameter-shift``, which PennyLane cannot apply to broadcast tapes, and
    useful as an independent reference for the broadcast path.
    """
    theta = gen.angles(xi)  # (batch, n_angles), differentiable wrt W, b
    rows = [stack_expvals(gen._qnode(theta[i])) for i in range(theta.shape[0])]
    return torch.stack(rows, dim=0).to(xi.dtype)


def _flat_grad(gen: QuantumGenerator) -> torch.Tensor:
    return torch.cat([gen.W.grad.reshape(-1), gen.b.grad.reshape(-1)])


def _probe_grad(
    gen: QuantumGenerator, xi: torch.Tensor, probe: torch.Tensor, forward=None
) -> torch.Tensor:
    gen.zero_grad(set_to_none=True)
    out = gen(xi) if forward is None else forward(gen, xi)
    (out * probe).sum().backward()
    return _flat_grad(gen)


def gradient_report(
    gen: QuantumGenerator, xi: torch.Tensor, probe_seed: int = 0
) -> GradReport:
    """Gradient of the probe loss wrt every (W, b); `gen` must be float64."""
    probe = make_probe_loss(gen.latent_dim, probe_seed)
    flat = _probe_grad(gen, xi, probe)
    n_angles = gen.W.numel()
    absg = flat.abs()
    dead = [
        f"{'W' if i < n_angles else 'b'}[{i % n_angles}]"
        for i in torch.nonzero(absg < DEAD_TOL).flatten().tolist()
    ]
    return GradReport(
        n_params=flat.numel(),
        all_finite=bool(torch.isfinite(flat).all()),
        n_dead=len(dead),
        abs_mean=float(absg.mean()),
        abs_min=float(absg.min()),
        abs_max=float(absg.max()),
        dead_names=tuple(dead),
    )


def _finite_difference(
    gen: QuantumGenerator, xi: torch.Tensor, probe: torch.Tensor, eps: float
) -> torch.Tensor:
    """Central difference of the probe loss wrt the concatenated (W, b)."""
    params = [gen.W, gen.b]
    out = []
    with torch.no_grad():
        for p in params:
            for i in range(p.numel()):
                orig = p[i].item()
                p[i] = orig + eps
                plus = (gen(xi) * probe).sum().item()
                p[i] = orig - eps
                minus = (gen(xi) * probe).sum().item()
                p[i] = orig
                out.append((plus - minus) / (2 * eps))
    return torch.tensor(out, dtype=torch.float64)


def compare_gradients(
    gen: QuantumGenerator, xi: torch.Tensor, *, fd_eps: float = 1e-6, probe_seed: int = 0
) -> GradComparison:
    """Backprop gradient (``gen``) vs parameter-shift vs central finite difference.

    Builds a second generator with identical weights but ``diff_method="parameter-shift"``,
    evaluated one sample at a time (PennyLane does not support parameter-shift on
    broadcast tapes). ``gen`` itself is left untouched apart from its ``.grad`` fields.
    """
    probe = make_probe_loss(gen.latent_dim, probe_seed)
    g_bp = _probe_grad(gen, xi, probe).detach().clone()

    shift = QuantumGenerator(
        gen.ansatz.n_qb,
        gen.ansatz.n_l,
        gen.ansatz.ansatz_type,
        readout=gen.readout,
        angle_mode=gen.angle_mode,
        weight_init_std=gen.weight_init_std,
        diff_method="parameter-shift",
    ).double()
    shift.load_state_dict(gen.state_dict())
    g_ps = _probe_grad(shift, xi, probe, forward=forward_unbroadcast).detach().clone()

    g_fd = _finite_difference(gen, xi, probe, fd_eps)
    return GradComparison(
        max_abs_backprop_vs_shift=float((g_bp - g_ps).abs().max()),
        max_abs_backprop_vs_fd=float((g_bp - g_fd).abs().max()),
        grad_scale=float(g_bp.abs().max()),
    )


def batch_consistency(
    gen: QuantumGenerator, xi: torch.Tensor, probe_seed: int = 0
) -> tuple[float, float]:
    """(max |out_batched - out_looped|, max |grad_batched - grad_looped|).

    The "looped" path calls the QNode once per sample on 1-D angles (no parameter
    broadcasting) and sums the per-sample probe losses, the same scalar as the batched path.
    """
    probe = make_probe_loss(gen.latent_dim, probe_seed)

    gen.zero_grad(set_to_none=True)
    out_b = gen(xi)
    (out_b * probe).sum().backward()
    grad_b = _flat_grad(gen).detach().clone()

    gen.zero_grad(set_to_none=True)
    out_l = forward_unbroadcast(gen, xi)
    (out_l * probe).sum().backward()
    grad_l = _flat_grad(gen).detach().clone()

    return (
        float((out_b - out_l).abs().max().detach()),
        float((grad_b - grad_l).abs().max()),
    )


def critic_gradient(
    gen: QuantumGenerator, critic: nn.Module, xi: torch.Tensor
) -> GradReport:
    """Generator-step gradient: loss = -critic(gen(xi)).mean(), as in WGAN-GP."""
    gen.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    loss = -critic(gen(xi)).mean()
    loss.backward()
    flat = _flat_grad(gen)
    n_angles = gen.W.numel()
    absg = flat.abs()
    dead = [
        f"{'W' if i < n_angles else 'b'}[{i % n_angles}]"
        for i in torch.nonzero(absg < DEAD_TOL).flatten().tolist()
    ]
    return GradReport(
        n_params=flat.numel(),
        all_finite=bool(torch.isfinite(flat).all()),
        n_dead=len(dead),
        abs_mean=float(absg.mean()),
        abs_min=float(absg.min()),
        abs_max=float(absg.max()),
        dead_names=tuple(dead),
    )