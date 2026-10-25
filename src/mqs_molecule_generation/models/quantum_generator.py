"""Quantum generator (paper §2.2.2): Eq. 4 angle map + ansatz QNode + readout, as an nn.Module.

    xi (batch, n_qb)  --Eq. 4-->  theta (batch, n_angles)  --ansatz QNode-->  (batch, D_l)

The module's only trainable parameters are the scalar W and b per angle, so
``sum(p.numel() for p in module.parameters()) == 2 * n_angles`` -- the paper's
N_params (e.g. 110 for BEL with n_qb=5, n_l=2).

Interface notes for the QGAN trainer (Phase 3 step 14), which is deliberately
NOT changed here:
  * ``noise_dim`` (= n_qb) is NOT ``latent_dim`` (= D_l). The classical trainer
    samples noise of size ``config.latent_dim``; for the quantum generator it must
    sample ``generator.noise_dim`` instead.
  * Outputs are expectation values, so they lie in [-1, 1]. Real VAE latents are
    unbounded; whether/how the paper rescales is not something this module decides.
  * The paper does not (in the text I could access) state the W/b initialisation or
    the noise distribution; ``weight_init_std`` and the trainer's noise are choices.
"""

from __future__ import annotations

import torch
from torch import nn

from mqs_molecule_generation.circuits.angle_map import ANGLE_MODES, eq4_angles
from mqs_molecule_generation.circuits.ansatz import AnsatzParams
from mqs_molecule_generation.circuits.pennylane_circuits import build_qnode, stack_expvals
from mqs_molecule_generation.circuits.readout import readout_dim


class QuantumGenerator(nn.Module):
    """Style-based quantum generator: noise enters every rotation angle (Eq. 4)."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz_type: str = "bel",
        readout: str = "dual",
        angle_mode: str = "literal",
        weight_init_std: float = 1.0,
        device: str = "default.qubit",
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()
        if angle_mode not in ANGLE_MODES:
            raise ValueError(f"angle_mode must be one of {ANGLE_MODES}, got {angle_mode!r}")

        self.ansatz = AnsatzParams(n_qb=n_qubits, n_l=n_layers, ansatz_type=ansatz_type)
        self.readout = readout
        self.angle_mode = angle_mode
        self.weight_init_std = weight_init_std
        self.noise_dim = self.ansatz.noise_dim
        self.latent_dim = readout_dim(n_qubits, readout)  # validates `readout`

        n_angles = self.ansatz.n_angles
        self.W = nn.Parameter(torch.empty(n_angles))
        self.b = nn.Parameter(torch.empty(n_angles))
        self.register_buffer(
            "noise_qubit", torch.tensor(self.ansatz.noise_qubit_per_angle, dtype=torch.long)
        )
        self.reset_parameters()

        self._qnode = build_qnode(
            self.ansatz, readout, interface="torch", diff_method=diff_method, device=device
        )

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.W.normal_(0.0, self.weight_init_std)
            self.b.zero_()

    def angles(self, xi: torch.Tensor) -> torch.Tensor:
        """Eq. 4: (batch, n_qb) noise -> (batch, n_angles) rotation angles."""
        return eq4_angles(xi, self.W, self.b, self.noise_qubit, self.angle_mode)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        if xi.ndim != 2 or xi.shape[1] != self.noise_dim:
            raise ValueError(
                f"expected noise of shape (batch, {self.noise_dim}) -- one component per "
                f"qubit, not per output dimension -- got {tuple(xi.shape)}"
            )
        out = stack_expvals(self._qnode(self.angles(xi)))
        return out.to(xi.dtype)

    def extra_repr(self) -> str:
        a = self.ansatz
        return (
            f"{a.ansatz_type}, n_qb={a.n_qb}, n_l={a.n_l}, readout={self.readout}, "
            f"angle_mode={self.angle_mode}, noise_dim={self.noise_dim}, "
            f"latent_dim={self.latent_dim}, n_angles={a.n_angles}, n_params={a.n_params}"
        )