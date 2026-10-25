"""Map VAE latents into the quantum generator's output range, and back.

The quantum generator outputs expectation values, so every component lies in [-1, 1].
The VAE's latent vectors (``encode_mu``) are unbounded. The critic has to compare the
two, so either the real latents are brought into [-1, 1] or the generator output is
stretched to the latent range. Both are the same affine map seen from opposite sides.
Here it is applied to the REAL latents before training, and inverted on generated
samples before they go to the VAE decoder.

The paper text available to this project does not state which (if any) scaling was
used. This is therefore a documented choice, switchable from the command line:

  * ``minmax`` (default): per dimension, [min, max] of the real latents -> [-bound, bound].
    Every real latent is reachable by the generator. ``bound`` < 1 leaves headroom,
    because reaching exactly +-1 needs the qubit in a Z/X eigenstate.
  * ``zscore``: per dimension, (z - mean) / (k * std). With k = 3 about 99.7% of each
    dimension lands in [-1, 1]; the tails are NOT reachable by the generator.
  * ``none``: identity. The generator can only cover the part of latent space inside
    [-1, 1]^D_l -- kept so the effect of scaling can be measured, not as a recommendation.

The statistics are fitted once on the real training latents and saved with the run, so
evaluation decodes generated samples with exactly the map used during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["SCALING_MODES", "LatentScaler"]

SCALING_MODES = ("minmax", "zscore", "none")


@dataclass
class LatentScaler:
    """Per-dimension affine map: scaled = (z - shift) / scale."""

    mode: str
    shift: torch.Tensor  # (D_l,)
    scale: torch.Tensor  # (D_l,), strictly positive

    @classmethod
    def fit(
        cls, real_latents: torch.Tensor, mode: str = "minmax", bound: float = 1.0, k: float = 3.0
    ) -> LatentScaler:
        if mode not in SCALING_MODES:
            raise ValueError(f"mode must be one of {SCALING_MODES}, got {mode!r}")
        if real_latents.ndim != 2:
            raise ValueError(f"expected (n, D_l) latents, got shape {tuple(real_latents.shape)}")
        if not 0 < bound <= 1:
            raise ValueError(f"bound must be in (0, 1], got {bound}")
        z = real_latents.detach().to(torch.float64)
        d = z.shape[1]
        if mode == "none":
            shift = torch.zeros(d, dtype=torch.float64)
            scale = torch.ones(d, dtype=torch.float64)
        elif mode == "minmax":
            lo, hi = z.min(dim=0).values, z.max(dim=0).values
            shift = (hi + lo) / 2
            scale = (hi - lo) / (2 * bound)
        else:  # zscore
            shift = z.mean(dim=0)
            scale = k * z.std(dim=0)
        # A constant dimension (fully collapsed latent) would give scale 0.
        scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        return cls(mode=mode, shift=shift, scale=scale)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        """VAE latent -> generator range."""
        return ((z - self.shift.to(z)) / self.scale.to(z)).to(z.dtype)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Generator output -> VAE latent."""
        return (y * self.scale.to(y) + self.shift.to(y)).to(y.dtype)

    def state_dict(self) -> dict:
        return {"mode": self.mode, "shift": self.shift.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> LatentScaler:
        return cls(
            mode=state["mode"],
            shift=torch.tensor(state["shift"], dtype=torch.float64),
            scale=torch.tensor(state["scale"], dtype=torch.float64),
        )