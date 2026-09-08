"""Torch-free pieces of the VAE training config (paper §2.4.1).

Split out from vae_trainer.py so KLAnnealer's arithmetic and TrainerConfig's
epoch-count derivation are testable without torch installed -- vae_trainer.py
needs torch at import time (CosineAnnealingLRWithRestart subclasses
torch.optim.lr_scheduler.LRScheduler), which would otherwise make even these
torch-independent pieces fail to import in an environment without torch.
"""

from __future__ import annotations

from dataclasses import dataclass


class KLAnnealer:
    """Linear KL-weight ramp from kl_w_start to kl_w_end (moses/vae/misc.py)."""

    def __init__(self, n_epoch: int, kl_start: int, kl_w_start: float, kl_w_end: float) -> None:
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = n_epoch
        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, epoch: int) -> float:
        k = (epoch - self.i_start) if epoch >= self.i_start else 0
        return self.w_start + k * self.inc


@dataclass(frozen=True)
class TrainerConfig:
    """VAE training hyperparameters (paper §2.4.1).

    Matches MOSES's raw config surface (moses/vae/config.py) rather than
    inventing a simplified one, to keep every knob traceable to a documented
    MOSES default or an explicit paper override.

    Total epoch count is NOT a direct parameter in MOSES's own config --
    it's derived (see ``n_epoch``) as
    ``lr_n_period * sum(lr_n_mult**i for i in range(lr_n_restarts))``.
    MOSES's own defaults (period=10, restarts=10, mult=1) give exactly 100
    epochs, matching the paper's "N_ep=100: default value from the MOSES
    library". ``nominal()``/``tuned()`` below pick the simplest
    factorization (a single, non-restarting cycle: n_period=n_epochs,
    n_restarts=1) to hit a target epoch count -- the paper doesn't state
    which factorization it used for its own N_ep in {100, 250, 800, 1000}
    study, so this is a documented choice, not a verified one.
    """

    lr_start: float = 3e-4  # paper: nominal lr = 3e-4 (MOSES default)
    lr_end: float = 3e-4  # nominal: same as lr_start (constant lr)
    lr_n_period: int = 100
    lr_n_restarts: int = 1
    lr_n_mult: int = 1

    kl_start: int = 0
    kl_w_start: float = 0.0
    kl_w_end: float = 0.05

    clip_grad: float = 50.0  # paper: gradient clipping threshold = 50
    batch_size: int = 64  # paper's override of MOSES's own default (512)
    n_last: int = 1000  # running-average window for logged losses

    @property
    def n_epoch(self) -> int:
        """Total epochs, matching MOSES's own derivation exactly."""
        return self.lr_n_period * sum(self.lr_n_mult**i for i in range(self.lr_n_restarts))

    @classmethod
    def nominal(cls, n_epochs: int = 1000) -> TrainerConfig:
        """Paper's "nominal" VAE training: constant lr=3e-4 (Table 1)."""
        return cls(lr_start=3e-4, lr_end=3e-4, lr_n_period=n_epochs, lr_n_restarts=1)

    @classmethod
    def tuned(cls, n_epochs: int = 1000) -> TrainerConfig:
        """Paper's "tuned" VAE training: cosine 1e-3 -> 1e-5 (Table 1)."""
        return cls(lr_start=1e-3, lr_end=1e-5, lr_n_period=n_epochs, lr_n_restarts=1)


@dataclass
class EpochMetrics:
    """Summary of one epoch's training or validation pass."""

    epoch: int
    mode: str  # "train" or "eval"
    kl_weight: float
    lr: float
    kl_loss: float
    recon_loss: float
    loss: float