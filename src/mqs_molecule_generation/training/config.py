"""Training configuration dataclass for LatentStyleGAN."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters for LatentStyleGAN.

    All values have reasonable defaults for a smoke test.
    """

    # Problem definition
    n_qubits: int = 4
    n_layers: int = 2
    ansatz_type: str = "simple"
    latent_dim: int = 8

    # Optimizer settings
    generator_lr: float = 0.0001
    discriminator_lr: float = 0.0001
    n_critic: int = 5
    gradient_penalty: float = 10.0

    # Training loop
    max_iterations: int = 5000
    batch_size: int = 64
    checkpoint_every: int = 500
    eval_every: int = 100

    # Evaluation
    n_eval_samples: int = 1000

    # Reproducibility
    seed: int = 0
    dry_run: bool = False
    backend_name: str = "latent_style_pennylane"

    # Discriminator readout
    readout: str = "single"  # "single" (D_l=n_qb) or "dual" (D_l=2*n_qb)

    def __post_init__(self) -> None:
        if self.ansatz_type not in ("simple", "bel"):
            raise ValueError(f"ansatz_type must be 'simple' or 'bel', got {self.ansatz_type!r}")
        if self.readout not in ("single", "dual"):
            raise ValueError(f"readout must be 'single' or 'dual', got {self.readout!r}")
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
