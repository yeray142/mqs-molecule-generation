"""LatentStyleGAN training loop (Wasserstein GAN with gradient penalty).

Stubbed: actual training raises NotImplementedError with TODO(paper §X) references.
"""

from __future__ import annotations

from dataclasses import dataclass

from mqs_molecule_generation.circuits.ansatz import AnsatzParams
from mqs_molecule_generation.circuits.discriminator import build_discriminator_circuit
from mqs_molecule_generation.circuits.generator import build_generator_circuit
from mqs_molecule_generation.utils.manifest import RunManifest
from mqs_molecule_generation.utils.seeding import set_seed


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    iteration: int
    d_loss: float
    g_loss: float
    gradient_penalty: float | None = None
    sa_score_mean: float | None = None
    diversity: float | None = None
    validity: float | None = None


class LatentStyleGANTraining:
    """Latent Style-based Quantum Wasserstein GAN training.

    Implements the WGAN objective with gradient penalty from arXiv:2603.22399.

    This is a STUB implementation - training methods raise NotImplementedError
    with TODO(paper §X) references to the relevant paper sections.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz_type: str,
        seed: int,
        dry_run: bool = False,
        manifest: RunManifest | None = None,
        max_iterations: int = 5000,
        batch_size: int = 64,
        latent_dim: int = 8,
        generator_lr: float = 0.0001,
        discriminator_lr: float = 0.0001,
        n_critic: int = 5,
        gradient_penalty: float = 10.0,
        checkpoint_every: int = 500,
        eval_every: int = 100,
        n_eval_samples: int = 1000,
        backend_name: str = "latent_style_pennylane",
        readout: str = "single",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.seed = seed
        self.dry_run = dry_run
        self.manifest = manifest
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.n_critic = n_critic
        self.gradient_penalty = gradient_penalty
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.n_eval_samples = n_eval_samples
        self.backend_name = backend_name
        self.readout = readout

        # Derived
        self.ansatz_params = AnsatzParams(
            n_qb=n_qubits, n_l=n_layers, ansatz_type=ansatz_type
        )

        # Build circuits
        self.generator_qasm, _ = build_generator_circuit(n_qubits, n_layers, ansatz_type)
        self.discriminator_qasm = build_discriminator_circuit(
            n_qubits, n_layers, ansatz_type, readout=readout
        )

    def run(self) -> list[TrainingMetrics]:
        """Run full training loop. Returns list of metrics per eval iteration.

        This is a STUB - it returns dummy metrics without training.
        """
        set_seed(self.seed)

        metrics_history: list[TrainingMetrics] = []
        metrics = TrainingMetrics(iteration=0, d_loss=0.0, g_loss=0.0)
        metrics_history.append(metrics)

        for iteration in range(self.max_iterations):
            # ---- Discriminator update (n_critic times) ----
            for _ in range(self.n_critic):
                d_loss, gp = self._discriminator_step()

            # ---- Generator update ----
            g_loss = self._generator_step()

            # ---- Periodic evaluation ----
            if iteration % self.eval_every == 0:
                metrics = TrainingMetrics(
                    iteration=iteration,
                    d_loss=d_loss,
                    g_loss=g_loss,
                    gradient_penalty=gp,
                )
                if not self.dry_run:
                    sa_scores, diversity, validity = self._evaluate()
                    metrics.sa_score_mean = sa_scores
                    metrics.diversity = diversity
                    metrics.validity = validity

                metrics_history.append(metrics)

            # ---- Checkpoint ----
            if iteration > 0 and iteration % self.checkpoint_every == 0:
                self._checkpoint(iteration, metrics_history)

        return metrics_history

    def _discriminator_step(self) -> tuple[float, float]:
        """One discriminator training step. Returns (loss, gradient_penalty).

        Raises:
            NotImplementedError: TODO(paper §4) - discriminator step implementation.
        """
        if self.dry_run:
            return 0.1, 0.01

        raise NotImplementedError(
            "TODO(paper §4): discriminator step implementation. "
            "See arXiv:2603.22399 §4 for the WGAN discriminator update."
        )

    def _generator_step(self) -> float:
        """One generator training step. Returns loss.

        Raises:
            NotImplementedError: TODO(paper §4) - generator step implementation.
        """
        if self.dry_run:
            return 0.1

        raise NotImplementedError(
            "TODO(paper §4): generator step implementation. "
            "See arXiv:2603.22399 §4 for the WGAN generator update."
        )

    def _evaluate(self) -> tuple[float, float, float]:
        """Evaluate generated molecules: SA score, diversity, validity.

        Returns:
            Tuple of (sa_score_mean, diversity, validity).

        Raises:
            NotImplementedError: TODO(paper §5) - sample generation for evaluation.
        """
        # Generate samples
        smiles_samples = self._generate_samples(self.n_eval_samples)

        # SA score
        try:
            from mqs_molecule_generation.metrics.sa_score import compute_sa_scores_batch

            sa_scores = compute_sa_scores_batch(smiles_samples)
            sa_mean = float(
                sum(sa_scores.values()) / len(sa_scores)
                if sa_scores
                else 0.0
            )
        except ImportError:
            sa_mean = 0.0

        # Diversity
        try:
            from mqs_molecule_generation.metrics.diversity import compute_diversity

            diversity = compute_diversity(smiles_samples)
        except ImportError:
            diversity = 0.0

        # Validity
        try:
            from mqs_molecule_generation.metrics.validity import compute_validity

            _, _, validity = compute_validity(smiles_samples)
        except ImportError:
            validity = 0.0

        return sa_mean, diversity, validity

    def _generate_samples(self, n: int) -> list[str]:
        """Generate n molecular samples from the generator.

        Args:
            n: Number of samples to generate.

        Returns:
            List of SMILES strings.

        Raises:
            NotImplementedError: TODO(paper §3) - quantum circuit execution for generation.
        """
        if self.dry_run:
            # Return dummy samples for dry run
            return ["CCO"] * min(n, 10)

        raise NotImplementedError(
            "TODO(paper §3): quantum circuit execution for molecular generation. "
            "See arXiv:2603.22399 §3 for the generator architecture."
        )

    def _checkpoint(
        self, iteration: int, metrics: list[TrainingMetrics]
    ) -> None:
        """Save checkpoint.

        Args:
            iteration: Current training iteration.
            metrics: Metrics history.
        """
        # Stub - no checkpoint saving in stub implementation
        pass
