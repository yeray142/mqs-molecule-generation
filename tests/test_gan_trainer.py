"""Tests for the classical GAN trainer (paper §2.2.2/§2.4.2)."""

from __future__ import annotations

import math

import pytest

from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig


class TestGANTrainerConfig:
    """Pure Python, no torch needed."""

    def test_nominal_preset_has_n_critic_five(self) -> None:
        config = GANTrainerConfig.nominal()
        assert config.n_critic == 5

    def test_tuned_preset_has_n_critic_one(self) -> None:
        config = GANTrainerConfig.tuned()
        assert config.n_critic == 1

    def test_shared_hyperparameters_match_paper(self) -> None:
        for config in (GANTrainerConfig.nominal(), GANTrainerConfig.tuned()):
            assert config.lr == pytest.approx(2e-4)
            assert config.b1 == pytest.approx(0.5)
            assert config.b2 == pytest.approx(0.9)  # paper's override of MOSES's 0.999
            assert config.lambda_gp == pytest.approx(10.0)
            assert config.n_epochs == 100
            assert config.batch_size == 64

    def test_default_gamma_makes_scheduler_a_no_op(self) -> None:
        # gamma=1 means StepLR multiplies lr by 1 every step_size epochs --
        # a no-op, matching the paper's stated single flat lr.
        config = GANTrainerConfig()
        assert config.gamma == pytest.approx(1.0)

    def test_latent_dim_propagates_to_presets(self) -> None:
        for d_l in (10, 20, 30):
            assert GANTrainerConfig.nominal(latent_dim=d_l).latent_dim == d_l
            assert GANTrainerConfig.tuned(latent_dim=d_l).latent_dim == d_l


class TestGANTrainerMechanics:
    """Requires torch. End-to-end mechanical checks on tiny synthetic data."""

    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    @pytest.fixture
    def models(self, torch):
        from mqs_molecule_generation.models.classical_gan import Discriminator, Generator

        latent_dim = 10
        return Generator(latent_dim), Discriminator(latent_dim), latent_dim

    def test_sample_noise_is_uniform_minus_one_to_one(self, torch) -> None:
        from mqs_molecule_generation.training.gan_trainer import sample_noise

        noise = sample_noise(1000, 10, torch.device("cpu"))
        assert noise.min().item() >= -1.0
        assert noise.max().item() <= 1.0
        # Should span most of the range with 1000 samples, not cluster near 0.
        assert noise.min().item() < -0.9
        assert noise.max().item() > 0.9

    def test_fit_runs_and_produces_finite_losses(self, torch, models) -> None:
        from mqs_molecule_generation.training.gan_trainer import GANTrainer
        from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig

        generator, discriminator, latent_dim = models
        real_latents = torch.randn(100, latent_dim)
        config = GANTrainerConfig(n_epochs=3, batch_size=16, n_critic=5)
        trainer = GANTrainer(config)
        history = trainer.fit(generator, discriminator, real_latents, seed=0)

        assert len(history) == 3
        for entry in history:
            assert math.isfinite(entry.d_loss)
            assert math.isfinite(entry.g_loss)
            assert math.isfinite(entry.gradient_penalty)
            assert entry.gradient_penalty >= 0.0

    def test_n_critic_one_trains_generator_every_batch(self, torch, models) -> None:
        # With n_critic=1, every batch index i satisfies i % 1 == 0, so the
        # generator should train on every batch the discriminator does.
        from mqs_molecule_generation.training.gan_trainer import GANTrainer
        from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig

        generator, discriminator, latent_dim = models
        real_latents = torch.randn(64, latent_dim)  # exactly 4 batches of 16
        config = GANTrainerConfig(n_epochs=1, batch_size=16, n_critic=1)
        trainer = GANTrainer(config)

        # Snapshot generator params before/after; with n_critic=1 they must move.
        before = [p.clone() for p in generator.parameters()]
        trainer.fit(generator, discriminator, real_latents, seed=0, log_every=0)
        after = list(generator.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_reproducible_with_same_seed(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Discriminator, Generator
        from mqs_molecule_generation.training.gan_trainer import GANTrainer
        from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig

        latent_dim = 10
        real_latents = torch.randn(64, latent_dim)
        config = GANTrainerConfig(n_epochs=2, batch_size=16, n_critic=5)

        torch.manual_seed(123)
        gen1, disc1 = Generator(latent_dim), Discriminator(latent_dim)
        h1 = GANTrainer(config).fit(gen1, disc1, real_latents.clone(), seed=42, log_every=0)

        torch.manual_seed(123)
        gen2, disc2 = Generator(latent_dim), Discriminator(latent_dim)
        h2 = GANTrainer(config).fit(gen2, disc2, real_latents.clone(), seed=42, log_every=0)

        assert h1[-1].d_loss == pytest.approx(h2[-1].d_loss)
        assert h1[-1].g_loss == pytest.approx(h2[-1].g_loss)

    def test_csv_log_has_one_row_per_epoch(self, torch, models, tmp_path) -> None:
        from mqs_molecule_generation.training.gan_trainer import GANTrainer
        from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig

        generator, discriminator, latent_dim = models
        real_latents = torch.randn(64, latent_dim)
        config = GANTrainerConfig(n_epochs=4, batch_size=16, n_critic=5)
        trainer = GANTrainer(config)

        csv_path = tmp_path / "gan_history.csv"
        trainer.fit(generator, discriminator, real_latents, seed=0, csv_path=csv_path)

        rows = csv_path.read_text().strip().split("\n")
        assert len(rows) == 1 + 4  # header + 4 epochs
        assert rows[0] == "epoch,d_loss,g_loss,gradient_penalty,lr"

    def test_log_every_prints_expected_number_of_lines(self, torch, models, capsys) -> None:
        from mqs_molecule_generation.training.gan_trainer import GANTrainer
        from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig

        generator, discriminator, latent_dim = models
        real_latents = torch.randn(64, latent_dim)
        config = GANTrainerConfig(n_epochs=6, batch_size=16, n_critic=5)
        trainer = GANTrainer(config)
        trainer.fit(generator, discriminator, real_latents, seed=0, log_every=2)

        out = capsys.readouterr().out
        printed = [line for line in out.splitlines() if line.startswith("epoch ")]
        assert len(printed) == 3  # epochs 0, 2, 4

    def test_skips_batch_of_size_one(self, torch, models) -> None:
        # BatchNorm1d in the generator requires batch_size > 1 in train mode
        # -- a dataset size that leaves a final batch of exactly 1 must not
        # crash.
        from mqs_molecule_generation.training.gan_trainer import GANTrainer
        from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig

        generator, discriminator, latent_dim = models
        real_latents = torch.randn(17, latent_dim)  # 16 + 1 leftover at batch_size=16
        config = GANTrainerConfig(n_epochs=1, batch_size=16, n_critic=5)
        trainer = GANTrainer(config)
        history = trainer.fit(generator, discriminator, real_latents, seed=0, log_every=0)
        assert len(history) == 1
        assert math.isfinite(history[0].d_loss)