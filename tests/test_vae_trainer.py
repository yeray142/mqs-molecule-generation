"""Tests for the VAE training loop (paper §2.4.1)."""

from __future__ import annotations

import math

import pytest

from mqs_molecule_generation.training.trainer_config import KLAnnealer, TrainerConfig


class TestKLAnnealer:
    """Pure Python, no torch needed -- exact linear-ramp arithmetic."""

    def test_starts_at_w_start(self) -> None:
        annealer = KLAnnealer(n_epoch=100, kl_start=0, kl_w_start=0.0, kl_w_end=0.05)
        assert annealer(0) == 0.0

    def test_increments_linearly(self) -> None:
        annealer = KLAnnealer(n_epoch=100, kl_start=0, kl_w_start=0.0, kl_w_end=0.05)
        inc = 0.05 / 100
        assert annealer(1) == pytest.approx(inc)
        assert annealer(10) == pytest.approx(10 * inc)
        assert annealer(50) == pytest.approx(50 * inc)

    def test_approaches_but_does_not_reach_w_end_at_last_training_epoch(self) -> None:
        # At epoch n_epoch-1 (the last actual training epoch), weight is
        # (n_epoch-1)/n_epoch * w_end -- strictly less than w_end.
        annealer = KLAnnealer(n_epoch=100, kl_start=0, kl_w_start=0.0, kl_w_end=0.05)
        last = annealer(99)
        assert last < 0.05
        assert last == pytest.approx(99 / 100 * 0.05)

    def test_kl_start_delays_ramp(self) -> None:
        annealer = KLAnnealer(n_epoch=100, kl_start=10, kl_w_start=0.0, kl_w_end=0.05)
        assert annealer(0) == 0.0
        assert annealer(10) == 0.0  # ramp begins exactly at kl_start
        assert annealer(11) == pytest.approx(0.05 / (100 - 10))

    def test_nonzero_w_start(self) -> None:
        annealer = KLAnnealer(n_epoch=100, kl_start=0, kl_w_start=0.01, kl_w_end=0.05)
        assert annealer(0) == pytest.approx(0.01)


class TestTrainerConfigEpochDerivation:
    """n_epoch = lr_n_period * sum(lr_n_mult**i for i in range(lr_n_restarts))."""

    def test_moses_default_gives_100_epochs(self) -> None:
        # MOSES's own defaults: period=10, restarts=10, mult=1 -> 100 epochs,
        # matching the paper's "N_ep=100: default value from the MOSES library".
        config = TrainerConfig(lr_n_period=10, lr_n_restarts=10, lr_n_mult=1)
        assert config.n_epoch == 100

    def test_single_cycle_equals_period(self) -> None:
        config = TrainerConfig(lr_n_period=1000, lr_n_restarts=1, lr_n_mult=1)
        assert config.n_epoch == 1000

    def test_for_target_epoch_counts(self) -> None:
        for target in (100, 250, 800, 1000):
            config = TrainerConfig.nominal(n_epochs=target)
            assert config.n_epoch == target

    def test_nominal_preset_is_constant_lr(self) -> None:
        config = TrainerConfig.nominal(n_epochs=1000)
        assert config.lr_start == config.lr_end == pytest.approx(3e-4)

    def test_tuned_preset_matches_paper_endpoints(self) -> None:
        config = TrainerConfig.tuned(n_epochs=1000)
        assert config.lr_start == pytest.approx(1e-3)
        assert config.lr_end == pytest.approx(1e-5)


class TestCosineAnnealingLRWithRestart:
    """Requires torch (subclasses torch.optim.lr_scheduler.LRScheduler)."""

    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    def _make_optimizer(self, torch, lr: float):
        model = torch.nn.Linear(2, 2)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def test_constant_lr_when_start_equals_end(self, torch) -> None:
        from mqs_molecule_generation.training.vae_trainer import CosineAnnealingLRWithRestart

        opt = self._make_optimizer(torch, lr=3e-4)
        sched = CosineAnnealingLRWithRestart(opt, n_period=10, n_mult=1, lr_end=3e-4)
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert all(lr == pytest.approx(3e-4) for lr in lrs)

    def test_single_cycle_reaches_lr_end_on_last_epoch(self, torch) -> None:
        # Verified by independent hand-simulation before writing this test
        # (see vae_trainer.py's module docstring): a single n_period=5 cycle
        # reaches lr_end EXACTLY on the 5th step() call.
        from mqs_molecule_generation.training.vae_trainer import CosineAnnealingLRWithRestart

        opt = self._make_optimizer(torch, lr=1e-3)
        sched = CosineAnnealingLRWithRestart(opt, n_period=5, n_mult=1, lr_end=1e-5)
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-5, abs=1e-9)

    def test_first_epoch_lr_is_already_slightly_decayed(self, torch) -> None:
        # The documented off-by-one: __init__ triggers one internal step(),
        # so the LR used for the FIRST training epoch is computed at
        # current_epoch=1, not 0 -- it is NOT exactly lr_start.
        from mqs_molecule_generation.training.vae_trainer import CosineAnnealingLRWithRestart

        opt = self._make_optimizer(torch, lr=1e-3)
        CosineAnnealingLRWithRestart(opt, n_period=5, n_mult=1, lr_end=1e-5)
        first_lr = opt.param_groups[0]["lr"]
        expected = 1e-5 + (1e-3 - 1e-5) * (1 + math.cos(math.pi * 1 / 5)) / 2
        assert first_lr == pytest.approx(expected)
        assert first_lr != pytest.approx(1e-3)

    def test_restart_returns_toward_base_lr(self, torch) -> None:
        from mqs_molecule_generation.training.vae_trainer import CosineAnnealingLRWithRestart

        opt = self._make_optimizer(torch, lr=1e-3)
        sched = CosineAnnealingLRWithRestart(opt, n_period=3, n_mult=1, lr_end=1e-5)
        for _ in range(3):
            sched.step()
        at_restart = opt.param_groups[0]["lr"]
        sched.step()
        after_restart = opt.param_groups[0]["lr"]
        assert after_restart > at_restart  # jumped back up toward base_lr


class TestVAETrainerMechanics:
    """End-to-end mechanical checks on tiny synthetic data. Requires torch."""

    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    @pytest.fixture
    def tokenizer(self):
        from mqs_molecule_generation.data.tokenizer import SmilesTokenizer

        return SmilesTokenizer.from_data(["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC"])

    @pytest.fixture
    def model(self, torch, tokenizer):
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        return VAE(tokenizer, VAEConfig.nominal(d_z=8))

    def test_grad_clipping_caps_gradient_norm(self, torch, model) -> None:
        from torch.nn.utils import clip_grad_norm_

        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 1000.0)  # artificially huge gradient
        total_norm = clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=50.0
        )
        assert total_norm > 50.0  # confirms the pre-clip norm really was large
        post_clip_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in model.parameters() if p.requires_grad)
        )
        assert post_clip_norm == pytest.approx(50.0, rel=1e-4)

    def test_fit_runs_and_produces_finite_losses(self, torch, model, tokenizer) -> None:
        from torch.utils.data import DataLoader

        from mqs_molecule_generation.training.vae_trainer import VAETrainer, build_collate_fn

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC"] * 4  # 20 samples
        collate = build_collate_fn(tokenizer, device=torch.device("cpu"))
        loader = DataLoader(smiles, batch_size=5, shuffle=True, collate_fn=collate)

        config = TrainerConfig(lr_n_period=3, lr_n_restarts=1, batch_size=5)
        trainer = VAETrainer(config)
        history = trainer.fit(model, loader)

        assert len(history) == 3  # one entry per epoch (no val_loader)
        for entry in history:
            assert math.isfinite(entry.kl_loss)
            assert math.isfinite(entry.recon_loss)
            assert math.isfinite(entry.loss)
            assert entry.mode == "train"

    def test_fit_with_val_loader_doubles_history_length(self, torch, model, tokenizer) -> None:
        from torch.utils.data import DataLoader

        from mqs_molecule_generation.training.vae_trainer import VAETrainer, build_collate_fn

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC"] * 4
        collate = build_collate_fn(tokenizer, device=torch.device("cpu"))
        train_loader = DataLoader(smiles, batch_size=5, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(smiles[:5], batch_size=5, shuffle=False, collate_fn=collate)

        config = TrainerConfig(lr_n_period=2, lr_n_restarts=1, batch_size=5)
        trainer = VAETrainer(config)
        history = trainer.fit(model, train_loader, val_loader)

        assert len(history) == 4  # 2 epochs x (train + eval)
        modes = [entry.mode for entry in history]
        assert modes == ["train", "eval", "train", "eval"]

    def test_checkpoint_fn_called_at_configured_frequency(self, torch, model, tokenizer) -> None:
        from torch.utils.data import DataLoader

        from mqs_molecule_generation.training.vae_trainer import VAETrainer, build_collate_fn

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC"] * 4
        collate = build_collate_fn(tokenizer, device=torch.device("cpu"))
        loader = DataLoader(smiles, batch_size=5, shuffle=True, collate_fn=collate)

        checkpointed_epochs: list[int] = []
        config = TrainerConfig(lr_n_period=6, lr_n_restarts=1, batch_size=5)
        trainer = VAETrainer(config)
        trainer.fit(
            model,
            loader,
            checkpoint_fn=lambda m, epoch: checkpointed_epochs.append(epoch),
            checkpoint_every=2,
        )
        assert checkpointed_epochs == [0, 2, 4]