"""GANTrainer with the quantum generator (Phase 3, step 14), plus a classical regression."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from mqs_molecule_generation.models.classical_gan import (  # noqa: E402
    Discriminator,
    Generator,
)
from mqs_molecule_generation.models.quantum_generator import QuantumGenerator  # noqa: E402
from mqs_molecule_generation.training import gan_trainer as gt  # noqa: E402
from mqs_molecule_generation.training.gan_trainer import GANTrainer, noise_dim_of  # noqa: E402
from mqs_molecule_generation.training.gan_trainer_config import GANTrainerConfig  # noqa: E402


def _real(n=128, d=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, d, generator=g) * 1.6 - 0.8).float()


def _qgen(n_qb=5, n_l=2, ansatz="bel", readout="dual"):
    torch.manual_seed(0)
    return QuantumGenerator(n_qb, n_l, ansatz, readout=readout)


def test_noise_dim_of():
    assert noise_dim_of(_qgen(), default=10) == 5
    assert noise_dim_of(Generator(10), default=10) == 10


def test_quantum_generator_trains_and_sees_n_qubit_noise(monkeypatch):
    seen = []
    orig = gt.sample_noise

    def spy(n, dim, device):
        seen.append(dim)
        return orig(n, dim, device)

    monkeypatch.setattr(gt, "sample_noise", spy)

    gen = _qgen()
    disc = Discriminator(gen.latent_dim)
    w0 = gen.W.detach().clone()
    cfg = GANTrainerConfig(n_epochs=2, n_critic=1, latent_dim=10)
    hist = GANTrainer(cfg).fit(gen, disc, _real(), log_every=0, seed=0)

    assert len(hist) == 2
    assert all(torch.isfinite(torch.tensor([h.d_loss, h.g_loss, h.gradient_penalty])).all()
               for h in hist)
    assert set(seen) == {5}, "quantum generator must get n_qb-dimensional noise, not D_l"
    assert not torch.allclose(gen.W, w0), "generator parameters did not move"


def test_single_readout_d_l_equals_n_qubits():
    gen = _qgen(n_qb=4, n_l=1, ansatz="simple", readout="single")
    cfg = GANTrainerConfig(n_epochs=1, n_critic=1, latent_dim=4)
    hist = GANTrainer(cfg).fit(gen, Discriminator(4), _real(64, 4), log_every=0, seed=0)
    assert len(hist) == 1


def test_dimension_mismatch_is_rejected():
    gen = _qgen()  # D_l = 10
    cfg = GANTrainerConfig(n_epochs=1, latent_dim=10)
    with pytest.raises(ValueError, match="10-dim latents but real latents are 8-dim"):
        GANTrainer(cfg).fit(gen, Discriminator(10), _real(64, 8), log_every=0)


def test_classical_generator_path_unchanged():
    """Phase 2 regression: classical Generator still trains with D_l-dimensional noise."""
    torch.manual_seed(0)
    gen, disc = Generator(10), Discriminator(10)
    cfg = GANTrainerConfig(n_epochs=2, n_critic=5, latent_dim=10)
    hist = GANTrainer(cfg).fit(gen, disc, _real(), log_every=0, seed=0)
    assert len(hist) == 2
    assert all(h.gradient_penalty >= 0 for h in hist)


def test_csv_history_written(tmp_path):
    gen = _qgen(n_qb=3, n_l=1, readout="dual")
    cfg = GANTrainerConfig(n_epochs=2, n_critic=1, latent_dim=6)
    csv_path = tmp_path / "h.csv"
    GANTrainer(cfg).fit(gen, Discriminator(6), _real(64, 6), log_every=0, csv_path=csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "epoch,d_loss,g_loss,gradient_penalty,lr"
    assert len(lines) == 3