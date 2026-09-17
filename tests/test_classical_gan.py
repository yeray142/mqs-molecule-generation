"""Tests for the classical GAN architecture (paper §2.2.2, N_params gate)."""

from __future__ import annotations

import pytest

from mqs_molecule_generation.models.gan_param_count import (
    batchnorm1d_param_count,
    discriminator_param_count,
    generator_param_count,
    linear_param_count,
)

# Paper Table 2 (and the scenario studies): N_params at D_l = 10 / 20 / 30.
TARGETS: dict[int, int] = {10: 705_162, 20: 716_692, 30: 728_222}


class TestLinearAndBatchNormParamCount:
    def test_linear_includes_bias(self) -> None:
        assert linear_param_count(in_features=10, out_features=5) == 10 * 5 + 5

    def test_batchnorm1d_is_weight_and_bias_only(self) -> None:
        # No running_mean/running_var -- those are buffers, not parameters.
        assert batchnorm1d_param_count(256) == 2 * 256


class TestGeneratorParamCount:
    """The paper's reported N_params -- exact match required, not approximate."""

    @pytest.mark.parametrize(("d_l", "target"), list(TARGETS.items()))
    def test_matches_paper_exactly(self, d_l: int, target: int) -> None:
        assert generator_param_count(d_l) == target

    def test_scales_with_d_l(self) -> None:
        # Larger D_l should mean more parameters (bigger first and last layers).
        assert generator_param_count(10) < generator_param_count(20) < generator_param_count(30)

    def test_manual_breakdown_at_d_l_10(self) -> None:
        d_l = 10
        expected = (
            linear_param_count(d_l, 128)  # block 1, no batchnorm
            + linear_param_count(128, 256)
            + batchnorm1d_param_count(256)
            + linear_param_count(256, 512)
            + batchnorm1d_param_count(512)
            + linear_param_count(512, 1024)
            + batchnorm1d_param_count(1024)
            + linear_param_count(1024, d_l)  # output layer, no batchnorm
        )
        assert generator_param_count(d_l) == expected == 705_162


class TestDiscriminatorParamCount:
    """Not the paper's reported metric, but verified for its own sake."""

    def test_manual_breakdown_at_d_l_10(self) -> None:
        d_l = 10
        expected = (
            linear_param_count(d_l, 512) + linear_param_count(512, 256) + linear_param_count(256, 1)
        )
        assert discriminator_param_count(d_l) == expected

    def test_does_not_match_generator_targets(self) -> None:
        # Confirms the paper's N_params is specifically the generator, not
        # the discriminator -- guards against silently mixing them up later.
        for d_l, target in TARGETS.items():
            assert discriminator_param_count(d_l) != target

    def test_scales_with_d_l(self) -> None:
        assert discriminator_param_count(10) < discriminator_param_count(20)


class TestGanModule:
    """Real torch.nn.Module verification -- requires torch, skipped if unavailable."""

    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    @pytest.mark.parametrize(("d_l", "target"), list(TARGETS.items()))
    def test_generator_param_count_matches_formula(self, torch, d_l, target) -> None:
        from mqs_molecule_generation.models.classical_gan import Generator

        model = Generator(latent_dim=d_l)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == target == generator_param_count(d_l)

    def test_discriminator_param_count_matches_formula(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Discriminator

        model = Discriminator(latent_dim=10)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == discriminator_param_count(10)

    def test_generator_forward_shape(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Generator

        model = Generator(latent_dim=10)
        model.eval()  # BatchNorm1d needs batch_size > 1 in train mode
        z = torch.rand(1, 10) * 2 - 1  # uniform(-1, 1), matching the paper's noise
        out = model(z)
        assert out.shape == (1, 10)

    def test_generator_forward_shape_batched(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Generator

        model = Generator(latent_dim=20)
        z = torch.rand(8, 20) * 2 - 1
        out = model(z)
        assert out.shape == (8, 20)

    def test_discriminator_forward_shape(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Discriminator

        model = Discriminator(latent_dim=30)
        z = torch.randn(8, 30)
        out = model(z)
        assert out.shape == (8, 1)

    def test_discriminator_no_batchnorm_layers(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Discriminator

        model = Discriminator(latent_dim=10)
        assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())

    def test_generator_has_exactly_three_batchnorm_layers(self, torch) -> None:
        from mqs_molecule_generation.models.classical_gan import Generator

        model = Generator(latent_dim=10)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 3

    def test_generator_batchnorm_eps_quirk_preserved(self, torch) -> None:
        # The documented eriklindernoren/MOSES artifact: eps=0.8, not the
        # PyTorch default of 1e-5. If this regresses to the default, training
        # dynamics silently diverge from the paper's, even though nothing
        # else (including parameter counts) would catch it.
        from mqs_molecule_generation.models.classical_gan import Generator

        model = Generator(latent_dim=10)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for bn in bn_layers:
            assert bn.eps == pytest.approx(0.8)