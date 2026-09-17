"""Tests for WGAN-GP losses (paper §2.2.2/§2.4.2). Requires torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mqs_molecule_generation.training.wgan_gp import (  # noqa: E402
    critic_loss,
    generator_loss,
    gradient_penalty,
)


class _LinearCritic(torch.nn.Module):
    """D(x) = w . x + b, with w fixed and known -- gradient is CONSTANT (=w)
    everywhere, independent of the input. Makes the gradient penalty an
    exact, closed-form, hand-checkable number rather than something only
    approximately verifiable.
    """

    def __init__(self, weight: torch.Tensor, bias: float = 0.0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(weight.numel(), 1)
        with torch.no_grad():
            self.linear.weight.copy_(weight.unsqueeze(0))
            self.linear.bias.fill_(bias)

    def forward(self, x):
        return self.linear(x)


class TestGradientPenaltyAnalytic:
    """The stated gate: exact analytic values, not approximate."""

    def test_unit_norm_weight_gives_zero_penalty(self) -> None:
        # ||w|| = 1 -> gradient is constant, norm 1, everywhere ->
        # (||grad|| - 1)^2 = 0 for every sample, exactly.
        d_l = 5
        weight = torch.zeros(d_l)
        weight[0] = 1.0  # a one-hot unit vector, ||w||=1 exactly
        critic = _LinearCritic(weight)

        real = torch.randn(8, d_l)
        fake = torch.randn(8, d_l)
        gp = gradient_penalty(critic, real, fake)

        assert gp.item() == pytest.approx(0.0, abs=1e-6)

    def test_unit_norm_weight_zero_regardless_of_samples(self) -> None:
        # Since the gradient of a linear function doesn't depend on WHERE
        # it's evaluated, the penalty must be exactly 0 for ANY real/fake
        # pair, not just one lucky draw. Guards against a formula that only
        # coincidentally lands near 0 for a specific input.
        d_l = 4
        weight = torch.tensor([0.6, 0.8, 0.0, 0.0])  # ||w|| = 1 exactly (3-4-5 triple)
        assert torch.linalg.norm(weight).item() == pytest.approx(1.0, abs=1e-7)
        critic = _LinearCritic(weight)

        for seed in range(5):
            torch.manual_seed(seed)
            real = torch.randn(6, d_l) * (seed + 1)  # varying scale
            fake = torch.randn(6, d_l) * (seed + 1) + seed
            gp = gradient_penalty(critic, real, fake)
            assert gp.item() == pytest.approx(0.0, abs=1e-6)

    def test_norm_two_weight_gives_penalty_exactly_one(self) -> None:
        # ||w|| = 2 -> gradient norm is constant 2 everywhere ->
        # (2 - 1)^2 = 1, exactly. A second, non-zero closed-form target:
        # a broken implementation that always returns 0 would pass the
        # unit-norm tests above but fail this one.
        d_l = 3
        weight = torch.tensor([2.0, 0.0, 0.0])  # ||w|| = 2 exactly
        critic = _LinearCritic(weight)

        real = torch.randn(10, d_l)
        fake = torch.randn(10, d_l)
        gp = gradient_penalty(critic, real, fake)

        assert gp.item() == pytest.approx(1.0, abs=1e-6)

    def test_norm_half_weight_gives_penalty_exactly_quarter(self) -> None:
        # ||w|| = 0.5 -> (0.5 - 1)^2 = 0.25, exactly.
        d_l = 2
        weight = torch.tensor([0.5, 0.0])
        critic = _LinearCritic(weight)

        real = torch.randn(6, d_l)
        fake = torch.randn(6, d_l)
        gp = gradient_penalty(critic, real, fake)

        assert gp.item() == pytest.approx(0.25, abs=1e-6)

    def test_penalty_is_nonnegative_for_a_real_nonlinear_network(self) -> None:
        # (||grad|| - 1)^2 is a square -- must be >= 0 for any network, not
        # just the hand-constructed linear ones above.
        from mqs_molecule_generation.models.classical_gan import Discriminator

        critic = Discriminator(latent_dim=10)
        real = torch.randn(8, 10)
        fake = torch.randn(8, 10)
        gp = gradient_penalty(critic, real, fake)
        assert gp.item() >= 0.0

    def test_requires_matching_batch_sizes_via_broadcast_error(self) -> None:
        d_l = 4
        weight = torch.zeros(d_l)
        weight[0] = 1.0
        critic = _LinearCritic(weight)
        real = torch.randn(8, d_l)
        fake = torch.randn(4, d_l)  # mismatched batch size
        with pytest.raises(RuntimeError):
            gradient_penalty(critic, real, fake)


class TestCriticLoss:
    def test_matches_hand_computed_formula(self) -> None:
        real_validity = torch.tensor([1.0, 2.0, 3.0])
        fake_validity = torch.tensor([0.5, 0.5, 0.5])
        gp = torch.tensor(0.1)
        loss = critic_loss(real_validity, fake_validity, gp, lambda_gp=10.0)
        expected = -2.0 + 0.5 + 10.0 * 0.1  # -mean(real) + mean(fake) + lambda*gp
        assert loss.item() == pytest.approx(expected)

    def test_zero_gp_and_equal_validity_gives_zero_loss(self) -> None:
        real_validity = torch.tensor([1.0, 1.0])
        fake_validity = torch.tensor([1.0, 1.0])
        gp = torch.tensor(0.0)
        loss = critic_loss(real_validity, fake_validity, gp)
        assert loss.item() == pytest.approx(0.0)

    def test_default_lambda_is_ten(self) -> None:
        real_validity = torch.tensor([0.0])
        fake_validity = torch.tensor([0.0])
        gp = torch.tensor(1.0)
        loss = critic_loss(real_validity, fake_validity, gp)
        assert loss.item() == pytest.approx(10.0)


class TestGeneratorLoss:
    def test_matches_hand_computed_formula(self) -> None:
        fake_validity = torch.tensor([1.0, 2.0, 3.0])
        loss = generator_loss(fake_validity)
        assert loss.item() == pytest.approx(-2.0)

    def test_negative_of_mean(self) -> None:
        fake_validity = torch.tensor([-1.0, -3.0])
        loss = generator_loss(fake_validity)
        assert loss.item() == pytest.approx(2.0)