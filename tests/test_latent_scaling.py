"""LatentScaler: VAE latents <-> the quantum generator's [-1, 1] range."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mqs_molecule_generation.training.latent_scaling import (  # noqa: E402
    SCALING_MODES,
    LatentScaler,
)


@pytest.fixture
def latents():
    g = torch.Generator().manual_seed(0)
    # Unbounded, per-dimension different location and spread, like real VAE latents.
    return torch.randn(2000, 10, generator=g) * torch.linspace(0.2, 4.0, 10) + torch.arange(10)


@pytest.mark.parametrize("mode", SCALING_MODES)
def test_inverse_round_trip(latents, mode):
    s = LatentScaler.fit(latents, mode)
    assert torch.allclose(s.inverse(s.transform(latents)), latents, atol=1e-5)


@pytest.mark.parametrize("bound", [1.0, 0.9])
def test_minmax_maps_real_latents_onto_bound(latents, bound):
    y = LatentScaler.fit(latents, "minmax", bound=bound).transform(latents)
    assert torch.allclose(y.min(0).values, torch.full((10,), -bound), atol=1e-5)
    assert torch.allclose(y.max(0).values, torch.full((10,), bound), atol=1e-5)


def test_zscore_centres_and_scales(latents):
    y = LatentScaler.fit(latents, "zscore", k=3.0).transform(latents)
    assert torch.allclose(y.mean(0), torch.zeros(10), atol=1e-5)
    assert torch.allclose(y.std(0), torch.full((10,), 1 / 3), atol=1e-4)


def test_none_is_identity(latents):
    s = LatentScaler.fit(latents, "none")
    assert torch.equal(s.transform(latents), latents)


def test_constant_dimension_does_not_divide_by_zero(latents):
    latents = latents.clone()
    latents[:, 3] = 1.5  # a fully collapsed latent dimension
    for mode in SCALING_MODES:
        y = LatentScaler.fit(latents, mode).transform(latents)
        assert torch.isfinite(y).all()


def test_state_dict_round_trip(latents):
    s = LatentScaler.fit(latents, "minmax", bound=0.9)
    t = LatentScaler.from_state_dict(s.state_dict())
    assert t.mode == "minmax"
    assert torch.allclose(t.transform(latents), s.transform(latents))


def test_preserves_dtype(latents):
    s = LatentScaler.fit(latents, "minmax")
    assert s.transform(latents).dtype == torch.float32
    assert s.inverse(latents.double()).dtype == torch.float64


def test_rejects_bad_inputs(latents):
    with pytest.raises(ValueError, match="mode"):
        LatentScaler.fit(latents, "bogus")
    with pytest.raises(ValueError, match="bound"):
        LatentScaler.fit(latents, "minmax", bound=1.5)
    with pytest.raises(ValueError, match="latents"):
        LatentScaler.fit(latents[0], "minmax")