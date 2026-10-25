"""QuantumGenerator nn.Module: gates 11 and 12. Requires torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mqs_molecule_generation.circuits.ansatz import AnsatzParams  # noqa: E402
from mqs_molecule_generation.circuits.paper_configs import (  # noqa: E402
    CONFIGS_NAMED_IN_PAPER_TEXT,
    N_PARAMS_TABLE,
    N_PARAMS_TABLE_EXPECTED_ROWS,
    PAPER_STATED_N_PARAMS,
)
from mqs_molecule_generation.circuits.readout import readout_dim  # noqa: E402
from mqs_molecule_generation.models.quantum_generator import QuantumGenerator  # noqa: E402

TWO_PI = 2 * torch.pi
TOL = 1e-6  # float32 rounds 2*pi up by ~1.7e-7


def n_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters())


class TestGate11AngleMapAndReadout:
    @pytest.mark.parametrize("readout", ["single", "dual"])
    @pytest.mark.parametrize("n_qb", [2, 5, 10])
    def test_output_dimension(self, n_qb, readout) -> None:
        g = QuantumGenerator(n_qb, 1, "bel", readout)
        expected = n_qb if readout == "single" else 2 * n_qb
        assert g.latent_dim == expected == readout_dim(n_qb, readout)
        out = g(torch.rand(3, n_qb) * 2 - 1)
        assert out.shape == (3, expected)

    @pytest.mark.parametrize("mode", ["shifted", "wrapped"])
    @pytest.mark.parametrize("kind", ["simple", "bel"])
    def test_angles_in_zero_to_two_pi_for_all_inputs(self, kind, mode) -> None:
        g = QuantumGenerator(5, 2, kind, "dual", angle_mode=mode)
        torch.manual_seed(0)
        with torch.no_grad():
            g.W.copy_(torch.randn_like(g.W) * 50)
            g.b.copy_(torch.randn_like(g.b) * 50)
        for scale in (1e-3, 1.0, 1e3, 1e6):
            theta = g.angles(torch.randn(256, 5) * scale)
            assert theta.min() >= -TOL and theta.max() <= TWO_PI + TOL, (mode, scale)

    def test_literal_eq4_goes_negative_so_the_stated_domain_needs_a_non_default_mode(self) -> None:
        g = QuantumGenerator(5, 2, "bel", "dual", angle_mode="literal")
        theta = g.angles(torch.randn(256, 5) * 3)
        assert theta.min() < 0, "literal Eq. 4 is odd in its argument"
        assert theta.min() > -TWO_PI - TOL and theta.max() < TWO_PI + TOL

    def test_noise_is_per_qubit_not_per_output_dimension(self) -> None:
        g = QuantumGenerator(5, 2, "bel", "dual")
        assert g.noise_dim == 5 and g.latent_dim == 10
        with pytest.raises(ValueError, match=r"\(batch, 5\)"):
            g(torch.rand(3, 10))  # a D_l-sized noise vector is the classical trainer's convention

    def test_invalid_options_rejected(self) -> None:
        with pytest.raises(ValueError, match="angle_mode"):
            QuantumGenerator(5, 2, "bel", "dual", angle_mode="nope")
        with pytest.raises(ValueError, match="readout"):
            QuantumGenerator(5, 2, "bel", "triple")
        with pytest.raises(ValueError, match="ansatz"):
            QuantumGenerator(5, 2, "strongly", "dual")


class TestGate12ParameterCounts:
    @pytest.mark.parametrize("kind", ["simple", "bel"])
    @pytest.mark.parametrize(("n_qb", "n_l"), [(2, 1), (3, 2), (5, 2), (5, 4), (10, 2), (10, 6)])
    def test_trainable_parameters_equal_two_per_angle(self, kind, n_qb, n_l) -> None:
        g = QuantumGenerator(n_qb, n_l, kind, "dual")
        ap = AnsatzParams(n_qb, n_l, kind)
        assert n_trainable(g) == ap.n_params == 2 * ap.n_angles

    def test_nominal_bel_is_the_papers_110(self) -> None:
        g = QuantumGenerator(5, 2, "bel", "dual")
        assert n_trainable(g) == PAPER_STATED_N_PARAMS[("bel", 5, 2)] == 110

    def test_only_w_and_b_are_trainable(self) -> None:
        g = QuantumGenerator(5, 2, "bel", "dual")
        assert {n for n, _ in g.named_parameters()} == {"W", "b"}
        assert "noise_qubit" in g.state_dict() and "noise_qubit" not in dict(g.named_parameters())

    @pytest.mark.parametrize("kind", ["simple", "bel"])
    @pytest.mark.parametrize(
        "config", CONFIGS_NAMED_IN_PAPER_TEXT, ids=lambda c: f"{c.n_qb}q-{c.n_l}l-{c.readout}"
    )
    def test_every_config_the_paper_names(self, kind, config) -> None:
        g = QuantumGenerator(config.n_qb, config.n_l, kind, config.readout)
        assert n_trainable(g) == AnsatzParams(config.n_qb, config.n_l, kind).n_params
        assert g.latent_dim == config.latent_dim
        out = g(torch.rand(2, config.n_qb) * 2 - 1)
        assert out.shape == (2, config.latent_dim)


class TestPaperNParamsTable:
    """Becomes the real 11-row gate once circuits/paper_configs.N_PARAMS_TABLE is filled in."""

    def test_table_has_been_entered(self) -> None:
        if not N_PARAMS_TABLE:
            pytest.skip(
                f"paper's N_params table not entered yet (0/{N_PARAMS_TABLE_EXPECTED_ROWS} rows); "
                "paste its rows into circuits/paper_configs.py::N_PARAMS_TABLE"
            )
        assert len(N_PARAMS_TABLE) == N_PARAMS_TABLE_EXPECTED_ROWS

    @pytest.mark.parametrize("row", N_PARAMS_TABLE, ids=lambda r: f"{r.n_qb}q-{r.n_l}l-{r.readout}")
    def test_row_matches_module_parameter_count(self, row) -> None:
        for kind in ("simple", "bel"):
            expected = getattr(row, kind)
            if expected is not None:
                assert (
                    n_trainable(QuantumGenerator(row.n_qb, row.n_l, kind, row.readout)) == expected
                )


class TestBehaviour:
    def test_forward_is_deterministic_bounded_float32(self) -> None:
        g = QuantumGenerator(5, 2, "bel", "dual")
        xi = torch.rand(6, 5) * 2 - 1
        a, b = g(xi), g(xi)
        assert torch.equal(a, b)
        assert a.dtype == torch.float32
        assert a.abs().max() <= 1 + 1e-5

    def test_gradients_reach_every_w_and_b(self) -> None:
        g = QuantumGenerator(5, 2, "bel", "dual", angle_mode="shifted")
        g(torch.rand(8, 5) * 2 - 1).pow(2).sum().backward()
        assert torch.isfinite(g.W.grad).all() and torch.isfinite(g.b.grad).all()
        assert (g.W.grad != 0).all(), "a W has no influence on the output"
        assert (g.b.grad != 0).any()

    def test_batch_of_one(self) -> None:
        g = QuantumGenerator(3, 1, "simple", "single")
        assert g(torch.rand(1, 3)).shape == (1, 3)

    def test_state_dict_roundtrip_reproduces_outputs(self) -> None:
        g1 = QuantumGenerator(4, 2, "bel", "dual")
        g2 = QuantumGenerator(4, 2, "bel", "dual")
        g2.load_state_dict(g1.state_dict())
        xi = torch.rand(5, 4) * 2 - 1
        assert torch.allclose(g1(xi), g2(xi), atol=1e-6)

    def test_largest_paper_config_runs(self) -> None:
        g = QuantumGenerator(15, 2, "bel", "dual")
        out = g(torch.rand(2, 15) * 2 - 1)
        assert out.shape == (2, 30)
        out.sum().backward()
        assert torch.isfinite(g.W.grad).all()