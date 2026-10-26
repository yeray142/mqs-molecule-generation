"""Checks tied directly to numbers and figures in arXiv:2603.22399.

Every expected value here is quoted from the paper, not recomputed from the code, so a
failure means the implementation drifted from the paper rather than from itself.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from mqs_molecule_generation.circuits.ansatz import AnsatzParams
from mqs_molecule_generation.circuits.paper_configs import (
    N_PARAMS_EXTRAPOLATED,
    N_PARAMS_TABLE,
    N_PARAMS_TABLE_EXPECTED_ROWS,
    PAPER_STATED_N_PARAMS,
    TABLE5_Z0,
)
from mqs_molecule_generation.circuits.readout import readout_dim
from mqs_molecule_generation.metrics.significance import MetricStat, average_significance


class TestPaperNParams:
    """Tables 4, 5 and the supplementary parameter counts."""

    def test_table_is_complete(self):
        assert len(N_PARAMS_TABLE) == N_PARAMS_TABLE_EXPECTED_ROWS

    @pytest.mark.parametrize("row", N_PARAMS_TABLE)
    def test_row_matches_implementation(self, row):
        for ansatz, printed in (("simple", row.simple), ("bel", row.bel)):
            if printed is None:
                continue
            got = AnsatzParams(n_qb=row.n_qb, n_l=row.n_l, ansatz_type=ansatz).n_params
            assert got == printed, f"{ansatz} n_qb={row.n_qb} n_l={row.n_l}"

    @pytest.mark.parametrize("row", N_PARAMS_TABLE)
    def test_readout_gives_paper_latent_dim(self, row):
        assert readout_dim(row.n_qb, row.readout) in (10, 20, 30)

    def test_nominal_count_is_the_one_quoted_in_the_text(self):
        """"...110 trainable parameters, compared to the 705,162 ... of the classical GAN"."""
        assert PAPER_STATED_N_PARAMS[("bel", 5, 2)] == 110
        assert AnsatzParams(n_qb=5, n_l=2, ansatz_type="bel").n_params == 110

    def test_only_the_one_extrapolated_entry_is_flagged(self):
        assert frozenset({("simple", 10, 6)}) == N_PARAMS_EXTRAPOLATED


class TestFigureLayouts:
    """Structure read directly from Figures 2 and 3."""

    def test_bel_starts_with_a_hadamard_on_every_qubit(self):
        """Fig. 3 draws an H on each of the five wires before any rotation."""
        layout = AnsatzParams(n_qb=5, n_l=2, ansatz_type="bel").layout
        assert [g.name for g in layout[:5]] == ["H"] * 5
        assert {g.wires[0] for g in layout[:5]} == set(range(5))
        assert all(g.name != "H" for g in layout[5:])

    def test_hadamard_layer_costs_no_parameters(self):
        assert AnsatzParams(n_qb=5, n_l=2, ansatz_type="bel").n_params == 110

    def test_simple_entangles_even_with_one_layer(self):
        """Fig. 2 (one layer) draws the CNOT staircase, so n_l=1 must not be a product state."""
        layout = AnsatzParams(n_qb=5, n_l=1, ansatz_type="simple").layout
        cx = [g for g in layout if g.name == "CX"]
        assert [g.wires for g in cx] == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_simple_staircase_repeats_per_layer(self):
        layout = AnsatzParams(n_qb=4, n_l=3, ansatz_type="simple").layout
        assert sum(g.name == "CX" for g in layout) == 3 * 3

    def test_legacy_layout_restores_previous_behaviour(self):
        bel = AnsatzParams(n_qb=5, n_l=2, ansatz_type="bel", figure_layout=False).layout
        simple = AnsatzParams(n_qb=5, n_l=1, ansatz_type="simple", figure_layout=False).layout
        assert all(g.name != "H" for g in bel)
        assert all(g.name != "CX" for g in simple)

    @pytest.mark.parametrize("figure_layout", [True, False])
    @pytest.mark.parametrize("ansatz", ["simple", "bel"])
    def test_angle_indices_stay_exact_either_way(self, ansatz, figure_layout):
        ap = AnsatzParams(n_qb=5, n_l=2, ansatz_type=ansatz, figure_layout=figure_layout)
        used = sorted(g.param for g in ap.layout if g.param is not None)
        assert used == list(range(ap.n_angles))


class TestTable4Significance:
    """Our Eq. 5 implementation against the paper's own published Table 4."""

    #: Table 4, mean +- std over five seeds. Reference column = Tuned (classical) GAN.
    TUNED_GAN: ClassVar[dict[str, tuple[float, float]]] = {
        "eps_d": (0.481, 0.031), "eps_v": (0.917, 0.012), "eps_u": (0.986, 0.003),
        "novelty": (0.536, 0.015), "intdiv": (0.898, 0.004), "filters": (0.721, 0.011),
        "eps_logp": (0.898, 0.007), "sa": (2.391, 0.076), "qed": (0.599, 0.007),
        "weight": (203.9, 10.6),
    }
    SIMPLE_QGAN: ClassVar[dict[str, tuple[float, float]]] = {
        "eps_d": (0.529, 0.003), "eps_v": (0.831, 0.002), "eps_u": (0.994, 0.000),
        "novelty": (0.572, 0.003), "intdiv": (0.882, 0.000), "filters": (0.737, 0.003),
        "eps_logp": (0.883, 0.001), "sa": (2.575, 0.007), "qed": (0.641, 0.001),
        "weight": (294.8, 0.8),
    }
    BEL_QGAN: ClassVar[dict[str, tuple[float, float]]] = {
        "eps_d": (0.523, 0.003), "eps_v": (0.875, 0.003), "eps_u": (0.993, 0.001),
        "novelty": (0.548, 0.003), "intdiv": (0.883, 0.000), "filters": (0.719, 0.002),
        "eps_logp": (0.896, 0.002), "sa": (2.448, 0.007), "qed": (0.643, 0.001),
        "weight": (261.4, 0.8),
    }

    @staticmethod
    def _stats(d):
        return {k: MetricStat(*v) for k, v in d.items()}

    @pytest.mark.parametrize(
        ("scenario", "key"),
        [("SIMPLE_QGAN", ("simple", 5, 2, "dual")), ("BEL_QGAN", ("bel", 5, 2, "dual"))],
    )
    def test_reproduces_published_z0(self, scenario, key):
        """Within 0.1 of the published value; the residual is the table's own rounding.

        The published sigmas are given to three decimals and several read "0.000", which a
        division by the combined sigma amplifies, so exact agreement is not achievable from
        the printed numbers alone.
        """
        _, z0 = average_significance(
            self._stats(self.TUNED_GAN), self._stats(getattr(self, scenario))
        )
        assert z0 == pytest.approx(TABLE5_Z0[key], abs=0.1)

    def test_bel_scores_better_than_simple(self):
        """"As expected, the simple circuit performs worse than the BEL circuit." (Sec. 3.1)"""
        ref = self._stats(self.TUNED_GAN)
        _, z_simple = average_significance(ref, self._stats(self.SIMPLE_QGAN))
        _, z_bel = average_significance(ref, self._stats(self.BEL_QGAN))
        assert z_bel > z_simple

    def test_sign_convention_is_degradation_negative(self):
        """Table 5's QGAN entries are all negative: worse than the classical reference."""
        assert all(z < 0 for z in TABLE5_Z0.values())