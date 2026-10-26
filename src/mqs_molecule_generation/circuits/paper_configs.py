"""Configurations and parameter counts the paper states, with where each comes from.

Two kinds of fact live here, kept apart on purpose:

CONFIGS_NAMED_IN_PAPER_TEXT -- (n_qb, n_l, readout) combinations the paper's text
and table captions name explicitly (quoted in ``source``). Each gives a
D_l in {10, 20, 30}, the three latent sizes the classical GAN is also run at.

N_PARAMS_TABLE -- the parameter counts the paper actually prints, now read from the
PDF (Tables 4 and 5 plus the supplementary tables, which agree). Table 5 is laid out
by latent dimension rather than as 11 flat rows, so it yields 7 (n_qb, n_l, readout)
configurations covering 13 printed counts. Every one is reproduced exactly by
n_angles = n_qb*n_l (simple) / n_qb*(5*n_l+1) (BEL) with N_params = 2*n_angles.

PAPER_STATED_N_PARAMS -- the one count the paper's running text states outright.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    n_qb: int
    n_l: int
    readout: str
    source: str

    @property
    def latent_dim(self) -> int:
        return self.n_qb if self.readout == "single" else 2 * self.n_qb


CONFIGS_NAMED_IN_PAPER_TEXT: tuple[PaperConfig, ...] = (
    PaperConfig(5, 2, "dual", 'nominal: "n_qb = 5, n_l = 2 and dual readout" (Table 6 text)'),
    PaperConfig(5, 4, "dual", '"The quantum GANs use n_qb = 5, n_l = 4, and dual readout."'),
    PaperConfig(10, 2, "single", '"The quantum GANs use n_qb = 10, n_l = 2, and single readout."'),
    PaperConfig(
        10, 4, "single", 'Table 17: "QGANs with n_qb = 10, n_l = 4, 6, and single readout"'
    ),
    PaperConfig(
        10, 6, "single", 'Table 17: "QGANs with n_qb = 10, n_l = 4, 6, and single readout"'
    ),
    PaperConfig(10, 2, "dual", '"The quantum GANs use n_qb = 10, n_l = 2, and dual readout."'),
    PaperConfig(15, 2, "dual", '"The quantum GANs use n_qb = 15, n_l = 2, and dual readout."'),
)

#: (ansatz_type, n_qb, n_l) -> N_params stated in the paper's running text:
#: "...to 110 trainable parameters, compared to the 705,162 trainable parameters of the
#: classical GAN pipeline" (BEL QGAN, nominal setting n_qb=5, n_l=2).
PAPER_STATED_N_PARAMS: dict[tuple[str, int, int], int] = {("bel", 5, 2): 110}


@dataclass(frozen=True)
class NParamsRow:
    """One row of the paper's N_params table."""

    n_qb: int
    n_l: int
    readout: str
    simple: int | None = None
    bel: int | None = None


#: The paper's printed parameter counts, read from Table 4 (nominal setting), Table 5
#: (all QGAN scenarios) and the supplementary tables, which agree with each other.
#: ``simple``/``bel`` are N_params, i.e. 2 x n_angles (W and b per angle, Eq. 4).
N_PARAMS_TABLE: tuple[NParamsRow, ...] = (
    NParamsRow(5, 2, "dual", simple=20, bel=110),      # Tables 4 and 5, nominal setting
    NParamsRow(5, 4, "dual", simple=40, bel=210),      # Table 5
    NParamsRow(10, 2, "single", simple=40, bel=220),   # Table 5, D_l = 10
    NParamsRow(10, 4, "single", simple=80, bel=420),   # Table 5
    NParamsRow(10, 6, "single", simple=120, bel=620),  # Table 5 (simple n_l=6 not printed)
    NParamsRow(10, 2, "dual", simple=40, bel=220),     # Table 5, D_l = 20
    NParamsRow(15, 2, "dual", simple=60, bel=330),     # Table 5, D_l = 30
)
#: Table 5 prints no simple/n_l=6 entry; its 120 above is this project's own extrapolation
#: of the formula, flagged here rather than passed off as a published number.
N_PARAMS_EXTRAPOLATED: frozenset[tuple[str, int, int]] = frozenset({("simple", 10, 6)})
N_PARAMS_TABLE_EXPECTED_ROWS = 7

#: Paper Table 4 / Table 5 average significance, reference = the TUNED CLASSICAL GAN.
#: Key: (ansatz, n_qb, n_l, readout) -> <Z0>.
TABLE5_Z0: dict[tuple[str, int, int, str], float] = {
    ("simple", 5, 2, "dual"): -1.00,
    ("simple", 5, 4, "dual"): -1.01,
    ("bel", 5, 2, "dual"): -0.26,
    ("bel", 5, 4, "dual"): -0.27,
    ("simple", 10, 2, "single"): -0.84,
    ("simple", 10, 4, "single"): -0.83,
    ("bel", 10, 2, "single"): -0.10,
    ("bel", 10, 4, "single"): -0.08,
    ("bel", 10, 6, "single"): -0.11,
    ("simple", 10, 2, "dual"): -0.28,
    ("bel", 10, 2, "dual"): -0.32,
    ("simple", 15, 2, "dual"): -0.39,
    ("bel", 15, 2, "dual"): -0.32,
}