"""Configurations and parameter counts the paper states, with where each comes from.

Two kinds of fact live here, kept apart on purpose:

CONFIGS_NAMED_IN_PAPER_TEXT -- (n_qb, n_l, readout) combinations the paper's text
and table captions name explicitly (quoted in ``source``). Each gives a
D_l in {10, 20, 30}, the three latent sizes the classical GAN is also run at.

N_PARAMS_TABLE -- the paper's printed N_params table (the gate calls it 11 rows).
I could NOT read that table from the text available to me, so it is deliberately
empty rather than filled with numbers recomputed from the formula (which would
make "gate passed" mean "the formula agrees with itself"). Paste its rows in as
``NParamsRow``s and ``tests/test_quantum_generator.py::TestPaperNParamsTable``
starts checking every one against the module's real parameter count.

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


#: Paste the paper's printed rows here. Expected: 11.
N_PARAMS_TABLE: tuple[NParamsRow, ...] = ()
N_PARAMS_TABLE_EXPECTED_ROWS = 11