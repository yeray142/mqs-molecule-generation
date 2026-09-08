"""Structural-alert filter registry (paper §2.1, "RDKit filters").

The paper reports a filter pass rate of 0.734 on the training subsample
(Table 3) but does not say which filter definition produced it. This module
exposes several candidate definitions behind one interface so the correct one
can be identified empirically -- see ``scripts/00_prepare_data.py calibrate``.

Two families of candidate are registered:

``moses_native``
    The filter MOSES itself uses to construct and evaluate its dataset: 22
    hand-curated medicinal-chemistry alerts (``mcf.csv``) plus 480 PAINS
    SMARTS (``wehi_pains.csv``), both vendored verbatim from
    ``molecularsets/moses`` (MIT license; see ``vendor/README.md``), applied
    together with a neutral-charge check, an allowed-atom check, and a
    reject-rings-of-8-or-more-atoms check. This reproduces
    ``moses.metrics.utils.mol_passes_filters`` without depending on the
    ``molsets`` package, which is frequently broken on modern Python/pandas.

    IMPORTANT: MOSES's released dataset was already filtered through this
    exact function during its own construction, so running it against MOSES
    training data is expected to land near 1.0, *not* 0.734. If calibration
    shows ``moses_native`` near 1.0 rather than 0.734, that is evidence the
    paper's "Filters" metric is not this function at all -- see the
    ``calibrate`` command's guidance for what to try next.

RDKit ``FilterCatalog`` sets (``pains``, ``brenk``, ``nih``, ...)
    Off-the-shelf structural-alert catalogues bundled with RDKit, offered as
    alternates in case the paper used a different definition than MOSES's own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import cache
from importlib import resources

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog

RDLogger.DisableLog("rdApp.*")

_VENDOR_PKG = "mqs_molecule_generation.data.vendor"
_Catalogs = FilterCatalog.FilterCatalogParams.FilterCatalogs

#: Named filter sets built from RDKit's bundled structural-alert catalogues.
CATALOG_SETS: dict[str, tuple[str, ...]] = {
    "pains": ("PAINS",),
    "brenk": ("BRENK",),
    "nih": ("NIH",),
    "zinc": ("ZINC",),
    "chembl": ("CHEMBL",),
    "pains_brenk": ("PAINS", "BRENK"),
    "pains_nih": ("PAINS", "NIH"),
    "pains_brenk_nih": ("PAINS", "BRENK", "NIH"),
    "pains_brenk_zinc": ("PAINS", "BRENK", "ZINC"),
    "pains_brenk_nih_zinc": ("PAINS", "BRENK", "NIH", "ZINC"),
}

#: Elements permitted by MOSES's own dataset construction procedure.
MOSES_ALLOWED_ATOMS: frozenset[str] = frozenset({"C", "N", "S", "O", "F", "Cl", "Br", "H"})

#: MOSES rejects any molecule with a ring of 8 or more atoms (rings up to
#: size 7 are allowed). Source: moses/metrics/utils.py::mol_passes_filters.
MOSES_MAX_RING_SIZE = 7


@cache
def _build_catalog(catalog_names: tuple[str, ...]) -> FilterCatalog.FilterCatalog:
    """Build (and memoise) a combined RDKit FilterCatalog."""
    params = FilterCatalog.FilterCatalogParams()
    for name in catalog_names:
        if not hasattr(_Catalogs, name):
            raise ValueError(f"Unknown RDKit filter catalog {name!r}")
        if not params.AddCatalog(getattr(_Catalogs, name)):
            raise RuntimeError(f"RDKit refused to add catalog {name!r}")
    return FilterCatalog.FilterCatalog(params)


def _catalog_predicate(catalog_names: tuple[str, ...]) -> Callable[[Chem.Mol], bool]:
    def passes(mol: Chem.Mol) -> bool:
        return not _build_catalog(catalog_names).HasMatch(mol)

    return passes


@cache
def _moses_native_patterns() -> tuple[Chem.Mol, ...]:
    """Load and parse the vendored MCF + PAINS SMARTS (MOSES's own alert set).

    Mirrors ``moses/metrics/utils.py``'s ``_mcf.append(_pains, sort=True)``,
    using ``pd.concat`` since ``DataFrame.append`` was removed in pandas 2.0.
    """
    mcf_csv = resources.files(_VENDOR_PKG).joinpath("mcf.csv")
    pains_csv = resources.files(_VENDOR_PKG).joinpath("wehi_pains.csv")
    with resources.as_file(mcf_csv) as path:
        mcf = pd.read_csv(path)
    with resources.as_file(pains_csv) as path:
        pains = pd.read_csv(path, names=["smarts", "names"])

    combined = pd.concat([mcf[["smarts"]], pains[["smarts"]]], ignore_index=True, sort=False)
    patterns: list[Chem.Mol] = []
    for smarts in combined["smarts"]:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise RuntimeError(f"MOSES-native filter: unparseable SMARTS {smarts!r}")
        patterns.append(pattern)
    return tuple(patterns)


def moses_native_passes(mol: Chem.Mol) -> bool:
    """The real MOSES filter: MCF + PAINS alerts, charge, atoms, ring size.

    Equivalent to ``moses.metrics.utils.mol_passes_filters`` (see module
    docstring). Expect ~1.0 on genuine MOSES data.
    """
    ring_sizes = (len(ring) for ring in mol.GetRingInfo().AtomRings())
    if any(size > MOSES_MAX_RING_SIZE for size in ring_sizes):
        return False
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in MOSES_ALLOWED_ATOMS for atom in mol.GetAtoms()):
        return False
    h_mol = Chem.AddHs(mol)
    return not any(h_mol.HasSubstructMatch(pattern) for pattern in _moses_native_patterns())


def get_filter(name: str) -> Callable[[Chem.Mol], bool]:
    """Return a predicate that is True when ``mol`` passes the named filter set."""
    if name == "none":
        return lambda mol: True
    if name == "moses_native":
        return moses_native_passes
    if name in CATALOG_SETS:
        return _catalog_predicate(CATALOG_SETS[name])
    raise ValueError(f"Unknown filter set {name!r}. Available: {', '.join(available())}")


def available() -> list[str]:
    """Names accepted by :func:`get_filter`."""
    return ["none", "moses_native", *sorted(CATALOG_SETS)]


def passes_filters(smiles: Iterable[str], name: str) -> list[bool]:
    """Apply the named filter set to an iterable of SMILES.

    Unparseable SMILES count as failures rather than raising, so a single bad
    row cannot abort a 12,000-molecule sweep.
    """
    predicate = get_filter(name)
    result: list[bool] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        result.append(False if mol is None else predicate(mol))
    return result


def pass_rate(smiles: Sequence[str], name: str) -> float:
    """Fraction of ``smiles`` passing the named filter set (paper Table 3: 0.734)."""
    if not smiles:
        return float("nan")
    flags = passes_filters(smiles, name)
    return sum(flags) / len(flags)