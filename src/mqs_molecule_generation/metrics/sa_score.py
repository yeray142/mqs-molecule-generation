"""SA score computation via RDKit's contrib SA_Score script (paper §2.3).

SA score (Synthetic Accessibility) is computed by RDKit's own contrib
implementation at ``RDConfig.RDContribDir / 'SA_Score' / 'sascorer.py'``. That
script is not on ``sys.path`` and ships no package ``__init__.py``, so it must
be loaded dynamically rather than imported normally.

Its public API, confirmed against the installed RDKit rather than assumed, is
a single module-level function ``calculateScore(mol) -> float``. It has no
class-based API and no ``.scoreMol()`` method -- earlier code in this project
called such a method and would have raised ``AttributeError`` on first use.
``calculateScore`` lazily loads its own fragment-score table on first call, so
no separate initialisation step is needed here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from rdkit import Chem, RDConfig

_SASCORER_MODULE: ModuleType | None = None


def _load_sascorer() -> ModuleType:
    """Lazily import RDKit's contrib ``sascorer`` module (and memoise it)."""
    global _SASCORER_MODULE
    if _SASCORER_MODULE is not None:
        return _SASCORER_MODULE

    sa_score_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    if not sa_score_path.exists():
        raise FileNotFoundError(
            f"SA score script not found at {sa_score_path}. This ships with "
            "RDKit's Contrib directory; a minimal RDKit build may omit it."
        )

    spec = importlib.util.spec_from_file_location("sascorer", sa_score_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SA score module from {sa_score_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["sascorer"] = module
    spec.loader.exec_module(module)
    _SASCORER_MODULE = module
    return module


def compute_sa_score(smiles: str) -> float:
    """Compute the SA score for one SMILES string.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        SA score in roughly [1, 10] (lower = easier to synthesise).

    Raises:
        ValueError: If SMILES is invalid or has no atoms (RDKit's
            ``calculateScore`` returns None for an empty molecule).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    score = _load_sascorer().calculateScore(mol)
    if score is None:
        raise ValueError(f"SA score undefined for empty molecule: {smiles!r}")
    return float(score)


def compute_sa_scores_batch(smiles_list: list[str]) -> list[float]:
    """Compute SA scores for a batch of molecules, skipping invalid SMILES.

    Returns a plain list, not a dict keyed by SMILES: a dict would silently
    collapse duplicate SMILES strings into one entry, corrupting the mean of
    the resulting distribution when the input (e.g. generated molecules)
    legitimately contains repeats.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        SA scores for every parseable, non-empty molecule, in input order.
        Invalid or empty-molecule entries are silently skipped (duplicates
        are not).
    """
    sascorer = _load_sascorer()
    scores: list[float] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        score = sascorer.calculateScore(mol)
        if score is not None:
            scores.append(float(score))
    return scores