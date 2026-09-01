"""SA score computation via RDKit.

SA score (Synthetic Accessibility) is computed using the RDKit's open-source
implementation at RDConfig.RDContribDir / 'SA_Score'.
"""

from __future__ import annotations

import sys
from pathlib import Path

# SA score generator - loaded lazily
_SA_SCORE_GENERATOR: object | None = None


def _load_sa_score_generator() -> object:
    """Lazily load SA score generator from RDConfig.RDContribDir."""
    global _SA_SCORE_GENERATOR
    if _SA_SCORE_GENERATOR is not None:
        return _SA_SCORE_GENERATOR

    try:
        from rdkit import RDConfig
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SA score computation. "
            "Install with: uv sync --extra rdkit"
        ) from exc

    sa_score_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascore.py"
    if not sa_score_path.exists():
        raise FileNotFoundError(
            f"SA score generator not found at {sa_score_path}. "
            "Ensure RDKit contrib data is installed."
        )

    # Load the SA score generator module dynamically
    import importlib.util

    spec = importlib.util.spec_from_file_location("sascore", sa_score_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SA score generator from {sa_score_path}")

    sa_module = importlib.util.module_from_spec(spec)
    sys.modules["sascore"] = sa_module
    spec.loader.exec_module(sa_module)
    _SA_SCORE_GENERATOR = sa_module.SAScore()
    return _SA_SCORE_GENERATOR


def compute_sa_score(molecule_smiles: str) -> float:
    """Compute SA score for a molecule SMILES string.

    Args:
        molecule_smiles: SMILES string of the molecule.

    Returns:
        SA score (lower = more synthetically accessible, typical range 1-10).

    Raises:
        ValueError: If SMILES is invalid.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {molecule_smiles!r}")

    generator = _load_sa_score_generator()
    return generator.scoreMol(mol)  # type: ignore[attr-defined]


def compute_sa_scores_batch(smiles_list: list[str]) -> dict[str, float]:
    """Compute SA scores for a batch of molecules.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Dict mapping SMILES -> SA score. Invalid SMILES are excluded with a warning.
    """
    from rdkit import Chem

    generator = _load_sa_score_generator()
    scores: dict[str, float] = {}

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # skip invalid
        scores[smi] = generator.scoreMol(mol)  # type: ignore[attr-defined]

    return scores
