"""Molecular validity checks via RDKit."""

from __future__ import annotations


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is chemically valid.

    Args:
        smiles: SMILES string to check.

    Returns:
        True if valid, False otherwise.
    """
    if not smiles:
        return False
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        return False


def compute_validity(smiles_list: list[str]) -> tuple[int, int, float]:
    """Compute validity fraction for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Tuple of (valid_count, total_count, validity_fraction).
    """
    total = len(smiles_list)
    if total == 0:
        return 0, 0, 0.0
    valid = sum(1 for smi in smiles_list if is_valid_smiles(smi))
    return valid, total, valid / total
