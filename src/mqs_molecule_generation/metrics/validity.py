"""Molecular validity checks via RDKit (paper §2.3, "Fraction of valid molecules").

RDKit is a core dependency of this project, not optional. Functions here do
NOT catch ImportError: if RDKit is missing, that is an environment problem
that should surface immediately and loudly, not be silently reinterpreted as
"every molecule is invalid". A caught-and-swallowed ImportError returning
False/0.0 is indistinguishable from a real validity failure and would corrupt
every downstream metric without anyone noticing RDKit isn't installed.
"""

from __future__ import annotations

from rdkit import Chem


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is chemically valid (RDKit parses and sanitizes it).

    Args:
        smiles: SMILES string to check.

    Returns:
        True if valid, False otherwise. Never raises on a bad *string* --
        only on a missing RDKit installation.
    """
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def compute_validity(smiles_list: list[str]) -> tuple[int, int, float]:
    """Compute validity fraction for a list of SMILES (paper's epsilon_v).

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