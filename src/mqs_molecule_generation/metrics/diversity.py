"""Molecular diversity metric: internal diversity of generated set."""

from __future__ import annotations


def compute_diversity(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Compute internal diversity as average Tanimoto distance within set.

    Args:
        smiles_list: List of valid SMILES strings.
        radius: Morgan fingerprint radius.
        n_bits: Morgan fingerprint bit length.

    Returns:
        Average Tanimoto distance (higher = more diverse).
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        return 0.0

    if len(smiles_list) < 2:
        return 0.0

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)  # type: ignore[attr-defined]
        fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # Compute Tanimoto distances
    n = len(fps)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim_val = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total += 1.0 - sim_val
            count += 1

    return total / count if count > 0 else 0.0
