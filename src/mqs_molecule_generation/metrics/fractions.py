"""Distinctness, uniqueness, and novelty fractions (paper §2.3, Eq. 6).

These three fractions are defined at different stages of the same pipeline
(paper Eq. 6): N_Canonical = N_Generated * epsilon_d * epsilon_v * epsilon_u.

- epsilon_d (distinct): fraction of RAW generated strings that are not exact
  duplicates of an earlier string in the same batch -- computed BEFORE
  parsing/canonicalisation, since it is meant to catch a generator literally
  emitting the same string twice.
- epsilon_v (valid): see ``metrics.validity`` -- not duplicated here.
- epsilon_u (unique): fraction of the VALID molecules, compared as CANONICAL
  SMILES, that are not duplicates of one another. Applied only to the valid
  subset so that N_Generated * epsilon_d * epsilon_v * epsilon_u reproduces
  the paper's N_Canonical directly when the three fractions are chained in
  this order.

Novelty is a separate, reference-dependent fraction: what proportion of the
(valid, canonical) generated molecules are absent from the training set.
"""

from __future__ import annotations

from rdkit import Chem


def distinct_fraction(raw_smiles_list: list[str]) -> float:
    """epsilon_d: fraction of exact-string-unique entries in the raw batch.

    Deliberately operates on the strings as given, before any canonicalisation
    -- two different (non-canonical) SMILES for the same molecule count as
    distinct here, since this fraction is about literal repeated generator
    output, not chemical identity (that is epsilon_u's job).
    """
    if not raw_smiles_list:
        return float("nan")
    return len(set(raw_smiles_list)) / len(raw_smiles_list)


def unique_fraction(valid_smiles_list: list[str]) -> float:
    """epsilon_u: fraction of canonically-unique molecules among valid SMILES.

    Args:
        valid_smiles_list: SMILES already known to be RDKit-parseable (i.e.
            already filtered through ``metrics.validity``). Unparseable
            entries are defensively skipped rather than raising, but should
            not be present if used as documented.
    """
    canonical = [
        Chem.MolToSmiles(mol)
        for mol in (Chem.MolFromSmiles(s) for s in valid_smiles_list)
        if mol is not None
    ]
    if not canonical:
        return float("nan")
    return len(set(canonical)) / len(canonical)


def novelty_fraction(valid_smiles_list: list[str], reference_smiles: list[str]) -> float:
    """Fraction of (valid) generated molecules absent from the reference/training set.

    Both sides are compared as canonical SMILES so that different string
    representations of the same molecule are correctly treated as a match.

    Args:
        valid_smiles_list: SMILES already known to be RDKit-parseable.
        reference_smiles: The training set to check novelty against. Assumed
            already canonical (true for this project's frozen ``train.smi``);
            re-canonicalised anyway so the function is correct even if not.
    """
    reference_canonical = {
        Chem.MolToSmiles(mol)
        for mol in (Chem.MolFromSmiles(s) for s in reference_smiles)
        if mol is not None
    }
    generated_canonical = [
        Chem.MolToSmiles(mol)
        for mol in (Chem.MolFromSmiles(s) for s in valid_smiles_list)
        if mol is not None
    ]
    if not generated_canonical:
        return float("nan")
    novel = sum(1 for smi in generated_canonical if smi not in reference_canonical)
    return novel / len(generated_canonical)