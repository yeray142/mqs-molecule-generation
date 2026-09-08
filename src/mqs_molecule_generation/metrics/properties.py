"""Molecular property metrics (paper §2.3: LogP, molecular weight, QED).

Batch functions return plain lists, not SMILES-keyed dicts, for the same
reason as ``sa_score.compute_sa_scores_batch``: a dict would silently collapse
duplicate SMILES and corrupt the resulting distribution's mean. Invalid
SMILES are skipped silently; validity should already have been checked
separately via ``metrics.validity`` before these are called on "generated"
output, and skipping keeps a single bad molecule from raising an unrelated
report.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

#: Paper Table 3 caption: "the fraction of the samples with |LogP| < 5".
#: Section 2.3's prose separately says "a typical good value ... is between
#: 0 and 5" (Lipinski's Rule of 5) -- an inconsistency in the paper's own
#: text. The Table 3 caption is the more operational statement (it describes
#: what was actually computed for the reported numbers), so |LogP| < 5 is
#: what is implemented here. Flagging this as an interpretation choice, not
#: an unambiguous fact.
LOGP_RANGE = (-5.0, 5.0)


def compute_logp(smiles: str) -> float:
    """RDKit/Crippen LogP for one molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return float(Crippen.MolLogP(mol))


def compute_qed(smiles: str) -> float:
    """QED (Quantitative Estimate of Drug-likeness) for one molecule, in [0, 1]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return float(QED.qed(mol))


def compute_molecular_weight(smiles: str) -> float:
    """Molecular weight for one molecule (paper: "sum of atomic weights")."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return float(Descriptors.MolWt(mol))


def _batch(smiles_list: list[str], fn: object) -> list[float]:
    values: list[float] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        values.append(float(fn(mol)))  # type: ignore[operator]
    return values


def compute_logps_batch(smiles_list: list[str]) -> list[float]:
    """LogP for every parseable molecule, in input order (duplicates kept)."""
    return _batch(smiles_list, Crippen.MolLogP)


def compute_qeds_batch(smiles_list: list[str]) -> list[float]:
    """QED for every parseable molecule, in input order (duplicates kept)."""
    return _batch(smiles_list, QED.qed)


def compute_molecular_weights_batch(smiles_list: list[str]) -> list[float]:
    """Molecular weight for every parseable molecule, in input order (duplicates kept)."""
    return _batch(smiles_list, Descriptors.MolWt)


def logp_in_range_fraction(
    smiles_list: list[str], logp_range: tuple[float, float] = LOGP_RANGE
) -> float:
    """Paper's epsilon_LogP: fraction of (valid) molecules with LogP in range.

    Invalid SMILES are excluded from both the numerator and the denominator,
    consistent with epsilon_LogP being reported as a property of the *valid*
    generated set, not a validity check in its own right.
    """
    logps = compute_logps_batch(smiles_list)
    if not logps:
        return float("nan")
    low, high = logp_range
    in_range = sum(1 for v in logps if low < v < high)
    return in_range / len(logps)