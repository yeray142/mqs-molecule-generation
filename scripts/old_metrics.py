#!/usr/bin/env python
"""Step 1 of 2: run MOSES's ORIGINAL metrics in the period-matched environment.

Run this inside the Python 3.7 conda env (rdkit 2019.09.3, pytorch 1.1.0,
molsets 0.3.1, fcd_torch 1.0.7 -- see the versions taken from MOSES's own
Dockerfile):

    conda run -n <your-env> python scripts/old_metrics.py

It deliberately imports NOTHING from this project. The project's code uses
Python 3.9+/3.10+ features throughout (functools.cache, zip(strict=True),
str.removeprefix, built-in generics under pydantic, ...), so importing it on
3.7 fails one incompatibility at a time. This script only does what only the
old environment can do -- call MOSES's own unmodified functions -- and writes
the per-molecule results to JSON. Step 2 (scripts/compare_old_metrics.py, run
in your normal uv environment) computes our values on the same SMILES and
compares.

Per-molecule values are saved, not just means: if the two implementations
disagree, the JSON shows exactly which molecules and by how much.

Must stay valid Python 3.7 syntax and stdlib usage.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-file", type=Path, default=Path("data/processed/train.smi"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/moses_original_values.json"))
    args = parser.parse_args(argv)

    import moses
    import moses.metrics as moses_metrics
    import rdkit
    from moses.utils import get_mol, mapper
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    print("python  :", platform.python_version())
    print("rdkit   :", rdkit.__version__)
    print("moses   :", moses.__version__)

    smiles_list = args.smiles_file.read_text().split()
    print(f"Loaded {len(smiles_list)} SMILES from {args.smiles_file}")

    mols = mapper(args.n_jobs)(get_mol, smiles_list)
    valid_mols = [m for m in mols if m is not None]
    n_invalid = len(mols) - len(valid_mols)
    if n_invalid:
        print(f"WARNING: {n_invalid} SMILES failed to parse via moses.utils.get_mol")

    def per_molecule(func, cast):
        """Apply func to valid mols, re-align to the full list (None for invalid)."""
        values = iter(mapper(args.n_jobs)(func, valid_mols))
        return [cast(next(values)) if m is not None else None for m in mols]

    results = {
        "weight": per_molecule(moses_metrics.weight, float),
        "qed": per_molecule(moses_metrics.QED, float),
        "sa": per_molecule(moses_metrics.SA, float),
        "logp": per_molecule(moses_metrics.logP, float),
        "passes_filters": per_molecule(moses_metrics.mol_passes_filters, bool),
    }

    print(f"\nMOSES original, means over {len(valid_mols)} valid molecules:")
    for name in ("weight", "qed", "sa", "logp"):
        vals = [v for v in results[name] if v is not None]
        print(f"  {name:<8}{sum(vals) / len(vals):.4f}")
    passes = [v for v in results["passes_filters"] if v is not None]
    print("  {:<8}{:.4f}".format("filters", sum(passes) / len(passes)))

    payload = {
        "meta": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "moses": moses.__version__,
            "smiles_file": str(args.smiles_file),
            "n_smiles": len(smiles_list),
            "n_invalid": n_invalid,
        },
        "smiles": smiles_list,
    }
    payload.update(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f"\nWrote {args.output}")
    print("Next: run scripts/compare_old_metrics.py in your normal (uv) environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())