"""MOSES dataset acquisition and subsampling (paper §2.1).

Reproduces the paper's dataset construction: the MOSES v1 dataset is downloaded,
canonicalised with RDKit, and randomly subsampled to 12,000 training and 4,087
validation molecules. The subsamples are drawn from the MOSES *native* splits --
pooling and re-splitting would erase the train/test distribution differences
visible in the paper's Figure 12.

Outputs are frozen to disk; every downstream phase reads the frozen files and
never re-samples.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger

from mqs_molecule_generation.data.filters import pass_rate

RDLogger.DisableLog("rdApp.*")

#: Git-LFS media endpoint. ``raw.githubusercontent.com`` serves the LFS *pointer*
#: (a 130-byte text stub), not the file, which is a common and confusing failure.
DATASET_URL = (
    "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
)

#: Verified against the LFS pointer metadata in the upstream repository.
DATASET_SHA256 = "bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b"
DATASET_BYTES = 84_482_588

#: Paper §2.1.
N_TRAIN = 12_000
N_VALID = 4_087

_LFS_POINTER_MAGIC = b"version https://git-lfs"


@dataclass(frozen=True)
class PrepareConfig:
    """Configuration for :func:`prepare`."""

    raw_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/processed")
    seed: int = 0
    n_train: int = N_TRAIN
    n_valid: int = N_VALID
    filter_set: str = "moses_native"
    train_split: str = "train"
    valid_split: str = "test"
    url: str = DATASET_URL
    verify_checksum: bool = True


@dataclass
class SplitManifest:
    """Provenance record written alongside the frozen splits."""

    created_utc: str
    seed: int
    filter_set: str
    n_train: int
    n_valid: int
    train_pass_rate: float
    valid_pass_rate: float
    source_url: str
    source_sha256: str
    source_rows: int
    parse_failures: int
    duplicates_removed: dict[str, int]
    train_valid_overlap: int
    rdkit_version: str
    python_version: str
    git_sha: str | None
    extra: dict[str, object] = field(default_factory=dict)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def download(dest: Path, url: str = DATASET_URL, *, verify: bool = True) -> Path:
    """Download MOSES ``dataset_v1.csv``, skipping if a valid copy already exists.

    Raises if the server returned a Git-LFS pointer instead of the payload, which
    is what happens when the URL points at ``raw.githubusercontent.com``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and (not verify or _sha256(dest) == DATASET_SHA256):
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)

    with tmp.open("rb") as fh:
        if fh.read(len(_LFS_POINTER_MAGIC)) == _LFS_POINTER_MAGIC:
            tmp.unlink()
            raise RuntimeError(
                f"{url} returned a Git-LFS pointer, not the dataset. Use the "
                "media.githubusercontent.com endpoint, or clone the repository "
                "with git-lfs installed."
            )

    if verify:
        got = _sha256(tmp)
        if got != DATASET_SHA256:
            tmp.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: expected {DATASET_SHA256}, got {got}"
            )

    tmp.replace(dest)
    return dest


def canonicalize(smiles: str) -> str | None:
    """Return the canonical SMILES, or None if RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_raw(path: Path) -> pd.DataFrame:
    """Load ``dataset_v1.csv`` and validate its shape."""
    frame = pd.read_csv(path)
    missing = {"SMILES", "SPLIT"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {sorted(missing)}")
    return frame


def subsample(pool: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """Draw ``n`` molecules without replacement, reconstructible from the seed alone."""
    if n > len(pool):
        raise ValueError(f"Requested {n} molecules but the pool holds only {len(pool)}")
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in np.sort(idx)]


def prepare(cfg: PrepareConfig) -> SplitManifest:
    """Run the full preparation pipeline and freeze the results to ``cfg.out_dir``."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = download(cfg.raw_dir / "dataset_v1.csv", cfg.url, verify=cfg.verify_checksum)
    frame = load_raw(raw_path)

    pools: dict[str, list[str]] = {}
    duplicates: dict[str, int] = {}
    parse_failures = 0

    for label, split in (("train", cfg.train_split), ("valid", cfg.valid_split)):
        raw_smiles = frame.loc[frame["SPLIT"] == split, "SMILES"].tolist()
        if not raw_smiles:
            raise ValueError(
                f"Split {split!r} is empty. Available: {sorted(frame['SPLIT'].unique())}"
            )
        canonical = [canonicalize(s) for s in raw_smiles]
        parse_failures += sum(1 for c in canonical if c is None)
        kept = [c for c in canonical if c is not None]
        # Deduplicate within each split only. Cross-split overlap is reported,
        # not silently repaired -- novelty depends on that number being honest.
        deduped = list(dict.fromkeys(kept))
        duplicates[label] = len(kept) - len(deduped)
        pools[label] = deduped

    rng = np.random.default_rng(cfg.seed)
    train = subsample(pools["train"], cfg.n_train, rng)
    valid = subsample(pools["valid"], cfg.n_valid, rng)

    (cfg.out_dir / "train.smi").write_text("\n".join(train) + "\n")
    (cfg.out_dir / "valid.smi").write_text("\n".join(valid) + "\n")

    manifest = SplitManifest(
        created_utc=datetime.now(UTC).isoformat(),
        seed=cfg.seed,
        filter_set=cfg.filter_set,
        n_train=len(train),
        n_valid=len(valid),
        train_pass_rate=pass_rate(train, cfg.filter_set),
        valid_pass_rate=pass_rate(valid, cfg.filter_set),
        source_url=cfg.url,
        source_sha256=_sha256(raw_path),
        source_rows=len(frame),
        parse_failures=parse_failures,
        duplicates_removed=duplicates,
        train_valid_overlap=len(set(train) & set(valid)),
        rdkit_version=rdkit.__version__,
        python_version=sys.version.split()[0],
        git_sha=_git_sha(),
    )
    (cfg.out_dir / "split_manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    return manifest


def load_splits(out_dir: Path) -> tuple[list[str], list[str]]:
    """Read the frozen splits produced by :func:`prepare`."""
    train = (out_dir / "train.smi").read_text().split()
    valid = (out_dir / "valid.smi").read_text().split()
    return train, valid