"""Per-run manifest: git SHA, lockfile hash, dependency versions.

Attached to every BenchmarkRecord via tags or hook.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pydantic


class RunManifest(pydantic.BaseModel):
    """Per-run metadata manifest.

    Captures: git SHA, lockfile hash, Python version, key package versions.
    Attached to every BenchmarkRecord via tags or hook.
    """

    git_sha: str
    git_dirty: bool
    lockfile_hash: str
    python_version: str
    package_versions: dict[str, str]

    @classmethod
    def build(cls) -> RunManifest:
        """Build manifest from current environment."""
        # Git SHA
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "diff", "--stat"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        except subprocess.CalledProcessError:
            git_sha = "unknown"
            git_dirty = False

        # Lockfile hash
        lockfile = Path("uv.lock")
        if lockfile.exists():
            lockfile_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()[:16]
        else:
            lockfile_hash = "no-lockfile"

        import sys

        python_version = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )

        # Package versions
        package_versions = _get_package_versions()

        return cls(
            git_sha=git_sha,
            git_dirty=git_dirty,
            lockfile_hash=lockfile_hash,
            python_version=python_version,
            package_versions=package_versions,
        )


def _get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    import importlib

    packages = ["torch", "pennylane", "rdkit", "hydra", "numpy", "click"]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            v: str = getattr(mod, "__version__", "unknown")
            versions[pkg] = v
        except ImportError:
            versions[pkg] = "not-installed"
    return versions


# Global singleton built at import time
MANIFEST: RunManifest = RunManifest.build()


def build_manifest() -> RunManifest:
    """Build a fresh manifest (useful for explicit manifest creation)."""
    return RunManifest.build()
