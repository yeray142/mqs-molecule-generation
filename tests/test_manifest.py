"""Tests for manifest generation."""

import pytest

from mqs_molecule_generation.utils.manifest import RunManifest, build_manifest


class TestRunManifest:
    """Test RunManifest."""

    def test_build_manifest(self) -> None:
        """Test manifest building."""
        manifest = build_manifest()
        assert isinstance(manifest, RunManifest)
        assert manifest.git_sha is not None
        assert manifest.lockfile_hash is not None
        assert manifest.python_version is not None

    def test_manifest_fields(self) -> None:
        """Test manifest has all required fields."""
        manifest = build_manifest()
        assert manifest.git_sha
        assert isinstance(manifest.git_dirty, bool)
        assert manifest.lockfile_hash
        assert manifest.python_version
        assert isinstance(manifest.package_versions, dict)

    def test_manifest_model_dump(self) -> None:
        """Test manifest can be dumped to dict."""
        manifest = build_manifest()
        data = manifest.model_dump()
        assert isinstance(data, dict)
        assert "git_sha" in data
        assert "lockfile_hash" in data

    def test_manifest_package_versions(self) -> None:
        """Test package versions are captured."""
        manifest = build_manifest()
        # Should have entries for numpy, click at minimum
        assert "numpy" in manifest.package_versions
        assert "click" in manifest.package_versions

    def test_manifest_not_installed_packages(self) -> None:
        """Test not-installed packages show 'not-installed'."""
        manifest = build_manifest()
        # Packages that are optional should show as not-installed
        assert manifest.package_versions.get("pennylane") in (
            "not-installed",
            "unknown",
        )
