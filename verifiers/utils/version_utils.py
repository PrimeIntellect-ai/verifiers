"""Utilities for detecting verifiers and environment version/commit info."""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from pathlib import Path

from verifiers.types import VersionInfo

logger = logging.getLogger(__name__)


def get_commit_for_path(path: Path) -> str | None:
    """
    Get the git commit hash for a file or directory path.

    Walks up the directory tree to find a git repository and returns the
    HEAD commit hash.
    """
    try:
        directory = path if path.is_dir() else path.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(directory),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_package_source_path(package_name: str) -> Path | None:
    """Get the source directory for an installed package."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, "__file__") and module.__file__:
            return Path(module.__file__).parent
    except Exception:
        pass
    return None


def get_vf_version() -> str:
    """Return the verifiers framework version."""
    import verifiers

    return verifiers.__version__


def get_vf_commit() -> str | None:
    """Return the git commit hash of the verifiers package, or None."""
    source = get_package_source_path("verifiers")
    if source is None:
        return None
    return get_commit_for_path(source)


def get_env_version(env_id: str) -> str | None:
    """Return the installed version of an environment package, or None."""
    module_name = env_id.replace("-", "_").split("/")[-1]
    if not module_name:
        return None
    try:
        return importlib.metadata.version(module_name)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


def get_env_commit(env_id: str) -> str | None:
    """Return the git commit hash of an environment package, or None."""
    module_name = env_id.replace("-", "_").split("/")[-1]
    if not module_name:
        return None
    source = get_package_source_path(module_name)
    if source is None:
        return None
    return get_commit_for_path(source)


def get_version_info(env_id: str) -> VersionInfo:
    """Get version and commit info for the verifiers framework and an environment."""
    return VersionInfo(
        vf_version=get_vf_version(),
        vf_commit=get_vf_commit(),
        env_version=get_env_version(env_id),
        env_commit=get_env_commit(env_id),
    )
