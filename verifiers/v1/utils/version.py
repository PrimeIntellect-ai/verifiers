"""Build provenance: the installed verifiers package's git commit, when resolvable."""

import json
import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


@cache
def verifiers_commit() -> str | None:
    """The git commit of the installed verifiers, when resolvable: a git-pinned
    install records it in `direct_url.json` (`vcs_info.commit_id`); an editable or
    source checkout answers `git rev-parse`. None otherwise (e.g. a PyPI wheel)."""
    try:
        direct_url = distribution("verifiers").read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except (PackageNotFoundError, ValueError):
        pass
    package_dir = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
