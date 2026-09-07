import os
import tempfile
from pathlib import Path


def home_dir() -> Path:
    """Best-effort home directory; fall back to the temp dir so import never fails."""
    try:
        return Path.home()
    except RuntimeError:
        return Path(tempfile.gettempdir())


def cache_dir() -> Path:
    if override := os.environ.get("VERIFIERS_CACHE_DIR"):
        return Path(override).expanduser()
    return home_dir() / ".cache" / "verifiers"


CACHE_DIR = cache_dir()
