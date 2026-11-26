import tempfile
from pathlib import Path


def home_dir() -> Path:
    """Best-effort home directory; fall back to the temp dir so import never fails."""
    try:
        return Path.home()
    except RuntimeError:
        return Path(tempfile.gettempdir())


CACHE_DIR = home_dir() / ".cache" / "verifiers"
