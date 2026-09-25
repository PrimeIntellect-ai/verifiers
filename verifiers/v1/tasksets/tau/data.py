"""Fetch the data omitted from Tau's wheels, at the package's pinned revision."""

import fcntl
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen


def prepare_data(repository: str, revision: str) -> Path:
    if configured := os.environ.get("TAU2_DATA_DIR"):
        return Path(configured)
    root = Path.home() / ".cache" / "verifiers-tau" / repository / revision
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root / "data").is_dir():
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                with (
                    urlopen(
                        f"https://github.com/{repository}/archive/{revision}.tar.gz",
                        timeout=120,
                    ) as response,
                    tarfile.open(fileobj=response, mode="r|gz") as archive,
                ):
                    for member in archive:
                        parts = Path(member.name).parts
                        if len(parts) > 1 and parts[1] == "data":
                            member.name = str(Path(*parts[1:]))
                            archive.extract(member, temporary, filter="data")
                (Path(temporary) / "data").rename(root / "data")
    return root / "data"
