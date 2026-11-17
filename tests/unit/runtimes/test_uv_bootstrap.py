"""The uv bootstrap's standalone-installer step, with the download stubbed out."""

import os
import subprocess
from pathlib import Path

from verifiers.v1.runtimes.base import _DOWNLOAD_UV


def test_standalone_uv_install_is_unmanaged(tmp_path: Path) -> None:
    """The installer must run unmanaged, into ~/.local/bin: a managed install writes a
    receipt to $HOME/.config/uv and fails when another user with the same HOME (root,
    installing a harness) already owns one."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    # Stands in for the downloaded installer: reports what the real one would read.
    curl.write_text("#!/bin/sh\necho 'echo \"unmanaged=$UV_UNMANAGED_INSTALL\"'\n")
    curl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["sh", "-c", _DOWNLOAD_UV],
        env={"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"unmanaged={home}/.local/bin"
