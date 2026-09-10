"""Runtime shell snippets, run under a real `sh` with stubbed tools: the harness setup keeps a
`uv` already on PATH and only installs one when none is found."""

import os
import stat
import subprocess
from pathlib import Path

from verifiers.v1.runtimes.base import _ENSURE_UV


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _ensure_uv(tmp_path: Path, *, with_uv: bool) -> bool:
    """Run `_ENSURE_UV` with `pip` stubbed to leave a marker; whether the install ran."""
    bin_dir, marker = tmp_path / "bin", tmp_path / "pip-ran"
    bin_dir.mkdir(parents=True)
    # `pip install --user uv` stands in: it drops a `uv` into ~/.local/bin and leaves a marker.
    _stub(
        bin_dir,
        "pip",
        f"touch {marker}; mkdir -p $HOME/.local/bin; cp $0 $HOME/.local/bin/uv",
    )
    if with_uv:
        _stub(bin_dir, "uv", "exit 0")
    subprocess.run(
        ["sh", "-c", f"{_ENSURE_UV}; command -v uv >/dev/null"],
        env={"HOME": str(tmp_path), "PATH": f"{bin_dir}:{os.defpath}"},
        check=True,
    )
    return marker.exists()


def test_ensure_uv_keeps_an_installed_uv(tmp_path):
    assert _ENSURE_UV.index("command -v uv") < _ENSURE_UV.index("pip install")
    assert not _ensure_uv(tmp_path / "present", with_uv=True)
    assert _ensure_uv(tmp_path / "absent", with_uv=False)
