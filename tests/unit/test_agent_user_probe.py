"""The agent-user home lookup works in images that delete awk, cut and getent."""

import os
import shutil
import subprocess

import pytest

from verifiers.v1.rollout import _HOME_PROBE

SHELLS = [s for s in ("/bin/dash", "/bin/bash", "/bin/sh") if os.path.exists(s)]


@pytest.fixture
def bare(tmp_path):
    """A PATH holding only `sh`, and a passwd file (last line without a newline)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sh").symlink_to(shutil.which("sh"))
    home = tmp_path / "home" / "agent"
    home.mkdir(parents=True)
    passwd = tmp_path / "passwd"
    passwd.write_text(
        "root:x:0:0:root:/root:/bin/bash\n"
        f"agentx:x:1002:1002::{tmp_path}/missing:/bin/sh\n"
        f"agent:x:1001:1001::{home}:/bin/bash\n"
        f"last:x:1003:1003::{home}:/bin/sh"
    )
    return bin_dir, passwd, home


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("user", "found", "ok"),
    [
        ("agent", "home", True),
        ("last", "home", True),
        ("agentx", "missing", False),
        ("age", "", False),
        ("a*", "", False),
        ("nobody-here", "", False),
    ],
)
def test_home_probe_without_awk(bare, shell, user, found, ok):
    bin_dir, passwd, home = bare
    probe = _HOME_PROBE.replace("/etc/passwd", str(passwd))
    result = subprocess.run(
        [shell, "-c", probe, "sh", user],
        capture_output=True,
        text=True,
        env={"PATH": str(bin_dir)},
        check=False,
    )
    paths = {"home": str(home), "missing": str(passwd.parent / "missing"), "": ""}
    expected = paths[found]
    assert result.stdout.strip() == expected
    assert (result.returncode == 0) is ok
