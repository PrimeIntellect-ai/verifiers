import hashlib
import os
import shlex
import subprocess
from pathlib import PurePosixPath

import pytest

from verifiers.v1.harnesses.prime_agent.harness import (
    INSTALL,
    STATE_ROOT,
    PrimeAgentHarness,
    _guarded_install,
)
from verifiers.v1.trace import Trace


def test_trace_root_encodes_untrusted_trace_id() -> None:
    trace = Trace.model_construct(id="../../etc")

    root = PurePosixPath(PrimeAgentHarness.trace_root(trace))

    assert root.parent == PurePosixPath(STATE_ROOT)
    assert root.name == hashlib.sha256(trace.id.encode()).hexdigest()[:32]


@pytest.mark.parametrize("owner", [str(os.getpid()), f"{os.getpid()}:0"])
def test_install_lock_recovers_from_live_reused_pid(tmp_path, owner: str) -> None:
    lock = tmp_path / "install.lock"
    marker = tmp_path / "installed"
    lock.symlink_to(owner)
    command = f": > {shlex.quote(str(marker))}"

    result = subprocess.run(
        ["sh", "-c", _guarded_install(str(tmp_path), str(lock), command)],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert not lock.is_symlink()


@pytest.mark.parametrize("failed_move", [0, 1, 2])
def test_install_activation_is_transactional(tmp_path, failed_move: int) -> None:
    root = tmp_path / "install"
    old_bin = root / "node_modules" / ".bin" / "prime-agent"
    old_bin.parent.mkdir(parents=True)
    old_bin.write_text("old install")
    old_bin.chmod(0o700)
    (root / ".installed").write_text("old digest")

    tarball = tmp_path / "prime-agent.tgz"
    tarball.write_bytes(b"new package")
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    curl = fake_bin / "curl"
    curl.write_text('#!/bin/sh\ncp "$TEST_TARBALL" "$4"\n')
    npm = fake_bin / "npm"
    npm.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --prefix ]; then prefix=$2; shift 2; else shift; fi\n'
        "done\n"
        'mkdir -p "$prefix/node_modules/.bin"\n'
        'printf %s new-install > "$prefix/node_modules/.bin/prime-agent"\n'
        'chmod 700 "$prefix/node_modules/.bin/prime-agent"\n'
    )
    mv = fake_bin / "mv"
    mv.write_text(
        "#!/bin/sh\n"
        'count=$(cat "$TEST_MV_COUNT" 2>/dev/null || printf 0)\n'
        "count=$((count + 1))\n"
        'printf %s "$count" > "$TEST_MV_COUNT"\n'
        'if [ "$count" -eq "$TEST_FAILED_MOVE" ]; then\n'
        '  /bin/mv "$@"\n'
        "  exit 1\n"
        "fi\n"
        'exec /bin/mv "$@"\n'
    )
    for executable in (curl, npm, mv):
        executable.chmod(0o700)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_FAILED_MOVE": str(failed_move),
        "TEST_MV_COUNT": str(tmp_path / "mv-count"),
        "TEST_TARBALL": str(tarball),
        "VF_NODE_BIN_DIR": str(fake_bin),
        "VF_PRIME_AGENT_BIN": str(old_bin),
        "VF_PRIME_AGENT_INSTALL_DIR": str(root),
        "VF_PRIME_AGENT_KERNEL_VENV": str(tmp_path / "kernel"),
        "VF_PRIME_AGENT_TARBALL": "https://example.invalid/prime-agent.tgz",
        "VF_PRIME_AGENT_TARBALL_SHA256": digest,
    }
    result = subprocess.run(
        ["sh", "-c", INSTALL],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    if failed_move:
        assert result.returncode != 0
        assert old_bin.read_text() == "old install"
        assert (root / ".installed").read_text() == "old digest"
    else:
        assert result.returncode == 0, result.stderr
        assert old_bin.read_text() == "new-install"
        assert (root / ".installed").read_text() == digest
    assert not list(tmp_path.glob("install.previous.*"))
    assert not list(tmp_path.glob("install.staging.*"))
