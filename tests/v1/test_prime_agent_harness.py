import hashlib
import os
import shlex
import subprocess
from pathlib import PurePosixPath

import pytest

from verifiers.v1.harnesses.prime_agent.harness import (
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
