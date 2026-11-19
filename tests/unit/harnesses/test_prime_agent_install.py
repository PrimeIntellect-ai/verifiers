"""prime-agent's install runs once per runtime: a second setup finds it ready and skips
the script, so it reaches neither the release download nor the package index again."""

import asyncio
import os
from pathlib import Path

import pytest

from verifiers.v1.acp import ACPHarness
from verifiers.v1.harnesses import PrimeAgentHarness, PrimeAgentHarnessConfig
from verifiers.v1.harnesses.prime_agent import harness as prime_agent
from verifiers.v1.runtimes import ProgramResult

UV = """#!/bin/sh
# Passes the uv bootstrap's feature probe; logs every install.
case "$*" in
  *--help*) echo "--script" ;;
  "pip install"*) echo "$*" >> "$UV_LOG" ;;
esac
"""

PRIME_AGENT = """#!/bin/sh
case "$1" in
  --version) echo "$PRIME_AGENT_RELEASE_VERSION" ;;
  --prime-agent-bootstrap)
    mkdir -p "$PRIME_AGENT_KERNEL_VENV/bin"
    printf '#!/bin/sh\\n' > "$PRIME_AGENT_KERNEL_VENV/bin/python"
    chmod +x "$PRIME_AGENT_KERNEL_VENV/bin/python" ;;
esac
"""


class LocalRuntime:
    """Runs setup's commands on the host, as root would in a sandbox."""

    user = None

    def __init__(self, path: str, uv_log: Path) -> None:
        self.env = {"PATH": path, "HOME": str(uv_log.parent), "UV_LOG": str(uv_log)}

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            env={**self.env, **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        return ProgramResult(process.returncode or 0, out.decode(), err.decode())


@pytest.fixture
def installed(tmp_path: Path, monkeypatch):
    """A harness whose release is already downloaded (one bundled skill), a stub uv
    that logs installs, and a runtime that runs on the host."""
    root = tmp_path / "vf-prime-agent"
    monkeypatch.setattr(prime_agent, "PRIME_AGENT_DIR", str(root))

    async def no_session_setup(self, runtime) -> None:
        return None

    monkeypatch.setattr(ACPHarness, "setup", no_session_setup)
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig())
    prefix = root / harness.config.commit
    (prefix / "bin" / "skills" / "kernel").mkdir(parents=True)
    (prefix / "bin" / "skills" / "kernel" / "pyproject.toml").write_text("")
    (prefix / "bin" / "prime-agent").write_text(PRIME_AGENT)
    (prefix / "bin" / "prime-agent").chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "uv").write_text(UV)
    (tools / "uv").chmod(0o755)
    uv_log = tmp_path / "uv.log"
    runtime = LocalRuntime(f"{tools}{os.pathsep}/usr/bin:/bin", uv_log)
    installs = lambda: len(uv_log.read_text().splitlines())
    return harness, runtime, prefix, installs


async def test_a_second_setup_skips_the_install(installed) -> None:
    harness, runtime, prefix, installs = installed
    await harness.setup(runtime)
    assert installs() == 1 and (prefix / ".ready").exists()
    await harness.setup(runtime)
    assert installs() == 1  # ready: no second install


@pytest.mark.parametrize("missing", [".ready", "kernel-venv/bin/python"])
async def test_an_unfinished_install_runs_again(installed, missing) -> None:
    harness, runtime, prefix, installs = installed
    await harness.setup(runtime)
    (prefix / missing).unlink()
    await harness.setup(runtime)
    assert installs() == 2
