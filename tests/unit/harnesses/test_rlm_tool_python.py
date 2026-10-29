"""Root installs rlm for the agent user on a Python that user can run.

install.sh takes the first `python3` on root's PATH for rlm's tool venv. CyberStrike's
langgraph-checkpoint-isolation-defensive puts /opt/rk/bin first, a venv on uv's Python
3.13 under root's 0700 home, so its `auditor` agent got "Permission denied" running
`bin/rlm` and every eval failed before turn 1 (2026-10-29)."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

from verifiers.v1.harnesses.rlm import harness
from verifiers.v1.harnesses.rlm.harness import (
    _USER_PYTHON,
    USER_PYTHON_FALLBACK,
    RLMHarness,
    RLMHarnessConfig,
)


def probe(path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", _USER_PYTHON],
        env={"PATH": path},
        capture_output=True,
        check=False,
    )


def test_the_probe_skips_a_python_the_user_cannot_run(tmp_path):
    denied, usable = tmp_path / "rk-bin", tmp_path / "usr-bin"
    denied.mkdir()
    usable.mkdir()
    (denied / "python3").write_text("#!/bin/sh\nexit 126\n")  # as /opt/rk/bin/python3
    (denied / "python3").chmod(0o755)
    os.symlink(sys.executable, usable / "python3")
    result = probe(f"{denied}:{usable}")
    assert (result.returncode, result.stdout.decode()) == (0, f"{usable}/python3")


def test_the_probe_finds_nothing_without_a_usable_python(tmp_path):
    assert probe(str(tmp_path)).returncode == 1


@dataclass
class Runtime:
    user: str | None
    answer: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(exit_code=0, stdout="/usr/bin/python3")
    )
    calls: list = field(default_factory=list)

    async def run(self, argv, env):
        self.calls.append((self.user, argv))
        return self.answer


async def setup(monkeypatch, runtime) -> dict:
    seen = {}

    async def installed(runtime, **kwargs):
        seen.update(kwargs)

    async def acp_setup(self, runtime):
        pass

    monkeypatch.setattr(harness, "ensure_installed", installed)
    monkeypatch.setattr(harness.ACPHarness, "setup", acp_setup)
    await RLMHarness(RLMHarnessConfig()).setup(runtime)  # type: ignore[arg-type]
    return seen


async def test_an_agent_user_gets_rlm_on_a_python_it_can_run(monkeypatch):
    runtime = Runtime("auditor")
    seen = await setup(monkeypatch, runtime)
    assert runtime.calls == [("auditor", ["sh", "-c", _USER_PYTHON])]
    assert seen["env"]["RLM_TOOL_PYTHON"] == "/usr/bin/python3"
    # An earlier install on another Python is not reused, and this one records its own.
    assert "= /usr/bin/python3 ]" in seen["ready"]
    assert "printf %s /usr/bin/python3 > " in seen["install"]


async def test_a_user_with_no_usable_python_gets_one_installed(monkeypatch):
    runtime = Runtime("auditor", SimpleNamespace(exit_code=1, stdout=""))
    seen = await setup(monkeypatch, runtime)
    assert seen["env"]["RLM_TOOL_PYTHON"] == USER_PYTHON_FALLBACK


async def test_root_and_an_explicit_choice_keep_install_sh_selection(monkeypatch):
    runtime = Runtime(None)
    seen = await setup(monkeypatch, runtime)
    assert runtime.calls == [] and "RLM_TOOL_PYTHON" not in seen["env"]
    runtime = Runtime("auditor")
    config = RLMHarnessConfig(env={"RLM_TOOL_PYTHON": "/opt/py/bin/python3"})
    seen = {}

    async def installed(runtime, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(harness, "ensure_installed", installed)
    await RLMHarness(config).setup(runtime)  # type: ignore[arg-type]
    assert runtime.calls == []
    assert seen["env"]["RLM_TOOL_PYTHON"] == "/opt/py/bin/python3"
