"""Deterministic guard tests for the Prime Agent live capability fixtures."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from prime_agent_failed_turn_v1 import has_raised_provider_failure
from prime_agent_ipython_cell_v1 import CELL, SENTINEL, has_ipython_cell_call


def test_failed_turn_guard_rejects_a_clean_stop_reason():
    """A provider failure must be visible even when no ModelCall was recorded.

    The request can fail before any call is committed, and an errored rollout
    reports stop_condition "error" rather than None -- so requiring a recorded
    call, or None, rejected the very failure this fixture exists to prove.
    """
    provider_error = SimpleNamespace(type="ProviderError")
    failed = SimpleNamespace(
        ok=False,
        errors=[provider_error],
        stop_condition="error",
        calls=[],
    )
    assert has_raised_provider_failure(failed)

    # Also valid: the failure recorded on a committed call.
    on_call = SimpleNamespace(
        ok=False,
        errors=[provider_error],
        stop_condition=None,
        calls=[SimpleNamespace(error=provider_error)],
    )
    assert has_raised_provider_failure(on_call)

    # A successful rollout must never satisfy it.
    assert not has_raised_provider_failure(
        SimpleNamespace(ok=True, errors=[], stop_condition="agent_completed", calls=[])
    )
    # Nor a clean agent stop that merely lacks errors.
    assert not has_raised_provider_failure(
        SimpleNamespace(ok=False, errors=[], stop_condition="error", calls=[])
    )
    # Nor a non-provider failure, which would mean something else broke.
    assert not has_raised_provider_failure(
        SimpleNamespace(
            ok=False,
            errors=[SimpleNamespace(type="HarnessError")],
            stop_condition="error",
            calls=[],
        )
    )


def test_ipython_cell_guard_requires_verbatim_code_and_real_execution():
    """A fabricated call plus the right reply must not score.

    The reward needs both halves: the cell submitted verbatim AND the sentinel in
    a tool RESULT, which only a real kernel execution produces.
    """

    def trace_with(code: str, outputs: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            info={
                "prime_agent_segments": [
                    {
                        "last_reply": "DONE",
                        "terminated": False,
                        "tool_calls": [
                            {"name": "ipython", "arguments": json.dumps({"code": code})}
                        ],
                        "tool_outputs": outputs,
                    }
                ]
            }
        )

    assert has_ipython_cell_call(trace_with(CELL, [SENTINEL]))
    # Claimed but never executed: no tool result carries the sentinel.
    assert not has_ipython_cell_call(trace_with(CELL, []))
    assert not has_ipython_cell_call(trace_with(CELL, ["something else"]))
    # Executed something else, even if it printed the sentinel itself.
    assert not has_ipython_cell_call(trace_with(f"{CELL}\nprint('extra')", [SENTINEL]))


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _PersistenceRuntime:
    def __init__(self, existing_paths: set[str]):
        self.existing_paths = existing_paths
        self.calls: list[tuple[list[str], dict]] = []

    async def run(self, command: list[str], environment: dict):
        self.calls.append((command, environment))
        # `test -e PATH` exits 0 when the path EXISTS.
        return SimpleNamespace(exit_code=0 if command[-1] in self.existing_paths else 1)


class _PersistenceAgent:
    def __init__(self, runtime, interaction):
        self.runtime = runtime
        self._interaction = interaction

    def provision(self, task):
        return _AsyncContext(self.runtime)

    def interaction(self, task, *, runtime):
        assert runtime is self.runtime
        return _AsyncContext(self._interaction)


class _PersistenceInteraction:
    def __init__(self, trace):
        self.trace = trace
        self._segments = [
            SimpleNamespace(last_reply="READY", messages=[], terminated=False),
            SimpleNamespace(last_reply="marker", messages=[], terminated=False),
        ]

    async def turn(self, prompt):
        return self._segments.pop(0)


def _prime_agent_trace_root(trace_id: str) -> str:
    return (
        "/tmp/vf-prime-agent-state/"
        f"{hashlib.sha256(trace_id.encode()).hexdigest()[:32]}"
    )


@pytest.mark.asyncio
async def test_persistence_fixture_sees_state_present_while_the_session_runs():
    from prime_agent_persistence_v1 import PrimeAgentPersistenceEnv

    trace = SimpleNamespace(id="trace-that-remains", info={})
    root = _prime_agent_trace_root(trace.id)
    runtime = _PersistenceRuntime(existing_paths={root})
    interaction = _PersistenceInteraction(trace)
    agents = SimpleNamespace(agent=_PersistenceAgent(runtime, interaction))

    await PrimeAgentPersistenceEnv.run(
        object.__new__(PrimeAgentPersistenceEnv), None, agents
    )

    assert runtime.calls == [(["test", "-e", root], {})]
    assert trace.info["prime_agent_state_present_during_run"] is True


class _GuardRuntime:
    supports_live_processes = False

    def __init__(self, wrapper: str | None = None):
        self.wrapper = wrapper
        self.calls: list[list[str]] = []

    async def run(self, command, environment):
        self.calls.append(command)
        failed = self.wrapper is not None and command == ["chmod", "700", self.wrapper]
        return SimpleNamespace(
            exit_code=1 if failed else 0, stderr="chmod denied", stdout=""
        )

    async def write(self, path, content):
        return None


@pytest.mark.asyncio
async def test_prime_agent_refuses_non_live_runtime_before_fresh_relaunch():
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    with pytest.raises(RuntimeError, match="requires runtime live-process support"):
        await harness.session(
            None, None, _GuardRuntime(), "endpoint", "secret", {}, None
        )


@pytest.mark.asyncio
async def test_prime_agent_wrapper_chmod_failure_is_reported():
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    trace = SimpleNamespace(id="chmod-guard")
    runtime = _GuardRuntime(f"{harness.trace_root(trace)}/prime-agent")
    ctx = SimpleNamespace(model="model", sampling=SimpleNamespace(max_tokens=None))
    with pytest.raises(RuntimeError, match="wrapper permissions failed"):
        await harness._prepare(ctx, trace, runtime, "endpoint", None)


@pytest.mark.asyncio
async def test_prime_agent_launch_preserves_typed_rollout_error(monkeypatch):
    from verifiers.v1.errors import SandboxError
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    error = SandboxError("sandbox unavailable")

    async def prepare(*args, **kwargs):
        return ["prime-agent"]

    async def run(*args, **kwargs):
        raise error

    async def tail(*args, **kwargs):
        return "daemon tail"

    monkeypatch.setattr(harness, "_prepare", prepare)
    monkeypatch.setattr(harness, "daemon_log_tail", tail)
    monkeypatch.setattr(
        "verifiers.v1.harnesses.prime_agent.harness.PRIME_AGENT_ACP.run", run
    )
    data = SimpleNamespace(prompt="hello", system_prompt=None)
    trace = SimpleNamespace(id="typed-error")
    with pytest.raises(SandboxError) as raised:
        await harness.launch(None, trace, object(), "endpoint", "secret", {}, data)
    assert raised.value is error


def test_prime_agent_installer_bootstraps_https_certificates_and_tools():
    """CA roots must accompany the tools, or every HTTPS download fails."""
    installer = Path("verifiers/v1/harnesses/prime_agent/install.sh").read_text()
    # Both package managers install CA roots alongside whatever is missing.
    assert (
        "apt-get install -y --no-install-recommends ca-certificates $missing"
        in installer
    )
    assert "apk add --no-cache ca-certificates $missing" in installer
    # git is provisioned too: a coding taskset that clones fails deep inside a
    # rollout without it, which reads as a bad score rather than a setup gap.
    assert 'command -v git >/dev/null 2>&1 || missing="$missing git"' in installer
    # uv must be pinned through the versioned installer URL; the generic
    # installer ignores UV_VERSION and would silently install a different build.
    assert "https://astral.sh/uv/${uv_version}/install.sh" in installer


def test_prime_agent_installer_handles_musl_without_glibc_download():
    installer = Path("verifiers/v1/harnesses/prime_agent/install.sh").read_text()
    assert "ldd --version 2>&1 | grep -qi musl" in installer
    assert "apk add --no-cache nodejs-current npm" in installer
    assert "Alpine/musl requires nodejs-current and npm" in installer


@pytest.mark.asyncio
async def test_prime_agent_setup_forwards_resolved_environment(monkeypatch):
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="prime-agent", env={"HTTPS_PROXY": "http://proxy"})
    )
    calls = []

    async def run(command, environment):
        calls.append((command, environment))
        return SimpleNamespace(exit_code=0, stderr="", stdout="")

    async def setup(*args):
        return None

    runtime = SimpleNamespace(run=run)
    monkeypatch.setattr(
        "verifiers.v1.harnesses.prime_agent.harness.PRIME_AGENT_ACP.setup", setup
    )
    await harness.setup(runtime)
    assert calls[0][1]["HTTPS_PROXY"] == "http://proxy"


def test_prime_agent_cleanup_removes_state_and_tmpdir_together():
    source = Path("verifiers/v1/harnesses/prime_agent/harness.py").read_text()
    assert 'runtime.run(["rm", "-rf", root, self.tmp_dir(trace)]' in source


def test_version_validator_rejects_malformed_semver():
    """A bad version becomes a 404 on the release URL instead of a clear error.

    The permissive suffix pattern accepted empty dot-separated identifiers, so
    values like "1.2.3-." validated and then failed as a download error far from
    the actual mistake.
    """
    from pydantic import ValidationError

    from verifiers.v1.utils.loaders import harness_config_type

    config = harness_config_type("prime-agent")
    digest = "a" * 64
    for good in ("0.7.0", "1.2.3-rc.1", "1.2.3+build.5"):
        config(id="x", version=good, tarball_sha256=digest)
    for bad in ("1.2.3-.", "1.2.3-foo..bar", "1.2.3+.", "01.2.3", "1.2"):
        with pytest.raises(ValidationError):
            config(id="x", version=bad, tarball_sha256=digest)


@pytest.mark.asyncio
async def test_daemon_log_tail_never_masks_the_original_failure():
    """Diagnostics run on the failure path, so they must not raise themselves.

    A sandbox that is already gone would otherwise replace the real error with a
    SandboxError from the log read, losing the attribution entirely.
    """
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    class _DeadRuntime:
        async def run(self, command, environment):
            raise RuntimeError("sandbox is gone")

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    trace = SimpleNamespace(id="trace-for-log-tail")
    assert await harness.daemon_log_tail(_DeadRuntime(), trace) == ""
