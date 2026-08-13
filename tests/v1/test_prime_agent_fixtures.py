"""Deterministic guard tests for the Prime Agent live capability fixtures."""

import asyncio
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
    command, environment = calls[0]
    assert environment["HTTPS_PROXY"] == "http://proxy"

    # With a command argument, flock's operand is a lock-file path, not an
    # already-open descriptor. It must therefore name the shared install lock,
    # rather than the accidental relative file named "9".
    guarded = command[-1]
    assert "flock -x 9 sh -c" not in guarded
    assert guarded.startswith(
        "mkdir -p /var/tmp/vf-prime-agent && "
        "flock -x /var/tmp/vf-prime-agent/install.lock sh -c "
    )
    assert "9>/var/tmp/vf-prime-agent/install.lock" not in guarded


class _CleanupRuntime:
    def __init__(self, results):
        self.results = iter(results)
        self.calls: list[tuple[list[str], dict]] = []

    async def run(self, command, environment):
        self.calls.append((command, environment))
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def prime_agent_cleanup():
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    trace = SimpleNamespace(id="cleanup-trace")
    return harness, trace


def _cleanup_result(exit_code=0, stderr="", *, timed_out=False):
    return SimpleNamespace(exit_code=exit_code, stderr=stderr, timed_out=timed_out)


@pytest.mark.asyncio
async def test_prime_agent_cleanup_checks_both_deletion_results(prime_agent_cleanup):
    harness, trace = prime_agent_cleanup
    runtime = _CleanupRuntime(
        [_cleanup_result(1), _cleanup_result(), _cleanup_result()]
    )

    await harness.cleanup(trace, runtime)

    root = harness.trace_root(trace)
    tmp_dir = harness.tmp_dir(trace)
    assert [call[0] for call in runtime.calls[1:]] == [
        ["rm", "-rf", tmp_dir],
        ["rm", "-rf", root],
    ]


@pytest.mark.asyncio
async def test_prime_agent_cleanup_first_deletion_failure_preserves_retry_paths(
    prime_agent_cleanup,
):
    from verifiers.v1.errors import SandboxError

    harness, trace = prime_agent_cleanup
    runtime = _CleanupRuntime(
        [_cleanup_result(1), _cleanup_result(exit_code=23, stderr="no")]
    )

    with pytest.raises(SandboxError, match="removal of per-trace tmp") as raised:
        await harness.cleanup(trace, runtime)

    root = harness.trace_root(trace)
    tmp_dir = harness.tmp_dir(trace)
    assert f"state root {root}; per-trace tmp {tmp_dir}" in str(raised.value)
    assert "Confirmed removed this attempt: none" in str(raised.value)
    assert [call[0] for call in runtime.calls[1:]] == [["rm", "-rf", tmp_dir]]


@pytest.mark.asyncio
async def test_prime_agent_cleanup_second_deletion_failure_reports_partial_cleanup(
    prime_agent_cleanup,
):
    from verifiers.v1.errors import SandboxError

    harness, trace = prime_agent_cleanup
    runtime = _CleanupRuntime(
        [_cleanup_result(1), _cleanup_result(), _cleanup_result(exit_code=24)]
    )

    with pytest.raises(SandboxError, match="removal of state root") as raised:
        await harness.cleanup(trace, runtime)

    root = harness.trace_root(trace)
    tmp_dir = harness.tmp_dir(trace)
    assert f"Retry paths: state root {root}; per-trace tmp {tmp_dir}" in str(
        raised.value
    )
    assert f"Confirmed removed this attempt: {tmp_dir}" in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (None, "no authoritative result"),
        (TimeoutError(), "timed out"),
    ],
)
async def test_prime_agent_cleanup_never_treats_unknown_deletion_as_success(
    prime_agent_cleanup, result, reason
):
    from verifiers.v1.errors import SandboxError

    harness, trace = prime_agent_cleanup
    runtime = _CleanupRuntime([_cleanup_result(1), result])

    with pytest.raises(SandboxError, match=reason):
        await harness.cleanup(trace, runtime)

    # An unknown result for the first deletion must not advance to the root.
    assert len(runtime.calls) == 2


@pytest.mark.asyncio
async def test_prime_agent_cleanup_retry_is_idempotent_after_partial_failure(
    prime_agent_cleanup,
):
    from verifiers.v1.errors import SandboxError

    harness, trace = prime_agent_cleanup
    # The first attempt confirms tmp removal but cannot confirm root removal.
    first = _CleanupRuntime([_cleanup_result(1), _cleanup_result(), _cleanup_result(1)])
    with pytest.raises(SandboxError):
        await harness.cleanup(trace, first)

    # `rm -rf` makes retrying tmp safe even though the first attempt removed it.
    retry = _CleanupRuntime([_cleanup_result(1), _cleanup_result(), _cleanup_result()])
    await harness.cleanup(trace, retry)
    assert [call[0] for call in retry.calls[1:]] == [
        ["rm", "-rf", harness.tmp_dir(trace)],
        ["rm", "-rf", harness.trace_root(trace)],
    ]


@pytest.mark.asyncio
async def test_prime_agent_cleanup_stop_failure_prevents_deletion(prime_agent_cleanup):
    from verifiers.v1.errors import SandboxError

    harness, trace = prime_agent_cleanup
    runtime = _CleanupRuntime(
        [_cleanup_result(0), _cleanup_result(exit_code=1, stderr="still live")]
    )

    with pytest.raises(SandboxError, match="daemon stop"):
        await harness.cleanup(trace, runtime)

    assert len(runtime.calls) == 2
    assert runtime.calls[0][0][:2] == ["test", "-S"]
    assert runtime.calls[1][0][:2] == ["sh", "-c"]


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


class _SocketCleanupRuntime:
    def __init__(
        self,
        *,
        socket_exists: bool,
        socket_exit_code: int | None = None,
        stop_exit_code: int = 0,
        rm_exit_code: int = 0,
    ):
        self.socket_exists = socket_exists
        self.socket_exit_code = socket_exit_code
        self.stop_exit_code = stop_exit_code
        self.rm_exit_code = rm_exit_code
        self.calls: list[list[str]] = []

    async def run(self, command, environment):
        self.calls.append(command)
        if command[:2] == ["test", "-S"]:
            return SimpleNamespace(
                exit_code=(
                    self.socket_exit_code
                    if self.socket_exit_code is not None
                    else (0 if self.socket_exists else 1)
                ),
                stderr="socket check denied",
                stdout="",
            )
        if command[:2] == ["rm", "-rf"]:
            return SimpleNamespace(
                exit_code=self.rm_exit_code, stderr="rm denied", stdout=""
            )
        return SimpleNamespace(
            exit_code=self.stop_exit_code, stderr="stop denied", stdout=""
        )


@pytest.mark.asyncio
async def test_prime_agent_cleanup_skips_missing_socket_and_remains_idempotent():
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    trace = SimpleNamespace(id="no-daemon")
    runtime = _SocketCleanupRuntime(socket_exists=False)

    await harness.cleanup(trace, runtime)
    await harness.cleanup(trace, runtime)

    assert [call[:2] for call in runtime.calls] == [
        ["test", "-S"],
        ["rm", "-rf"],
        ["rm", "-rf"],
        ["test", "-S"],
        ["rm", "-rf"],
        ["rm", "-rf"],
    ]
    assert not any(call[:2] == ["sh", "-c"] for call in runtime.calls)


@pytest.mark.asyncio
async def test_prime_agent_cleanup_retains_state_when_stop_or_rm_fails():
    from verifiers.v1.errors import SandboxError
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    trace = SimpleNamespace(id="cleanup-errors")
    stop_failed = _SocketCleanupRuntime(socket_exists=True, stop_exit_code=1)
    with pytest.raises(SandboxError, match="daemon stop"):
        await harness.cleanup(trace, stop_failed)
    assert not any(call[:2] == ["rm", "-rf"] for call in stop_failed.calls)

    rm_failed = _SocketCleanupRuntime(socket_exists=False, rm_exit_code=1)
    with pytest.raises(SandboxError, match="removal of per-trace tmp"):
        await harness.cleanup(trace, rm_failed)


@pytest.mark.asyncio
async def test_prime_agent_stateful_turn_errors_include_daemon_log_and_keep_rollout_type(
    monkeypatch,
):
    from verifiers.v1.errors import SandboxError
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    trace = SimpleNamespace(id="stateful-turn")

    async def prepare(*args, **kwargs):
        return ["prime-agent"]

    async def tail(*args, **kwargs):
        return "daemon tail"

    def session(*args, **kwargs):
        return kwargs["on_error"]

    monkeypatch.setattr(harness, "_prepare", prepare)
    monkeypatch.setattr(harness, "daemon_log_tail", tail)
    monkeypatch.setattr(
        "verifiers.v1.harnesses.prime_agent.harness.PRIME_AGENT_ACP.session", session
    )
    callback = await harness.session(
        SimpleNamespace(model="model", sampling=SimpleNamespace(max_tokens=None)),
        trace,
        SimpleNamespace(supports_live_processes=True),
        "endpoint",
        "secret",
        {},
        SimpleNamespace(prompt="hello", system_prompt=None),
    )
    with pytest.raises(RuntimeError, match="prime-agent daemon log:\\ndaemon tail"):
        await callback(RuntimeError("ACP failed"))

    typed = SandboxError("sandbox unavailable")
    assert await callback(typed) is None


@pytest.mark.asyncio
async def test_prime_agent_cleanup_retains_state_when_socket_preflight_is_indeterminate():
    from verifiers.v1.errors import SandboxError
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    runtime = _SocketCleanupRuntime(socket_exists=False, socket_exit_code=2)
    with pytest.raises(SandboxError, match="daemon socket check"):
        await harness.cleanup(SimpleNamespace(id="indeterminate-socket"), runtime)
    assert not any(call[:2] == ["rm", "-rf"] for call in runtime.calls)


@pytest.mark.asyncio
async def test_stateful_turn_tail_failure_preserves_original_error(monkeypatch):
    from verifiers.v1.acp import ACPHarnessSession
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    original = RuntimeError("ACP failed")
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))

    async def failed_tail(*args, **kwargs):
        raise RuntimeError("sandbox is gone")

    monkeypatch.setattr(harness, "daemon_log_tail", failed_tail)
    session = SimpleNamespace(
        on_error=lambda error: harness._session_error(
            SimpleNamespace(), SimpleNamespace(id="tail-failure"), error
        )
    )
    with pytest.raises(RuntimeError) as raised:
        await ACPHarnessSession._raise_error(session, original)
    assert raised.value is original


@pytest.mark.asyncio
async def test_stateful_turn_callback_failure_preserves_typed_rollout_error():
    from verifiers.v1.acp import ACPHarnessSession
    from verifiers.v1.errors import SandboxError

    original = SandboxError("sandbox unavailable")

    async def failed_diagnostic(error):
        raise RuntimeError("diagnostic failure")

    session = SimpleNamespace(on_error=failed_diagnostic)
    with pytest.raises(SandboxError) as raised:
        await ACPHarnessSession._raise_error(session, original)
    assert raised.value is original


@pytest.mark.asyncio
async def test_stateful_turn_cancellation_skips_error_diagnostics():
    from verifiers.v1.acp import ACPHarnessSession

    original = asyncio.CancelledError()
    callback_invoked = False

    async def failed_diagnostic(error):
        nonlocal callback_invoked
        callback_invoked = True
        raise RuntimeError("diagnostic failure")

    session = SimpleNamespace(on_error=failed_diagnostic)
    with pytest.raises(asyncio.CancelledError) as raised:
        await ACPHarnessSession._raise_error(session, original)
    assert raised.value is original
    assert type(raised.value) is asyncio.CancelledError
    assert not callback_invoked
