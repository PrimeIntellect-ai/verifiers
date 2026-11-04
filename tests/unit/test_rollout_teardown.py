"""Harness teardown distinguishes an already-gone sandbox from real failures."""

import logging
from types import SimpleNamespace

from prime_sandboxes.core import APIError
from prime_sandboxes.exceptions import SandboxNotRunningError

from verifiers.v1.errors import SandboxError
from verifiers.v1.rollout import Rollout


def _stream_gone_error() -> APIError:
    return APIError(
        "process stream RPC failed (not_found): The sandbox has been terminated"
    )


def _cleanup_gone_error() -> SandboxError:
    cause = SandboxNotRunningError(
        "sandbox-test",
        status="TERMINATED",
        error_type="SANDBOX_NOT_FOUND",
        message="sandbox is no longer running. The sandbox was terminated.",
    )
    error = SandboxError(f"prime exec failed: {cause}")
    error.__cause__ = cause
    return error


def _rollout() -> Rollout:
    rollout = object.__new__(Rollout)
    rollout.trace = SimpleNamespace(
        id="rollout-test",
        stop_condition="sandbox_error",
        last_error=RuntimeError("primary failure"),
        info={"primary": "unchanged"},
    )
    rollout._harness_cleaned = False
    rollout.runtime = None
    return rollout


def _teardown_records(caplog):
    return [
        record for record in caplog.records if record.name == "verifiers.v1.rollout"
    ]


async def test_close_ignores_only_already_gone_stream(caplog):
    rollout = _rollout()
    primary = rollout.trace.last_error

    async def close():
        raise _stream_gone_error()

    async def aclose():
        return None

    rollout._harness_session = SimpleNamespace(close=close)
    rollout._stack = SimpleNamespace(aclose=aclose)
    caplog.set_level(logging.DEBUG, logger="verifiers.v1.rollout")

    await rollout._close_harness()

    records = _teardown_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].exc_info is None
    assert "already closed" in records[0].getMessage()
    assert rollout.trace.last_error is primary
    assert rollout.trace.stop_condition == "sandbox_error"
    assert rollout.trace.info == {"primary": "unchanged"}


async def test_cleanup_ignores_wrapped_terminated_sandbox(caplog):
    rollout = _rollout()

    async def cleanup(trace, runtime):
        raise _cleanup_gone_error()

    rollout.harness = SimpleNamespace(cleanup=cleanup)
    caplog.set_level(logging.DEBUG, logger="verifiers.v1.rollout")

    await rollout._cleanup_harness(SimpleNamespace())

    records = _teardown_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].exc_info is None
    assert "already complete" in records[0].getMessage()
    assert rollout._harness_cleaned


async def test_close_still_warns_for_real_failure(caplog):
    rollout = _rollout()

    async def close():
        raise RuntimeError("close handshake failed")

    async def aclose():
        return None

    rollout._harness_session = SimpleNamespace(close=close)
    rollout._stack = SimpleNamespace(aclose=aclose)
    caplog.set_level(logging.DEBUG, logger="verifiers.v1.rollout")

    await rollout._close_harness()

    records = _teardown_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is RuntimeError


async def test_cleanup_still_warns_for_unavailable_stream(caplog):
    rollout = _rollout()

    async def cleanup(trace, runtime):
        raise APIError(
            "process stream RPC failed (unavailable): The sandbox has been terminated"
        )

    rollout.harness = SimpleNamespace(cleanup=cleanup)
    caplog.set_level(logging.DEBUG, logger="verifiers.v1.rollout")

    await rollout._cleanup_harness(SimpleNamespace())

    records = _teardown_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is APIError
    assert rollout._harness_cleaned


async def test_scoped_close_quiesces_before_metrics():
    events = []
    rollout = _rollout()

    async def finish():
        events.append("thaw")

    async def command(*argv):
        events.append(argv)

    async def close():
        events.append("close")

    async def kill_agent():
        await command("kill", "agent")

    scope = SimpleNamespace(
        finish_verification=finish, kill_agent_after_close=kill_agent
    )
    rollout.runtime = SimpleNamespace(
        backend=SimpleNamespace(cache=SimpleNamespace(process_scope=scope))
    )
    rollout._agent_runtime = None
    rollout._harness_session = SimpleNamespace(close=close)
    await rollout._close_harness_session()
    assert events == ["thaw", "close", ("kill", "agent")]


async def test_scoped_close_kills_agent_if_handshake_fails():
    import pytest

    events = []
    rollout = _rollout()

    async def finish():
        events.append("thaw")

    async def command(*argv):
        events.append(argv)

    async def close():
        raise RuntimeError("handshake failed")

    async def kill_agent():
        await command("kill", "agent")

    scope = SimpleNamespace(
        finish_verification=finish, kill_agent_after_close=kill_agent
    )
    rollout.runtime = SimpleNamespace(
        backend=SimpleNamespace(cache=SimpleNamespace(process_scope=scope))
    )
    rollout._agent_runtime = None
    rollout._harness_session = SimpleNamespace(close=close)
    with pytest.raises(RuntimeError, match="handshake failed"):
        await rollout._close_harness_session()
    assert events == ["thaw", ("kill", "agent")]


async def test_scoped_close_kills_agent_if_scorer_reset_fails():
    import pytest

    events = []
    rollout = _rollout()

    async def finish():
        raise RuntimeError("scorer reset failed")

    async def command(*argv):
        events.append(argv)

    async def close():
        events.append("close")

    async def kill_agent():
        await command("kill", "agent")

    scope = SimpleNamespace(
        finish_verification=finish, kill_agent_after_close=kill_agent
    )
    rollout.runtime = SimpleNamespace(
        backend=SimpleNamespace(cache=SimpleNamespace(process_scope=scope))
    )
    rollout._agent_runtime = None
    rollout._harness_session = SimpleNamespace(close=close)
    with pytest.raises(RuntimeError, match="scorer reset failed"):
        await rollout._close_harness_session()
    assert events == [("kill", "agent")]


async def test_scoped_close_finishes_cleanup_before_reraising_cancellation():
    import asyncio

    import pytest

    entered, release = asyncio.Event(), asyncio.Event()
    events = []
    rollout = _rollout()

    async def finish():
        events.append("thaw")

    async def kill_agent():
        events.append("kill")

    async def close():
        entered.set()
        await release.wait()
        events.append("close")

    scope = SimpleNamespace(
        finish_verification=finish, kill_agent_after_close=kill_agent
    )
    rollout.runtime = SimpleNamespace(
        backend=SimpleNamespace(cache=SimpleNamespace(process_scope=scope))
    )
    rollout._agent_runtime = None
    rollout._harness_session = SimpleNamespace(close=close)
    task = asyncio.create_task(rollout._close_harness_session())
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["thaw", "close", "kill"]
    assert rollout._scope_closed
