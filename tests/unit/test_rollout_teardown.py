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
