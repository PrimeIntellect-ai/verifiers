import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from prime_sandboxes.core import APIError
from prime_sandboxes.exceptions import SandboxNotRunningError

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig
from verifiers.v1.runtimes.backends import prime


def unavailable():
    error = APIError("RPC failed (unavailable): RESPONSE_SECRET")
    error.__cause__ = ConnectError(Code.UNAVAILABLE, "RESPONSE_SECRET")
    return error


def result(*, completed=True, output_error=None, exit_code=7):
    return SimpleNamespace(
        completed=completed,
        stdout_error=output_error,
        stderr_error=None,
        exit_code=exit_code,
        stdout="STDOUT_SECRET",
        stderr="STDERR_SECRET",
    )


class Client:
    def __init__(self, outcomes, launch_error=None):
        self.outcomes = iter(outcomes)
        self.launch_error = launch_error
        self.launches = 0
        self.reads = []
        self.job = object()

    async def start_background_job(self, *args, **kwargs):
        self.launches += 1
        if self.launch_error is not None:
            raise self.launch_error
        return self.job

    async def get_background_job(self, sandbox_id, job, **kwargs):
        self.reads.append((sandbox_id, job, kwargs))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def backend(monkeypatch, client):
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    instance = PrimeBackend(PrimeConfig(), name="trace-123")
    instance.info.id = "sandbox-123"
    instance._client = client
    return instance


def runtime():
    return SimpleNamespace(user=None, process_env=lambda env: env)


@pytest.fixture
def jitter(monkeypatch):
    bounds = []

    def sample(low, high):
        bounds.append((low, high))
        return 0

    monkeypatch.setattr(prime.random, "uniform", sample)
    return bounds


async def test_unavailable_retries_same_handle_without_relaunch(
    monkeypatch, jitter, caplog
):
    client = Client([unavailable(), unavailable(), result()])
    instance = backend(monkeypatch, client)
    caplog.set_level(logging.INFO, logger=prime.__name__)

    completed = await instance.run(runtime(), ["COMMAND_SECRET"], {})

    assert completed.exit_code == 7  # A user failure is a result, not a retry.
    assert completed.stdout == "STDOUT_SECRET"
    assert client.launches == 1
    assert (
        client.reads
        == [("sandbox-123", client.job, {"timeout": prime._OUTPUT_DEADLINE_SECONDS})]
        * 3
    )
    assert jitter == [(0.5, 1), (0.5, 1)]
    records = [
        json.loads(item.message.removeprefix("prime_operation "))
        for item in caplog.records
        if item.message.startswith("prime_operation ")
    ]
    retries = [record for record in records if record["event"] == "poll_retry"]
    assert [record["poll_retries"] for record in retries] == [1, 2]
    assert all(record["phase"] == "poll_retry_backoff" for record in retries)
    poll_end = records[-1]
    assert poll_end["poll_count"] == 3
    assert poll_end["poll_retries"] == 2
    assert poll_end["output_retries"] == 0
    assert poll_end["terminal_error_type"] is None
    for secret in (
        "COMMAND_SECRET",
        "RESPONSE_SECRET",
        "STDOUT_SECRET",
        "STDERR_SECRET",
    ):
        assert secret not in caplog.text


async def test_retry_budget_is_total_and_preserves_terminal_cause(monkeypatch, jitter):
    faults = [unavailable() for _ in range(prime._POLL_TRANSPORT_RETRIES + 1)]
    # A successful incomplete status must not reset the retry budget.
    outcomes = []
    for fault in faults:
        outcomes.extend([fault, result(completed=False)])
    client = Client(outcomes)

    with pytest.raises(SandboxError) as caught:
        await backend(monkeypatch, client).run(runtime(), ["true"], {})

    assert caught.value.__cause__ is faults[-1]
    assert client.launches == 1
    assert len(client.reads) == 2 * prime._POLL_TRANSPORT_RETRIES + 1
    assert jitter == [(0.5, 1)] * 3


async def test_launch_unavailable_is_never_retried(monkeypatch, jitter):
    fault = unavailable()
    client = Client([], launch_error=fault)

    with pytest.raises(SandboxError) as caught:
        await backend(monkeypatch, client).run(runtime(), ["true"], {})

    assert caught.value.__cause__ is fault
    assert client.launches == 1
    assert not client.reads
    assert not jitter


def negative_errors():
    implicit = APIError("not a transport cause")
    implicit.__context__ = ConnectError(Code.UNAVAILABLE, "gateway unavailable")
    terminated = SandboxNotRunningError("sandbox-123", status="TERMINATED")
    terminated.__cause__ = ConnectError(Code.UNAVAILABLE, "gateway unavailable")
    wrapped_terminated = APIError("sandbox wrapper")
    wrapped_terminated.__cause__ = terminated
    gone = APIError("RPC failed (not_found): sandbox terminated")
    gone.__cause__ = ConnectError(Code.UNAVAILABLE, "gateway unavailable")
    conflicting = ConnectError(Code.NOT_FOUND, "sandbox missing")
    conflicting.__cause__ = ConnectError(Code.UNAVAILABLE, "gateway unavailable")
    return [
        APIError("RPC failed (unavailable): Bad Gateway"),
        APIError("Background job exited without recording an exit code"),
        implicit,
        terminated,
        wrapped_terminated,
        gone,
        conflicting,
        ConnectError(Code.NOT_FOUND, "sandbox missing"),
        ConnectError(Code.INTERNAL, "bad gateway"),
        ConnectError(Code.DEADLINE_EXCEEDED, "request timed out"),
        ConnectError(Code.UNAUTHENTICATED, "authentication failed"),
        ConnectError(Code.PERMISSION_DENIED, "permission denied"),
        TimeoutError("request timed out"),
    ]


@pytest.mark.parametrize("fault", negative_errors())
async def test_non_transport_or_semantic_errors_are_not_retried(
    monkeypatch, jitter, fault
):
    client = Client([fault, result()])

    with pytest.raises(SandboxError) as caught:
        await backend(monkeypatch, client).run(runtime(), ["true"], {})

    assert caught.value.__cause__ is fault
    assert client.launches == len(client.reads) == 1
    assert not jitter


async def test_unavailable_then_terminated_stops_without_another_retry(
    monkeypatch, jitter
):
    gone = SandboxNotRunningError("sandbox-123", status="TERMINATED")
    client = Client([unavailable(), gone, result()])

    with pytest.raises(SandboxError) as caught:
        await backend(monkeypatch, client).run(runtime(), ["true"], {})

    assert caught.value.__cause__ is gone
    assert client.launches == 1
    assert len(client.reads) == 2
    assert jitter == [(0.5, 1)]


async def test_output_errors_keep_separate_existing_budget(monkeypatch, jitter):
    monkeypatch.setattr(prime, "_OUTPUT_RETRIES", 1)
    client = Client([result(output_error="missing"), result(output_error="missing")])

    with pytest.raises(SandboxError, match="output unavailable"):
        await backend(monkeypatch, client).run(runtime(), ["true"], {})

    assert client.launches == 1
    assert len(client.reads) == 2
    assert not jitter


async def test_completed_without_exit_code_is_not_retried(monkeypatch, jitter):
    client = Client([result(exit_code=None), result()])

    with pytest.raises(SandboxError, match="without an exit code"):
        await backend(monkeypatch, client).run(runtime(), ["true"], {})

    assert client.launches == len(client.reads) == 1
    assert not jitter


async def test_cancellation_during_retry_backoff_stops_reads(monkeypatch):
    backoff_started = asyncio.Event()
    client = Client([unavailable(), result()])

    def sample(low, high):
        backoff_started.set()
        return high

    monkeypatch.setattr(prime.random, "uniform", sample)
    task = asyncio.create_task(
        backend(monkeypatch, client).run(runtime(), ["true"], {})
    )
    await backoff_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert client.launches == len(client.reads) == 1


async def test_outer_deadline_cancels_an_inflight_read(monkeypatch):
    polling = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockedClient(Client):
        async def get_background_job(self, sandbox_id, job, **kwargs):
            self.reads.append((sandbox_id, job, kwargs))
            polling.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    client = BlockedClient([])
    instance = backend(monkeypatch, client)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(None) as deadline:
            task = asyncio.create_task(instance.run(runtime(), ["true"], {}))
            await polling.wait()
            deadline.reschedule(asyncio.get_running_loop().time())
            await task

    assert task.cancelled()
    assert cancelled.is_set()
    assert client.launches == len(client.reads) == 1


def test_classifier_accepts_direct_typed_error_and_stops_on_cause_cycle():
    assert prime._is_unavailable_poll_error(ConnectError(Code.UNAVAILABLE, "gateway"))
    cyclic = APIError("no transport error")
    cyclic.__cause__ = cyclic
    assert not prime._is_unavailable_poll_error(cyclic)
