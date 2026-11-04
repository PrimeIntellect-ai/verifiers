"""Prime teardown recognizes explicit sandbox-gone responses, not other errors."""

import json
import logging

import pytest
from prime_sandboxes.core import APIError
from prime_sandboxes.exceptions import SandboxNotRunningError

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig, is_sandbox_gone_error
from verifiers.v1.runtimes.backends import prime


def _stream_gone_error() -> APIError:
    return APIError(
        "process stream RPC failed (not_found): The sandbox has been terminated"
    )


def _terminated_error() -> SandboxNotRunningError:
    return SandboxNotRunningError(
        "sandbox-test",
        status="TERMINATED",
        error_type="SANDBOX_NOT_FOUND",
    )


@pytest.mark.parametrize("error", [_stream_gone_error(), _terminated_error()])
def test_sandbox_gone_error_matches_explicit_provider_responses(error):
    assert is_sandbox_gone_error(error)


def test_sandbox_gone_error_follows_wrapped_cause():
    error = SandboxError("prime exec failed")
    error.__cause__ = _terminated_error()

    assert is_sandbox_gone_error(error)


@pytest.mark.parametrize(
    "error",
    [
        APIError("HTTP 404: model not found"),
        APIError(
            "process stream RPC failed (unavailable): The sandbox has been terminated"
        ),
        RuntimeError(str(_stream_gone_error())),
        SandboxNotRunningError("sandbox-test", status="RUNNING"),
    ],
)
def test_sandbox_gone_error_rejects_unrelated_or_transient_failures(error):
    assert not is_sandbox_gone_error(error)


def test_sandbox_gone_error_does_not_use_suppressed_context():
    error = RuntimeError("real cleanup failure")
    error.__context__ = _stream_gone_error()
    error.__suppress_context__ = True

    assert not is_sandbox_gone_error(error)


@pytest.mark.parametrize(
    ("error", "expected_level"),
    [
        (_stream_gone_error(), logging.DEBUG),
        (RuntimeError("delete failed"), logging.WARNING),
    ],
)
async def test_prime_delete_handles_only_already_gone_as_idempotent(
    monkeypatch, caplog, error, expected_level
):
    class Client:
        async def delete(self, sandbox_id):
            raise error

    released = []

    async def release(client):
        released.append(client)

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_release_client", release)
    backend = PrimeBackend(PrimeConfig())
    backend.info.id = "sandbox-test"
    client = Client()
    backend._client = client
    caplog.set_level(logging.DEBUG, logger=prime.__name__)

    await backend.teardown()

    lifecycle = [
        json.loads(record.message.removeprefix("prime_lifecycle "))
        for record in caplog.records
        if record.message.startswith("prime_lifecycle ")
    ]
    assert [record["event"] for record in lifecycle] == ["delete_start", "delete_error"]
    assert all(record["initiator"] == "async_teardown" for record in lifecycle)
    assert lifecycle[-1]["error_type"] == type(error).__name__
    records = [
        record
        for record in caplog.records
        if record.name == prime.__name__
        and not record.message.startswith("prime_lifecycle ")
    ]
    assert len(records) == 1
    assert records[0].levelno == expected_level
    if expected_level == logging.DEBUG:
        assert records[0].exc_info is None
    assert backend._client is None
    assert released == [client]


async def test_delete_success_logs_once_without_error_payload(monkeypatch, caplog):
    calls = []

    class Client:
        async def delete(self, sandbox_id):
            calls.append(sandbox_id)

    async def release(client):
        pass

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_release_client", release)
    backend = PrimeBackend(PrimeConfig())
    backend.info.id = "sandbox-test"
    backend._client = Client()
    caplog.set_level(logging.INFO, logger=prime.__name__)
    await backend.teardown()
    await backend.teardown()
    events = [
        json.loads(r.message.removeprefix("prime_lifecycle ")) for r in caplog.records
    ]
    assert calls == ["sandbox-test"]
    assert [e["event"] for e in events] == ["delete_start", "delete_end"]
    assert all(e["sandbox_id"] == "sandbox-test" for e in events)
    assert all(e["error_type"] is None for e in events)


def test_atexit_delete_logs_initiator_and_only_error_type(monkeypatch, caplog):
    import prime_sandboxes
    import prime_sandboxes.core

    class Client:
        def delete(self, sandbox_id):
            raise ValueError("private-error-payload")

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime_sandboxes, "SandboxClient", lambda client: Client())
    monkeypatch.setattr(prime_sandboxes.core, "APIClient", lambda: object())
    backend = PrimeBackend(PrimeConfig())
    backend.info.id = "sandbox-test"
    caplog.set_level(logging.INFO, logger=prime.__name__)
    backend.cleanup()
    events = [
        json.loads(r.message.removeprefix("prime_lifecycle ")) for r in caplog.records
    ]
    assert [e["event"] for e in events] == ["delete_start", "delete_error"]
    assert all(e["initiator"] == "atexit_backstop" for e in events)
    assert events[-1]["error_type"] == "ValueError"
    assert "private-error-payload" not in caplog.text
