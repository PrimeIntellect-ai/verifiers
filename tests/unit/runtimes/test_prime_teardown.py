"""Prime teardown recognizes explicit sandbox-gone responses, not other errors."""

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

    records = [record for record in caplog.records if record.name == prime.__name__]
    assert len(records) == 1
    assert records[0].levelno == expected_level
    if expected_level == logging.DEBUG:
        assert records[0].exc_info is None
    assert backend._client is None
    assert released == [client]
