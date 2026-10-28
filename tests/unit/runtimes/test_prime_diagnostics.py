import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig
from verifiers.v1.runtimes.backends import prime

REQUIRED_FIELDS = {
    "rollout_id",
    "operation_id",
    "phase",
    "elapsed_s",
    "sdk_wait_s",
    "poll_count",
    "output_retries",
    "last_stream_activity_age_s",
    "terminal_error_type",
}


def records(caplog):
    return [
        json.loads(record.getMessage().removeprefix("prime_operation "))
        for record in caplog.records
        if record.getMessage().startswith("prime_operation ")
    ]


def make_backend(monkeypatch, client):
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    backend = PrimeBackend(PrimeConfig(), name="trace-123")
    backend.info.id = "sandbox-123"
    backend._client = client
    return backend


def runtime():
    return SimpleNamespace(
        user=None,
        process_env=lambda env: {**env, "DERIVED_ENV_SECRET": "hidden"},
    )


def job_result(*, output_error=None):
    return SimpleNamespace(
        completed=True,
        stdout_error=output_error,
        stderr_error=None,
        exit_code=7,
        stdout="STDOUT_SECRET",
        stderr="STDERR_SECRET",
    )


async def test_background_job_diagnostics_cover_polling_and_output_retries(
    monkeypatch, caplog
):
    class Client:
        def __init__(self):
            self.results = iter(
                [job_result(output_error="OUTPUT_SECRET"), job_result()]
            )
            self.contexts = []

        async def start_background_job(self, *args, **kwargs):
            self.contexts.append(prime.current_prime_operation())
            return "job-123"

        async def get_background_job(self, *args, **kwargs):
            self.contexts.append(prime.current_prime_operation())
            return next(self.results)

    client = Client()
    backend = make_backend(monkeypatch, client)
    caplog.set_level(logging.INFO, logger=prime.__name__)

    result = await backend.run(runtime(), ["ARGV_SECRET"], {"ENV_SECRET": "hidden"})

    assert result.exit_code == 7
    assert result.stdout == "STDOUT_SECRET"
    assert result.stderr == "STDERR_SECRET"
    logged = records(caplog)
    assert len(logged) == 4  # start/end for launch and the whole polling loop
    assert {record["operation"] for record in logged} == {
        "start_background_job",
        "poll_output",
    }
    assert all(REQUIRED_FIELDS <= record.keys() for record in logged)
    assert all(record["rollout_id"] == "trace-123" for record in logged)
    assert all(record["terminal_error_type"] is None for record in logged)
    poll_end = next(
        record
        for record in logged
        if record["operation"] == "poll_output" and record["event"] == "end"
    )
    assert poll_end["poll_count"] == 2
    assert poll_end["output_retries"] == 1
    assert poll_end["phase"] == "output_read"
    assert poll_end["elapsed_s"] >= poll_end["sdk_wait_s"] >= 0
    assert [context.operation for context in client.contexts] == [
        "start_background_job",
        "poll_output",
        "poll_output",
    ]
    assert prime.current_prime_operation() is None
    for secret in (
        "ARGV_SECRET",
        "ENV_SECRET",
        "DERIVED_ENV_SECRET",
        "STDOUT_SECRET",
        "STDERR_SECRET",
        "OUTPUT_SECRET",
    ):
        assert secret not in caplog.text


class FakeStream:
    def __init__(self, chunks, contexts):
        self.chunks = iter(chunks)
        self.contexts = contexts

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.contexts.append(prime.current_prime_operation())
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeProcess:
    def __init__(self, exit_code=7):
        self.contexts = []
        self.exit_code = exit_code
        self.stdout = FakeStream([b"STDOUT_SECRET", b"_MORE"], self.contexts)
        self.stderr = FakeStream([b"STDERR_SECRET"], self.contexts)
        self.returncode = None

    async def wait(self):
        self.contexts.append(prime.current_prime_operation())
        self.returncode = self.exit_code
        return self.returncode

    async def aclose(self):
        pass


async def test_process_wait_and_stream_diagnostics_are_not_per_chunk(
    monkeypatch, caplog
):
    class Client:
        def __init__(self):
            self.process = FakeProcess()

        async def open_process(self, *args, **kwargs):
            self.process.contexts.append(prime.current_prime_operation())
            return self.process

    client = Client()
    backend = make_backend(monkeypatch, client)
    caplog.set_level(logging.INFO, logger=prime.__name__)

    process = await backend.exec(runtime(), ["ARGV_SECRET"], {"ENV_SECRET": "hidden"})
    stdout = b"".join([chunk async for chunk in process.stdout])
    stderr = b"".join([chunk async for chunk in process.stderr])

    assert stdout == b"STDOUT_SECRET_MORE"
    assert stderr == b"STDERR_SECRET"
    assert await process.wait() == 7
    logged = records(caplog)
    assert len(logged) == 8  # start/end for open, two streams, and wait
    assert {record["operation"] for record in logged} == {
        "open_process",
        "stdout_stream",
        "stderr_stream",
        "process_wait",
    }
    for operation in ("stdout_stream", "stderr_stream"):
        stream_records = [
            record for record in logged if record["operation"] == operation
        ]
        assert [record["event"] for record in stream_records] == ["start", "end"]
        assert all(
            record["last_stream_activity_age_s"] is not None
            for record in stream_records
        )
    assert {context.operation for context in client.process.contexts} == {
        "open_process",
        "stdout_stream",
        "stderr_stream",
        "process_wait",
    }
    assert prime.current_prime_operation() is None
    assert "ARGV_SECRET" not in caplog.text
    assert "ENV_SECRET" not in caplog.text
    assert "STDOUT_SECRET" not in caplog.text
    assert "STDERR_SECRET" not in caplog.text


async def test_heartbeat_and_terminal_error_type_are_metadata_only(monkeypatch, caplog):
    class Client:
        async def start_background_job(self, *args, **kwargs):
            await asyncio.sleep(0.035)
            raise TimeoutError("ERROR_PAYLOAD_SECRET")

    backend = make_backend(monkeypatch, Client())
    monkeypatch.setattr(prime, "_OPERATION_HEARTBEAT_SECONDS", 0.01)
    caplog.set_level(logging.INFO, logger=prime.__name__)

    with pytest.raises(SandboxError, match="prime background launch failed"):
        await backend.run_background(
            runtime(), ["ARGV_SECRET"], {"ENV_SECRET": "hidden"}, "LOG_SECRET"
        )

    logged = records(caplog)
    assert [record["event"] for record in logged].count("heartbeat") >= 2
    assert logged[0]["event"] == "start"
    assert all(
        record["sdk_wait_s"] >= 0.009
        for record in logged
        if record["event"] == "heartbeat"
    )
    assert logged[-1]["event"] == "end"
    assert logged[-1]["terminal_error_type"] == "TimeoutError"
    assert logged[-1]["elapsed_s"] >= logged[-1]["sdk_wait_s"] >= 0.03
    assert len({record["operation_id"] for record in logged}) == 1
    assert prime.current_prime_operation() is None
    assert "ARGV_SECRET" not in caplog.text
    assert "ENV_SECRET" not in caplog.text
    assert "LOG_SECRET" not in caplog.text
    assert "ERROR_PAYLOAD_SECRET" not in caplog.text


async def test_streaming_file_read_uses_process_diagnostics(monkeypatch, caplog):
    class Client:
        def __init__(self):
            self.process = FakeProcess(exit_code=0)

        async def open_process(self, *args, **kwargs):
            return self.process

    backend = make_backend(monkeypatch, Client())
    caplog.set_level(logging.INFO, logger=prime.__name__)

    data = await backend.read(runtime(), "READ_PATH_SECRET", max_bytes=100)

    assert data == b"STDOUT_SECRET_MORE"
    logged = records(caplog)
    assert len(logged) == 8
    assert {record["operation"] for record in logged} == {
        "open_process",
        "stdout_stream",
        "stderr_stream",
        "process_wait",
    }
    assert "READ_PATH_SECRET" not in caplog.text
    assert "STDOUT_SECRET" not in caplog.text
    assert "STDERR_SECRET" not in caplog.text


async def test_cancellation_is_recorded_without_replacing_it(monkeypatch, caplog):
    class Client:
        async def start_background_job(self, *args, **kwargs):
            await asyncio.sleep(60)

    backend = make_backend(monkeypatch, Client())
    caplog.set_level(logging.INFO, logger=prime.__name__)

    task = asyncio.create_task(
        backend.run_background(runtime(), ["sleep", "1"], {}, "log")
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    logged = records(caplog)
    assert [record["event"] for record in logged] == ["start", "end"]
    assert logged[-1]["terminal_error_type"] == "CancelledError"
    assert prime.current_prime_operation() is None
