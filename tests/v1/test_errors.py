"""Sandbox fault codes: typed evidence (an exception type, an HTTP status) names the fault, the
code rides the `SandboxError` onto the trace, a missing path reads as `not_found`, and a
cancelled task's own cancellation is never a sandbox fault."""

import asyncio
import contextlib
import errno
from collections.abc import Callable

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from prime_sandboxes import APIError, AsyncSandboxProcess, SandboxFileNotFoundError
from prime_sandboxes import process as sdk_process
from prime_sandboxes._proto.command_session import command_session_pb2

import verifiers.v1 as vf
from verifiers.v1.errors import sandbox_fault_code
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime, _fault_code
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://sandboxes")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("http", request=request, response=response)


def _sdk_http_error(status: int) -> APIError:
    # The SDK's shape: `APIError("HTTP <status>: ...")` raised inside the httpx handler
    # without `from`, so the typed status is only on `__context__`.
    try:
        raise _http_error(status)
    except httpx.HTTPStatusError:
        try:
            raise APIError(f"HTTP {status}: detail")
        except APIError as e:
            return e


def test_sandbox_error_code_survives_to_trace():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="p")),
    )
    tr.record_error(vf.SandboxError("prime exec failed: gone", code="not_found"))
    assert tr.last_error is not None and tr.last_error.code == "not_found"
    rt = vf.Trace.model_validate(tr.to_record())
    assert rt.last_error is not None
    assert (rt.last_error.type, rt.last_error.code) == ("SandboxError", "not_found")
    # An untyped fault, and a non-sandbox error, carry no code.
    tr.record_error(vf.SandboxError("read 'x': exit 1"))
    assert tr.last_error is not None and tr.last_error.code is None
    tr.record_error(vf.ProviderError("upstream 502", status_code=502))
    assert tr.last_error is not None and tr.last_error.code is None


def test_fault_codes_come_from_typed_evidence_only():
    assert sandbox_fault_code(FileNotFoundError(errno.ENOENT, "missing")) == "not_found"
    assert sandbox_fault_code(OSError(errno.ENOSPC, "full")) == "disk_full"
    assert sandbox_fault_code(TimeoutError()) == "timeout"
    assert sandbox_fault_code(_http_error(503)) == "unavailable"
    assert sandbox_fault_code(RuntimeError("No such file or directory")) is None
    # prime: SDK types, and the status hidden in a bare `APIError`'s text, read typed.
    assert _fault_code(SandboxFileNotFoundError("File not found: x")) == "not_found"
    assert _fault_code(_sdk_http_error(404)) == "not_found"
    assert _fault_code(_sdk_http_error(402)) == "denied"
    assert _fault_code(_sdk_http_error(503)) == "unavailable"
    assert _fault_code(_sdk_http_error(409)) is None
    assert _fault_code(APIError("Sandbox x is being deleted")) is None
    # A cancel is not a fault code: the runtime re-raises it (below), never names it.
    assert _fault_code(_rpc_cancelled(asyncio.CancelledError())) is None


async def test_missing_path_reads_as_not_found(tmp_path):
    runtime = SubprocessRuntime(SubprocessConfig())
    runtime.workdir = tmp_path
    with pytest.raises(vf.SandboxError) as info:
        await runtime.read("missing.txt", max_bytes=16)
    assert info.value.code == "not_found"


def _rpc_cancelled(cancel: asyncio.CancelledError) -> APIError:
    # What a cancelled task gets back from a VM RPC: connectrpc turns the `CancelledError`
    # into `ConnectError(CANCELED)`, the SDK re-wraps that as `APIError`.
    try:
        raise ConnectError(Code.CANCELED, "Request was cancelled") from cancel
    except ConnectError as rpc:
        try:
            raise APIError(
                f"Connect RPC failed ({rpc.code.value}): {rpc.message}"
            ) from rpc
        except APIError as e:
            return e


def _bare_swallow(cancel: asyncio.CancelledError) -> APIError:
    # A swallow that keeps no typed trace of the cancel at all: only `Task.cancelling()` says.
    return APIError("Request failed")


class _FakeSandboxClient:
    """An SDK client whose RPC parks until the task is cancelled, then reports the cancel as an
    ordinary error (`swallow`), or fails outright with `error`."""

    def __init__(
        self,
        swallow: Callable[[asyncio.CancelledError], BaseException] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.swallow = swallow
        self.error = error
        self.entered = asyncio.Event()

    async def start_background_job(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancel:
            assert self.swallow is not None
            raise self.swallow(cancel) from None


def _prime_runtime(monkeypatch, client) -> PrimeRuntime:
    monkeypatch.setenv("PRIME_API_KEY", "test")
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = "sandbox"
    runtime._client = client
    return runtime


@pytest.mark.parametrize("swallow", [_rpc_cancelled, _bare_swallow])
async def test_cancelled_rpc_is_the_cancellation_not_a_sandbox_fault(
    monkeypatch, swallow
):
    client = _FakeSandboxClient(swallow=swallow)
    runtime = _prime_runtime(monkeypatch, client)
    task = asyncio.create_task(runtime.run(["true"], {}))
    await client.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_sdk_fault_without_a_cancel_stays_a_sandbox_fault(monkeypatch):
    runtime = _prime_runtime(
        monkeypatch, _FakeSandboxClient(error=_sdk_http_error(503))
    )
    with pytest.raises(vf.SandboxError) as info:
        await runtime.run(["true"], {})
    assert info.value.code == "unavailable"


def _event(**fields) -> command_session_pb2.StartResponse:
    response = command_session_pb2.StartResponse()
    for name, value in fields.items():
        setattr(
            getattr(response.event, name),
            "pid" if name == "start" else "exit_code",
            value,
        )
    return response


class _ClosableClient:
    async def close(self) -> None:
        pass


async def _live_process(
    stream, reconnect, reattached: list[bool]
) -> AsyncSandboxProcess:
    async def counted(*args):
        reattached.append(True)
        async for response in reconnect():
            yield response

    return await AsyncSandboxProcess._create(
        _ClosableClient(), stream(), None, None, reconnect=counted
    )


async def test_stream_pump_does_not_reattach_on_its_own_cancel(monkeypatch):
    monkeypatch.setattr(sdk_process, "_STREAM_RECONNECT_BACKOFF_SECONDS", 0)
    reattached: list[bool] = []

    async def stream():
        yield _event(start=7)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancel:
            raise ConnectError(Code.CANCELED, "Request was cancelled") from cancel

    async def reconnect():
        yield _event(start=7)
        await asyncio.Event().wait()

    process = await _live_process(stream, reconnect, reattached)
    process._pump_task.cancel()
    try:
        # A re-attaching pump streams on forever; a terminal cancel ends it at once.
        await asyncio.wait_for(asyncio.shield(process._pump_task), 2)
    finally:
        process._pump_task.cancel()
        with contextlib.suppress(BaseException):
            await process._pump_task
    assert reattached == []
    with pytest.raises(APIError, match="canceled"):
        await process.wait()


async def test_stream_pump_still_reattaches_on_a_dropped_link(monkeypatch):
    monkeypatch.setattr(sdk_process, "_STREAM_RECONNECT_BACKOFF_SECONDS", 0)
    reattached: list[bool] = []

    async def stream():
        yield _event(start=7)
        raise ConnectError(Code.UNAVAILABLE, "link reset")

    async def reconnect():
        yield _event(end=0)

    process = await _live_process(stream, reconnect, reattached)
    assert await process.wait() == 0
    assert reattached == [True]
