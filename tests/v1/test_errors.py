"""Sandbox fault codes: typed evidence (an exception type, an HTTP status) names the fault, the
code rides the `SandboxError` onto the trace, and a missing path reads as `not_found`."""

import errno

import httpx
import pytest
from prime_sandboxes import APIError, SandboxFileNotFoundError

import verifiers.v1 as vf
from verifiers.v1.errors import sandbox_fault_code
from verifiers.v1.runtimes.prime import _fault_code
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


async def test_missing_path_reads_as_not_found(tmp_path):
    runtime = SubprocessRuntime(SubprocessConfig())
    runtime.workdir = tmp_path
    with pytest.raises(vf.SandboxError) as info:
        await runtime.read("missing.txt", max_bytes=16)
    assert info.value.code == "not_found"
