"""Typed sandbox errors: typed evidence (an exception type, an errno, an HTTP status, an RPC code)
names the `SandboxError` subclass, the class's `code` rides onto the trace, a base class in a retry
policy matches its subclasses, and message text never counts."""

import errno

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from prime_sandboxes import (
    APIError,
    CommandTimeoutError,
    SandboxFileNotFoundError,
    SandboxImagePullError,
    SandboxNotRunningError,
    UnauthorizedError,
)

import verifiers.v1 as vf
from verifiers.v1.errors import sandbox_error
from verifiers.v1.runtimes.prime import _error
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime
from verifiers.v1.utils.retries import _retryable


def _http(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://sandboxes")
    return httpx.HTTPStatusError(
        "http", request=request, response=httpx.Response(status, request=request)
    )


def _sdk_http(status: int) -> APIError:
    # The SDK's shape: `APIError("HTTP <status>: ...")` raised inside the httpx handler
    # without `from`, so the typed status is only on `__context__`.
    try:
        raise _http(status)
    except httpx.HTTPStatusError:
        try:
            raise APIError(f"HTTP {status}: detail")
        except APIError as e:
            return e


def _rpc(code: Code, message: str) -> APIError:
    # The SDK's shape for a VM RPC failure: `APIError(...) from ConnectError`.
    try:
        raise ConnectError(code, message)
    except ConnectError as rpc:
        try:
            raise APIError(
                f"Connect RPC failed ({rpc.code.value}): {rpc.message}"
            ) from rpc
        except APIError as e:
            return e


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (FileNotFoundError(errno.ENOENT, "missing"), vf.SandboxNotFoundError),
        (OSError(errno.ENOSPC, "full"), vf.SandboxDiskFullError),
        (TimeoutError(), vf.SandboxTimeoutError),
        (_http(503), vf.SandboxUnavailableError),
        (_http(429), vf.SandboxUnavailableError),
        (httpx.ConnectError("refused"), vf.SandboxUnavailableError),
        (_http(404), vf.SandboxNotFoundError),
        (_http(402), vf.SandboxDeniedError),
        (_http(409), vf.SandboxError),
        (RuntimeError("No such file or directory"), vf.SandboxError),
        (_sdk_http(503), vf.SandboxUnavailableError),
        (
            _rpc(
                Code.UNAVAILABLE, "The sandbox is being placed on a node; retry shortly"
            ),
            vf.SandboxUnavailableError,
        ),
    ],
)
def test_sandbox_error_reads_typed_evidence_down_the_chain(cause, expected):
    error = sandbox_error("prime exec failed", cause)
    assert type(error) is expected
    assert str(error) == f"prime exec failed: {cause}"


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (SandboxFileNotFoundError("File not found: x"), vf.SandboxNotFoundError),
        (UnauthorizedError("API key unauthorized"), vf.SandboxDeniedError),
        (CommandTimeoutError("sb", "true", 5), vf.SandboxTimeoutError),
        (
            SandboxImagePullError("sb", "ERROR", "IMAGE_PULL_FAILED"),
            vf.SandboxProvisioningError,
        ),
        (
            SandboxNotRunningError("sb", "TERMINATED", "SANDBOX_NOT_FOUND"),
            vf.SandboxNotFoundError,
        ),
        (APIError("Sandbox x is being deleted"), vf.SandboxError),
    ],
)
def test_prime_reads_the_sdk_types_first(cause, expected):
    assert type(_error("prime exec failed", cause)) is expected


def test_prime_provisioning_default_covers_a_bare_creation_failure():
    never_up = SandboxNotRunningError(
        "sb", message="Sandbox did not reach RUNNING within 60s"
    )
    error = _error(
        "prime sandbox provisioning failed",
        never_up,
        default=vf.SandboxProvisioningError,
    )
    assert type(error) is vf.SandboxProvisioningError
    # Typed evidence still wins over the default.
    denied = _error(
        "prime sandbox provisioning failed",
        _sdk_http(402),
        default=vf.SandboxProvisioningError,
    )
    assert type(denied) is vf.SandboxDeniedError


def test_error_code_comes_from_the_class_and_round_trips():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="p")),
    )
    trace.record_error(vf.SandboxNotFoundError("prime exec failed: gone"))
    assert trace.last_error is not None
    assert (trace.last_error.type, trace.last_error.code) == (
        "SandboxNotFoundError",
        "not_found",
    )
    loaded = vf.Trace.model_validate(trace.to_record())
    assert loaded.last_error is not None and loaded.last_error.code == "not_found"
    trace.record_error(vf.ProviderError("upstream 502", status_code=502))
    assert trace.last_error is not None and trace.last_error.code is None
    assert vf.Error(type="SandboxError", message="an old record").code is None


def test_retry_policy_matches_the_error_family():
    unavailable = vf.Error(type="SandboxUnavailableError", message="m")
    denied = vf.Error(type="SandboxDeniedError", message="m")
    assert _retryable(unavailable, vf.RetryConfig(include=["SandboxError"]))
    assert not _retryable(
        denied, vf.RetryConfig(include=["SandboxError"], exclude=["SandboxDeniedError"])
    )
    assert not _retryable(unavailable, vf.RetryConfig(include=["SandboxDeniedError"]))
    unknown = vf.Error(type="ValueError", message="m")
    assert _retryable(unknown, vf.RetryConfig(include=["ValueError"]))
    assert not _retryable(unknown, vf.RetryConfig(include=["Exception"]))


async def test_subprocess_missing_path_is_not_found(tmp_path):
    runtime = SubprocessRuntime(SubprocessConfig())
    runtime.workdir = tmp_path
    with pytest.raises(vf.SandboxNotFoundError):
        await runtime.read("missing.txt", max_bytes=16)
