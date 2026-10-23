"""Outages: infrastructure failures typed from evidence, never from message text, on a
live exception or on an error record."""

import aiohttp
import httpx

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError, model_error, outage_kind


def test_provider_and_transport_failures_are_outages_by_kind():
    assert (
        outage_kind(vf.ProviderError("<html>bad gateway</html>", status_code=502))
        == "io"
    )
    assert outage_kind(vf.ProviderError("no", status_code=402)) == "account"
    assert outage_kind(vf.ProviderError("slow down", status_code=429)) == "quota"
    assert outage_kind(vf.ProviderError("timeout", status_code=408)) == "io"
    assert outage_kind(vf.ProviderError("context overflow", status_code=400)) is None
    assert outage_kind(model_error("ReadTimeout", status_code=504)) == "io"
    assert outage_kind(httpx.ReadTimeout("read timed out")) == "io"
    assert outage_kind(aiohttp.ClientConnectionResetError("reset by peer")) == "io"


def test_only_sandbox_kinds_that_wait_for_something_are_outages():
    assert outage_kind(SandboxError("burst", kind="quota")) == "quota"
    assert outage_kind(SandboxError("billing", kind="account")) == "account"
    assert outage_kind(SandboxError("dropped", kind="io")) == "io"
    for kind in ("box_lost", "missing", "disk_full", "exec", "unknown"):
        assert outage_kind(SandboxError("x", kind=kind)) is None


def test_message_text_never_makes_an_outage():
    assert outage_kind(RuntimeError("401 unauthorized: invalid api key")) is None
    assert outage_kind(ValueError("upstream unavailable")) is None
    assert outage_kind(vf.TaskError("503 in the task's own output")) is None


def test_error_records_are_typed_by_type_and_status_or_kind():
    record = vf.Error(type="ProviderError", message="upstream 503", status_code=503)
    assert outage_kind(record) == "io"
    sandbox = vf.Error(type="SandboxError", message="x", kind="account")
    assert outage_kind(sandbox) == "account"
    assert outage_kind(vf.Error(type="SandboxError", message="no kind")) is None
    assert outage_kind(vf.Error(type="TaskError", message="503")) is None

    class CallFailed(Exception):
        """A caller's own exception carrying the failed run's record."""

        def __init__(self, error: vf.Error) -> None:
            super().__init__(error.message)
            self.error = error

    assert outage_kind(CallFailed(sandbox)) == "account"
    assert outage_kind(CallFailed(vf.Error(type="HarnessError", message="x"))) is None


def test_interception_and_tunnel_failures_are_io_live_or_recorded():
    """A failed host link is an outage whether it is raised or only its record is at
    hand: a record has no cause chain to read."""
    assert outage_kind(vf.TunnelError("host tunnel (port 8000) failed")) == "io"
    assert outage_kind(vf.InterceptionError("harness lost the model")) == "io"
    for name in ("TunnelError", "InterceptionError"):
        record = vf.Error(type=name, message="failed")
        assert outage_kind(record) == "io"

    class CallFailed(Exception):
        def __init__(self, error: vf.Error) -> None:
            super().__init__(error.message)
            self.error = error

    assert outage_kind(CallFailed(vf.Error(type="TunnelError", message="x"))) == "io"


def raised_from(outer: BaseException, inner: BaseException) -> BaseException:
    outer.__cause__ = inner
    return outer


def status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.example/x")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


def test_a_status_comes_from_the_error_never_its_message():
    assert outage_kind(RuntimeError("HTTP 401 calling my api")) is None
    assert outage_kind(RuntimeError("HTTP 503: unavailable")) is None
    assert outage_kind(status_error(503)) == "io"
    assert outage_kind(status_error(402)) == "account"


def test_a_tasks_own_failure_is_never_an_outage():
    """A grader's own 503 or timeout is the task's failure: holding the job is right,
    pausing or retrying the run is not."""
    for cause in (
        RuntimeError("HTTP 503 from grader"),
        TimeoutError(),
        status_error(503),
    ):
        assert outage_kind(raised_from(vf.TaskError("grader failed"), cause)) is None
    # A harness that could not install over a dead network is infrastructure.
    harness = raised_from(vf.HarnessError("install failed"), ConnectionResetError())
    assert outage_kind(harness) == "io"


def test_a_failed_tunnel_is_its_causes_outage_else_io():
    assert outage_kind(raised_from(vf.TunnelError("x"), status_error(401))) == "account"
    assert outage_kind(raised_from(vf.TunnelError("x"), status_error(429))) == "quota"
    assert outage_kind(raised_from(vf.TunnelError("x"), ValueError("?"))) == "io"


def test_a_provider_record_without_a_status_is_not_an_outage():
    assert outage_kind(vf.Error(type="ProviderError", message="?")) is None
