"""Typed sandbox errors (`SandboxError.kind`) against fake Prime SDK clients: no
sandbox is created."""

import errno
import pickle
from types import SimpleNamespace

import httpx
import prime_sandboxes
import pytest
from prime_sandboxes.core.client import (
    APIError,
    APITimeoutError,
    PaymentRequiredError,
    UnauthorizedError,
)
from prime_sandboxes.exceptions import (
    SandboxFileNotFoundError,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
)

import verifiers.v1 as vf
from verifiers.v1.errors import (
    SandboxError,
    error_kind,
    outage_kind,
    sandbox_error_kind,
)
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig
from verifiers.v1.runtimes.backends import prime
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.trace import TraceTask


def test_sandbox_error_defaults_to_unknown_and_keeps_its_message():
    error = SandboxError("prime exec failed: boom")
    assert error.kind == "unknown"
    assert str(error) == "prime exec failed: boom"
    typed = SandboxError("write 'x': refused", kind="account")
    assert typed.kind == "account"
    restored = pickle.loads(pickle.dumps(typed))
    assert (str(restored), restored.kind) == ("write 'x': refused", "account")


def raised_from(outer: Exception, inner: BaseException) -> Exception:
    outer.__cause__ = inner
    return outer


def status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example/sandbox")
    response = httpx.Response(status, json={}, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


def connect_error(code: str) -> Exception:
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    return ConnectError(getattr(Code, code), "rpc failed")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (PaymentRequiredError("Payment required. Check billing status."), "account"),
        (UnauthorizedError("API key unauthorized. Check PRIME_API_KEY."), "account"),
        (APIError("HTTP 403: forbidden for this team"), "account"),
        (APIError("HTTP 429: sandbox creation burst exceeded"), "quota"),
        (APIError("HTTP 410: gone"), "box_lost"),
        (SandboxNotRunningError("sb1", status="TERMINATED"), "box_lost"),
        (SandboxOOMError("sb1", "ERROR", "OOM_KILLED"), "box_lost"),
        # The SDK's stream RPC error for a deleted box, as seen live: no cause.
        (
            APIError(
                "process stream RPC failed (not_found): The sandbox has been terminated"
            ),
            "box_lost",
        ),
        (SandboxNotRunningError("sb1", status="PENDING"), "io"),
        # The SDK's boot-wait timeouts: no status, or RUNNING but not reachable.
        (
            SandboxNotRunningError("sb1", message="did not reach RUNNING within 355s"),
            "io",
        ),
        (
            SandboxNotRunningError(
                "sb1",
                status="RUNNING",
                error_type="GATEWAY_REACHABILITY_TIMEOUT",
                message="reached RUNNING but did not become reachable",
            ),
            "io",
        ),
        (APITimeoutError("Request timed out: read"), "io"),
        (APIError("HTTP 502: Bad Gateway"), "io"),
        (RuntimeError("failed to route request (unavailable)"), "io"),
        (httpx.ReadError("connection reset by peer"), "io"),
        (ConnectionResetError("reset"), "io"),
        (SandboxFileNotFoundError("HTTP 404: missing"), "missing"),
        (FileNotFoundError("/app/out.txt"), "missing"),
        (OSError(errno.ENOSPC, "No space left on device"), "disk_full"),
        (OSError(errno.ENOENT, "gone"), "missing"),
        (status_error(507), "disk_full"),
        (status_error(404), "missing"),
        (status_error(400), "unknown"),
        (status_error(413), "unknown"),
        (status_error(422), "unknown"),
        (status_error(500), "io"),
        (connect_error("NOT_FOUND"), "missing"),
        (connect_error("UNAVAILABLE"), "io"),
        (connect_error("RESOURCE_EXHAUSTED"), "quota"),
        (connect_error("PERMISSION_DENIED"), "account"),
        (connect_error("CANCELED"), "exec"),
        (RuntimeError("write failed: No space left on device"), "disk_full"),
        (SandboxImagePullError("sb1", "ERROR", "IMAGE_PULL_FAILED"), "exec"),
        (RuntimeError("something else"), "exec"),
    ],
)
def test_sandbox_error_kind_classifies_sdk_failures(error, kind):
    assert sandbox_error_kind(error, "exec") == kind


def test_sandbox_error_kind_reads_causes_groups_and_typed_errors():
    payment = PaymentRequiredError("Payment required. Check billing status.")
    # A wrapper's own text says nothing; its cause decides.
    assert sandbox_error_kind(raised_from(RuntimeError("create"), payment)) == "account"
    group = ExceptionGroup("outer", [ExceptionGroup("inner", [payment])])
    assert sandbox_error_kind(group) == "account"
    assert sandbox_error_kind(SandboxError("x", kind="box_lost"), "io") == "box_lost"
    assert sandbox_error_kind(RuntimeError("?")) == "unknown"
    assert sandbox_error_kind(RuntimeError("?"), "io") == "io"


def test_typed_evidence_beats_message_text():
    wrapped = raised_from(RuntimeError("payment required?"), httpx.ReadError("reset"))
    assert sandbox_error_kind(wrapped) == "io"
    missing = raised_from(RuntimeError("sandbox_not_found"), status_error(404))
    assert sandbox_error_kind(missing) == "missing"
    # Text decides only when nothing typed does.
    assert sandbox_error_kind(RuntimeError("sandbox_not_found")) == "box_lost"


def test_recorded_errors_carry_a_known_kind_only():
    trace = vf.Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
    )
    trace.record_error(SandboxError("prime exec failed: gone", kind="box_lost"))
    trace.record_error(SandboxError("prime exec failed: ?"))
    trace.record_error(ValueError("not a sandbox error"))
    assert [e.kind for e in trace.errors] == ["box_lost", None, None]
    errors = trace.to_record()["errors"]
    assert errors[0]["kind"] == "box_lost"
    # Records without a kind keep the shape they had before kinds existed.
    assert "kind" not in errors[1] and "kind" not in errors[2]
    assert error_kind(SandboxError("x")) is None


def test_episode_errors_carry_a_known_kind_only():
    from verifiers.v1.env import _as_error

    for error, kind in [
        (SandboxError("box gone", kind="box_lost"), "box_lost"),
        (SandboxError("?"), None),
        (ValueError("not a sandbox error"), None),
    ]:
        try:
            raise error
        except (SandboxError, ValueError) as e:
            assert _as_error(e).kind == kind


def test_records_written_before_kinds_existed_still_load():
    old = {"type": "SandboxError", "message": "boom", "status_code": None}
    error = vf.Error.model_validate(old)
    assert error.kind is None
    assert error.model_dump(mode="json") == {**old, "traceback": None}


class FakeSandboxClient:
    """A Prime SDK client whose named calls raise."""

    def __init__(self, **raises):
        self.raises = raises

    def __getattr__(self, name):
        if name in self.raises:

            async def fail(*args, **kwargs):
                raise self.raises[name]

            return fail
        raise AttributeError(name)

    async def aclose(self):
        pass


@pytest.fixture
def runtime(monkeypatch) -> PrimeBackend:
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    return PrimeBackend(PrimeConfig())


async def test_prime_provisioning_refused_for_billing_is_an_account_error(
    runtime, monkeypatch
):
    payment = PaymentRequiredError("Payment required. Check billing status.")
    monkeypatch.setattr(
        prime_sandboxes,
        "AsyncSandboxClient",
        lambda **_: FakeSandboxClient(create=payment),
    )
    with pytest.raises(SandboxError) as caught:
        await runtime.start()
    await runtime.teardown()
    assert caught.value.kind == "account"
    assert str(caught.value) == (
        "prime sandbox provisioning failed: Payment required. Check billing status."
    )


async def test_prime_io_and_exec_failures_are_typed(runtime):
    runtime.info.id = "sb1"
    runtime._client = FakeSandboxClient(
        upload_bytes=httpx.WriteError("connection reset"),
        download_file=SandboxFileNotFoundError("HTTP 404: no such file"),
        start_background_job=SandboxNotRunningError("sb1", status="TERMINATED"),
    )
    with pytest.raises(SandboxError) as write:
        await runtime.bind().write("out.txt", b"data")
    assert (write.value.kind, str(write.value)) == (
        "io",
        "write 'out.txt': connection reset",
    )
    with pytest.raises(SandboxError) as read:
        await runtime.bind().read("missing.txt")
    assert read.value.kind == "missing"
    with pytest.raises(SandboxError) as run:
        await runtime.bind().run(["true"], {})
    assert run.value.kind == "box_lost"


def test_disk_quota_is_a_full_disk_not_a_rate_limit():
    assert sandbox_error_kind(RuntimeError("write: Disk quota exceeded")) == "disk_full"
    assert sandbox_error_kind(RuntimeError("HTTP 4xx: quota exceeded")) == "quota"


class FailedProcess:
    """A streamed command that printed `stderr` and exited 1."""

    def __init__(self, stderr: str):
        self.stdout = self._chunks(b"")
        self.stderr = self._chunks(stderr.encode())

    @staticmethod
    async def _chunks(data: bytes):
        if data:
            yield data

    async def wait(self):
        return 1

    async def aclose(self):
        pass


@pytest.mark.parametrize(
    ("stderr", "kind"),
    [
        ("cat: /app/forbidden_words.txt: No such file or directory", "missing"),
        ("cat: /app/unauthorized.txt: No such file or directory", "missing"),
        ("cat: /app/rate_limit_quota_exceeded.txt: Permission denied", "unknown"),
        ("head: error reading '/app/out': Disk quota exceeded", "disk_full"),
    ],
)
async def test_a_failed_read_is_typed_by_its_stderr_never_by_the_path(
    runtime, stderr, kind
):
    """A path a task names may hold any marker word; only a missing file or a full
    disk may type a failed read, so no read of a user path becomes an outage."""

    async def open_process(*args, **kwargs):
        return FailedProcess(stderr)

    runtime.info.id = "sb1"
    runtime._client = SimpleNamespace(open_process=open_process)
    with pytest.raises(SandboxError) as caught:
        await runtime.bind().read("/app/x", max_bytes=100)
    assert caught.value.kind == kind
    assert outage_kind(caught.value) is None


async def test_a_users_failed_write_is_typed(runtime):
    async def upload_bytes(*args, **kwargs):
        pass

    async def run(view, argv, env):
        failed = argv[:2] == ["sh", "-c"]  # the copy into place, as the user
        return ProgramResult(int(failed), "", "No space left on device" * failed)

    runtime.info.id = "sb1"
    runtime._client = SimpleNamespace(upload_bytes=upload_bytes)
    runtime.run = run
    with pytest.raises(SandboxError) as caught:
        await runtime.bind().with_user("agent").write("out.txt", b"data")
    assert caught.value.kind == "disk_full"


def gateway_error(status: int, body: str, what: str) -> APIError:
    """The SDK's gateway failure as raised live: an `APIError` whose text quotes the
    status, raised while handling the httpx error that carries it."""
    request = httpx.Request("GET", "https://gateway.example/sb1/download")
    response = httpx.Response(status, text=body, request=request)
    error = httpx.HTTPStatusError(str(status), request=request, response=response)
    return raised_from(APIError(f"{what} failed: HTTP {status}: {body}"), error)


async def test_live_gateway_and_stream_failures_are_typed(runtime):
    """Two failures seen on a live Prime box: a download of a directory (400) is the
    request's own fault, not an outage; a streamed read of a deleted box is the box
    lost, as a download, write or run of it already is."""
    directory = gateway_error(
        400, '{"code":400,"message":"path \'/app\' is a directory"}', "Download"
    )
    terminated = APIError(
        "process stream RPC failed (not_found): The sandbox has been terminated"
    )

    async def download_file(*args, **kwargs):
        raise directory

    async def open_process(*args, **kwargs):
        raise terminated

    runtime.info.id = "sb1"
    runtime._client = SimpleNamespace(
        download_file=download_file, open_process=open_process
    )
    with pytest.raises(SandboxError) as read_dir:
        await runtime.bind().read("/app")
    assert read_dir.value.kind == "unknown"
    assert outage_kind(read_dir.value) is None
    with pytest.raises(SandboxError) as read_gone:
        await runtime.bind().read("/app/x", max_bytes=100)
    assert read_gone.value.kind == "box_lost"
    assert outage_kind(read_gone.value) is None
