"""Typed sandbox errors (`SandboxError.kind`) and checkpoint deletion, against fake
Prime SDK clients: no sandbox is created."""

import asyncio
import errno
import json
import pickle

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
from verifiers.v1.errors import SandboxError, error_kind, sandbox_error_kind
from verifiers.v1.runtimes import prime
from verifiers.v1.runtimes.base import ProgramResult, Runtime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.trace import TraceTask


def test_sandbox_error_defaults_to_unknown_and_keeps_its_message():
    error = SandboxError("prime exec failed: boom")
    assert error.kind == "unknown"
    assert str(error) == "prime exec failed: boom"
    typed = SandboxError("write 'x': refused", kind="account")
    assert typed.kind == "account"
    assert str(typed) == "write 'x': refused"
    restored = pickle.loads(pickle.dumps(typed))
    assert (str(restored), restored.kind) == ("write 'x': refused", "account")


def raised_from(outer: Exception, inner: BaseException) -> Exception:
    outer.__cause__ = inner
    return outer


def status_error(status: int, body: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example/sandbox")
    response = httpx.Response(status, json=body or {}, request=request)
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
        (APITimeoutError("Request timed out: read"), "io"),
        (APIError("HTTP 502: Bad Gateway"), "io"),
        (RuntimeError("failed to route request (unavailable)"), "io"),
        (httpx.ReadError("connection reset by peer"), "io"),
        (ConnectionResetError("reset"), "io"),
        (SandboxFileNotFoundError("HTTP 404: missing"), "missing"),
        (FileNotFoundError("/app/out.txt"), "missing"),
        (SandboxNotRunningError("sb1", status="PENDING"), "io"),
        (OSError(errno.ENOSPC, "No space left on device"), "disk_full"),
        (OSError(errno.ENOENT, "gone"), "missing"),
        (status_error(507), "disk_full"),
        (status_error(404), "missing"),
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
    # a wrapper's own text says nothing; its cause decides
    assert sandbox_error_kind(raised_from(RuntimeError("create"), payment)) == "account"
    # an exception group is classified by its leaf
    group = ExceptionGroup("outer", [ExceptionGroup("inner", [payment])])
    assert sandbox_error_kind(group) == "account"
    # an already-typed SandboxError keeps its kind
    assert sandbox_error_kind(SandboxError("x", kind="box_lost"), "io") == "box_lost"
    # no sign at all: the operation's default
    assert sandbox_error_kind(RuntimeError("?")) == "unknown"
    assert sandbox_error_kind(RuntimeError("?"), "io") == "io"


def test_typed_evidence_beats_message_text():
    """The same words from two providers mean two things: a status, errno or code
    anywhere in the chain decides before any marker does."""
    wrapped = raised_from(RuntimeError("payment required?"), httpx.ReadError("reset"))
    assert sandbox_error_kind(wrapped) == "io"
    # A 404 that says "sandbox_not_found" in its text is still a status first.
    assert (
        sandbox_error_kind(
            raised_from(RuntimeError("sandbox_not_found"), status_error(404))
        )
        == "missing"
    )
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
    assert "kind" not in errors[1] and "kind" not in errors[2]
    assert errors[0]["message"] == "prime exec failed: gone"
    assert error_kind(SandboxError("x")) is None


# --- Prime: kinds at the source, checkpoint deletion -------------------------------


class FakeAPI:
    """The SDK's `AsyncAPIClient.request`, answering from a script."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    async def request(self, method, endpoint, **_):
        self.calls.append((method, endpoint))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeSandboxClient:
    def __init__(self, answers=(), **raises):
        self.client = FakeAPI(answers)
        self.raises = raises
        self.closed = False

    def __getattr__(self, name):
        if name in self.raises:

            async def fail(*args, **kwargs):
                raise self.raises[name]

            return fail
        raise AttributeError(name)

    async def aclose(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


@pytest.fixture
def runtime(monkeypatch) -> PrimeRuntime:
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    return PrimeRuntime(PrimeConfig())


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    real_sleep = prime.asyncio.sleep

    async def sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(prime.asyncio, "sleep", sleep)
    return sleeps


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


async def test_prime_io_and_exec_failures_are_typed(runtime, no_sleep):
    runtime.info.id = "sb1"
    runtime._client = FakeSandboxClient(
        upload_bytes=httpx.WriteError("connection reset"),
        execute_command=SandboxFileNotFoundError("HTTP 404: no such file"),
        start_background_job=SandboxNotRunningError("sb1", status="TERMINATED"),
    )
    with pytest.raises(SandboxError) as write:
        await runtime.write("out.txt", b"data")
    assert (write.value.kind, str(write.value)) == (
        "io",
        "write 'out.txt': connection reset",
    )
    with pytest.raises(SandboxError) as read:
        await runtime.read("missing.txt")
    assert read.value.kind == "missing"
    assert str(read.value) == "read 'missing.txt': HTTP 404: no such file"
    with pytest.raises(SandboxError) as run:
        await runtime.run(["true"], {})
    assert run.value.kind == "box_lost"
    assert str(run.value).startswith("prime exec failed: ")


async def test_prime_exec_errors_without_a_sign_are_exec(runtime):
    runtime.info.id = "sb1"
    runtime._client = FakeSandboxClient(start_background_job=RuntimeError("bad argv"))
    with pytest.raises(SandboxError) as run:
        await runtime.run(["true"], {})
    assert (run.value.kind, str(run.value)) == ("exec", "prime exec failed: bad argv")


async def test_prime_delete_checkpoint_retries_409_until_the_empty_body(
    runtime, no_sleep
):
    empty = json.JSONDecodeError("Expecting value", "", 0)
    runtime._client = client = FakeSandboxClient(
        [APIError("HTTP 409: in use"), APIError("HTTP 409: in use"), empty]
    )
    await runtime.delete_checkpoint("ck1")
    assert client.client.calls == [("DELETE", "/sandbox/checkpoints/ck1")] * 3
    assert no_sleep == [1.0, 2.0]


async def test_prime_delete_checkpoint_treats_404_and_a_body_as_success(runtime):
    runtime._client = client = FakeSandboxClient(
        [APIError("HTTP 404: no such checkpoint"), {"ok": True}]
    )
    await runtime.delete_checkpoint("gone")
    await runtime.delete_checkpoint("ck2")
    assert [endpoint for _, endpoint in client.client.calls] == [
        "/sandbox/checkpoints/gone",
        "/sandbox/checkpoints/ck2",
    ]


async def test_prime_delete_checkpoint_gives_up_on_409_after_its_deadline(
    runtime, monkeypatch, no_sleep
):
    monkeypatch.setattr(prime, "_CHECKPOINT_DELETE_DEADLINE_SECONDS", 2.5)
    runtime._client = FakeSandboxClient([APIError("HTTP 409: in use")] * 10)
    with pytest.raises(SandboxError, match="prime checkpoint delete failed: HTTP 409"):
        await runtime.delete_checkpoint("ck1")
    # 1s and 2s waits fit the 2.5s deadline; the next (4s) would not
    assert no_sleep == [1.0, 2.0]


async def test_prime_delete_checkpoint_fails_fast_on_an_account_refusal(runtime):
    payment = PaymentRequiredError("Payment required. Check billing status.")
    runtime._client = FakeSandboxClient([payment])
    with pytest.raises(SandboxError) as caught:
        await runtime.delete_checkpoint("ck1")
    assert caught.value.kind == "account"


async def test_prime_delete_checkpoint_works_without_a_started_box(
    runtime, monkeypatch
):
    made: list[FakeSandboxClient] = []

    def make(**_):
        made.append(FakeSandboxClient([{}]))
        return made[-1]

    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", make)
    assert runtime._client is None
    await runtime.delete_checkpoint("ck1")
    (client,) = made
    assert client.client.calls == [("DELETE", "/sandbox/checkpoints/ck1")]
    assert client.closed


class _Box(Runtime):
    async def start(self) -> None:
        pass

    async def run(self, argv, env) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def _read(self, path, max_bytes=None) -> bytes:
        return b""

    async def write(self, path, data) -> None:
        pass


class _CheckpointOnly(_Box):
    async def checkpoint(self) -> str:
        return "ck"


class _Checkpointing(_CheckpointOnly):
    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        pass


async def test_supports_checkpoint_needs_both_operations(runtime):
    assert runtime.supports_checkpoint
    assert not _Box().supports_checkpoint
    assert not _CheckpointOnly().supports_checkpoint
    assert _Checkpointing().supports_checkpoint
    with pytest.raises(NotImplementedError):
        await _Box().delete_checkpoint("ck")


async def test_a_wrapped_exception_keeps_its_type_on_the_record():
    from verifiers.v1.errors import EnvError, boundary, error_cause
    from verifiers.v1.trace import Error

    with pytest.raises(EnvError) as caught:
        async with boundary(EnvError, "Env.run()"):
            raise KeyError("missing")
    assert error_cause(caught.value) == "KeyError"
    assert error_cause(vf.SandboxError("gone", kind="box_lost")) is None
    try:
        raise vf.TaskError("verdict invalid") from ValueError("no JSON")
    except vf.TaskError as typed:
        assert error_cause(typed) is None
    record = Error(type="EnvError", message="m")
    assert "cause" not in record.model_dump(exclude_none=False)


class _CheckpointClient:
    """The SDK client's checkpoint calls: requested ids come back, the wait for one
    to become restorable fails or hangs as scripted."""

    def __init__(self, wait):
        self.wait = wait
        self.client = FakeAPI([{}])

    async def checkpoint(self, sandbox_id):
        return type("Requested", (), {"id": "ck1"})()

    async def wait_for_checkpoint(self, checkpoint_id, timeout_seconds):
        return await self.wait(checkpoint_id)


async def test_prime_checkpoint_that_never_becomes_restorable_is_deleted(runtime):
    async def refused(checkpoint_id):
        raise APIError("HTTP 500: checkpoint failed")

    runtime.info.id = "sb1"
    runtime._client = client = _CheckpointClient(refused)
    with pytest.raises(SandboxError, match="prime checkpoint failed"):
        await runtime.checkpoint()
    await asyncio.gather(*prime._discards)
    assert client.client.calls == [("DELETE", "/sandbox/checkpoints/ck1")]


async def test_prime_checkpoint_cancelled_while_waiting_is_deleted(runtime):
    waiting = asyncio.Event()

    async def hang(checkpoint_id):
        waiting.set()
        await asyncio.Event().wait()

    runtime.info.id = "sb1"
    runtime._client = client = _CheckpointClient(hang)
    task = asyncio.create_task(runtime.checkpoint())
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*prime._discards)
    assert client.client.calls == [("DELETE", "/sandbox/checkpoints/ck1")]


class Probed(Runtime):
    """A runtime whose `run` raises a scripted error: what `alive()` makes of it."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def start(self) -> None: ...

    async def run(self, argv, env) -> ProgramResult:
        raise self.error

    async def _read(self, path, max_bytes=None) -> bytes:
        raise NotImplementedError

    async def write(self, path, data) -> None: ...


async def test_alive_tells_a_gone_box_from_an_unreachable_platform():
    assert not await Probed(SandboxError("gone", kind="box_lost")).alive()
    assert not await Probed(SandboxError("?")).alive()
    assert not await Probed(RuntimeError("?")).alive()
    # The probe never reached the box: that says nothing about it, so it raises.
    for kind in ("io", "quota", "account"):
        with pytest.raises(SandboxError) as caught:
            await Probed(SandboxError("unreachable", kind=kind)).alive()
        assert caught.value.kind == kind


async def test_subprocess_reads_and_writes_raise_typed_errors(tmp_path):
    from verifiers.v1.runtimes import SubprocessConfig, SubprocessRuntime

    runtime = SubprocessRuntime(SubprocessConfig())
    await runtime.start()
    try:
        with pytest.raises(SandboxError) as missing:
            await runtime.read("nope.txt")
        assert missing.value.kind == "missing"
        with pytest.raises(SandboxError) as capped:
            await runtime.read("nope.txt", max_bytes=10)
        assert capped.value.kind == "missing"
        await runtime.write("dir/file.txt", b"x")
        with pytest.raises(SandboxError) as clash:
            await runtime.write("dir/file.txt/below", b"x")  # a file is not a dir
        assert clash.value.kind == "io"
    finally:
        await runtime.stop()
