"""`PrimeRuntime`'s holds and bounds over a scripted SDK client: an idempotent call held
through a platform outage for `outage_budget_s` (never a bare fault, never a cancelled
one), the provisioning bounded by `create_timeout_s` (the box it made deleted), the
creation pace's backlog bound `creates_backlog_s`; no sandbox."""

import asyncio
import time
from types import SimpleNamespace

import prime_sandboxes
import pytest
from prime_sandboxes import APIError

import verifiers.v1 as vf
from verifiers.v1.runtimes import limiters, prime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "k")
    monkeypatch.setattr(prime, "HOLD_START_S", 0.001)
    monkeypatch.setattr(prime, "HOLD_MAX_S", 0.002)


def _unavailable() -> APIError:
    """The SDK's shape for a VM RPC failure: `APIError(...)` from the RPC error."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    try:
        raise ConnectError(Code.UNAVAILABLE, "routing catalog unavailable")
    except ConnectError as rpc:
        try:
            raise APIError("Connect RPC failed (unavailable): routing") from rpc
        except APIError as e:
            return e


async def test_an_idempotent_call_is_held_through_an_outage_within_the_budget(fast):
    runtime = PrimeRuntime(PrimeConfig(outage_budget_s=5))
    runtime.info.id = "box-1"
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise _unavailable()
        return "ok"

    assert await runtime._held("read", flaky) == "ok" and calls == 4
    spent = PrimeRuntime(PrimeConfig(outage_budget_s=0))
    calls = 0
    with pytest.raises(APIError):
        await spent._held("read", flaky)
    assert calls == 1  # no budget: the SDK's own retries alone
    bare = 0

    async def refused():
        nonlocal bare
        bare += 1
        raise APIError("Sandbox x is being deleted")

    with pytest.raises(APIError):
        await runtime._held("read", refused)
    assert bare == 1  # a bare fault is never held


async def test_a_cancelled_task_is_not_held(fast):
    runtime = PrimeRuntime(PrimeConfig(outage_budget_s=60))
    runtime.info.id = "box-1"
    started = asyncio.Event()

    async def swallowing():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise _unavailable() from None  # connectrpc answers a cancel this way

    task = asyncio.create_task(runtime._held("read", swallowing))
    await started.wait()
    task.cancel()
    with pytest.raises((APIError, asyncio.CancelledError)):
        await task


class _Client:
    """The SDK client `start` drives: a create that answers after `create_after`
    seconds, a boot wait that takes `boot` seconds, deletes recorded."""

    def __init__(self, create_after: float = 0, boot: float = 0):
        self.create_after, self.boot, self.deleted = create_after, boot, []
        self.commands: list[str] = []

    async def create(self, request):
        await asyncio.sleep(self.create_after)
        return SimpleNamespace(id="sb-1", pending_image_build_id=None)

    async def wait_for_creation(self, sandbox_id, max_attempts):
        await asyncio.sleep(self.boot)

    async def execute_command(self, sandbox_id, command, **kw):
        self.commands.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def delete(self, sandbox_id):
        self.deleted.append(sandbox_id)

    async def aclose(self):
        pass


def _with_client(monkeypatch, config: PrimeConfig, client: _Client) -> PrimeRuntime:
    monkeypatch.setattr(prime, "_shared_clients", {})
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", lambda: client)
    return PrimeRuntime(config)


async def test_the_provisioning_is_bounded_by_create_timeout_s_and_a_late_box_is_deleted(
    fast, monkeypatch
):
    client = _Client(boot=0.5)
    runtime = _with_client(
        monkeypatch, PrimeConfig(create_timeout_s=0.05, workdir="/w"), client
    )
    with pytest.raises(vf.SandboxTimeoutError, match="no box within 0.05s"):
        await runtime.start()
    await runtime.stop()
    assert client.deleted == ["sb-1"]  # the id was captured: the box does not leak
    quick = _Client()
    runtime = _with_client(
        monkeypatch, PrimeConfig(create_timeout_s=5, workdir="/w"), quick
    )
    await runtime.start()
    assert runtime.info.id == "sb-1" and quick.commands == ["mkdir -p /w"]
    await runtime.stop()


async def test_the_creation_pace_backlog_is_the_configs(fast, monkeypatch, tmp_path):
    """A create waits at most `creates_backlog_s` for its slot under
    `creates_per_min`; a longer backlog is a typed timeout before any box exists."""
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    monkeypatch.setattr(limiters, "_creation_limiters", {})
    (tmp_path / "prime-sandbox.bucket").write_text(repr(time.time() + 3600))
    client = _Client()
    runtime = _with_client(
        monkeypatch,
        PrimeConfig(creates_per_min=60, creates_backlog_s=1, workdir="/w"),
        client,
    )
    with pytest.raises(vf.SandboxTimeoutError, match="backlog of .* exceeds 1s"):
        await runtime.start()
    await runtime.stop()
    assert client.deleted == []  # nothing was created
