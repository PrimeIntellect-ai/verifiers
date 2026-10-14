"""Riding out provider outages on Prime (`PrimeConfig.outage_budget_s`,
`create_timeout_s`, `creates_backlog_s`) and telling a lost placement from an outage,
against fake SDK clients: no sandbox is created."""

import asyncio
import uuid
from types import SimpleNamespace

import httpx
import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import prime
from verifiers.v1.runtimes.limiters import CreationBacklogFull, CreationLimiter
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime


def status_error(status: int, body: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.example/sandbox/sb1")
    response = httpx.Response(status, json=body or {}, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def make(monkeypatch, **config) -> PrimeRuntime:
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    runtime = PrimeRuntime(PrimeConfig(**config))
    runtime.info.id = "sb1"
    return runtime


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Every sleep's delay, none of them waited: the hold's clock advances by it."""
    waited: list[float] = []
    real = asyncio.sleep

    async def sleep(delay, *args):
        waited.append(delay)
        await real(0)

    monkeypatch.setattr(prime.asyncio, "sleep", sleep)
    monkeypatch.setattr(prime, "_clock", lambda: sum(waited))
    return waited


class Flaky:
    """An idempotent SDK call failing `errors` in order, then answering."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


async def test_an_outage_is_held_within_the_budget(monkeypatch, sleeps):
    runtime = make(monkeypatch, outage_budget_s=600)
    call = Flaky(*(httpx.ConnectError("no route") for _ in range(12)))
    assert await runtime._held("read", call) == "ok"
    assert call.calls == 13
    # After the quick tries, the hold waits 1 s doubling to 30 s.
    held = [delay for delay in sleeps if delay in (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)]
    assert held[-3:] == [1.0, 2.0, 4.0]


async def test_without_a_budget_an_outage_fails_after_the_quick_tries(
    monkeypatch, sleeps
):
    runtime = make(monkeypatch)
    call = Flaky(*(httpx.ConnectError("no route") for _ in range(12)))
    with pytest.raises(httpx.ConnectError):
        await runtime._held("read", call)
    assert call.calls == prime._API_ATTEMPTS


async def test_a_rate_limit_is_held_but_a_refusal_or_full_disk_is_not(
    monkeypatch, sleeps
):
    runtime = make(monkeypatch, outage_budget_s=600)
    quota = Flaky(status_error(429), status_error(429))
    assert await runtime._held("create", quota, attempts=1) == "ok"
    for final in (status_error(401), status_error(507), status_error(413)):
        call = Flaky(final)
        with pytest.raises(httpx.HTTPStatusError):
            await runtime._held("write", call)
        assert call.calls == 1


async def test_the_hold_ends_with_its_budget(monkeypatch, sleeps):
    runtime = make(monkeypatch, outage_budget_s=5)
    call = Flaky(*(status_error(503) for _ in range(50)))
    with pytest.raises(httpx.HTTPStatusError):
        await runtime._held("poll", call, attempts=1)
    assert sum(d for d in sleeps if d >= 1.0) <= 5


async def test_a_placed_box_whose_placement_is_lost_is_gone_at_once(
    monkeypatch, sleeps
):
    lost = {"error": prime.NOT_PLACED, "message": "no placement"}
    runtime = make(monkeypatch, outage_budget_s=600)
    # While the box comes up, no placement is an outage like any other.
    coming_up = Flaky(status_error(503, lost))
    assert await runtime._held("boot", coming_up) == "ok"
    runtime.placed = True
    call = Flaky(*(status_error(503, lost) for _ in range(5)))
    with pytest.raises(SandboxError) as caught:
        await runtime._held("poll", call)
    assert caught.value.kind == "box_lost"
    assert call.calls == 1
    assert prime.NOT_PLACED in str(caught.value)
    # Another 503 on a placed box is still an outage.
    assert await runtime._held("poll", Flaky(status_error(503))) == "ok"


class Client:
    """The SDK client calls `PrimeRuntime.start` makes."""

    def __init__(self, create=None, boot=None) -> None:
        self.create_calls = 0
        self.create_keys: list[str | None] = []
        self._create = create
        self._boot = boot
        self.deleted: list[str] = []

    async def create(self, request):
        self.create_calls += 1
        self.create_keys.append(request.idempotency_key)
        if self._create is not None:
            await self._create()
        return SimpleNamespace(id="sb1", pending_image_build_id=None)

    async def wait_for_creation(self, sandbox_id, max_attempts):
        if self._boot is not None:
            await self._boot()

    async def execute_command(self, sandbox_id, command, **_):
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def delete(self, sandbox_id):
        self.deleted.append(sandbox_id)

    async def aclose(self):
        pass


def install(monkeypatch, client: Client) -> None:
    import prime_sandboxes

    monkeypatch.setattr(
        prime_sandboxes, "AsyncSandboxClient", lambda **_: client, raising=True
    )
    monkeypatch.setattr(prime, "_shared_clients", {})


async def test_start_marks_the_box_placed(monkeypatch):
    client = Client()
    install(monkeypatch, client)
    runtime = make(monkeypatch)
    await runtime.start()
    assert runtime.placed
    await runtime.teardown()
    assert client.deleted == ["sb1"]


async def test_a_create_held_through_an_outage(monkeypatch, sleeps):
    outage = Flaky(httpx.ConnectError("api down"), status_error(503))
    client = Client(create=outage)
    install(monkeypatch, client)
    runtime = make(monkeypatch, outage_budget_s=300)
    await runtime.start()
    assert client.create_calls == 3 and runtime.placed
    await runtime.teardown()


async def test_a_create_whose_reply_was_lost_is_resent_with_its_idempotency_key(
    monkeypatch, sleeps
):
    """A create that timed out after the platform made the box is sent again with the
    same idempotency key, so the platform hands back that box instead of a second."""
    boxes: dict[str, str] = {}

    class Platform(Client):
        async def create(self, request):
            # Without a key of the caller's, the SDK mints a fresh one per send.
            key = request.idempotency_key or uuid.uuid4().hex
            self.create_keys.append(key)
            box = boxes.setdefault(key, f"sb{len(boxes) + 1}")
            if len(self.create_keys) == 1:
                raise httpx.ReadTimeout("reply lost after the box was made")
            return SimpleNamespace(id=box, pending_image_build_id=None)

    client = Platform()
    install(monkeypatch, client)
    runtime = make(monkeypatch, outage_budget_s=300)
    runtime.info.id = None
    await runtime.start()
    assert len(client.create_keys) == 2
    assert boxes == {client.create_keys[0]: "sb1"}, "one box, never a second"
    await runtime.teardown()
    assert client.deleted == ["sb1"]


async def test_a_boot_wait_that_ran_out_is_held_not_a_lost_box(monkeypatch, sleeps):
    """The SDK's boot-wait timeouts (no status, or RUNNING but unreachable through the
    gateway) are an outage the boot hold rides out, not a lost box."""
    from prime_sandboxes.exceptions import SandboxNotRunningError

    boot = Flaky(
        SandboxNotRunningError("sb1", message="did not reach RUNNING within 355s"),
        SandboxNotRunningError(
            "sb1", status="RUNNING", error_type="GATEWAY_REACHABILITY_TIMEOUT"
        ),
    )
    client = Client(boot=boot)
    install(monkeypatch, client)
    runtime = make(monkeypatch, outage_budget_s=300)
    await runtime.start()
    assert boot.calls == 3 and runtime.placed
    await runtime.teardown()


async def test_provisioning_past_its_bound_fails_and_the_late_box_is_deleted(
    monkeypatch,
):
    async def never():
        await asyncio.Event().wait()

    client = Client(boot=never)
    install(monkeypatch, client)
    runtime = make(monkeypatch, create_timeout_s=0.05)
    with pytest.raises(SandboxError) as caught:
        await runtime.start()
    assert caught.value.kind == "io"
    assert "no box within 0.05s" in str(caught.value)
    assert not runtime.placed
    await runtime.stop()
    assert client.deleted == ["sb1"]


async def test_a_full_creation_backlog_is_a_quota_refusal(monkeypatch, tmp_path):
    monkeypatch.setattr("verifiers.v1.runtimes.limiters.LIMITER_DIR", tmp_path)
    limiter = CreationLimiter("prime-sandbox", "run", per_sec=1 / 600, max_backlog_s=60)
    async with limiter:
        pass  # the first slot is now
    with pytest.raises(CreationBacklogFull):
        async with limiter:
            pass
    # The refused create took no slot: a later one waits no longer for it.
    assert tmp_path.joinpath("prime-sandbox-run.bucket").read_text()

    client = Client()
    install(monkeypatch, client)
    runtime = make(monkeypatch, creates_per_min=1, creates_backlog_s=1)
    monkeypatch.setattr(
        prime,
        "creation_limiter",
        lambda per_sec, name, scope, backlog: CreationLimiter(
            "prime-sandbox", "run", per_sec, backlog
        ),
    )
    with pytest.raises(SandboxError) as caught:
        await runtime.start()
    assert caught.value.kind == "quota"
    assert client.create_calls == 0
    await runtime.stop()


async def test_a_command_on_a_box_whose_placement_is_lost_is_box_lost(monkeypatch):
    lost = {"error": prime.NOT_PLACED, "message": "no placement"}

    class Gone(Client):
        async def start_background_job(self, *args, **kwargs):
            raise status_error(503, lost)

    install(monkeypatch, Gone())
    runtime = make(monkeypatch)
    await runtime.start()
    with pytest.raises(SandboxError) as caught:
        await runtime.run(["true"], {})
    assert caught.value.kind == "box_lost"
    await runtime.teardown()
