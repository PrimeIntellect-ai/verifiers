"""Prime provisioning bounds: an outage is ridden out for `outage_budget_s`, a lost
placement fails at once, and `create_timeout_s` and `creates_backlog_s` bound a create.
Every bound is off by default."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from prime_sandboxes.exceptions import SandboxNotRunningError

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeConfig, limiters, provision_backend
from verifiers.v1.runtimes.backends import prime
from verifiers.v1.runtimes.base import _LIVE


def http_error(status: int, body: dict | None = None) -> Exception:
    """The SDK's `HTTP <status>` error, raised from the httpx error that carries the
    response."""
    request = httpx.Request("GET", "https://api.example/sandbox")
    response = httpx.Response(status, json=body or {}, request=request)
    try:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {status}: {response.text}") from e
    except RuntimeError as e:
        return e
    raise AssertionError("unreachable")


NOT_PLACED = {"error": "sandbox_not_placed"}


class FakePrimeClient:
    """Fails each call with the errors queued for it, then answers."""

    def __init__(self):
        self.fail: dict[str, list[Exception]] = {}
        self.calls: dict[str, list] = {}
        self.boot = asyncio.Event()
        self.boot.set()
        self.swallow: asyncio.Event | None = None

    async def _call(self, name, *args):
        self.calls.setdefault(name, []).append(args)
        if self.fail.get(name):
            raise self.fail[name].pop(0)

    async def create(self, request):
        await self._call("create", request.idempotency_key)
        return SimpleNamespace(id="sb1", pending_image_build_id=None)

    async def wait_for_creation(self, sandbox_id, max_attempts):
        await self.boot.wait()
        await self._call("boot")

    async def execute_command(self, sandbox_id, command):
        await self._call("workdir")

    async def download_file(self, sandbox_id, target, local):
        await self._call("download")
        if self.swallow is not None:  # like connectrpc: a cancel becomes an error
            entered, self.swallow = self.swallow, None
            entered.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise ConnectionResetError("request cancelled") from None

    async def delete(self, sandbox_id):
        await self._call("delete", sandbox_id)


@pytest.fixture
def client(monkeypatch, tmp_path):
    client = FakePrimeClient()

    async def release(_):
        pass

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_lease_client", lambda: client)
    monkeypatch.setattr(prime, "_release_client", release)
    monkeypatch.setattr(prime, "HOLD_START_S", 0.01)
    monkeypatch.setattr(prime, "HOLD_MAX_S", 0.02)
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    yield client
    # The fake boxes never existed: keep the atexit backstop from deleting them.
    for backend in list(_LIVE):
        if isinstance(backend, prime.PrimeBackend):
            _LIVE.discard(backend)


async def test_an_outage_is_ridden_out_within_its_budget(client):
    client.fail = {
        "create": [http_error(503), http_error(429)],
        "boot": [ConnectionResetError()],
    }
    async with provision_backend(PrimeConfig(outage_budget_s=5)):
        pass
    keys = [args[0] for args in client.calls["create"]]
    assert len(keys) == 3 and len(set(keys)) == 1  # one box, however often it is sent
    assert len(client.calls["boot"]) == 2

    # A refusal or a stopped box is no outage, and an outage past its budget reaches
    # the caller.
    stopped = SandboxNotRunningError("sb1", status="TERMINATED")
    for config, phase, error, kind in [
        (PrimeConfig(outage_budget_s=5), "create", http_error(404), "missing"),
        (PrimeConfig(outage_budget_s=5), "boot", stopped, "box_lost"),
        (PrimeConfig(outage_budget_s=0.02), "create", http_error(503), "io"),
    ]:
        client.fail, client.calls = {phase: [error] * 10}, {}
        with pytest.raises(SandboxError, match="provisioning failed") as failed:
            async with provision_backend(config):
                pass
        assert failed.value.kind == kind
        assert len(client.calls[phase]) == (2 if kind == "io" else 1)


async def test_without_a_budget_every_call_is_sent_once(client):
    client.fail = {"create": [http_error(503)]}
    with pytest.raises(SandboxError, match="HTTP 503"):
        async with provision_backend(PrimeConfig()):
            pass
    assert len(client.calls["create"]) == 1


async def test_a_lost_placement_is_held_only_while_the_box_comes_up(client):
    client.fail = {"workdir": [http_error(503, NOT_PLACED)]}
    async with provision_backend(PrimeConfig(outage_budget_s=5)) as backend:
        assert len(client.calls["workdir"]) == 2
        client.fail = {"download": [http_error(503, NOT_PLACED)]}
        with pytest.raises(SandboxError, match="sandbox_not_placed") as lost:
            await backend.read(SimpleNamespace(user=None), "/app/out.txt")
        assert lost.value.kind == "box_lost"
        assert len(client.calls["download"]) == 1


async def test_a_swallowed_cancellation_is_never_held_through(client):
    async with provision_backend(PrimeConfig(outage_budget_s=5)) as backend:
        client.swallow = entered = asyncio.Event()
        read = asyncio.create_task(
            backend.read(SimpleNamespace(user=None), "/app/out.txt")
        )
        await entered.wait()
        read.cancel()
        with pytest.raises((SandboxError, asyncio.CancelledError)):
            await read
        assert len(client.calls["download"]) == 1


async def test_the_teardown_delete_is_held_through_an_outage(client):
    client.fail = {"delete": [http_error(503)]}
    async with provision_backend(PrimeConfig(outage_budget_s=5)):
        pass
    assert len(client.calls["delete"]) == 2


async def test_a_create_is_bounded_in_time_and_backlog(client):
    client.boot.clear()  # the box never comes up
    with pytest.raises(SandboxError, match="no sandbox within 0.05s") as late:
        async with provision_backend(PrimeConfig(create_timeout_s=0.05)):
            pass
    assert late.value.kind == "io"
    assert client.calls["delete"] == [("sb1",)]  # the late box is deleted

    client.boot.set()
    config = PrimeConfig(creates_per_min=1, creates_backlog_s=30)
    async with provision_backend(config):
        pass  # takes the free slot; the next is a minute off
    with pytest.raises(SandboxError, match="creation backlog") as full:
        async with provision_backend(config):
            pass
    assert full.value.kind == "quota"
