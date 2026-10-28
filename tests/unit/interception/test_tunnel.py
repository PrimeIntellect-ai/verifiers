"""The prime tunnel watchdog restarts frpc when the tunnel service has the tunnel
disconnected or frpc has exited, leaves it alone otherwise, and stops watching a tunnel
whose registration is gone."""

import asyncio
from contextlib import asynccontextmanager
from typing import ClassVar

import pytest
from prime_tunnel import TunnelGoneError, TunnelStatus

from verifiers.v1.interception.tunnel import prime

GONE = (None, "terminated", "expired")


class FakeClient:
    def __init__(self, statuses, running=True, restart_error=None):
        self.statuses = list(statuses)
        self.is_running = running
        self.is_gone = False
        self.tunnel_id = "t-0-test"
        self.restart_error = restart_error
        self.restarts = 0
        self.checks = 0

    async def restart(self):
        self.restarts += 1
        if self.restart_error is not None:
            raise self.restart_error
        self.is_running = True

    async def status(self):
        self.checks += 1
        registration = self.statuses.pop(0) if self.statuses else "connected"
        if isinstance(registration, Exception):
            raise registration
        self.is_gone = self.is_gone or registration in GONE
        return TunnelStatus(
            tunnel_id=self.tunnel_id,
            running=self.is_running,
            registration=registration,
            gone=self.is_gone,
        )


async def watch(client, checks, monkeypatch):
    """Run the watchdog for about `checks` checks; how many restarts it made."""
    monkeypatch.setattr(prime, "CHECK_SECONDS", 0.01)
    task = asyncio.create_task(prime._watch(client, "https://t"))
    await asyncio.sleep(0.0125 * checks)
    task.cancel()
    await asyncio.wait({task})
    return client.restarts


async def stops(client, monkeypatch):
    """Run the watchdog until it gives the tunnel up."""
    monkeypatch.setattr(prime, "CHECK_SECONDS", 0.01)
    await asyncio.wait_for(prime._watch(client, "https://t"), 1)


async def test_connected_is_left_alone(monkeypatch):
    assert await watch(FakeClient(["connected"] * 10), 8, monkeypatch) == 0


async def test_one_disconnected_check_is_not_enough(monkeypatch):
    client = FakeClient(["disconnected", "connected"] * 5)
    assert await watch(client, 8, monkeypatch) == 0


async def test_disconnected_twice_restarts_once(monkeypatch):
    client = FakeClient(["disconnected", "disconnected"])
    assert await watch(client, 8, monkeypatch) == 1


async def test_exited_frpc_restarts_without_asking(monkeypatch):
    client = FakeClient([RuntimeError("api down")] * 10, running=False)
    assert await watch(client, 4, monkeypatch) == 1


async def test_unreachable_service_is_no_verdict(monkeypatch):
    client = FakeClient([RuntimeError("api down")] * 10)
    assert await watch(client, 8, monkeypatch) == 0


@pytest.mark.parametrize("registration", GONE)
async def test_gone_registration_stops_watching(registration, monkeypatch):
    client = FakeClient([registration])
    await stops(client, monkeypatch)
    assert client.restarts == 0


async def test_gone_reported_by_frpc_stops_watching(monkeypatch):
    client = FakeClient([])
    client.is_gone = True
    await stops(client, monkeypatch)
    assert (client.checks, client.restarts) == (0, 0)


async def test_restart_refused_as_gone_stops_watching(monkeypatch):
    client = FakeClient([], running=False, restart_error=TunnelGoneError("inactive"))
    await stops(client, monkeypatch)
    assert client.restarts == 1


async def test_failed_restart_is_tried_again(monkeypatch):
    client = FakeClient([], running=False, restart_error=RuntimeError("frpc"))
    assert await watch(client, 6, monkeypatch) >= 2


class RegistrationClient:
    def __init__(self, ready):
        self.ready = ready

    async def check_registered(self):
        return self.ready


async def test_require_registered_accepts_ready_registration():
    await prime._require_registered(RegistrationClient(True), "test")


@pytest.mark.parametrize("ready", [False, None])
async def test_require_registered_rejects_unready_registration(ready):
    with pytest.raises(prime.TunnelError, match="registration not ready"):
        await prime._require_registered(RegistrationClient(ready), "test")


async def test_gone_registration_invokes_callback(monkeypatch):
    client = FakeClient(["terminated"])
    calls = 0

    async def on_gone():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(prime, "CHECK_SECONDS", 0.01)
    await asyncio.wait_for(prime._watch(client, "https://t", on_gone=on_gone), 1)
    assert calls == 1


@asynccontextmanager
async def unlimited():
    yield


def install_expose_client(monkeypatch, client_cls):
    """Use a fake tunnel service with no rate limit or retry backoff."""
    from verifiers.v1.utils import retries

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "tunnel_limiter", unlimited)
    monkeypatch.setattr("prime_tunnel.Tunnel", client_cls)
    client_cls.instances = []
    real_retrying = retries.retrying

    def retrying_without_wait(**kwargs):
        retry = real_retrying(**kwargs)
        retry.wait = lambda _: 0
        return retry

    monkeypatch.setattr(retries, "retrying", retrying_without_wait)


class ExposeClient:
    readiness: ClassVar[list[bool]] = []
    instances: ClassVar[list["ExposeClient"]] = []

    def __init__(self, local_port):
        self.local_port = local_port
        self.index = len(type(self).instances)
        self.tunnel_id = f"tunnel-{self.index}"
        self.is_running = True
        self.is_gone = False
        self.stops = 0
        type(self).instances.append(self)

    async def start(self):
        return f"https://tunnel-{self.index}.example/"

    async def check_registered(self):
        if self.index < len(self.readiness):
            return self.readiness[self.index]
        return True

    def sync_stop(self):
        self.stops += 1
        self.is_running = False


async def test_expose_retries_unready_registration_and_closes_failed_attempt(
    monkeypatch,
):
    class FalseThenReadyClient(ExposeClient):
        readiness: ClassVar[list[bool]] = [False, True]

    install_expose_client(monkeypatch, FalseThenReadyClient)
    tunnel = prime.PrimeTunnel()

    async with tunnel.expose(1234) as url:
        assert url == "https://tunnel-1.example"
        assert tunnel.tunnel_id == "tunnel-1"
        assert [client.stops for client in FalseThenReadyClient.instances] == [1, 0]

    assert [client.stops for client in FalseThenReadyClient.instances] == [1, 1]


async def test_expose_cancellation_closes_partially_started_client(monkeypatch):
    started = asyncio.Event()

    class CancelledClient(ExposeClient):
        async def check_registered(self):
            started.set()
            await asyncio.Future()

    install_expose_client(monkeypatch, CancelledClient)
    tunnel = prime.PrimeTunnel()
    exposure = tunnel.expose(1234)
    entering = asyncio.create_task(exposure.__aenter__())
    await asyncio.wait_for(started.wait(), 1)

    entering.cancel()
    with pytest.raises(asyncio.CancelledError):
        await entering
    assert [client.stops for client in CancelledClient.instances] == [1]


async def test_watch_callback_can_close_its_own_exposure(monkeypatch):
    class GoneClient(ExposeClient):
        async def status(self):
            self.is_gone = True
            return TunnelStatus(
                tunnel_id=self.tunnel_id,
                running=self.is_running,
                registration="terminated",
                gone=True,
            )

    install_expose_client(monkeypatch, GoneClient)
    monkeypatch.setattr(prime, "CHECK_SECONDS", 0.01)
    tunnel = prime.PrimeTunnel()
    exposure = tunnel.expose(1234)
    closed = asyncio.Event()

    async def close_from_watch():
        await exposure.__aexit__(None, None, None)
        closed.set()

    tunnel.on_gone = close_from_watch
    await exposure.__aenter__()
    await asyncio.wait_for(closed.wait(), 1)
    assert [client.stops for client in GoneClient.instances] == [1]
