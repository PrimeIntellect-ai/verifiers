"""The prime tunnel watchdog restarts frpc when the tunnel service has the tunnel
disconnected or frpc has exited, leaves it alone otherwise, and stops watching a tunnel
whose registration is gone."""

import asyncio

import pytest
from prime_tunnel import TunnelGoneError, TunnelStatus

from verifiers.v1.interception.tunnel import prime

pytestmark = pytest.mark.asyncio

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
