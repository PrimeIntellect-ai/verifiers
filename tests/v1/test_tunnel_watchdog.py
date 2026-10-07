"""The prime tunnel watchdog restarts frpc when the tunnel service has the tunnel
disconnected or frpc has exited, and leaves it alone otherwise."""

import asyncio
from types import SimpleNamespace

import pytest

from verifiers.v1.interception.tunnel import prime

pytestmark = pytest.mark.asyncio


class FakeClient:
    def __init__(self, statuses, running=True):
        self.statuses = list(statuses)
        self.is_running = running
        self.tunnel_id = "t-0-test"
        self._client = self

    async def get_tunnel(self, tunnel_id):
        status = self.statuses.pop(0) if self.statuses else "connected"
        if isinstance(status, Exception):
            raise status
        return None if status is None else SimpleNamespace(status=status)


async def watch(client, checks, monkeypatch):
    """Run the watchdog for about `checks` checks; the restarts it made."""
    restarts = []

    async def restart(c):
        restarts.append(c)
        c.is_running = True

    monkeypatch.setattr(prime, "CHECK_SECONDS", 0.01)
    monkeypatch.setattr(prime, "_restart", restart)
    task = asyncio.create_task(prime._watch(client, "https://t"))
    await asyncio.sleep(0.0125 * checks)
    task.cancel()
    await asyncio.wait({task})
    return restarts


async def test_connected_is_left_alone(monkeypatch):
    assert await watch(FakeClient(["connected"] * 10), 8, monkeypatch) == []


async def test_one_disconnected_check_is_not_enough(monkeypatch):
    client = FakeClient(["disconnected", "connected"] * 5)
    assert await watch(client, 8, monkeypatch) == []


async def test_disconnected_twice_restarts_once(monkeypatch):
    client = FakeClient(["disconnected", "disconnected"])
    assert await watch(client, 8, monkeypatch) == [client]


async def test_exited_frpc_restarts_without_asking(monkeypatch):
    client = FakeClient([RuntimeError("api down")] * 10, running=False)
    assert await watch(client, 4, monkeypatch) == [client]


async def test_unreachable_service_is_no_verdict(monkeypatch):
    client = FakeClient([RuntimeError("api down")] * 10)
    assert await watch(client, 8, monkeypatch) == []


async def test_gone_registration_stops_watching(monkeypatch):
    client = FakeClient([None])
    restarts = []

    async def restart(c):
        restarts.append(c)

    monkeypatch.setattr(prime, "CHECK_SECONDS", 0.01)
    monkeypatch.setattr(prime, "_restart", restart)
    await asyncio.wait_for(prime._watch(client, "https://t"), 1)
    assert restarts == []
