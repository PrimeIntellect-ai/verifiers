"""Quarantined elastic members drain before retirement, without lifetime callbacks."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from test_server import Session, network as socket_free_network

from verifiers.v1.configs.interception import StaticInterceptionPoolConfig
from verifiers.v1.errors import InterceptionError
from verifiers.v1.interception import server as server_module
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    StaticInterceptionPool,
)
from verifiers.v1.interception.server import InterceptionServer

network = socket_free_network


@pytest.mark.parametrize("active_count", [1, 32])
async def test_quarantine_replacement_and_drain(network, monkeypatch, active_count):
    monkeypatch.setattr(server_module, "HEALTH_CHECK_INTERVAL_SECONDS", 0)
    checks, completed = asyncio.Queue(), asyncio.Queue()
    sessions = [Session(f"active-{n}") for n in range(active_count)]
    async with ElasticInterceptionPool(requires_tunnel=True) as pool:
        async with AsyncExitStack() as active:
            slots = [
                await active.enter_async_context(pool.acquire(s)) for s in sessions
            ]
            old = pool.servers[0]
            probe = old._probe_health

            async def controlled_probe(scope, url):
                if scope == "public":
                    result = await checks.get()
                    completed.put_nowait(None)
                    return result
                return await probe(scope, url)

            monkeypatch.setattr(old, "_probe_health", controlled_probe)
            # A good round resets the streak; a single bad round never retires a member.
            for index, result in enumerate([False, True, False, False]):
                checks.put_nowait(result)
                await completed.get()
                assert old.healthy is (index < 3)
            await old._health_watch
            assert not old.healthy
            assert old.load == active_count and not old.tunnel.closed
            assert all(not s.released and s.error is None for s in sessions)
            with pytest.raises(InterceptionError):
                old.register(Session("rejected"))
            async with pool.acquire(Session("replacement")) as replacement:
                assert replacement[0] != slots[0][0]
                assert len(pool.servers) == 2
                assert old in pool.servers and not old.tunnel.closed
                assert all(not s.released for s in sessions)
            assert old in pool.servers
        assert old.load == 0 and old.tunnel.closed
        assert old not in pool.servers
        assert all(s.released and s.error is None for s in sessions)
        # Retired members have no bound callbacks retained on the pool's stack.
        assert len(pool.stack._exit_callbacks) == 0
    assert all(t.closed for t in network.tunnels)
    assert all(r.closed for r in network.runners)


async def test_cancelled_startup_candidate_is_owned_and_closed(network, monkeypatch):
    entered = asyncio.Event()

    async def blocked_probe(self, scope, url):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(InterceptionServer, "_probe_health", blocked_probe)
    pool = ElasticInterceptionPool(requires_tunnel=True)
    await pool.start()
    await entered.wait()
    candidate = pool.servers[0]
    assert not candidate.tunnel.closed
    # Cancel initial validation and unwind the native server stack.
    await pool.stop()
    assert pool._warm_task.cancelled()
    assert candidate.tunnel.closed and network.runners[0].closed
    assert pool.servers == []


async def test_failed_retirement_stays_owned_until_successful_cleanup(
    network, monkeypatch
):
    async with ElasticInterceptionPool(requires_tunnel=True) as pool:
        async with pool.acquire(Session("first")):
            old = pool.servers[0]
        stop = old.stop

        async def fail():
            raise RuntimeError("fixture cleanup failed")

        monkeypatch.setattr(old, "stop", fail)
        old.healthy = False
        await pool._prune()
        assert old in pool.servers and not old.tunnel.closed
        with pytest.raises(RuntimeError, match="fixture cleanup failed"):
            await pool.stop()
        assert old in pool.servers
        monkeypatch.setattr(old, "stop", stop)
        await pool.stop()
        assert old not in pool.servers and old.tunnel.closed


async def test_static_pool_selects_only_healthy_members(network):
    config = StaticInterceptionPoolConfig(servers=[{}, {}])
    async with StaticInterceptionPool(config, requires_tunnel=True) as pool:
        first, second = pool.servers
        first.healthy = False
        async with pool.acquire(Session("healthy")) as slot:
            assert slot[0] == second.base_url
        second.healthy = False
        with pytest.raises(InterceptionError, match="no healthy"):
            async with pool.acquire(Session("none")):
                pytest.fail("an unavailable member was selected")


async def test_startup_failure_removes_candidate_after_native_cleanup(network):
    network.override = lambda url, reply: server_module.httpx.Response(404)
    pool = ElasticInterceptionPool(requires_tunnel=True)
    with pytest.raises(InterceptionError, match="startup health"):
        await pool._server()
    assert pool.servers == []
    assert network.tunnels[0].closed and network.runners[0].closed
    await pool.stop()


async def test_stop_waits_for_startup_and_rejects_new_borrows(network, monkeypatch):
    pool = ElasticInterceptionPool(requires_tunnel=True)
    await pool.start()
    async with pool.acquire(Session("warm")):
        warm = pool.servers[0]
    warm.healthy = False
    setup_entered, resume_setup, stop_started = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    setup = server_module.web.AppRunner.setup

    async def blocked_setup(self):
        setup_entered.set()
        await resume_setup.wait()
        await setup(self)

    monkeypatch.setattr(server_module.web.AppRunner, "setup", blocked_setup)
    session = Session("shutdown-race")

    async def borrow():
        with pytest.raises(InterceptionError, match="pool is stopping"):
            async with pool.acquire(session):
                pytest.fail("shutdown admitted a new rollout")

    async def stop():
        stop_started.set()
        await pool.stop()

    borrowing = asyncio.create_task(borrow())
    await setup_entered.wait()
    candidate = pool.servers[0]
    stopping = asyncio.create_task(stop())
    await stop_started.wait()
    assert pool._stopping and pool._lock.locked()
    assert not stopping.done()
    assert candidate in pool.servers
    resume_setup.set()
    await asyncio.gather(borrowing, stopping)
    assert candidate.load == 0 and session.error is None
    assert pool.servers == []
    assert all(r.closed for r in network.runners)
    assert all(t.closed for t in network.tunnels)
    with pytest.raises(InterceptionError, match="pool is stopping"):
        async with pool.acquire(Session("after-shutdown")):
            pytest.fail("a stopped pool admitted a rollout")
    assert len(network.runners) == len(network.tunnels) == 2
