"""`RuntimePool`: the boxes `Agent.provision(task, reuse=key)` keeps between contexts —
hit/miss on key, config, TTL and liveness (probed under the new env); a box started with
no lease's env, each lease's env its own per-exec overlay; teardown on error, a cancelled
probe, caller stop, `discard`, `max_idle` and pool close (a lease queued behind it
refused); one box per key, its gate dropped when unused. Subprocess runtimes (a directory
each), no model."""

import asyncio
import gc
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import (
    Runtime,
    RuntimePool,
    RuntimePoolConfig,
    SubprocessRuntime,
)
from verifiers.v1.runtimes.base import _LIVE, cleanup_at_exit
from verifiers.v1.runtimes.subprocess import SubprocessConfig

SUBPROCESS = SubprocessConfig()


class OtherConfig(SubprocessConfig):
    """The same fields under another model type: pydantic equality tells them apart, so
    this is the cheapest "same key, drifted resolved config" a subprocess box can have."""


class EnvTask(vf.Task[vf.TaskData]):
    def runtime_env(self) -> dict[str, str]:
        return {"TASK": str(self.data.idx)}


def pool(**overrides) -> RuntimePool:
    return RuntimePool(RuntimePoolConfig(**overrides))


async def test_lease_reuses_the_box_under_a_key() -> None:
    async with pool() as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {"A": "1"}) as first:
            assert first.env == {"A": "1"} and first.info.id
        async with runtimes.lease("k", SUBPROCESS, {"A": "2"}) as second:
            assert second is first and second.env == {"A": "2"}
        async with runtimes.lease("other", SUBPROCESS, {}) as third:
            assert third is not first
    # Closing the pool stops every idle box.
    assert first.stopped and third.stopped


async def test_lease_replaces_a_box_that_no_longer_fits(monkeypatch) -> None:
    async with pool(ttl=60) as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {}) as a:
            pass
        async with runtimes.lease("k", OtherConfig(), {}) as b:
            assert b is not a and a.stopped  # the resolved config drifted
        async with runtimes.lease("k", OtherConfig(), {}) as same:
            assert same is b
        runtimes._idle["k"].since -= 61  # idle past `ttl`
        async with runtimes.lease("k", OtherConfig(), {}) as c:
            assert c is not b and b.stopped
        monkeypatch.setattr(SubprocessRuntime, "alive", lambda self: _false())
        async with runtimes.lease("k", OtherConfig(), {}) as d:
            assert d is not c and c.stopped  # a dead box is not handed back


async def _false() -> bool:
    return False


async def test_the_probe_runs_under_the_new_lease_env() -> None:
    async with pool() as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {"PATH": ""}) as box:
            assert not await box.alive()  # `true` is unreachable under an empty PATH
        async with runtimes.lease("k", SUBPROCESS, {}) as again:
            assert again is box  # probed under the new env, not the last lease's


async def test_a_pooled_box_starts_without_a_lease_env(monkeypatch) -> None:
    """Docker/prime/modal bake `runtime.env` into the box at creation, and a pooled box
    hosts later leases: it is started with no lease's env, and each lease's env is the
    per-exec overlay for that lease alone (parked, the box carries none)."""
    started_with: list[dict[str, str]] = []
    real_start = SubprocessRuntime.start

    async def spy_start(self: SubprocessRuntime) -> None:
        started_with.append(dict(self.env))  # what a container would bake in
        await real_start(self)

    monkeypatch.setattr(SubprocessRuntime, "start", spy_start)
    probe = ["sh", "-c", 'echo "${VF_LEASE_A-unset}" "${VF_LEASE_B-unset}"']
    async with pool() as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {"VF_LEASE_A": "a"}) as first:
            assert (await first.run(probe, {})).stdout.split() == ["a", "unset"]
        assert first.env == {}
        async with runtimes.lease("k", SUBPROCESS, {"VF_LEASE_B": "b"}) as second:
            assert second is first
            assert (await second.run(probe, {})).stdout.split() == ["unset", "b"]
    assert started_with == [{}]  # one box, started with nothing lease-specific


async def test_cancelling_a_lease_mid_probe_stops_the_popped_box(monkeypatch) -> None:
    probing = asyncio.Event()

    async def _slow_alive(self) -> bool:
        probing.set()
        await asyncio.sleep(60)
        return True

    async with pool() as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {}) as box:
            pass
        monkeypatch.setattr(SubprocessRuntime, "alive", _slow_alive)

        async def take() -> None:
            async with runtimes.lease("k", SUBPROCESS, {}):
                pass

        task = asyncio.create_task(take())
        await probing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Popped before the probe: the cancellation must stop it, not orphan it.
        assert box.stopped and not runtimes._idle and not runtimes._locks


async def test_sweeper_stops_an_expired_idle_box() -> None:
    async with pool(ttl=0.05) as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {}) as box:
            pass
        for _ in range(100):  # the sweeper wakes every `ttl / 4`
            if box.stopped:
                break
            await asyncio.sleep(0.01)
        assert box.stopped and not runtimes._idle


async def test_lease_tears_down_instead_of_keeping() -> None:
    async with pool() as runtimes:
        with pytest.raises(RuntimeError):
            async with runtimes.lease("k", SUBPROCESS, {}) as failed:
                raise RuntimeError("the rollout blew up")
        assert failed.stopped
        async with runtimes.lease("k", SUBPROCESS, {}) as released:
            await released.stop()  # the caller says "do not keep it"
        async with runtimes.lease("k", SUBPROCESS, {}) as fresh:
            assert fresh is not released
        await runtimes.discard("k")
        assert fresh.stopped
        await runtimes.discard("k")  # idempotent: nothing under the key
        async with runtimes.lease("k", SUBPROCESS, {}) as last:
            pass
    assert last.stopped  # closing the pool
    with pytest.raises(RuntimeError, match="runtime pool is closed"):
        async with runtimes.lease("k", SUBPROCESS, {}):
            pass  # a lease after the pool closed is refused, never started


async def test_idle_boxes_reach_the_atexit_backstop() -> None:
    async with pool() as runtimes:
        async with runtimes.lease("k", SUBPROCESS, {}) as box:
            workdir = Path(box.info.id)
        del box
        gc.collect()
        # The pool's reference keeps the idle box registered for a hard exit.
        assert any(r.info.id == str(workdir) for r in _LIVE)
        cleanup_at_exit()
        assert not workdir.exists()


async def test_max_idle_stops_the_oldest_box() -> None:
    async with pool(max_idle=1) as runtimes:
        async with runtimes.lease("a", SUBPROCESS, {}) as a:
            pass
        async with runtimes.lease("b", SUBPROCESS, {}) as b:
            pass
        assert a.stopped and not b.stopped


async def test_one_box_per_key_serialises_leases() -> None:
    order: list[str] = []
    async with pool() as runtimes:

        async def hold(name: str, gate: asyncio.Event | None) -> None:
            async with runtimes.lease("k", SUBPROCESS, {}):
                order.append(f"{name}:in")
                if gate is not None:
                    await gate.wait()
                order.append(f"{name}:out")

        gate = asyncio.Event()
        first = asyncio.create_task(hold("first", gate))
        await asyncio.sleep(0)
        second = asyncio.create_task(hold("second", None))
        await asyncio.sleep(0.05)
        assert order == ["first:in"]  # the second lease waits for the first to end
        assert runtimes._locks["k"].users == 2  # one holding, one waiting
        gate.set()
        await asyncio.gather(first, second)
        assert not runtimes._locks  # nothing holds or waits on the key: gate dropped
    assert order == ["first:in", "first:out", "second:in", "second:out"]


async def test_stop_refuses_a_lease_queued_on_a_key(monkeypatch) -> None:
    started: list[SubprocessRuntime] = []
    real_start = SubprocessRuntime.start

    async def counting_start(self: SubprocessRuntime) -> None:
        started.append(self)
        await real_start(self)

    runtimes = pool()
    async with runtimes:
        held, release = asyncio.Event(), asyncio.Event()
        boxes: list[Runtime] = []

        async def hold() -> None:
            async with runtimes.lease("k", SUBPROCESS, {}) as box:
                boxes.append(box)
                held.set()
                await release.wait()

        async def queued() -> None:
            async with runtimes.lease("k", SUBPROCESS, {}):
                pass

        holder = asyncio.create_task(hold())
        await held.wait()
        waiter = asyncio.create_task(queued())
        await asyncio.sleep(0)
        assert runtimes._locks["k"].users == 2  # the second lease waits on the gate
        monkeypatch.setattr(SubprocessRuntime, "start", counting_start)
        await runtimes.stop()  # closes the pool while the waiter is still queued
        release.set()
        await holder
        with pytest.raises(RuntimeError, match="runtime pool is closed"):
            await waiter
    assert boxes[0].stopped  # the live lease's box stops on release, never parks
    assert started == []  # the queued lease never started a box
    assert not runtimes._locks


def _agent(runtimes: RuntimePool | None) -> vf.Agent:
    config = vf.AgentConfig(
        model="m", harness=vf.HarnessConfig(id="null"), runtime=SUBPROCESS
    )
    return vf.Agent(config, runtimes=runtimes)


async def test_provision_reuse_rides_the_agent_pool() -> None:
    async with pool() as runtimes:
        agent = _agent(runtimes)
        task = EnvTask(vf.TaskData(idx=1, prompt="hi"))
        async with agent.provision(task, reuse="k") as first:
            assert first.env == {"TASK": "1"}
        task = EnvTask(vf.TaskData(idx=2, prompt="hi"))
        async with agent.provision(task, reuse="k") as second:
            assert second is first and second.env == {"TASK": "2"}
        async with agent.provision(task) as owned:  # reuse=None: as before
            assert owned is not first
        assert owned.stopped and not first.stopped
    assert first.stopped
    plain = _agent(None)  # no pool: `reuse` provisions and tears down as before
    async with plain.provision(task, reuse="k") as box:
        pass
    assert box.stopped
    async with plain.provision(task, reuse="k") as again:
        assert again is not box
