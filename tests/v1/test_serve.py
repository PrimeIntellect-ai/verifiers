"""The env server's request handling and failure paths (offline: no model calls)."""

import asyncio
import contextlib
import multiprocessing as mp

import pytest

import verifiers.v1 as vf
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.pool import serve_env
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import RunRolloutRequest, RunRolloutResponse
from verifiers.v1.trace import Episode, Trace, TraceTask


async def test_served_rollouts_ride_the_env_gate(monkeypatch):
    """`--env.max-concurrent` bounds a worker's agent runs: the server builds one
    semaphore and every served rollout's `run_slot` acquires it — the same gate
    semantics the in-process eval enforces."""
    server = EnvServer.__new__(EnvServer)
    config = vf.resolve_env_config({"taskset": {"id": "echo-v1"}, "max_concurrent": 3})
    server.env = vf.load_environment(config)
    server._gate = asyncio.Semaphore(config.max_concurrent)
    server._tasks = list(server.env.taskset.load())
    server._task_iter = iter(())
    seen = {}

    async def spy(slot, ctx, semaphore=None, on_complete=None):
        seen["gate"] = semaphore
        return Episode(env="echo-v1", task=TraceTask(type="Task", data=slot.task.data))

    monkeypatch.setattr(server.env, "run_slot", spy)
    monkeypatch.setattr(EnvServer, "_context", lambda self, *a: None)
    req = RunRolloutRequest(
        task_idx=0,
        client=vf.EvalClientConfig(),
        model="m",
        sampling=vf.SamplingConfig(),
    )
    await server._run_rollout(req)
    assert seen["gate"] is server._gate


def test_server_builds_the_gate_from_the_env_config():
    """The knob rides the `[env]` block (the only config a pool worker receives), so
    a served run and its workers agree without extra plumbing."""
    config = vf.resolve_env_config({"taskset": {"id": "echo-v1"}, "max_concurrent": 2})
    assert config.max_concurrent == 2
    rebuilt = vf.resolve_env_config(config.model_dump(mode="json"))
    assert rebuilt.max_concurrent == 2  # survives the worker-spawn wire


async def test_unencodable_response_still_answers_the_client(monkeypatch):
    """A response msgpack can't encode (e.g. a scoring hook wrote a `set` into
    `trace.info`) must come back as an error reply — serialization failures ride
    the failed-request-is-data contract, so the client raises instead of hanging
    on a reply that never arrives."""
    config = vf.resolve_env_config({"taskset": {"id": "echo-v1"}})
    server = EnvServer(config, address="tcp://127.0.0.1:0")
    task = server._task(0)
    trace_task = TraceTask(type=type(task).__name__, data=task.data)
    episode = Episode(
        env="echo-v1",
        task=trace_task,
        traces=[Trace(task=trace_task, info={"scratch": {1, 2, 3}})],
    )

    async def bad_rollout(self, req):
        return RunRolloutResponse.model_construct(episode=episode)

    monkeypatch.setattr(EnvServer, "_run_rollout", bad_rollout)
    server_task = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    try:
        await asyncio.wait_for(client.wait_for_server_startup(timeout=10), timeout=15)
        with pytest.raises(RuntimeError, match="serialization failed"):
            await asyncio.wait_for(
                client.run_rollout(0, vf.EvalClientConfig(), "m", vf.SamplingConfig()),
                timeout=30,
            )
    finally:
        await client.close()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


async def test_pool_fails_the_request_when_its_worker_dies_at_env_load():
    """The broker answers `health` itself, so a worker that dies loading its env
    (here: an unresolvable taskset id) makes startup look fine — the forwarded
    request must then fail back to the client instead of hanging forever."""
    mpctx = mp.get_context("spawn")
    address_queue = mpctx.Queue()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            max_workers=2,
            address="tcp://127.0.0.1:0",
            address_queue=address_queue,
            config_data={"taskset": {"id": "no-such-taskset-v1"}},
        ),
        daemon=False,
    )
    proc.start()
    client = None
    try:
        address = await asyncio.to_thread(address_queue.get, timeout=120)
        client = EnvClient(address)
        await client.wait_for_server_startup(timeout=60)
        with pytest.raises(RuntimeError, match="died"):
            await asyncio.wait_for(client.info(), timeout=60)
    finally:
        if client is not None:
            await client.close()
        proc.terminate()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()


async def test_close_fails_pending_requests_instead_of_stranding_them():
    """`close()` resolves in-flight futures: a caller still awaiting a reply gets
    a ConnectionError, never a silent hang."""
    client = EnvClient("tcp://127.0.0.1:1")  # nothing listens; the request pends
    request = asyncio.create_task(client.info())
    await asyncio.sleep(0.2)
    assert not request.done()
    await client.close()
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(request, timeout=10)
