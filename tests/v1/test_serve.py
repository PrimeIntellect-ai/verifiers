"""The env server's request handling and failure paths (offline: no model calls)."""

import asyncio


import verifiers.v1 as vf
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import RunRequest
from verifiers.v1.trace import EpisodeInfo, EpisodeRecord


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
        return EpisodeRecord(episode=EpisodeInfo(env="echo-v1"))

    monkeypatch.setattr(server.env, "run_slot", spy)
    monkeypatch.setattr(EnvServer, "_context", lambda self, *a: None)
    req = RunRequest(
        task_idx=0,
        client=vf.EvalClientConfig(),
        model="m",
        sampling=vf.SamplingConfig(),
    )
    await server._run(req)
    assert seen["gate"] is server._gate


def test_server_builds_the_gate_from_the_env_config():
    """The knob rides the `[env]` block (the only config a pool worker receives), so
    a served run and its workers agree without extra plumbing."""
    config = vf.resolve_env_config({"taskset": {"id": "echo-v1"}, "max_concurrent": 2})
    assert config.max_concurrent == 2
    rebuilt = vf.resolve_env_config(config.model_dump(mode="json"))
    assert rebuilt.max_concurrent == 2  # survives the worker-spawn wire
