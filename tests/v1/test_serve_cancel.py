"""Cancelling an `EnvClient.run` call aborts the rollout on the env server: the
client's cancel reaches the server over the wire, the server cancels the run task,
and the rollout's own cleanup (where a runtime tears its sandbox down) runs. This is
the path a runner relies on to stop an episode its taskset cancelled."""

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.serve import server as server_module
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.server import EnvServer

TIMEOUT_S = 10


class CountTask(vf.Task[vf.TaskData]):
    pass


class Stream(vf.Taskset[CountTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self):
        yield CountTask(vf.TaskData(id="slow"))


class HangingEnv:
    """An env whose rollout runs until cancelled, then cleans up like a runtime's
    shielded teardown."""

    def __init__(self) -> None:
        self.taskset = Stream(vf.TasksetConfig())
        self.config = SimpleNamespace(taskset=SimpleNamespace(task=None))
        self.started = asyncio.Event()
        self.torn_down = asyncio.Event()

    def slots(self, task):
        return [SimpleNamespace(task=task, traces=[])]

    async def run_slot(self, slot, ctx, gate, on_trace=None):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.shield(asyncio.sleep(0))
            self.torn_down.set()

    @contextlib.asynccontextmanager
    async def serving(self):
        yield


@pytest.mark.asyncio
async def test_cancelling_the_client_run_aborts_the_server_rollout(monkeypatch):
    env = HangingEnv()
    monkeypatch.setattr(server_module, "load_environment", lambda config: env)
    config = SimpleNamespace(taskset=SimpleNamespace(id="stub"))
    server = EnvServer(config, address="tcp://127.0.0.1:0")
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    try:
        run = asyncio.create_task(
            client.run(
                client=EvalClientConfig(),
                model="m",
                sampling=vf.SamplingConfig(),
                task_data=vf.TaskData(id="slow").model_dump(mode="json"),
            )
        )
        await asyncio.wait_for(env.started.wait(), TIMEOUT_S)
        assert len(server._running) == 1

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        await asyncio.wait_for(env.torn_down.wait(), TIMEOUT_S)
        assert server._running == {}
    finally:
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
