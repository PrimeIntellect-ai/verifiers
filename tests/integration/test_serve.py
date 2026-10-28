"""A served run: an env server runs the episode a client's run request asks for, and
the request's rollout group is on the episode while the env's hooks run."""

import asyncio

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.utils.loaders import resolve_env_config


async def test_finalize_sees_the_requests_group(scripted_model):
    config = resolve_env_config(
        {"taskset": {"id": "grouped"}, "agent": {"harness": {"id": "null"}}}
    )
    server = EnvServer(config, address="tcp://127.0.0.1:0")
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    task = next(iter(server.env.taskset))
    try:
        episode = await client.run(
            client=EvalClientConfig(**scripted_model),
            model="scripted",
            sampling=SamplingConfig(),
            task_data=task.data.model_dump(mode="json"),
            group=vf.GroupInfo(id="group-0"),
        )
    finally:
        await client.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
    assert episode.ok, [error.message for error in episode.errors]
    assert episode.group == vf.GroupInfo(id="group-0")
    (trace,) = episode.traces
    assert trace.info["group"] == "group-0"
