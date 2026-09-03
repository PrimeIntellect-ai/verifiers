"""Run a v1 env over a taskset for the e2e tests: in-process, or through a spawned
env-server worker pool. Evaluation as a product lives in prime-rl (`uv run eval`);
this keeps the suite's coverage of both execution paths, with the same on-disk output
(`configs/resolved/*.json` + `traces.jsonl`) the `replay` CLI reads."""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.cli.output import (
    append_episode,
    attempt_log_file,
    output_path,
    save_config,
)
from verifiers.v1.clients import ClientConfig, EvalClientConfig, ModelContext
from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.cli.run import RunConfig, default_run_name
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.serve import ServeConfig
from verifiers.v1.env import Env, RunSlot
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig
from verifiers.v1.episode import Episode, EvalRunInfo
from verifiers.v1.types import SamplingConfig

RunSlotFn = Callable[[RunSlot], Awaitable[Episode]]
OnComplete = Callable[[Episode], Awaitable[None]]


class RunnerConfig(BaseConfig):
    """The env, how it is hosted, the model, and the run's size."""

    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    serve: ServeConfig | None = None
    """None runs the rollouts in-process; a `ServeConfig` spawns an env-server pool."""
    run: RunConfig = Field(default_factory=RunConfig)
    model: str
    client: ClientConfig = EvalClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    num_tasks: int | None = None
    num_rollouts: int = 1
    max_concurrent: int | None = 128
    output_dir: Path = Path("outputs")

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))

    @model_validator(mode="after")
    def auto_setup_run_name(self):
        if self.run.name is None:
            self.run.name = default_run_name(self.env, self.model)
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self


@contextlib.asynccontextmanager
async def _in_process(
    env: Env,
    config: RunnerConfig,
    semaphore: asyncio.Semaphore | None,
    on_complete: OnComplete,
) -> AsyncIterator[RunSlotFn]:
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )

    async def run(slot: RunSlot) -> Episode:
        return await env.run_slot(slot, ctx, semaphore, on_complete)

    async with env.serving():
        yield run


@contextlib.asynccontextmanager
async def _server(
    config: RunnerConfig,
    serve: ServeConfig,
    semaphore: asyncio.Semaphore | None,
    on_complete: OnComplete,
) -> AsyncIterator[RunSlotFn]:
    """Each rollout is its own `run` request to a spawned env-server pool; the workers own
    the env, this process owns the taskset and the results."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.configs.serve import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env
    from verifiers.v1.utils.logging import setup_logging

    log_file = str(attempt_log_file(output_path(config)))
    mpctx = mp.get_context("spawn")
    address_queue: mp.Queue = mpctx.Queue()
    # Death pipe: serve_env self-terminates if this process dies abruptly.
    parent_conn, child_conn = mpctx.Pipe()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            **pool_serve_kwargs(serve.pool),
            address="tcp://127.0.0.1:0",
            address_queue=address_queue,
            death_pipe=child_conn,
            log_setup=partial(setup_logging, "INFO", log_file, True),
            config_data=env_config_data(config.env),
            max_concurrent=serve.max_concurrent
            if serve.max_concurrent is not None
            else config.max_concurrent,
        ),
        daemon=False,
    )
    proc.start()
    child_conn.close()
    try:
        address = await asyncio.to_thread(address_queue.get, timeout=600)
        client = EnvClient(address=address)
        try:
            await client.wait_for_server_startup(timeout=600)

            async def run(slot: RunSlot) -> Episode:
                async with semaphore or contextlib.nullcontext():
                    slot.started = time.time()
                    episode = await client.run(
                        client=config.client,
                        model=config.model,
                        sampling=config.sampling,
                        task_data=slot.task.data.model_dump(mode="json"),
                    )
                slot.traces = list(episode.traces)
                slot.episode = cast(Episode, episode)
                slot.done = True
                await on_complete(cast(Episode, episode))
                return cast(Episode, episode)

            yield run
        finally:
            await client.close()
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
        with contextlib.suppress(Exception):
            parent_conn.close()


async def run_episodes(config: RunnerConfig) -> list[Episode]:
    """Run `num_rollouts` episodes for the first `num_tasks` tasks and return them; the
    run dir gets the resolved config and one `traces.jsonl` line per episode."""
    from verifiers.v1.utils.loaders import load_environment, load_taskset

    env = None if config.serve is not None else load_environment(config.env)
    taskset = env.taskset if env is not None else load_taskset(config.env.taskset)
    selected = taskset if config.num_tasks is None else taskset.head(config.num_tasks)
    tasks = list(selected)
    out = output_path(config)
    save_config(config, out, "run.json")
    write_lock = asyncio.Lock()

    async def on_complete(episode: Episode) -> None:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        await append_episode(out, episode, write_lock)

    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    backend = (
        _in_process(env, config, semaphore, on_complete)
        if env is not None
        else _server(config, config.serve, semaphore, on_complete)
    )
    async with backend as run_slot:
        slots = [
            slot
            for task in tasks
            for slot in (
                env.slots(task, config.num_rollouts)
                if env is not None
                else [RunSlot(task) for _ in range(config.num_rollouts)]
            )
        ]
        return list(await asyncio.gather(*(run_slot(slot) for slot in slots)))
