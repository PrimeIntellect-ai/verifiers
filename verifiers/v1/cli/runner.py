"""The eval runner: fan rollouts out (one episode per task) with bounded concurrency."""

import asyncio
import contextlib
import logging
import random
import time

from verifiers.v1.clients import RolloutContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.output import append_trace, output_path, save_config
from verifiers.v1.env import Environment
from verifiers.v1.interception import InterceptionPool
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

_SHUFFLE_SEED = (
    0  # fixed so `--shuffle` samples the same tasks every run (reproducible)
)


async def run_eval(env: Environment, config: EvalConfig) -> list[Trace]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    client = resolve_client(config.client)
    tasks = env.taskset.load_tasks()
    if config.shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(tasks)
    tasks = tasks if config.num_tasks is None else tasks[: config.num_tasks]
    ctx = RolloutContext(client=client, model=config.model, sampling=config.sampling)
    # One episode of `num_rollouts` rollouts per task; the shared semaphore bounds total
    # concurrent rollouts (across episodes), so group rewards still see their whole episode.
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    episodes = [env.episode(task, ctx, n=config.num_rollouts) for task in tasks]
    logger.info(
        "running %dx%d rollouts on %s", len(tasks), config.num_rollouts, config.model
    )
    start = time.time()
    rollouts = [rollout for episode in episodes for rollout in episode.rollouts]
    display = (
        dashboard(rollouts, config, start) if config.rich else contextlib.nullcontext()
    )
    # Write config.toml up front, then persist each trace as it completes (so the results
    # are durable mid-run, not only at the end). `append_trace` is a sync single-line
    # append, safe to call from concurrent rollouts in the one event loop.
    out = output_path(config)
    save_config(config, out)
    logger.info("results: %s", out)

    def on_complete(trace: Trace) -> None:
        append_trace(out, trace)

    # A shared interception pool comes up once here too, so N concurrent rollouts share
    # ~N/multiplex servers + tunnels rather than one each (grown on demand). Always on
    # (multiplex >= 1; 1 = a server/tunnel per rollout).
    pool = InterceptionPool(env.harness.config.runtime, config.multiplex)
    # Shared tool servers (if any) come up once here and their URLs flow into every
    # rollout; non-shared ones start per rollout inside the episodes.
    async with (
        env.shared_tools(tasks) as shared_urls,
        pool,
        display,
    ):
        results = await asyncio.gather(
            *(
                episode.run(semaphore, shared_urls, on_complete, pool)
                for episode in episodes
            )
        )
    traces = [trace for episode_traces in results for trace in episode_traces]
    await client.close()
    return traces
