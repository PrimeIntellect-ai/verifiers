"""The eval runner: fan rollouts out (one episode per task) with bounded concurrency."""

import asyncio
import contextlib
import logging
import random
import time

from verifiers.v1.clients import RolloutContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.cli import resume
from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.output import append_trace, output_path, save_config
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.env import Environment
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

_SHUFFLE_SEED = (
    0  # fixed so `--shuffle` samples the same tasks every run (reproducible)
)


async def run_eval(env: Environment, config: EvalConfig) -> list[Trace]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    tasks = env.taskset.load_tasks()
    if config.shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(tasks)
    tasks = tasks if config.num_tasks is None else tasks[: config.num_tasks]
    base_urls = (
        [config.client.base_url]
        if isinstance(config.client.base_url, str)
        else config.client.base_url
    )
    clients = [resolve_client(config.client, i) for i in range(len(base_urls))]
    contexts = [
        RolloutContext(client=client, model=config.model, sampling=config.sampling)
        for client in clients
    ]
    # One episode of `num_rollouts` rollouts per task; the shared semaphore bounds total
    # concurrent rollouts (across episodes), so group rewards still see their whole episode.
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    out = output_path(config)
    # Write config.toml up front, then persist each trace as it completes (so the results are
    # durable mid-run, not only at the end). On resume, keep the saved config + good traces and
    # run only the owed rollouts. `append_trace` is a sync single-line append, safe to call from
    # concurrent rollouts in the one event loop.
    if config.resume is not None:
        group = bool(discover_decorated(env.taskset, "group_reward"))
        keep, owed = resume.plan(
            out, [t.idx for t in tasks], config.num_rollouts, group
        )
        if not owed:  # already complete - report it and exit successfully
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        tasks = [task for task in tasks if owed.get(task.idx)]
        episodes = [env.episode(task, contexts, n=owed[task.idx]) for task in tasks]
        resume.rewrite_results(out, keep)
        logger.info(
            "resuming %s: %d task(s), %d rollout(s) owed",
            out,
            len(tasks),
            sum(owed.values()),
        )
    else:
        episodes = [
            env.episode(task, contexts, n=config.num_rollouts) for task in tasks
        ]
        save_config(config, out)
        logger.info(
            "running %dx%d rollouts on %s",
            len(tasks),
            config.num_rollouts,
            config.model,
        )
    start = time.time()
    rollouts = [rollout for episode in episodes for rollout in episode.rollouts]
    display = (
        dashboard(rollouts, config, start) if config.rich else contextlib.nullcontext()
    )
    logger.info("results: %s", out)

    def on_complete(trace: Trace) -> None:
        append_trace(out, trace)

    # Shared tool servers (if any) come up once here and their URLs flow into every rollout
    # (non-shared ones start per rollout inside the episodes); the interception pool comes up
    # here too, so concurrent rollouts share its servers + tunnels rather than one each.
    async with (
        env.shared_tools(tasks) as shared_urls,
        env.interception_pool() as pool,
        display,
    ):
        results = await asyncio.gather(
            *(
                episode.run(semaphore, shared_urls, on_complete, pool)
                for episode in episodes
            )
        )
    traces = [trace for episode_traces in results for trace in episode_traces]
    await asyncio.gather(*(client.close() for client in clients))
    return traces


async def run_eval_server(config: EvalConfig) -> list[Trace]:
    """Eval through the env-server worker pool (`--num-workers > 0`). Spawns the pool
    (works for v1 and the v0 bridge), then drives rollouts by task idx over an
    `EnvClient` — the same path prime-rl trains through, so it exercises the
    router + workers end-to-end. Output matches `run_eval` (config.toml + results.jsonl)."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.cli.log import setup_logging
    from verifiers.v1.env import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env

    legacy = config.is_legacy
    server_kwargs = (
        {
            "env_id": config.id,
            "env_args": config.args,
            "extra_env_kwargs": config.extra_env_kwargs,
        }
        if legacy
        else {"config_data": env_config_data(config)}  # picklable across the spawn
    )
    # The pool broker + workers are spawned (fresh interpreters, no logging) — hand them
    # the same loguru setup the main process uses (stderr + the run's log file) so their
    # rollout logs come back and land in the output dir.
    level = "DEBUG" if config.verbose else "INFO"
    log_file = str(output_path(config) / "eval.log")
    mpctx = mp.get_context("spawn")
    address_queue: mp.Queue = mpctx.Queue()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            **pool_serve_kwargs(config.pool),
            legacy=legacy,
            address="tcp://127.0.0.1:0",
            address_queue=address_queue,
            log_setup=partial(setup_logging, level, log_file),
            **server_kwargs,
        ),
        daemon=False,
    )
    proc.start()
    try:
        address = await asyncio.to_thread(address_queue.get, timeout=600)
        client = EnvClient(address=address)
        await client.wait_for_server_startup(timeout=600)
        info = await client.info()
        idxs = list(range(info.num_tasks))
        if config.shuffle:
            random.Random(_SHUFFLE_SEED).shuffle(idxs)
        if config.num_tasks is not None:
            idxs = idxs[: config.num_tasks]
        out = output_path(config)
        if config.resume is not None:
            keep, owed = resume.plan(
                out, idxs, config.num_rollouts, info.requires_group_scoring
            )
            if not owed:  # already complete - report it and exit successfully
                print(resume.nothing_to_resume_msg(out, len(idxs), config.num_rollouts))
                raise SystemExit(0)
            resume.rewrite_results(out, keep)
            idxs = [idx for idx in idxs if owed.get(idx)]
            logger.info(
                "resuming %s: %d task(s), %d rollout(s) owed",
                out,
                len(idxs),
                sum(owed.values()),
            )
        else:
            owed = {idx: config.num_rollouts for idx in idxs}
            save_config(config, out)
            logger.info(
                "running %dx%d rollouts via the env-server %s pool on %s",
                len(idxs),
                config.num_rollouts,
                config.pool.type,
                config.model,
            )
        logger.info("results: %s", out)
        semaphore = (
            asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
        )

        async def run_group_unit(idx: int) -> list[Trace]:
            async with semaphore or contextlib.nullcontext():
                traces = await client.run_group(
                    task_idx=idx,
                    n=config.num_rollouts,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            for trace in traces:
                append_trace(out, trace)
            return traces

        async def run_rollout_unit(idx: int) -> list[Trace]:
            async with semaphore or contextlib.nullcontext():
                trace = await client.run_rollout(
                    task_idx=idx,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            append_trace(out, trace)
            return [trace]

        # A group-scored taskset must run each task's rollouts together (cross-rollout
        # scoring) → one `run_group` request per task (one worker). Otherwise the rollouts
        # are independent → one `run_rollout` request each, which the broker round-robins
        # (least-busy) across workers — mirrors the prime-rl dispatcher.
        if info.requires_group_scoring:
            units = [run_group_unit(i) for i in idxs]
        else:
            units = [run_rollout_unit(i) for i in idxs for _ in range(owed[i])]
        results = await asyncio.gather(*units)
        await client.close()
        return [trace for unit_traces in results for trace in unit_traces]
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
