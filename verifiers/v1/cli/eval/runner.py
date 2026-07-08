"""The eval runner: fan rollouts out (one episode per task) with bounded concurrency."""

import asyncio
import contextlib
import logging
import random
import time

from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import append_graph, append_trace, output_path, save_config
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment
from verifiers.v1.episode import requires_group_scoring
from verifiers.v1.topology import TopologyRunner
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

_SHUFFLE_SEED = 0  # fixed so `--shuffle` samples the same tasks every run (reproducible)


async def run_eval(env: Environment, config: EvalConfig) -> list[Trace]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    client = resolve_client(config.client)
    tasks = env.taskset.load_tasks()
    if config.shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(tasks)
    tasks = tasks if config.num_tasks is None else tasks[: config.num_tasks]
    ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
    # One episode of `num_rollouts` rollouts per task; the shared semaphore bounds total
    # concurrent rollouts (across episodes), so group rewards still see their whole episode.
    semaphore = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    out = output_path(config)
    # Write config.toml up front, then persist each trace as it completes (so the results are
    # durable mid-run, not only at the end). On resume, keep the saved config + good traces and
    # run only the owed rollouts. One lock serializes worker-thread appends from concurrent
    # rollouts while keeping large trace serialization off the event loop.
    owed: dict[int, int] | None = None
    # On resume, the kept (good) on-disk rollouts, reloaded as finished traces so the live
    # dashboard counts the whole run (progress, reward, err, usage/time) rather than only this
    # session's re-run rollouts. Only the --rich dashboard reads them, so skip the load otherwise.
    finished: list[Trace] = []
    if config.resume is not None:
        group = requires_group_scoring(tasks)
        keep, owed = resume.plan(out, [t.idx for t in tasks], config.num_rollouts, group)
        if not owed:  # already complete - report it and exit successfully
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        tasks = [task for task in tasks if owed.get(task.idx)]
        resume.rewrite_results(out, keep)
        if config.rich:
            finished = resume.load_kept(out, env.taskset)
        logger.info(
            "resuming %s: %d task(s), %d rollout(s) owed",
            out,
            len(tasks),
            sum(owed.values()),
        )
    else:
        save_config(config, out)
        logger.info(
            "running %dx%d rollouts on %s",
            len(tasks),
            config.num_rollouts,
            config.model,
        )
    start = time.time()
    logger.info("results: %s", out)

    write_lock = asyncio.Lock()

    async def on_complete(trace: Trace) -> None:
        await append_trace(out, trace, write_lock)

    # The serving resources — the lazy shared tool-server registry and the interception
    # pool — come up once here, so concurrent rollouts share servers + tunnels rather than
    # one each. Build episodes inside `serving` so each rollout is wired to those resources
    # at construction.
    async with env.serving():
        episodes = [env.episode(task, ctx, n=owed[task.idx] if owed else config.num_rollouts) for task in tasks]
        rollouts = [rollout for episode in episodes for rollout in episode.rollouts]
        display = dashboard(rollouts, config, start, finished=finished) if config.rich else contextlib.nullcontext()
        async with display:
            results = await asyncio.gather(*(episode.run(semaphore, on_complete) for episode in episodes))
    traces = [trace for episode_traces in results for trace in episode_traces]
    await client.close()
    return traces


async def run_topology_eval(env: TopologyRunner, config: EvalConfig) -> list[Trace]:
    """The topology counterpart of `run_eval`: one topology instance (`go`) per seed task ×
    `num_rollouts`, all instances concurrent, with the shared semaphore still bounding
    total rollouts in flight (episodes across instances and agents). An instance's traces
    persist together once the instance finishes — deferred (backward-arrow) rewards are
    recorded onto upstream traces during the instance, so nothing may land earlier."""
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    client = resolve_client(config.client)
    tasks = env.topology.load_tasks()
    if config.shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(tasks)
    tasks = tasks if config.num_tasks is None else tasks[: config.num_tasks]
    ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
    semaphore = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    out = output_path(config)
    save_config(config, out)
    logger.info(
        "running %dx%d topology instance(s) of %s on %s",
        len(tasks),
        config.num_rollouts,
        env.topology.config.id,
        config.model,
    )
    logger.info("results: %s", out)
    write_lock = asyncio.Lock()

    async def run_instance(task) -> list[Trace]:
        graph = await env.run_instance(task, semaphore)
        await append_graph(out, graph, write_lock)
        return graph.traces

    async with env.serving(ctx):
        instances = [run_instance(task) for task in tasks for _ in range(config.num_rollouts)]
        results = await asyncio.gather(*instances)
    traces = [trace for instance_traces in results for trace in instance_traces]
    await client.close()
    return traces


async def run_eval_server(config: EvalConfig) -> list[Trace]:
    """Eval through the env-server worker pool (`--num-workers > 0`). Spawns the pool
    (works for v1 and the v0 bridge), then drives rollouts by task idx over an
    `EnvClient` — the same path prime-rl trains through, so it exercises the
    router + workers end-to-end. Output matches `run_eval` (config.toml + results.jsonl)."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.env import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env
    from verifiers.v1.utils.logging import setup_logging

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
    # Death pipe: serve_env (and, transitively, its workers/tunnels/sandboxes) self-terminates
    # if this main process dies abruptly. We keep parent_conn; its close — even on our SIGKILL —
    # signals death to the child's watch (see _arm_teardown). Mirrors the broker -> worker pipe.
    parent_conn, child_conn = mpctx.Pipe()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            **pool_serve_kwargs(config.pool),
            legacy=legacy,
            address="tcp://127.0.0.1:0",
            address_queue=address_queue,
            death_pipe=child_conn,
            log_setup=partial(setup_logging, level, log_file),
            **server_kwargs,
        ),
        daemon=False,
    )
    proc.start()
    child_conn.close()  # the child holds its end; we keep parent_conn so our exit closes it
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
            keep, owed = resume.plan(out, idxs, config.num_rollouts, info.requires_group_scoring)
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
        request_concurrency = config.max_concurrent
        if request_concurrency and info.requires_group_scoring:
            # max_concurrent is a rollout resource bound, not a request-throughput target.
            # A group is indivisible, so one oversized group must still be allowed to run.
            request_concurrency = max(1, request_concurrency // config.num_rollouts)
        semaphore = asyncio.Semaphore(request_concurrency) if request_concurrency else None
        write_lock = asyncio.Lock()

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
                await append_trace(out, trace, write_lock)
            return traces

        async def run_rollout_unit(idx: int) -> list[Trace]:
            async with semaphore or contextlib.nullcontext():
                trace = await client.run_rollout(
                    task_idx=idx,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            await append_trace(out, trace, write_lock)
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
        with contextlib.suppress(Exception):
            parent_conn.close()
