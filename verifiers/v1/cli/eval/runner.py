"""The eval runner: fan rollouts out (one episode per task) with bounded concurrency."""

import asyncio
import contextlib
import logging
import time
from collections import defaultdict

from verifiers.v1.clients import RolloutContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.output import append_trace, output_path, save_config
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.env import Environment
from verifiers.v1.taskset import select_tasks
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


async def run_eval(env: Environment, config: EvalConfig) -> list[Trace]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    client = resolve_client(config.client)
    # Draw only the tasks this run needs from `load_tasks` (which may be a lazy/infinite
    # generator): without `--shuffle`, just the first `--num-tasks` are built.
    tasks = select_tasks(env.taskset, config.num_tasks, config.shuffle)
    ctx = RolloutContext(client=client, model=config.model, sampling=config.sampling)
    # One episode of `num_rollouts` rollouts per task; the shared semaphore bounds total
    # concurrent rollouts (across episodes), so group rewards still see their whole episode.
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    out = output_path(config)
    # Write config.toml up front, then persist each trace as it completes (so the results are
    # durable mid-run, not only at the end). On resume, keep the saved config + good traces and
    # run only the owed rollouts. One lock serializes worker-thread appends from concurrent
    # rollouts while keeping large trace serialization off the event loop.
    owed: dict[str, int] | None = None
    if config.resume is not None:
        group = bool(discover_decorated(env.taskset, "group_reward"))
        keep, owed = resume.plan(
            out, [t.idx for t in tasks], config.num_rollouts, group
        )
        if not owed:  # already complete - report it and exit successfully
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        tasks = [task for task in tasks if owed.get(task.idx)]
        resume.rewrite_results(out, keep)
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

    # Shared tool servers (if any) come up once here and their URLs flow into every rollout
    # (non-shared ones start per rollout inside the episodes); the interception pool comes up
    # here too, so concurrent rollouts share its servers + tunnels rather than one each. Build
    # episodes inside `serving` so each rollout is wired to those resources at construction.
    async with env.serving(tasks):
        episodes = [
            env.episode(task, ctx, n=owed[task.idx] if owed else config.num_rollouts)
            for task in tasks
        ]
        rollouts = [rollout for episode in episodes for rollout in episode.rollouts]
        display = (
            dashboard(rollouts, config, start)
            if config.rich
            else contextlib.nullcontext()
        )
        async with display:
            results = await asyncio.gather(
                *(episode.run(semaphore, on_complete) for episode in episodes)
            )
    traces = [trace for episode_traces in results for trace in episode_traces]
    await client.close()
    return traces


async def run_eval_server(config: EvalConfig) -> list[Trace]:
    """Eval through the env-server worker pool (`--num-workers > 0`). Spawns the pool
    (works for v1 and the v0 bridge), then drives rollouts by task idx over an
    `EnvClient` — the same path prime-rl trains through, so it exercises the
    router + workers end-to-end. Output matches `run_eval` (config.toml + results.jsonl)."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.utils.logging import setup_logging
    from verifiers.v1.env import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env

    legacy = config.is_legacy
    server_kwargs = (
        {
            "env_id": config.id,
            "env_args": config.args,
            "extra_env_kwargs": config.extra_env_kwargs,
            "shuffle": config.shuffle,
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
        if config.num_tasks is None and info.num_tasks is None:
            raise SystemExit(
                "this taskset is infinite (no task count); pass -n/--num-tasks to bound the eval"
            )
        # How many tasks to pull. The server owns task order (shuffle/epoch); the runner just
        # pulls this many — the requested cap, the taskset's count, or the smaller of the two.
        if config.num_tasks is None:
            total = info.num_tasks
        elif info.num_tasks is None:
            total = config.num_tasks
        else:
            total = min(config.num_tasks, info.num_tasks)
        out = output_path(config)
        if config.resume is not None:
            good: dict[int, int] = defaultdict(int)
            for _offset, idx, errored in resume._read_results(out / "results.jsonl"):
                if not errored:
                    good[idx] += 1
            complete = sum(1 for c in good.values() if c >= config.num_rollouts)
            if complete >= total:  # already done — report and exit successfully
                print(resume.nothing_to_resume_msg(out, total, config.num_rollouts))
                raise SystemExit(0)
            if complete:
                # Task-less pull can't target the specific missing tasks (the server owns order),
                # so a partially-complete --server eval resumes by re-running from scratch.
                logger.warning(
                    "--server eval resume re-runs from scratch (the server owns task order); "
                    "discarding %d partially-complete task(s)",
                    complete,
                )
            resume.rewrite_results(out, [])
        else:
            save_config(config, out)
        logger.info(
            "running %dx%d rollouts via the env-server %s pool on %s",
            total,
            config.num_rollouts,
            config.pool.type,
            config.model,
        )
        logger.info("results: %s", out)
        request_concurrency = config.max_concurrent
        if request_concurrency:
            # A unit pulls one task and runs `num_rollouts` of it together in one indivisible
            # `run_group` request, so it costs `num_rollouts` rollout slots; max_concurrent bounds
            # rollouts, not requests. A group can't be split, so the cap can't go below one group.
            if request_concurrency < config.num_rollouts:
                logger.warning(
                    "max_concurrent=%d < num_rollouts=%d: a group's rollouts run together in one "
                    "request, so %d run concurrently regardless of max_concurrent (a group is "
                    "indivisible under task-less pull).",
                    request_concurrency,
                    config.num_rollouts,
                    config.num_rollouts,
                )
            request_concurrency = max(1, request_concurrency // config.num_rollouts)
        semaphore = (
            asyncio.Semaphore(request_concurrency) if request_concurrency else None
        )
        write_lock = asyncio.Lock()

        async def run_unit() -> list[Trace]:
            # Each unit samples one task and runs `num_rollouts` of it as a group. The runner
            # never addresses a task — `sample()` pulls it; the served task is on `trace.task.idx`.
            async with semaphore or contextlib.nullcontext():
                task = await client.sample()
                traces = await client.run_group(
                    task=task,
                    n=config.num_rollouts,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            for trace in traces:
                await append_trace(out, trace, write_lock)
            return traces

        units = [run_unit() for _ in range(total)]
        results = await asyncio.gather(*units)
        await client.close()
        return [trace for unit_traces in results for trace in unit_traces]
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
        with contextlib.suppress(Exception):
            parent_conn.close()
