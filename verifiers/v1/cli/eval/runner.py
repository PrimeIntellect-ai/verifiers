"""The eval runner: fan episodes out with bounded concurrency."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Iterable
from typing import TypeVar, cast

from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import (
    append_episode,
    output_path,
    save_config,
)
from verifiers.v1.cli.resume import distribute
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.env import Env, RunSlot
from verifiers.v1.episode import Episode, EvalRunInfo
from verifiers.v1.utils.platform import PushState, abort_run, finish_run, open_run

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_rollouts(rollouts: Iterable[Awaitable[T]]) -> list[T]:
    """`asyncio.gather`, but one rollout failing stops the others too.

    Plain `gather` raises the first error and leaves the rest running. They then
    keep going while the caller is already handling that error — still uploading
    to a run it has just closed, still using an env it is tearing down.
    Cancelling them here, and waiting for each one to finish unwinding, keeps
    those two things from overlapping.

    The error is re-raised exactly as it arrived, which is why this is not an
    `asyncio.TaskGroup`: a TaskGroup wraps everything in an `ExceptionGroup`, and
    `main` would stop recognizing a `KeyboardInterrupt` as Ctrl-C."""
    tasks = [asyncio.ensure_future(rollout) for rollout in rollouts]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        # return_exceptions so this waits for all of them; without it the first
        # cancellation would raise and the rest would be left running again.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def run_eval(env: Env, config: EvalConfig) -> list[Episode]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    taskset = env.taskset
    if config.num_tasks is None and taskset.INFINITE:
        raise ValueError(
            f"{type(taskset).__name__} is infinite - bound the run with -n"
        )
    selected = taskset.shuffle() if config.shuffle else taskset
    if config.num_tasks is not None:
        selected = selected.head(config.num_tasks)
    tasks = list(selected)
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    out = output_path(config)
    # One (task, rollouts-to-run) pair per selected task; resume shrinks the counts.
    plan = [(task, config.num_rollouts) for task in tasks]
    # Kept on-disk rollouts rejoin the run as finished episodes; only owed ones re-run.
    finished: list[Episode] = []
    if config.resume:
        keys = [task.hash for task in tasks]
        loaded, owed = resume.load(
            out,
            keys,
            config.num_rollouts,
            lambda episode: env.complete(cast(Episode, episode)),
        )
        finished = [cast(Episode, episode) for episode in loaded]
        if not owed:  # already complete - report it and exit successfully
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        counts = distribute(keys, owed, config.num_rollouts)
        plan = [(task, n) for task, n in zip(tasks, counts) if n]
        logger.info(
            "resuming %s: %d task(s), %d rollout(s) owed",
            out,
            len(plan),
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
    push_state = PushState() if config.push and config.rich else None

    # Opened before the first rollout so the platform's id is the run's id
    run = open_run(config, push_state)
    config.run.adopt_id(run.id)
    # A resume's kept rollouts are part of this run too, so they carry its id and
    # go up with the rest
    for episode in finished:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
    run.log_episodes(finished)

    async def on_complete(episode: Episode) -> None:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        await append_episode(out, episode, write_lock)
        await asyncio.to_thread(run.log_episodes, [episode])

    # Serving resources (shared tool servers, interception) come up once for the
    # run; plan slots inside so the env's agents borrow them. Everything from
    # bringing those up to tearing them down is inside the try: a run that was
    # opened is closed out whatever breaks, so none of them sits at running.
    try:
        async with env.serving():
            planned = [slot for task, n in plan for slot in env.slots(task, n=n)]
            slots = [RunSlot.finished(episode) for episode in finished] + planned
            display = (
                dashboard(slots, config, start, push=push_state)
                if config.rich
                else contextlib.nullcontext()
            )
            async with display:
                results = await gather_rollouts(
                    env.run_slot(slot, ctx, semaphore, on_complete) for slot in planned
                )
                episodes = finished + list(results)
                # Drain and close out off the event loop so the view keeps refreshing.
                await asyncio.to_thread(finish_run, run, episodes, push_state)
    except BaseException as e:
        await asyncio.to_thread(abort_run, run, e, push_state)
        raise
    return episodes


async def run_eval_server(config: EvalConfig) -> list[Episode]:
    """Run evaluation through the env-server worker pool."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.configs.serve import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env
    from verifiers.v1.utils.loaders import load_taskset
    from verifiers.v1.utils.logging import setup_logging

    server_kwargs = {
        "config_data": env_config_data(config.env),  # picklable across the spawn
        # `-c` seeds each worker's episode bound unless `[serve]` pins one — so a
        # pool carries `workers * bound` episodes, as `multiplex` implies.
        "max_concurrent": config.worker_max_concurrent,
    }
    # The client owns the taskset: load it here, once — the server (and its pool
    # workers) never load data, they rebuild each dispatched task from its request.
    taskset = load_taskset(config.env.taskset)
    if config.num_tasks is None and taskset.INFINITE:
        raise ValueError(
            f"{type(taskset).__name__} is infinite - bound the run with -n"
        )
    selected = taskset.shuffle() if config.shuffle else taskset
    if config.num_tasks is not None:
        selected = selected.head(config.num_tasks)
    tasks = list(selected)
    # Spawned processes inherit no logging — hand them the main process's setup so
    # their rollout logs land in the output dir.
    level = "DEBUG" if config.verbose else "INFO"
    log_file = str(output_path(config) / "logs" / "eval.log")
    mpctx = mp.get_context("spawn")
    address_queue: mp.Queue = mpctx.Queue()
    # Death pipe: serve_env self-terminates if this process dies abruptly — we keep
    # parent_conn, whose close (even on our SIGKILL) signals the child's watch.
    parent_conn, child_conn = mpctx.Pipe()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            **pool_serve_kwargs(config.serve.pool),
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
        # A run dispatches — and resumes — tasks by content: the client owns them,
        # and `task.hash` is their identity.
        plan = [
            ({"task_data": task.data.model_dump(mode="json")}, config.num_rollouts)
            for task in tasks
        ]
        out = output_path(config)
        finished: list[Episode] = []
        if config.resume:
            keys = [task.hash for task in tasks]
            loaded, owed = resume.load(out, keys, config.num_rollouts)
            finished = [cast(Episode, episode) for episode in loaded]
            counts = distribute(keys, owed, config.num_rollouts)
            if not owed:  # already complete - report it and exit successfully
                print(resume.nothing_to_resume_msg(out, len(plan), config.num_rollouts))
                raise SystemExit(0)
            plan = [(payload, n) for (payload, _), n in zip(plan, counts) if n]
            logger.info(
                "resuming %s: %d task(s), %d rollout(s) owed",
                out,
                len(plan),
                sum(owed.values()),
            )
        else:
            save_config(config, out)
            logger.info(
                "running %dx%d rollouts via the env-server %s pool on %s",
                len(plan),
                config.num_rollouts,
                config.serve.pool.type,
                config.model,
            )
        logger.info("results: %s", out)
        semaphore = (
            asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
        )
        write_lock = asyncio.Lock()

        run = open_run(config)
        config.run.adopt_id(run.id)
        for episode in finished:
            episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        run.log_episodes(finished)

        async def run_unit(payload: dict) -> list[Episode]:
            async with semaphore or contextlib.nullcontext():
                episode = await client.run(
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                    **payload,
                )
            episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
            await append_episode(out, episode, write_lock)
            await asyncio.to_thread(run.log_episodes, [episode])
            return [cast(Episode, episode)]

        # Each rollout is its own `run` request, dispatched least-busy across workers.
        units = [run_unit(payload) for payload, n in plan for _ in range(n)]
        try:
            results = await gather_rollouts(units)
            await client.close()
            episodes = finished + [record for unit in results for record in unit]
            await asyncio.to_thread(finish_run, run, episodes)
        except BaseException as e:
            await asyncio.to_thread(abort_run, run, e)
            raise
        return episodes
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
        with contextlib.suppress(Exception):
            parent_conn.close()
