"""The eval runner: fan episodes out with bounded concurrency.

Rollouts run through the env-server worker pool by default (`[serve]` sizes it;
elastic — one worker, scaling on demand), the same path prime-rl trains through.
`--no-serve` runs them in-process instead. Both paths share this runner — task
selection, resume, persistence, the dashboard — and differ only in how one slot
becomes one episode: `env.run_slot` in-process, a `run` request to the pool
otherwise. The dashboard watches the same `RunSlot`s either way; a served slot
has no live traces, so its per-turn detail lands when the episode completes.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar, cast

from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import (
    append_episode,
    attempt_log_file,
    output_path,
    save_config,
)
from verifiers.v1.cli.resume import distribute
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.serve import ServeConfig
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


RunSlotFn = Callable[[RunSlot], Awaitable[Episode]]
OnComplete = Callable[[Episode], Awaitable[None]]


@contextlib.asynccontextmanager
async def _in_process(
    env: Env,
    config: EvalConfig,
    semaphore: asyncio.Semaphore | None,
    on_complete: OnComplete,
) -> AsyncIterator[RunSlotFn]:
    """Run slots in-process: serving resources (shared tool servers, interception)
    come up once for the run; the env's agents borrow them."""
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )

    async def run(slot: RunSlot) -> Episode:
        return await env.run_slot(slot, ctx, semaphore, on_complete)

    async with env.serving():
        yield run


@contextlib.asynccontextmanager
async def _server(
    config: EvalConfig,
    serve: ServeConfig,
    semaphore: asyncio.Semaphore | None,
    on_complete: OnComplete,
) -> AsyncIterator[RunSlotFn]:
    """Run slots through a spawned env-server worker pool: each rollout is its own
    `run` request, dispatched least-busy across workers. The workers own the env
    (and its serving resources); this process owns the taskset and the results."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.configs.serve import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env
    from verifiers.v1.utils.logging import setup_logging

    # Spawned processes inherit no logging — hand them the main process's setup so
    # their rollout logs land in the output dir. They share its stderr, so console
    # output follows the main process's choice: off under the dashboard (worker log
    # lines would print over the Live view and shift it), on otherwise.
    level = "DEBUG" if config.verbose else "INFO"
    log_file = str(attempt_log_file(output_path(config)))
    console = config.rich is None
    mpctx = mp.get_context("spawn")
    address_queue: mp.Queue = mpctx.Queue()
    # Death pipe: serve_env self-terminates if this process dies abruptly — we keep
    # parent_conn, whose close (even on our SIGKILL) signals the child's watch.
    parent_conn, child_conn = mpctx.Pipe()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            **pool_serve_kwargs(serve.pool),
            address="tcp://127.0.0.1:0",
            address_queue=address_queue,
            death_pipe=child_conn,
            log_setup=partial(setup_logging, level, log_file, console),
            config_data=env_config_data(config.env),  # picklable across the spawn
            # `-c` seeds each worker's episode bound unless `[serve]` pins one — so a
            # pool carries `workers * bound` episodes, as `multiplex` implies.
            max_concurrent=serve.max_concurrent
            if serve.max_concurrent is not None
            else config.max_concurrent,
        ),
        daemon=False,
    )
    proc.start()
    child_conn.close()  # the child holds its end; we keep parent_conn so our exit closes it
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


async def run_eval(config: EvalConfig) -> list[Episode]:
    from verifiers.v1.utils.loaders import load_environment, load_taskset

    # The env comes up in this process only for an in-process run; a served run's
    # workers each load their own, and this process owns just the taskset.
    env = None if config.serve is not None else load_environment(config.env)
    taskset = env.taskset if env is not None else load_taskset(config.env.taskset)
    if config.num_tasks is None and taskset.INFINITE:
        raise ValueError(
            f"{type(taskset).__name__} is infinite - bound the run with -n"
        )
    selected = taskset.shuffle() if config.shuffle else taskset
    if config.num_tasks is not None:
        selected = selected.head(config.num_tasks)
    tasks = list(selected)
    out = output_path(config)
    # One (task, rollouts-to-run) pair per selected task; resume shrinks the counts.
    plan = [(task, config.num_rollouts) for task in tasks]
    # Kept on-disk rollouts rejoin the run as finished episodes; only owed ones re-run.
    finished: list[Episode] = []
    if config.resume:
        keys = [task.hash for task in tasks]
        # In-process, the env's own keep-verdict decides what resumes; a served run
        # can't ask the worker-side env, so it keeps the default `episode.ok`.
        complete = (
            (lambda episode: env.complete(cast(Episode, episode)))
            if env is not None
            else None
        )
        loaded, owed = resume.load(out, keys, config.num_rollouts, complete)
        finished = [cast(Episode, episode) for episode in loaded]
        if not owed:  # already complete - report it and exit successfully
            print(
                f"nothing to resume in {out}: all {len(tasks)}x{config.num_rollouts} "
                "rollouts already completed without error"
            )
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
        via = (
            f" via the env-server {config.serve.pool.type} pool"
            if config.serve is not None
            else ""
        )
        logger.info(
            "running %dx%d rollouts on %s%s",
            len(plan),
            config.num_rollouts,
            config.model,
            via,
        )
    start = time.time()
    logger.info("results: %s", out)

    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    write_lock = asyncio.Lock()
    push_state = PushState() if config.push and config.rich is not None else None

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

    backend = (
        _in_process(env, config, semaphore, on_complete)
        if env is not None
        else _server(config, config.serve, semaphore, on_complete)
    )
    # Everything from bringing the backend up (serving resources, or the worker
    # pool) to tearing it down is inside the try: a run that was opened is closed
    # out whatever breaks, so none of them sits at running.
    try:
        async with backend as run_slot:
            # The display slots: in-process ones are the env's own (it fills their live
            # traces); a served rollout's is a client-side stand-in its worker never sees.
            planned = [
                slot
                for task, n in plan
                for slot in (
                    env.slots(task, n)
                    if env is not None
                    else [RunSlot(task) for _ in range(n)]
                )
            ]
            slots = [RunSlot.finished(episode) for episode in finished] + planned
            display = (
                dashboard(slots, config, start, push=push_state)
                if config.rich is not None
                else contextlib.nullcontext()
            )
            async with display:
                results = await gather_rollouts(run_slot(slot) for slot in planned)
                episodes = finished + list(results)
                # Drain and close out off the event loop so the view keeps refreshing.
                await asyncio.to_thread(finish_run, run, episodes, push_state)
    except BaseException as e:
        await asyncio.to_thread(abort_run, run, e, push_state)
        raise
    return episodes
