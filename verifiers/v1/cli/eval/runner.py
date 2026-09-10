"""The eval runner: fan episodes out with bounded concurrency.

Rollouts run through the env-server worker pool by default (`[serve]` sizes it;
elastic — one worker, scaling on demand), the same path prime-rl trains through.
`--no-serve` runs them in-process instead. Both paths share this runner — task
selection, resume, persistence, the dashboard — and differ only in how one slot
becomes one episode: `env.run_slot` in-process, a `run` request to the pool
otherwise. Tasks come off the taskset's `stream()`: a static taskset's are all
planned up front, a streaming one's as they appear, pulled while the concurrency
window has room. The dashboard watches the same `RunSlot`s either way; a served slot
has no live traces, so its per-turn detail lands when the episode completes.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
)
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
from verifiers.v1.task import Task
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.platform import (
    PushState,
    abort_run,
    finish_run,
    log_episodes,
    open_run,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


RunSlotFn = Callable[[RunSlot], Awaitable[Episode]]
OnComplete = Callable[[Episode], Awaitable[None]]


async def _aiter(items: Iterable[T]) -> AsyncGenerator[T, None]:
    for item in items:
        yield item


async def _take(tasks: AsyncIterator[T], n: int | None) -> AsyncGenerator[T, None]:
    """The first `n` tasks of an async source (all when None), without pulling an
    (n+1)th the source may still be waiting for; the source is closed after."""
    taken = 0
    try:
        async for task in tasks:
            yield task
            taken += 1
            if taken == n:
                break
    finally:
        # End the source now (its `finally` runs), not when it is collected.
        if (aclose := getattr(tasks, "aclose", None)) is not None:
            await aclose()


async def run_stream(
    groups: AsyncGenerator[list[RunSlot], None],
    run_slot: RunSlotFn,
    window: int | None,
) -> list[Episode]:
    """Run each group of slots as it arrives, pulling the next group only while fewer
    than `window` slots are in flight (None = pull freely), and return every episode
    in submission order once the stream has ended and the last slot is done. One
    slot failing cancels the rest and waits for them to unwind, so nothing keeps
    uploading into a run the caller is already closing; the stream is `aclose()`d
    on any exit. The pull is its own task, raced against the slots' first failure:
    a slot that fails while the feed is quiet aborts the run at once, not when the
    feed next yields (which may be never). Not a `TaskGroup`: that wraps errors in
    an `ExceptionGroup`, and `main` would no longer see a `KeyboardInterrupt` as
    Ctrl-C."""
    started: list[asyncio.Task[Episode]] = []
    pending: set[asyncio.Task[Episode]] = set()
    pull: asyncio.Task[list[RunSlot]] | None = None
    # Resolved with the first slot to fail (by a done-callback, so the wait on the
    # pull stays O(1) however many slots a static run has in flight).
    failed: asyncio.Future[asyncio.Task[Episode]] = (
        asyncio.get_running_loop().create_future()
    )

    def watch(task: asyncio.Task[Episode]) -> None:
        if not task.cancelled() and task.exception() is not None and not failed.done():
            failed.set_result(task)

    try:
        while True:
            # Back-pressure: the next group is pulled only once a slot has freed.
            while window is not None and len(pending) >= window:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()  # a failed slot raises here: cancel the rest
            pull = asyncio.ensure_future(anext(groups))
            await asyncio.wait({pull, failed}, return_when=asyncio.FIRST_COMPLETED)
            if failed.done():
                failed.result().result()  # raises the slot's error: cancel the rest
            try:
                group = pull.result()
            except StopAsyncIteration:
                break
            for slot in group:
                started.append(asyncio.ensure_future(run_slot(slot)))
                started[-1].add_done_callback(watch)
                pending.add(started[-1])
        return await asyncio.gather(*started)
    except BaseException:
        # The pull too: the stream must not still be running when it is closed.
        tasks = [*started, pull] if pull is not None else started
        for task in tasks:
            task.cancel()
        # return_exceptions: wait for every task, not just the first to cancel.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        await groups.aclose()


async def _plan_stream(
    tasks: AsyncGenerator[Task, None],
    plan_slots: Callable[[Task], list[RunSlot]],
    slots: list[RunSlot],
) -> AsyncGenerator[list[RunSlot], None]:
    """Plan each task as it arrives: its slots join the display (`slots`) at once,
    but are yielded one per group, so the window bounds the slots in flight rather
    than the tasks — a task's `-r` rollouts wait here, not as live coroutines."""
    async with contextlib.aclosing(tasks):
        async for task in tasks:
            planned = plan_slots(task)
            slots.extend(planned)  # the display grows with the stream
            for slot in planned:
                yield [slot]


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
    # A streaming taskset's tasks appear over time: nothing to list up front, so
    # no whole set to shuffle and no keys to resume against (`-n` still bounds it).
    streaming = taskset.streaming
    if streaming and (config.shuffle or config.resume):
        raise ValueError(
            f"{type(taskset).__name__} streams its tasks - cannot "
            f"{'shuffle' if config.shuffle else 'resume'}"
        )
    selected = taskset.shuffle() if config.shuffle else taskset
    if config.num_tasks is not None:
        selected = selected.head(config.num_tasks)
    tasks = [] if streaming else list(selected)
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
        count = config.num_tasks if streaming else len(plan)
        logger.info(
            "running %sx%d rollouts on %s%s",
            "streamed tasks " if count is None else count,
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
    push_state = PushState()

    # Opened before the first rollout so every episode streams as it lands.
    run = open_run(
        config, push_state, num_examples=config.num_tasks if streaming else len(tasks)
    )
    # Resumed rollouts are part of this run too.
    log_episodes(run, finished)

    async def on_complete(episode: Episode) -> None:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        await append_episode(out, episode, write_lock)
        await asyncio.to_thread(log_episodes, run, [episode])

    backend = (
        _in_process(env, config, semaphore, on_complete)
        if env is not None
        else _server(config, config.serve, semaphore, on_complete)
    )

    # The display slots: in-process ones are the env's own (it fills their live
    # traces); a served rollout's is a client-side stand-in its worker never sees.
    def plan_slots(task: Task, n: int) -> list[RunSlot]:
        if env is not None:
            return env.slots(task, n)
        return [RunSlot(task) for _ in range(n)]

    slots = [RunSlot.finished(episode) for episode in finished]

    # The run is closed out whatever breaks, backend setup and teardown included.
    try:
        async with backend as run_slot:
            if streaming:
                # Each task is planned as it arrives, pulled only as slots free up: the
                # feed decides what is owed, `-c` how many run at once, `-n` how many
                # in total.
                groups = _plan_stream(
                    _take(selected.stream(), config.num_tasks),
                    lambda task: plan_slots(task, config.num_rollouts),
                    slots,
                )
                window = config.max_concurrent
            else:
                # Every slot exists before the first rollout runs; the semaphore alone
                # bounds what runs, so nothing is held back.
                planned = [plan_slots(task, n) for task, n in plan]
                slots.extend(slot for group in planned for slot in group)
                groups, window = _aiter(planned), None
            display = (
                dashboard(slots, config, start, push=push_state)
                if config.rich is not None
                else contextlib.nullcontext()
            )
            async with display:
                results = await run_stream(groups, run_slot, window)
                episodes = finished + list(results)
                # Drain and close out off the event loop so the view keeps refreshing.
                # Shielded: a Ctrl-C here must not cancel the close-out before the
                # worker picks it up (a cancelled executor item never runs), so it
                # runs to completion first and the interrupt is re-raised after —
                # by which point the run is finished and `abort_run` has nothing to do.
                await run_shielded(
                    asyncio.to_thread(finish_run, run, episodes, push_state)
                )
    except BaseException as e:
        await asyncio.to_thread(abort_run, run, e, push_state)
        raise
    return episodes
