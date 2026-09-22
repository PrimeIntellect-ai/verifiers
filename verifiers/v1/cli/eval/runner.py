"""The eval runner: fan episodes out with bounded concurrency.

Rollouts run in-process: the env comes up once in this process and each slot becomes
one episode through `env.run_slot`. The dashboard watches the same `RunSlot`s the env
fills with live traces. Serving an env to many consumers is prime-rl's job (its `eval`
runs env servers); this CLI is the quick local path.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar, cast

from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.eval.hint import PRIME_RL_HINT
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


async def gather_rollouts(rollouts: Iterable[Awaitable[T]]) -> list[T]:
    """`asyncio.gather`, but one rollout failing cancels the rest and waits for them
    to unwind, so nothing keeps uploading into a run the caller is already closing.
    Not a `TaskGroup`: that wraps errors in an `ExceptionGroup`, and `main` would no
    longer see a `KeyboardInterrupt` as Ctrl-C."""
    tasks = [asyncio.ensure_future(rollout) for rollout in rollouts]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        # return_exceptions: wait for every task, not just the first to cancel.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


RunSlotFn = Callable[[RunSlot], Awaitable[Episode]]
OnComplete = Callable[[Episode], Awaitable[None]]


@contextlib.asynccontextmanager
async def _in_process(
    env: Env,
    config: EvalConfig,
    episodes: asyncio.Semaphore | None,
    on_complete: OnComplete,
) -> AsyncIterator[RunSlotFn]:
    """Run slots in-process: serving resources (shared tool servers, interception)
    come up once for the run; the env's agents borrow them."""
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )

    async def run(slot: RunSlot) -> Episode:
        return await env.run_slot(slot, ctx, episodes, on_complete)

    async with env.serving():
        yield run


async def run_eval(config: EvalConfig) -> list[Episode]:
    from verifiers.v1.utils.loaders import load_environment

    env = load_environment(config.env)
    taskset = env.taskset
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
        # the env's own keep-verdict decides what resumes
        loaded, owed = resume.load(
            out,
            keys,
            config.num_rollouts,
            lambda episode: env.complete(cast(Episode, episode)),
        )
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
        logger.info(
            "running %dx%d rollouts on %s", len(plan), config.num_rollouts, config.model
        )
        logger.info(PRIME_RL_HINT)
    start = time.time()
    logger.info("results: %s", out)

    episodes = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    env._agent_runs = (
        asyncio.Semaphore(config.max_agent_runs) if config.max_agent_runs else None
    )

    write_lock = asyncio.Lock()
    push_state = PushState()

    # Opened before the first rollout so every episode streams as it lands.
    run = open_run(config, push_state, num_examples=len(tasks))
    # Resumed rollouts are part of this run too.
    log_episodes(run, finished)

    async def on_complete(episode: Episode) -> None:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        await append_episode(out, episode, write_lock)
        await asyncio.to_thread(log_episodes, run, [episode])

    # The run is closed out whatever breaks, env setup and teardown included.
    try:
        async with _in_process(env, config, episodes, on_complete) as run_slot:
            # the env's own slots: it fills their live traces as the rollouts run
            planned = [slot for task, n in plan for slot in env.slots(task, n)]
            slots = [RunSlot.finished(episode) for episode in finished] + planned
            display = (
                dashboard(slots, config, start, push=push_state)
                if config.rich is not None
                else contextlib.nullcontext()
            )
            async with display:
                results = await gather_rollouts(run_slot(slot) for slot in planned)
                completed = finished + list(results)
                # Drain and close out off the event loop so the view keeps refreshing.
                # Shielded: a Ctrl-C here must not cancel the close-out before the
                # worker picks it up (a cancelled executor item never runs), so it
                # runs to completion first and the interrupt is re-raised after —
                # by which point the run is finished and `abort_run` has nothing to do.
                await run_shielded(
                    asyncio.to_thread(finish_run, run, completed, push_state)
                )
    except BaseException as e:
        await asyncio.to_thread(abort_run, run, e, push_state)
        raise
    return completed
