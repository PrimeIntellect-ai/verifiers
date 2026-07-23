"""The eval runner: fan episodes out with bounded concurrency."""

import asyncio
import contextlib
import logging
import time

from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.output import (
    append_episode,
    append_trace,
    output_path,
    save_config,
)
from verifiers.v1.env import Env, RunSlot
from verifiers.v1.episode import Episode
from verifiers.v1.trace import EvalRunInfo
from verifiers.v1.utils.sampling import sample

logger = logging.getLogger(__name__)


async def run_eval(env: Env, config: EvalConfig) -> list[Episode]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    client = resolve_client(config.client)
    tasks = env.taskset.select(config.num_tasks, config.shuffle)
    ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    out = output_path(config)
    # One (task, rollouts-to-run) pair per selected task; resume shrinks the counts.
    plan = [(task, config.num_rollouts) for task in tasks]
    # Kept on-disk rollouts rejoin the run as finished episodes; only owed ones re-run.
    finished: list[Episode] = []
    if config.resume is not None:
        keys = [
            resume.task_key(t.data.model_dump(mode="json", exclude_none=True))
            for t in tasks
        ]
        finished, owed = resume.load(out, keys, config.num_rollouts, env.complete)
        if not owed:  # already complete - report it and exit successfully
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        counts = resume.distribute(keys, owed, config.num_rollouts)
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

    async def on_complete(episode: Episode) -> None:
        for trace in episode.traces:
            trace.stamp(EvalRunInfo(id=config.uuid))
        await append_episode(out, episode, write_lock)

    # Serving resources (shared tool servers, interception) come up once for the
    # run; plan slots inside so the env's agents borrow them.
    async with env.serving():
        planned = [slot for task, n in plan for slot in env.slots(task, n=n)]
        slots = [RunSlot.finished(episode) for episode in finished] + planned
        push_state = None
        if config.push and config.rich:
            from verifiers.v1.push import PushState

            push_state = PushState()
        display = (
            dashboard(slots, config, start, push=push_state)
            if config.rich
            else contextlib.nullcontext()
        )
        async with display:
            results = await asyncio.gather(
                *(env.run_slot(slot, ctx, semaphore, on_complete) for slot in planned)
            )
            episodes = finished + list(results)
            if (
                push_state is not None
            ):  # upload off the event loop so the view keeps refreshing
                from verifiers.v1.push import push_traces

                push_state.started = True
                await asyncio.to_thread(push_traces, episodes, config, push_state)
    await client.close()
    return episodes


async def run_eval_server(config: EvalConfig) -> list[Episode]:
    """Run evaluation through the env-server worker pool."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.utils.logging import setup_logging
    from verifiers.v1.configs.env import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env

    legacy = config.is_legacy
    server_kwargs = (
        {
            "env_id": config.id,
            "env_args": config.args,
            "extra_env_kwargs": config.extra_env_kwargs,
        }
        if legacy
        else {"config_data": env_config_data(config.env)}  # picklable across the spawn
    )
    tasks = []
    if not legacy:
        from verifiers.v1.loaders import load_taskset

        # The client owns the taskset: load it here, once — the server (and its pool
        # workers) never load data, they rebuild each dispatched task from its request.
        tasks = load_taskset(config.env.taskset).select(
            config.num_tasks, config.shuffle
        )
    # Spawned processes inherit no logging — hand them the main process's setup so
    # their rollout logs land in the output dir.
    level = "DEBUG" if config.verbose else "INFO"
    log_file = str(output_path(config) / "eval.log")
    mpctx = mp.get_context("spawn")
    address_queue: mp.Queue = mpctx.Queue()
    # Death pipe: serve_env self-terminates if this process dies abruptly — we keep
    # parent_conn, whose close (even on our SIGKILL) signals the child's watch.
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
        # A v1 run dispatches — and resumes — tasks by content: the client owns them,
        # and `resume.task_key` is their identity. Only the legacy bridge is addressed
        # by dataset row (its dataset lives server-side, reported via `info`), and
        # only a legacy env group-scores; a v1 env scores siblings in its own rollout.
        if legacy:
            info = await client.info()
            group_scored = info.requires_group_scoring
            idxs = sample(list(range(info.num_tasks)), config.shuffle, config.num_tasks)
            plan = [({"task_idx": idx}, config.num_rollouts) for idx in idxs]
        else:
            group_scored = False
            plan = [
                ({"task_data": task.data.model_dump(mode="json")}, config.num_rollouts)
                for task in tasks
            ]
        out = output_path(config)
        finished: list[Episode] = []
        if config.resume is not None:
            if legacy:
                # A group is served and scored together, so a partially-kept task
                # redoes as a whole group — whole_task drops its kept rows.
                finished, owed = resume.load(
                    out,
                    idxs,
                    config.num_rollouts,
                    whole_task=group_scored,
                    key_of=lambda data: data.get("idx"),
                )
                counts = resume.distribute(idxs, owed, config.num_rollouts)
            else:
                keys = [
                    resume.task_key(t.data.model_dump(mode="json", exclude_none=True))
                    for t in tasks
                ]
                finished, owed = resume.load(out, keys, config.num_rollouts)
                counts = resume.distribute(keys, owed, config.num_rollouts)
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
                config.pool.type,
                config.model,
            )
        logger.info("results: %s", out)
        request_concurrency = config.max_concurrent
        if request_concurrency and group_scored:
            # (legacy only) max_concurrent bounds rollouts, not requests; a group is
            # indivisible, so one oversized group must still be allowed to run.
            request_concurrency = max(1, request_concurrency // config.num_rollouts)
        semaphore = (
            asyncio.Semaphore(request_concurrency) if request_concurrency else None
        )
        write_lock = asyncio.Lock()

        async def run_group_unit(idx: int) -> list[Episode]:
            async with semaphore or contextlib.nullcontext():
                traces = await client.run_group(
                    task_idx=idx,
                    n=config.num_rollouts,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            records = []
            for trace in traces:
                trace.stamp(EvalRunInfo(id=config.uuid))
                await append_trace(out, trace, write_lock, env=config.env_id)
                records.append(Episode.of(trace))
            return records

        async def run_unit(payload: dict) -> list[Episode]:
            async with semaphore or contextlib.nullcontext():
                episode = await client.run(
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                    **payload,
                )
            for trace in episode.traces:
                trace.stamp(EvalRunInfo(id=config.uuid))
            await append_episode(out, episode, write_lock)
            return [episode]

        # A group-scored legacy task runs its rollouts together (one `run_group`
        # request, one worker); otherwise each rollout is its own `run` request,
        # dispatched least-busy across workers.
        units = (
            [run_group_unit(payload["task_idx"]) for payload, _ in plan]
            if group_scored
            else [run_unit(payload) for payload, n in plan for _ in range(n)]
        )
        results = await asyncio.gather(*units)
        await client.close()
        return finished + [record for unit in results for record in unit]
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
        with contextlib.suppress(Exception):
            parent_conn.close()
