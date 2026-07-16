"""The eval runner: fan env-rollouts out with bounded concurrency."""

import asyncio
import contextlib
import logging
import time

from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.output import (
    append_record,
    append_trace,
    output_path,
    save_config,
)
from verifiers.v1.env import Environment, RunSlot
from verifiers.v1.trace import RolloutRecord, Trace
from verifiers.v1.utils.sampling import sample

logger = logging.getLogger(__name__)


async def run_eval(env: Environment, config: EvalConfig) -> list[Trace]:
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    client = resolve_client(config.client)
    tasks = env.taskset.select(config.num_tasks, config.shuffle)
    ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
    # `num_rollouts` independent env-rollouts per task; the shared semaphore bounds how
    # many are in flight at once.
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    out = output_path(config)
    # Write config.toml up front, then persist each record as it completes (so the results are
    # durable mid-run, not only at the end). One lock serializes worker-thread appends from
    # concurrent rollouts while keeping large trace serialization off the event loop.
    owed: dict[str, int] | None = None
    # On resume, the kept (good) on-disk rollouts rejoin the run as finished records: displayed,
    # returned, pushed, and printed alongside this session's, with only the owed rollouts re-run
    # — so the resumed run is indistinguishable from one that was never interrupted.
    finished: list[RolloutRecord] = []
    if config.resume is not None:
        finished, owed = resume.load(
            out, [t.data.idx for t in tasks], config.num_rollouts, env.complete
        )
        if not owed:  # already complete - report it and exit successfully
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        tasks = [task for task in tasks if owed.get(task.data.idx)]
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

    async def on_complete(record: RolloutRecord) -> None:
        await append_record(out, record, write_lock)

    # Shared tool servers (if any) come up once here and their URLs flow into every rollout
    # (non-shared ones start per rollout); the interception comes up here too, so
    # concurrent rollouts share its servers + tunnels rather than one each. Plan slots
    # inside `serving` so the env's agents borrow those resources when the runs build them.
    async with env.serving():
        planned = [
            slot
            for task in tasks
            for slot in env.slots(
                task, n=owed[task.data.idx] if owed else config.num_rollouts
            )
        ]
        slots = [RunSlot.finished(record) for record in finished] + planned
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
            records = finished + list(results)
            traces = [trace for record in records for trace in record.traces]
            if (
                push_state is not None
            ):  # upload off the event loop so the view keeps refreshing
                from verifiers.v1.push import push_traces

                push_state.started = True
                await asyncio.to_thread(push_traces, traces, config, push_state)
    await client.close()
    return traces


async def run_eval_server(config: EvalConfig) -> list[Trace]:
    """Run evaluation through the env-server worker pool."""
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
        # Only a legacy (v0) env can require group scoring — v1 envs score
        # sibling-dependent signals inside their own rollout (`Environment.score`).
        group_scored = info.requires_group_scoring
        if info.num_tasks is None:  # infinite taskset - the run must be bounded
            if config.num_tasks is None:
                raise ValueError(
                    f"{config.env_id} is infinite - bound the run with -n/--num-tasks"
                )
            if config.shuffle:
                logger.warning(
                    "shuffle is a no-op on an infinite taskset - "
                    "taking the first %d generated tasks",
                    config.num_tasks,
                )
            idxs = list(range(config.num_tasks))
        else:
            idxs = sample(list(range(info.num_tasks)), config.shuffle, config.num_tasks)
        out = output_path(config)
        finished: list[Trace] = []
        if config.resume is not None:
            records, owed = resume.load(out, idxs, config.num_rollouts)
            finished = [trace for record in records for trace in record.traces]
            if not owed:  # already complete - report it and exit successfully
                print(resume.nothing_to_resume_msg(out, len(idxs), config.num_rollouts))
                raise SystemExit(0)
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
        if request_concurrency and group_scored:
            # (legacy only) max_concurrent is a rollout resource bound, not a request-
            # throughput target. A group is indivisible, so one oversized group must
            # still be allowed to run.
            request_concurrency = max(1, request_concurrency // config.num_rollouts)
        semaphore = (
            asyncio.Semaphore(request_concurrency) if request_concurrency else None
        )
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
                await append_trace(out, trace, write_lock, env=config.env_id)
            return traces

        async def run_rollout_unit(idx: int) -> list[Trace]:
            async with semaphore or contextlib.nullcontext():
                record = await client.run_rollout(
                    task_idx=idx,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            await append_record(out, record, write_lock)
            return list(record.traces)

        # A group-scored legacy task must run its rollouts together (cross-rollout
        # scoring) → one `run_group` request per task (one worker); otherwise rollouts
        # are independent → one `run_rollout` request each, which the broker
        # round-robins (least-busy) across workers — mirrors the prime-rl dispatcher.
        units = (
            [run_group_unit(i) for i in idxs]
            if group_scored
            else [run_rollout_unit(i) for i in idxs for _ in range(owed[i])]
        )
        results = await asyncio.gather(*units)
        await client.close()
        return finished + [trace for unit_traces in results for trace in unit_traces]
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
        with contextlib.suppress(Exception):
            parent_conn.close()
