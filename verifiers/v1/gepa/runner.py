"""The GEPA runner: split tasks, drive `gepa.api.optimize` against a `GEPAAdapter`, save
results. Unlike the async sibling runners, `optimize()` is synchronous and blocking, so it runs
on the main thread and it manages one event loop by hand: `runner.serving()` is entered on
it once, each rollout batch runs via `loop.run_until_complete` (see `GEPAAdapter.evaluate`),
and everything is torn down in `finally`. Mirrors how v0 vf-gepa bridged sync GEPA to async
rollouts — a Ctrl-C raises on the main thread inside `optimize()` and unwinds through teardown."""

import asyncio
import logging

from gepa.api import optimize
from gepa.core.result import GEPAResult

from verifiers.v1.cli.output import append_trace, output_path, save_config
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.gepa.adapter import GEPAAdapter
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.gepa.dataset import (
    reject_group_reward_tasksets,
    resolve_gepa_seed_prompt,
    split_tasks,
)
from verifiers.v1.gepa.reflection import build_reflection_lm
from verifiers.v1.topology import TopologyRunner
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class _GEPALog:
    """Forwards GEPA's optimizer log lines to v1 logging. GEPA's `LoggerProtocol` needs only
    `.log(str)`, so its per-iteration progress rides the same loguru setup as the rest of the
    v1 CLIs instead of a bespoke dashboard."""

    def log(self, message: str) -> None:
        logger.info(message)


def run_gepa(runner: TopologyRunner, config: GEPAConfig) -> GEPAResult:
    logger.info("gepa config:\n%s", config.model_dump_json(indent=2))
    all_tasks = runner.select_tasks(config.num_train + config.num_val, config.shuffle)
    train_tasks, val_tasks = split_tasks(all_tasks, config.num_train, config.num_val)
    selected_tasks = [*train_tasks, *val_tasks]
    reject_group_reward_tasksets(
        selected_tasks
    )  # GEPA scores n=1 per task; groups need >=2
    # Seed from the tasks GEPA actually evaluates (train ∪ val), not the full pre-split pool —
    # a taskset with per-task system prompts could otherwise seed from a task in neither split.
    seed_prompt = resolve_gepa_seed_prompt(selected_tasks, config.initial_prompt)
    tasks_by_idx = {task.data.idx: task for task in selected_tasks}

    run_dir = output_path(config) if config.save_results else None
    if run_dir is not None:
        save_config(
            config, run_dir
        )  # config.toml + a fresh traces.jsonl (like run_eval)
        logger.info("results: %s", run_dir)

    # optimize() is synchronous and blocking, so it drives the run from this (main) thread. We
    # own one event loop: `runner.serving()` (shared tool servers + interception, built once
    # like run_eval) is entered on it, each rollout batch runs on it via `loop.run_until_complete`
    # (GEPAAdapter.evaluate), and it's all torn down in `finally`. Keeping optimize() on the
    # main thread means a Ctrl-C raises straight through it into this teardown.
    loop = asyncio.new_event_loop()
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    # Stream every rollout's trace to traces.jsonl as it finalizes — the same persist hook
    # run_eval passes to Episode.run (each trace records its candidate prompt).
    write_lock = asyncio.Lock()

    async def on_complete(trace: Trace) -> None:
        if run_dir is not None:
            await append_trace(run_dir, trace, write_lock)

    # The client opens an httpx pool at construction, so build it inside the try that closes it —
    # a failure while building ctx/reflection_lm must not leak the pool.
    client = None
    try:
        client = resolve_client(config.client)
        ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
        reflection_lm = build_reflection_lm(config)
        serving = runner.serving()
        loop.run_until_complete(serving.__aenter__())
        # Pair __aexit__ with a *successful* __aenter__: this inner block is reached only once
        # serving is up, so a startup failure propagates as-is instead of being masked by
        # tearing down a context that never entered.
        try:
            adapter = GEPAAdapter(
                runner=runner,
                ctx=ctx,
                tasks=tasks_by_idx,
                loop=loop,
                semaphore=semaphore,
                on_complete=on_complete,
                reflection_columns=config.reflection_columns,
            )
            optimize_kwargs: dict = dict(
                seed_candidate={"system_prompt": seed_prompt},
                trainset=[task.data.idx for task in train_tasks],
                valset=[task.data.idx for task in val_tasks],
                adapter=adapter,
                reflection_lm=reflection_lm,
                max_metric_calls=config.max_metric_calls,
                reflection_minibatch_size=config.reflection_minibatch_size,
                run_dir=str(run_dir) if run_dir is not None else None,
                seed=config.seed,
                display_progress_bar=False,
                skip_perfect_score=config.perfect_score is not None,
                logger=_GEPALog(),
            )
            if config.perfect_score is not None:
                optimize_kwargs["perfect_score"] = config.perfect_score
            return optimize(**optimize_kwargs)
        finally:
            loop.run_until_complete(serving.__aexit__(None, None, None))
    finally:
        if client is not None:
            loop.run_until_complete(client.close())
        loop.close()
