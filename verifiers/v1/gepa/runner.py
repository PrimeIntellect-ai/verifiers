"""The GEPA runner: split tasks, drive `gepa.api.optimize` against a `GEPAv1Adapter`, save
results. Async like the sibling runners (`run_eval`) — it holds `env.serving()` open with an
`async with` and runs the blocking `optimize()` in a worker thread (`asyncio.to_thread`) so the
loop stays free to drive the adapter's rollouts (see `GEPAv1Adapter`)."""

import asyncio
import logging

from gepa.api import optimize
from gepa.core.result import GEPAResult

from verifiers.v1.cli.output import append_trace, output_path, save_config
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.env import Environment
from verifiers.v1.gepa.adapter import GEPAv1Adapter
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.gepa.dataset import resolve_gepa_seed_prompt, split_tasks
from verifiers.v1.gepa.reflection import build_reflection_lm
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class _GEPALog:
    """Forwards GEPA's optimizer log lines to v1 logging. GEPA's `LoggerProtocol` needs only
    `.log(str)`, so its per-iteration progress rides the same loguru setup as the rest of the
    v1 CLIs instead of a bespoke dashboard."""

    def log(self, message: str) -> None:
        logger.info(message)


async def run_gepa(env: Environment, config: GEPAConfig) -> GEPAResult:
    logger.info("gepa config:\n%s", config.model_dump_json(indent=2))
    all_tasks = env.taskset.load()
    train_tasks, val_tasks = split_tasks(
        all_tasks, config.num_train, config.num_val, config.shuffle
    )
    selected_tasks = [*train_tasks, *val_tasks]
    # Seed from the tasks GEPA actually evaluates (train ∪ val), not the full pre-split pool —
    # a taskset with per-task system prompts could otherwise seed from a task in neither split.
    seed_prompt = resolve_gepa_seed_prompt(selected_tasks, config.initial_prompt)
    tasks_by_idx = {task.data.idx: task for task in selected_tasks}

    run_dir = output_path(config) if config.save_results else None
    if run_dir is not None:
        save_config(
            config, run_dir
        )  # config.toml + a fresh results.jsonl (like run_eval)
        logger.info("results: %s", run_dir)

    client = resolve_client(config.client)
    ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
    reflection_lm = build_reflection_lm(config)
    try:
        # Shared serving (tool servers + interception pool) comes up once, like run_eval.
        async with env.serving():
            semaphore = (
                asyncio.Semaphore(config.max_concurrent)
                if config.max_concurrent
                else None
            )
            # Stream every rollout's trace to results.jsonl as it finalizes — the same persist
            # hook run_eval passes to Episode.run (each trace records its candidate prompt).
            write_lock = asyncio.Lock()

            async def on_complete(trace: Trace) -> None:
                if run_dir is not None:
                    await append_trace(run_dir, trace, write_lock)

            adapter = GEPAv1Adapter(
                env=env,
                ctx=ctx,
                tasks=tasks_by_idx,
                loop=asyncio.get_running_loop(),
                semaphore=semaphore,
                on_complete=on_complete,
                state_columns=config.state_columns,
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
            # optimize() is synchronous and blocking; run it off-thread so this loop stays
            # free to serve the rollouts the adapter submits back to it.
            result = await asyncio.to_thread(optimize, **optimize_kwargs)
    finally:
        await client.close()
    return result
