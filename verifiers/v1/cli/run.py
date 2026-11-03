"""The `vf-run` entrypoint: run a taskset's episodes in-process and save their traces.

Prints one summary line and exits non-zero when any episode is not ok, so CI can call
it directly. For full evaluations, use prime-rl's `uv run eval`.
"""

import asyncio
import os
import sys
from pathlib import Path

from pydantic_config import cli

from verifiers.v1.cli.output import output_path, save_config, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.cli.run import RunnerConfig
from verifiers.v1.episode import Episode
from verifiers.v1.trace import EvalRunInfo
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.loaders import load_environment
from verifiers.v1.utils.log_scope import get_logger
from verifiers.v1.utils.logging import setup_logging
from verifiers.v1.utils.trace_store import TRACES_FILE, append_episode

logger = get_logger(__name__)

USAGE = "usage: uv run vf-run [<taskset-id>] [--env.id <id>] [-m <model>] [-n <tasks>] [-r <rollouts>] [options] [@ file.toml]"


def ensure_fresh(out: Path) -> None:
    """Refuse a run directory that already holds episodes: a new run would truncate them."""
    traces_file = out / TRACES_FILE
    if traces_file.exists() and traces_file.stat().st_size > 0:
        raise FileExistsError(
            f"{out} already contains results - pick another --run.dir"
        )


async def run_episodes(config: RunnerConfig) -> list[Episode]:
    """Run `num_rollouts` episodes per selected task and append each to `traces.jsonl`
    as it completes."""
    env = load_environment(config.env)
    taskset = env.taskset.select(config.select)
    if not taskset.bounded:
        raise ValueError(
            f"{type(taskset).__name__} is infinite - bound the run with -n"
        )
    out = output_path(config)
    ensure_fresh(out)
    save_config(config, out, "run.json")
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    write_lock = asyncio.Lock()

    async def on_complete(episode: Episode) -> None:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        await append_episode(out, episode, write_lock)

    async with env.serving():
        rollouts: list[asyncio.Future[Episode]] = []
        try:
            # Scheduled inside the try: a lazy taskset can raise after earlier tasks started.
            for task in taskset:
                for slot in env.slots(task, config.num_rollouts):
                    rollouts.append(
                        asyncio.ensure_future(
                            env.run_slot(slot, ctx, semaphore, on_complete)
                        )
                    )
            return list(await asyncio.gather(*rollouts))
        except BaseException:
            # Unwind every sibling before `serving()` tears down the resources they use.
            for rollout in rollouts:
                rollout.cancel()
            await asyncio.gather(*rollouts, return_exceptions=True)
            raise


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))
    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(narrow_config(RunnerConfig, argv))
        return
    if (
        not extract_id(argv, "env.taskset")
        and not references_config_file(argv)
        and not any(a.startswith("--env.") for a in argv)
    ):
        raise SystemExit(USAGE)

    with plugin_errors():
        config_type = narrow_config(RunnerConfig, argv)
        sys.argv = [sys.argv[0], *argv]
        config = cli(config_type)
    setup_logging("DEBUG" if config.verbose else "INFO")
    out = output_path(config)
    try:
        ensure_fresh(out)
    except FileExistsError as e:
        raise SystemExit(str(e)) from e
    if config.dry_run:
        logger.info("wrote config to %s", write_config(config, out, "run.json"))
        return

    # The run identity: every process this run spawns inherits it.
    os.environ.setdefault("VF_RUN_ID", config.run.id)
    # A first Ctrl-C raises KeyboardInterrupt so each rollout's teardown still runs.
    install_interrupt()
    try:
        episodes = asyncio.run(run_episodes(config))
    except KeyboardInterrupt:
        raise SystemExit(130)
    traces = [trace for episode in episodes for trace in episode.traces]
    ok = sum(episode.ok for episode in episodes)
    reward = sum(trace.reward for trace in traces) / len(traces) if traces else 0.0
    print(f"{ok}/{len(episodes)} episodes ok, mean reward {reward:.3f}: {out}")
    if ok < len(episodes):
        raise SystemExit(1)
