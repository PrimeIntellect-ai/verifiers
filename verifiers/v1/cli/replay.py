"""Offline scoring for saved traces.

Replay runs trace-only handlers and judges, preserving runtime and group scores recorded
by the source run. Its saved config is the base for replay-specific overrides.
"""

import asyncio
import contextlib
import logging
import signal
import sys
import time
import tomllib
from pathlib import Path

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.dashboard.replay import ReplayProgress, replay_dashboard
from verifiers.v1.cli.output import (
    CONFIG_FILE,
    append_trace,
    read_traces,
    save_config,
    write_config,
)
from verifiers.v1.configs.replay import ReplayConfig
from verifiers.v1.state import state_cls
from verifiers.v1.task import WireTaskData, task_data_cls
from verifiers.v1.trace import Trace
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run replay <output-dir> [options] [@ file.toml]\n"
    "       re-score a finished run's saved traces (judges + trace-only signals; no runtime)"
)


def _narrow(config_path: Path) -> type[ReplayConfig]:
    """Narrow replay config to the saved taskset's config type."""
    data = tomllib.loads(config_path.read_text())
    taskset_id = (data.get("taskset") or {}).get("id")
    if not taskset_id:
        return ReplayConfig
    ftype = vf.taskset_config_type(taskset_id)
    return type(
        "ReplayConfig",
        (ReplayConfig,),
        {"__annotations__": {"taskset": ftype}, "taskset": ftype(id=taskset_id)},
    )


def output_dir(config: ReplayConfig) -> Path:
    """Resolve a replay's output directory."""
    return config.output_dir or Path("outputs") / f"{config.name}--replay" / config.uuid


async def run_replay(config: ReplayConfig, source: Path, out: Path) -> list[Trace]:
    logger.debug("replay config:\n%s", config.model_dump_json(indent=2))
    task_cls = vf.task_type(config.taskset.id)
    data_cls = task_data_cls(task_cls)
    # Rebuild wire data as concrete TaskData so subclass scoring sees typed fields.
    traces = read_traces(source, Trace[WireTaskData, state_cls(task_cls)])
    if config.num_traces is not None:
        traces = traces[: config.num_traces]
    for trace in traces:
        trace.task = data_cls.model_validate(trace.task.model_dump())
    work = [t.model_copy(deep=True) for t in traces for _ in range(config.num_rescores)]

    save_config(config, out)
    logger.info(
        "replay: re-scoring %d trace(s) x%d from %s -> %s",
        len(traces),
        config.num_rescores,
        source,
        out,
    )
    start = time.time()

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    states = [ReplayProgress(idx=t.task.idx, name=t.task.name) for t in work]
    lock = asyncio.Lock()

    async def rescore(trace: Trace, st: ReplayProgress) -> None:
        async with sem or contextlib.nullcontext():
            st.start = time.time()
            # Generation failures have no complete transcript to score.
            if trace.stop_condition == "error":
                st.state, st.detail, st.end = "skipped", "rollout errored", time.time()
            else:
                st.state = "running"
                # Recompute trace-only scores after restoring runtime-only values below.
                trace.info.pop("judge", None)
                prior_rewards, prior_metrics = trace.rewards, trace.metrics
                trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
                task = task_cls(trace.task, config=config.taskset.task)
                try:
                    # Trace-only handlers may depend on restored runtime metrics.
                    task.restore_offline(trace, prior_rewards, prior_metrics)
                    await task.score(trace)
                    st.state, st.detail = "scored", f"reward {trace.reward:.3f}"
                except Exception as exc:
                    st.state, st.detail = "error", type(exc).__name__
                    trace.capture_error(exc)
                    if not config.rich:
                        logger.warning(
                            "replay: scoring failed for task %s",
                            trace.task.idx,
                            exc_info=True,
                        )
                # Harness and direct-write metrics have no attributable producer.
                for name, value in prior_metrics.items():
                    trace.metrics.setdefault(name, value)
                st.end = time.time()
        if not config.rich:
            logger.info(
                "idx=%s %s (%.1fs)", trace.task.idx, st.state, st.end - st.start
            )
        # Persist each result as it lands so an interrupted replay keeps its progress.
        await append_trace(out, trace, lock)

    display = (
        replay_dashboard(states, config.name, str(source), str(out), start)
        if config.rich
        else contextlib.nullcontext()
    )
    async with display:
        await asyncio.gather(*(rescore(t, s) for t, s in zip(work, states)))
    logger.info("replay: done in %.1fs -> %s", time.time() - start, out)
    return work


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not argv or any(a in ("-h", "--help") for a in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(ReplayConfig)  # full, typed pydantic-config option help
        return
    source = Path(argv.pop(0))
    config_path = source / CONFIG_FILE
    if not config_path.exists():
        raise SystemExit(f"{USAGE}\nno config.toml in {source}")

    layered = ["@", str(config_path), *argv]
    config_type = _narrow(config_path)
    sys.argv = [sys.argv[0], *layered]
    config = cli(config_type)
    source_out = tomllib.loads(config_path.read_text()).get("output_dir")
    # Clear the source run's output_dir unless the user overrode it.
    if config.output_dir is None or str(config.output_dir) == str(source_out):
        config = config.model_copy(update={"output_dir": None})

    out = output_dir(config)
    if out.resolve() == source.resolve():
        raise SystemExit(
            f"replay: --output-dir must differ from the source run ({source}); "
            "refusing to overwrite it"
        )
    level = "DEBUG" if config.verbose else "INFO"
    if config.dry_run:
        setup_logging(level)
        logger.info("wrote config to %s", write_config(config, out))
        return

    log_file = str(out / "replay.log")
    if config.rich:
        setup_logging(level, log_file=log_file, console=False)
        logging.lastResort = None
    else:
        setup_logging(level, log_file=log_file, console=True)
    # Translate SIGTERM so async finally blocks still tear down their resources.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    asyncio.run(run_replay(config, source, out))


if __name__ == "__main__":
    main()
