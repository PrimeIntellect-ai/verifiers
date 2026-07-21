"""Offline scoring for saved traces.

Replay clears each trace's scores and recomputes everything computable from the saved
transcript — trace-only handlers plus the layered config's judges. Runtime-requiring
signals don't run offline, so a replay carries offline scores only; the source run keeps
the runtime-recorded values. Its saved config is the base for replay-specific overrides.
Multi-agent runs don't support replay: re-scoring is per trace, and the env's
cross-trace `score()` can't run offline.
"""

import asyncio
import contextlib
import logging
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
from verifiers.v1.task import Task, WireTaskData, task_data_cls
from verifiers.v1.trace import Trace
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run replay <output-dir> [options] [@ file.toml]\n"
    "       re-score a finished run's saved traces (judges + trace-only signals; no runtime)"
)


def _narrow(config_path: Path) -> type[ReplayConfig]:
    """Narrow replay config to the saved taskset's config type. The source may be
    an eval run (taskset on the [env] block) or an earlier replay (root taskset)."""
    data = tomllib.loads(config_path.read_text())
    taskset = data.get("taskset") or (data.get("env") or {}).get("taskset") or {}
    taskset_id = taskset.get("id")
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
    # Refuse multi-agent up front, from the SOURCE run's saved config — the replay
    # layer's taskset is overridable and must not decide what the run was.
    # Everything below may assume plain single-agent episodes.
    saved = tomllib.loads((source / CONFIG_FILE).read_text())
    saved_env = saved.get("env") or {}
    saved_taskset = saved.get("taskset") or saved_env.get("taskset") or {}
    env_cls = vf.environment_class(
        saved_taskset.get("id") or "", saved_env.get("id") or ""
    )
    if not issubclass(env_cls, vf.SingleAgentEnv):
        raise SystemExit(
            f"replay: {source} is a multi-agent run ({env_cls.__name__}); offline "
            "re-scoring runs per trace and would drop the env's cross-trace "
            "score() — multi-agent runs don't support replay"
        )
    task_cls = vf.task_type(config.taskset.id)
    data_cls = task_data_cls(task_cls)
    # `WireTaskData` reads any taskset's saved task without importing its Task type.
    # Each trace keeps its own episode stamp through the copy, so the replay output
    # stays linkable to the source run's episodes.
    traces = read_traces(source, Trace[WireTaskData, state_cls(task_cls)])
    if config.num_traces is not None:
        traces = traces[: config.num_traces]

    # Re-scoring with the taskset's behavior needs the declared `TaskData` type, but
    # the rebuilt row feeds ONLY the behavior wrapper: `trace.task` keeps its wire
    # form — the trace persists through the `Trace[WireTaskData, ...]` schema it was
    # read as, and a sibling `TaskData` assigned onto it would have its subclass
    # fields silently dropped from the replay's own output. A row that can't be
    # rebuilt (a load-time-only field excluded from serialization) is scored by the
    # base `Task` on the wire row (judges + base signals only).
    def rebuild(trace: Trace) -> vf.TaskData:
        if trace.task.type != task_cls.__name__:
            logger.warning(
                "replay: task %s was produced by %s but re-scores as %s (the taskset's declared type)",
                trace.task.data.idx,
                trace.task.type,
                task_cls.__name__,
            )
        try:
            return data_cls.model_validate(trace.task.data.model_dump())
        except Exception:
            logger.warning(
                "replay: can't rebuild %s from the saved task %s; re-scoring it as a "
                "plain WireTaskData (judges + base-task signals only)",
                data_cls.__name__,
                trace.task.data.idx,
                exc_info=True,
            )
            return trace.task.data

    rows = [rebuild(trace) for trace in traces]
    # Judges are config, not wire data: `Task.score` resolves them from the replay
    # config's `taskset.task.judges`, so a re-tuned judge overrides the recorded
    # run's with no trace surgery. `num_rescores` copies each trace per re-score.
    work = [
        (t.model_copy(deep=True), row)
        for t, row in zip(traces, rows)
        for _ in range(config.num_rescores)
    ]

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
    states = [
        ReplayProgress(idx=t.task.data.idx, name=t.task.data.name) for t, _ in work
    ]
    lock = asyncio.Lock()

    async def rescore(trace: Trace, row: vf.TaskData, st: ReplayProgress) -> None:
        async with sem or contextlib.nullcontext():
            st.start = time.time()
            # Generation failures have no complete transcript to score.
            if trace.stop_condition == "error":
                st.state, st.detail, st.end = "skipped", "rollout errored", time.time()
            else:
                st.state = "running"
                # Clear the recorded scores; only offline-recomputable ones come back.
                trace.info.pop("judge", None)
                trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
                # The declared Task for a rebuilt row, the base Task for the
                # WireTaskData fallback.
                task = (task_cls if isinstance(row, data_cls) else Task)(
                    row, config=config.taskset.task
                )
                try:
                    await task.score(trace)
                    st.state, st.detail = "scored", f"reward {trace.reward:.3f}"
                except Exception as exc:
                    st.state, st.detail = "error", type(exc).__name__
                    trace.capture_error(exc)
                    if not config.rich:
                        logger.warning(
                            "replay: scoring failed for task %s",
                            trace.task.data.idx,
                            exc_info=True,
                        )
                st.end = time.time()
        if not config.rich:
            logger.info(
                "idx=%s %s (%.1fs)", trace.task.data.idx, st.state, st.end - st.start
            )
        # Persist each result as it lands so an interrupted replay keeps its progress.
        await append_trace(out, trace, lock)

    display = (
        replay_dashboard(states, config.name, str(source), str(out), start)
        if config.rich
        else contextlib.nullcontext()
    )
    async with display:
        await asyncio.gather(*(rescore(t, row, s) for (t, row), s in zip(work, states)))
    logger.info("replay: done in %.1fs -> %s", time.time() - start, out)
    return [t for t, _ in work]


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
            f"replay: --output-dir must differ from the source run ({source}); refusing to overwrite it"
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
    # Graceful shutdown: first Ctrl-C/SIGTERM unwinds the scoring teardown `finally`;
    # a second is swallowed so it can't orphan resources mid-cleanup.
    install_interrupt()
    asyncio.run(run_replay(config, source, out))


if __name__ == "__main__":
    main()
