"""The replay entrypoint: `uv run replay <output-dir> [options] [@ file.toml]`.

Registered as the `replay` console script — the offline sibling of `eval`. Where `eval` runs a
model rollout per task and scores it in a live runtime, `replay` loads a finished run's saved
traces (`<output-dir>/results.jsonl`) and re-runs **scoring only**, with no runtime. Its base
config is the run's own `config.toml`, so CLI flags / `@ file.toml` layer on top exactly like
`eval` — e.g. re-judge with a different judge model or `--taskset.judges` batching.

The previous scores are cleared and re-scoring runs fresh, so the replay output holds only what it
produced: the config-plugged judges and trace-only `@reward`/`@metric`s. Each trace is scored on
its own (group scoring is an eval concern, skipped here); `--num-rollouts`/`-r` re-scores every
selected trace that many times (each on its own copy) to sample judge variance. Signals that
declare a `runtime` parameter (in-sandbox verifiers like a SWE `solved` reward) can't run
offline and are skipped; those and any harness metrics are therefore not carried into the replay
output — the source run retains them. `trace.state` is transient runtime scratch that is not
persisted, so a signal that reads it re-scores against a fresh empty state — the built-in judges
grade from the transcript and are unaffected, but a state-dependent `@reward`/`@metric` should not
be relied on under replay. Results are written to a fresh output dir (config.toml + results.jsonl),
like `eval`, so the original run is never overwritten.
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
from verifiers.v1.cli.dashboard.replay import replay_dashboard
from verifiers.v1.cli.dashboard.validate import TaskProgress
from verifiers.v1.cli.output import append_trace, read_traces, save_config, write_config
from verifiers.v1.configs.replay import ReplayConfig
from verifiers.v1.state import state_cls
from verifiers.v1.task import WireTask
from verifiers.v1.trace import Trace
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run replay <output-dir> [options] [@ file.toml]\n"
    "       re-score a finished run's saved traces (judges + trace-only signals; no runtime)"
)


def _narrow(config_path: Path) -> type[ReplayConfig]:
    """`ReplayConfig` with `taskset` narrowed to the saved run's taskset type, so the `cli()`
    parse stays typed and CLI overrides of taskset fields validate (mirrors eval/validate)."""
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
    """Where a replay writes: `--output-dir`, else a fresh `outputs/<taskset>--replay/<uuid>`."""
    return config.output_dir or Path("outputs") / f"{config.name}--replay" / config.uuid


async def run_replay(config: ReplayConfig, source: Path, out: Path) -> list[Trace]:
    logger.debug("replay config:\n%s", config.model_dump_json(indent=2))
    taskset = vf.load_taskset(config.taskset)
    # Type the task as `WireTask`: it preserves the taskset's extra fields but requires none of
    # the runtime-only ones (e.g. a Harbor task's `task_dir`, set during setup and absent from
    # saved traces), so replay reads any taskset's traces without a runtime or importing its Task
    # type. Judges read `task.prompt_text`; runtime-dependent `@reward`s are skipped anyway.
    traces = read_traces(source, Trace[WireTask, state_cls(type(taskset))])
    if config.num_traces is not None:
        traces = traces[: config.num_traces]
    # Replay scores each trace independently (no `@group_reward` — grouping is an eval concern).
    # `num_rollouts` re-scores every selected trace that many times, each on its own deep copy so
    # the gradings don't clobber each other — e.g. to measure judge variance over one trace.
    work = [t.model_copy(deep=True) for t in traces for _ in range(config.num_rollouts)]

    save_config(config, out)
    logger.info(
        "replay: re-scoring %d trace(s) x%d from %s -> %s",
        len(traces),
        config.num_rollouts,
        source,
        out,
    )
    start = time.time()

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    states = [TaskProgress(idx=t.task.idx, name=t.task.name) for t in work]
    lock = asyncio.Lock()

    async def rescore(trace: Trace, st: TaskProgress) -> None:
        async with sem or contextlib.nullcontext():
            st.start, st.state = time.time(), "running"
            # Re-score fresh: drop the saved judge records and every score replay recomputes, so a
            # changed/removed judge leaves no stale (double-summed) entry. Runtime-only rewards
            # (e.g. a SWE `solved`) and harness metrics aren't recomputed offline, so they are not
            # carried into the replay output — the source run keeps them.
            if isinstance(trace.info, dict):
                trace.info.pop("judge", None)
            trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
            try:
                await taskset.score(trace)
                st.state = "scored"
            except Exception as exc:
                st.state = "error"
                trace.capture_error(exc)  # record the re-scoring failure on the trace
                if not config.rich:
                    logger.warning(
                        "replay: scoring failed for task %s",
                        trace.task.idx,
                        exc_info=True,
                    )
            st.end = time.time()
        if not config.rich:
            logger.info(
                "idx=%s %s (%.1fs)", trace.task.idx, st.state, st.end - st.start
            )
        # Persist as each trace finishes (like eval), so an interrupted replay keeps its progress.
        await append_trace(out, trace, lock)

    display = (
        replay_dashboard(states, config.name, str(source), start)
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
    source = Path(argv.pop(0))  # the finished run dir to replay
    config_path = source / "config.toml"
    if not config_path.exists():
        raise SystemExit(f"{USAGE}\nno config.toml in {source}")

    # The saved run's config is the base; user flags / @ file.toml layer on top.
    layered = ["@", str(config_path), *argv]
    config_type = _narrow(config_path)
    sys.argv = [sys.argv[0], *layered]
    config = cli(config_type)
    # Honor a user-set output_dir (via -o or an @ file); drop the *source* run's own output_dir
    # so a replay always writes to a fresh dir and never back over the run it re-scores.
    source_out = (tomllib.loads(config_path.read_text())).get("output_dir")
    if config.output_dir is None or str(config.output_dir) == str(source_out):
        config = config.model_copy(update={"output_dir": None})

    out = output_dir(config)
    if (
        out.resolve() == source.resolve()
    ):  # never write back over the run being replayed
        raise SystemExit(
            f"replay: --output-dir must differ from the source run ({source}); "
            "refusing to overwrite it"
        )
    level = "DEBUG" if config.verbose else "INFO"
    if config.dry_run:  # resolve + validate, write the config, and exit (no re-scoring)
        setup_logging(level)
        logger.info("wrote config to %s", write_config(config, out))
        return

    log_file = str(
        out / "replay.log"
    )  # tee the run's logs to its output dir, like eval
    if config.rich:  # the dashboard owns the screen, so keep logs off the console
        setup_logging(level, log_file=log_file, console=False)
        logging.lastResort = None
    else:
        setup_logging(level, log_file=log_file, console=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    asyncio.run(run_replay(config, source, out))


if __name__ == "__main__":
    main()
