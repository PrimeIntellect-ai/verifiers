"""The replay entrypoint: `uv run replay <output-dir> [options] [@ file.toml]`.

Offline sibling of `eval`: loads a finished run's saved traces and re-runs **scoring only**, no
runtime. The run's own `config.toml` is the base; CLI flags / `@ file.toml` layer on top (e.g. a
different judge model). Signals needing a `runtime` (in-sandbox verifiers like a SWE `solved`
reward) are skipped and left as the source recorded them; config-plugged judges and trace-only
`@reward`/`@metric`s re-run. `--num-rescores`/`-r` re-scores each trace N times to sample judge
variance. Results go to a fresh output dir, so the source run is never overwritten.
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
    # `WireTask` reads any taskset's saved task without a runtime or its Task type (see WireTask).
    traces = read_traces(source, Trace[WireTask, state_cls(type(taskset))])
    if config.num_traces is not None:
        traces = traces[: config.num_traces]
    # `num_rescores` re-scores each trace that many times, each on its own copy.
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
            st.start, st.state = time.time(), "running"
            # Clear prior scores so a changed/removed judge leaves no stale (double-summed) entry.
            if isinstance(trace.info, dict):
                trace.info.pop("judge", None)
            trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
            try:
                await taskset.score(trace)
                st.state, st.detail = "scored", f"reward {trace.reward:.3f}"
            except Exception as exc:
                st.state, st.detail = "error", type(exc).__name__
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
