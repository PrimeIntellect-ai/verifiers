"""The replay entrypoint: `uv run replay <output-dir> [options] [@ file.toml]`.

Offline sibling of `eval`: loads a finished run's saved traces and re-runs **scoring only**, no
runtime. The run's own `config.toml` is the base; CLI flags / `@ file.toml` layer on top (e.g. a
different judge model). Signals needing a `runtime` (in-sandbox verifiers like a SWE `solved`
reward) are skipped and left as the source recorded them — group rewards likewise (no group
context offline); the config's judges and trace-only `@reward`/`@metric`s re-run (judges are
config, so the layered replay config decides exactly what judges). Metrics the
re-score doesn't re-record (harness metrics, a skipped signal's direct writes) keep their source
values. Rollouts that errored during generation (`stop_condition == "error"`)
were never scored by eval, so they're copied through unchanged rather than re-scored on a broken
transcript. `--num-rescores`/`-r` re-scores each trace N times to sample judge variance. Results go
to a fresh output dir, so the source run is never overwritten.
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
from verifiers.v1.task import Task, WireTaskData, task_data_cls
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
    task_cls = vf.task_type(config.taskset.id)
    data_cls = task_data_cls(task_cls)
    # `WireTaskData` reads any taskset's saved task without a runtime or its Task type (see WireTaskData).
    traces = read_traces(source, Trace[WireTaskData, state_cls(task_cls)])
    if config.num_traces is not None:
        traces = traces[: config.num_traces]
    # `trace.task` is pure data; re-scoring with the taskset's behavior needs the declared
    # `TaskData` type — which every saved row is (one task type per taskset; its `load()`
    # constructs it). The rebuilt row feeds ONLY the behavior wrapper at score time —
    # `trace.task` keeps its wire form, because the trace persists through the
    # `Trace[WireTaskData, ...]` schema it was read as: a sibling `TaskData` assigned onto
    # it would have its subclass fields silently dropped from the replay's own output
    # (they're real fields, not `model_extra`). A row that can't be rebuilt from the wire
    # (a load-time-only field excluded from serialization, like harbor's `task_dir`) is
    # scored by the base `Task` on the wire row (judges + base signals only; the
    # subclass's own `@reward`s don't run — runtime-dependent ones would be skipped
    # offline anyway).
    def rebuild(trace: Trace) -> vf.TaskData:
        try:
            return data_cls.model_validate(trace.task.model_dump())
        except Exception:
            logger.warning(
                "replay: can't rebuild %s from the saved task %s; re-scoring it as a "
                "plain WireTaskData (judges + base-task signals only)",
                data_cls.__name__,
                trace.task.idx,
                exc_info=True,
            )
            return trace.task

    rows = [rebuild(trace) for trace in traces]
    # Judges are config, not wire data: `Task.score` resolves them from the replay
    # config's `taskset.task.judges` (the source run's config.toml layered under any CLI
    # overrides) — so a re-tuned judge overrides the recorded run's and a newly-plugged
    # one joins, with no trace surgery.
    # `num_rescores` re-scores each trace that many times, each on its own copy.
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
    states = [ReplayProgress(idx=t.task.idx, name=t.task.name) for t, _ in work]
    lock = asyncio.Lock()

    async def rescore(trace: Trace, row: vf.TaskData, st: ReplayProgress) -> None:
        async with sem or contextlib.nullcontext():
            st.start = time.time()
            # A rollout that errored during generation was never scored by eval (its scoring phase
            # never ran), so re-scoring would grade a broken/partial transcript. Copy it through
            # unchanged, mirroring eval. A *scoring* error leaves `stop_condition` as the
            # generation outcome, so those still re-score (the offline-recovery case).
            if trace.stop_condition == "error":
                st.state, st.detail, st.end = "skipped", "rollout errored", time.time()
            else:
                st.state = "running"
                # Clear prior scores so a changed/removed judge leaves no stale (double-summed) entry.
                if isinstance(trace.info, dict):
                    trace.info.pop("judge", None)
                prior_rewards, prior_metrics = trace.rewards, trace.metrics
                trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
                # The behavior wrapper around the rebuilt row: the declared Task for a
                # rebuilt row, the base Task (judges + base signals) for a WireTaskData
                # fallback; the replay config's task subtree supplies the knobs.
                task = (task_cls if isinstance(row, data_cls) else Task)(
                    row, config=config.taskset.task
                )
                try:
                    # Pre-fill the runtime-only values the offline `score` can't recompute
                    # BEFORE scoring — so trace-only signals that read them (e.g. a
                    # `@reward` reading a runtime `@metric`'s entry) see them, and a failed
                    # re-score still persists them.
                    task.restore_offline(trace, prior_rewards, prior_metrics)
                    await task.score(trace)
                    st.state, st.detail = "scored", f"reward {trace.reward:.3f}"
                except Exception as exc:
                    st.state, st.detail = "error", type(exc).__name__
                    trace.capture_error(
                        exc
                    )  # record the re-scoring failure on the trace
                    if not config.rich:
                        logger.warning(
                            "replay: scoring failed for task %s",
                            trace.task.idx,
                            exc_info=True,
                        )
                # Source metrics the re-score didn't re-record survive wholesale — harness
                # metrics and direct `record_metric` writes by skipped runtime signals are
                # unattributable to a producer (`Task.restore_offline` covers only returned
                # keys). Fill-if-missing: everything re-recorded above keeps its fresh
                # value. Rewards get no such sweep — they stay attributed, so a removed
                # judge's entry is not resurrected into the reward sum.
                for name, value in prior_metrics.items():
                    trace.metrics.setdefault(name, value)
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
        await asyncio.gather(
            *(rescore(t, row, s) for (t, row), s in zip(work, states))
        )
    logger.info("replay: done in %.1fs -> %s", time.time() - start, out)
    return [t for t, _ in work]


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not argv or any(a in ("-h", "--help") for a in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(ReplayConfig)  # full, typed pydantic-config option help
        return
    source = Path(argv.pop(0))  # the finished run dir to replay
    config_path = source / CONFIG_FILE
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
