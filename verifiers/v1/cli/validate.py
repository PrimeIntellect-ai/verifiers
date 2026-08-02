"""Model-free task-validation CLI."""

import asyncio
import contextlib
import json
import logging
import sys
import time
from uuid import uuid4

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.dashboard import TaskProgress, validate_dashboard
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_taskset_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.cli.validate_output import (
    LOG_FILE,
    SUMMARY_FILE,
    ResultRow,
    append_result,
    identity,
    load_results,
    load_resume_config,
    output_path,
    save_run,
    split_resume,
    summarize,
    validation_mode,
    write_summary,
)
from verifiers.v1.configs.cli.validate import ValidateConfig
from verifiers.v1.runtimes import make_runtime
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run validate [<taskset-id>] [--only-setup | --only-gold] "
    "[-o <output-dir>] [--runtime.type subprocess] [options] [@ file.toml]\n"
    "       uv run validate --resume <output-dir>\n"
    "       runs persisted gold and setup-only checks per task (no model)"
)


def _narrow(argv: list[str]) -> type[ValidateConfig]:
    """`ValidateConfig` with `taskset` narrowed to the config type of the id on the CLI — so
    the single `cli()` parse stays typed and `-h` renders the taskset's fields. Absent an id
    (a `@ file.toml` may carry it) the base type is left for the validator to resolve."""
    return narrow_taskset_config(ValidateConfig, extract_id(argv, "taskset"))


def _classify(valid: bool, exc: BaseException | None) -> str:
    if valid:
        return "valid"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if exc is not None:
        return "error"
    return "invalid"


def _row(
    task, mode: str, valid: bool, exc: BaseException | None, start: float
) -> ResultRow:
    return {
        "index": task.data.idx,
        "name": task.data.name,
        "mode": mode,
        "valid": bool(valid),
        "reason": _classify(valid, exc),
        "elapsed": round(time.time() - start, 2),
        "error": str(exc) if exc is not None else None,
        "error_type": type(exc).__name__ if exc is not None else None,
    }


async def _run_gold(task: Task, config: ValidateConfig) -> ResultRow:
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"validate-gold-{task.data.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.timeout.setup
        if config.timeout.setup is not None
        else task.data.timeout.setup
    )
    valid, exc = False, None
    try:
        trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            state=state_cls(type(task))(),
            # No agent runs here — the info only records the runtime policy.
            agent=vf.AgentInfo(
                config=vf.AgentConfig(runtime=config.runtime),
                name="validate",
                trainable=False,
            ),
        )
        await runtime.start()
        await asyncio.wait_for(
            invoke(task.setup, {"trace": trace, "runtime": runtime}),
            setup_timeout,
        )
        valid = await asyncio.wait_for(task.validate(runtime), config.timeout.total)
    except Exception as e:  # noqa: BLE001 - validation reports plugin failures per task
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning(
                "runtime teardown failed (task %s)", task.data.idx, exc_info=True
            )
    return _row(task, "gold", valid, exc, start)


async def _run_setup(task: Task, config: ValidateConfig) -> ResultRow:
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"validate-setup-{task.data.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.timeout.setup
        if config.timeout.setup is not None
        else task.data.timeout.setup
    )
    valid, exc = False, None
    try:
        trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            state=state_cls(type(task))(),
            # No agent runs here — the info only records the runtime policy.
            agent=vf.AgentInfo(
                config=vf.AgentConfig(runtime=config.runtime),
                name="validate",
                trainable=False,
            ),
        )
        await runtime.start()
        await asyncio.wait_for(
            invoke(task.setup, {"trace": trace, "runtime": runtime}),
            setup_timeout,
        )
        valid = True
    except Exception as e:  # noqa: BLE001 - validation reports plugin failures per task
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning(
                "runtime teardown failed (task %s)", task.data.idx, exc_info=True
            )
    return _row(task, "setup", valid, exc, start)


def _all_reason(rows: list[ResultRow]) -> str:
    if all(row["valid"] for row in rows):
        return "valid"
    reasons = {str(row["reason"]) for row in rows}
    if "error" in reasons:
        return "error"
    if "timeout" in reasons:
        return "timeout"
    return "invalid"


def _all_error(rows: list[ResultRow]) -> tuple[str | None, str | None]:
    failed = [row for row in rows if not row["valid"]]
    parts = [f"{row['mode']}: {row['error'] or row['reason']}" for row in failed]
    error_types = {str(row["error_type"]) for row in failed if row["error_type"]}
    return ("; ".join(parts) or None, "+".join(sorted(error_types)) or None)


async def _run_all(task: Task, config: ValidateConfig) -> ResultRow:
    start = time.time()
    gold = await _run_gold(task, config)
    setup = await _run_setup(task, config)
    rows = [gold, setup]
    error, error_type = _all_error(rows)
    return {
        "index": task.data.idx,
        "name": task.data.name,
        "mode": "all",
        "valid": all(row["valid"] for row in rows),
        "reason": _all_reason(rows),
        "elapsed": round(time.time() - start, 2),
        "error": error,
        "error_type": error_type,
        "gold": gold,
        "setup": setup,
    }


async def _validate_task(task: Task, config: ValidateConfig) -> ResultRow:
    if config.only_gold:
        return await _run_gold(task, config)
    if config.only_setup:
        return await _run_setup(task, config)
    return await _run_all(task, config)


async def run_validate(config: ValidateConfig) -> list[dict]:
    taskset = vf.load_taskset(config.taskset)
    if config.num_tasks is None and taskset.INFINITE:
        raise ValueError(
            f"{type(taskset).__name__} is infinite - bound the run with -n"
        )
    selected = taskset.shuffle() if config.shuffle else taskset
    if config.num_tasks is not None:
        selected = selected.head(config.num_tasks)
    tasks = list(selected)
    if isinstance(config.runtime, vf.SubprocessConfig) and any(
        type(t).NEEDS_CONTAINER or t.data.image for t in tasks
    ):
        raise SystemExit(
            "taskset needs a container runtime to validate - pass --runtime.type docker (or prime)"
        )
    mode = validation_mode(config)
    checks = "gold+setup" if mode == "all" else mode
    out = output_path(config)
    selected_identities = [
        identity(
            position,
            task.data.model_dump(mode="json", exclude_none=True),
        )
        for position, task in enumerate(tasks)
    ]
    if config.resume is None:
        save_run(config, out, len(tasks))
        rows: list[ResultRow] = []
        owed = list(range(len(tasks)))
    else:
        rows, owed = load_results(out, selected_identities, mode)
        write_summary(out, summarize(rows, len(tasks), mode))
    owed_set = set(owed)
    plan = [
        (position, task, key)
        for (position, task), (_, key) in zip(enumerate(tasks), selected_identities)
        if position in owed_set
    ]
    logger.info(
        "%s %d/%d task(s) from %s on the %s runtime (%s)",
        "resuming" if config.resume is not None else "validating",
        len(plan),
        len(tasks),
        config.name,
        config.runtime.type,
        checks,
    )
    logger.info("results: %s", out)

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    states = [TaskProgress(idx=t.data.idx, name=t.data.name) for t in tasks]
    for row in rows:
        state = states[row["task_position"]]
        state.state = row["reason"]

    write_lock = asyncio.Lock()

    async def _one(position: int, task: Task, key: str) -> ResultRow:
        st = states[position]
        async with sem or contextlib.nullcontext():
            st.start = time.time()
            st.state = "running"
            row = await _validate_task(task, config)
        row["task_position"] = position
        row["task_key"] = key

        async def persist() -> None:
            async with write_lock:
                await asyncio.to_thread(append_result, out, row)
                rows.append(row)
                await asyncio.to_thread(
                    write_summary,
                    out,
                    summarize(rows, len(tasks), mode),
                )

        await run_shielded(persist())
        st.end, st.state = time.time(), row["reason"]
        if not config.rich:  # the dashboard shows this live; otherwise log each task
            detail = f" - {row['error']}" if row["error"] else ""
            logger.info(
                "idx=%s valid=%s reason=%s (%.1fs)%s",
                row["index"],
                row["valid"],
                row["reason"],
                row["elapsed"],
                detail,
            )
        return row

    display = (
        validate_dashboard(states, config, time.time())
        if config.rich
        else contextlib.nullcontext()
    )
    async with display:
        if not plan:
            logger.info(
                "nothing to resume: all %d task(s) are valid or invalid", len(tasks)
            )
            return rows
        await asyncio.gather(
            *(_one(position, task, key) for position, task, key in plan)
        )
    rows.sort(key=lambda row: row["task_position"])
    write_summary(out, summarize(rows, len(tasks), mode))
    return rows


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(
        list(sys.argv[1:]) if argv is None else list(argv), flag="--taskset.id"
    )

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(_narrow(argv))  # full option help, narrowed to the given taskset
        return
    resume_dir, rest = split_resume(argv)
    if resume_dir is not None:
        if rest:
            raise SystemExit(
                f"{USAGE}\n--resume replays the saved config and takes no other arguments"
            )
        with plugin_errors():
            config = load_resume_config(resume_dir)
    else:
        if not extract_id(argv, "taskset") and not references_config_file(argv):
            raise SystemExit(
                USAGE
            )  # need a taskset (positional / --taskset.id) or a @ file.toml

        with plugin_errors():
            config_type = _narrow(argv)
            sys.argv = [
                sys.argv[0],
                *argv,
            ]  # let prime-pydantic-config render help/errors
            config = cli(config_type)
    out = output_path(config)
    setup_logging(
        "DEBUG" if config.verbose else "INFO",
        log_file=str(out / LOG_FILE),
        console=not config.rich,
    )
    if config.rich:
        logging.lastResort = None  # drop stdlib records that bypass loguru
    # Graceful shutdown: first Ctrl-C/SIGTERM unwinds each task's teardown `finally`
    # (containers/sandboxes); a second is swallowed so it can't orphan them mid-cleanup.
    install_interrupt()
    try:
        asyncio.run(run_validate(config))
    except KeyboardInterrupt:
        print(f"interrupted; partial results: {out}", file=sys.stderr)
        raise SystemExit(130)
    summary = json.loads((out / SUMMARY_FILE).read_text())
    print(f"results: {out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
