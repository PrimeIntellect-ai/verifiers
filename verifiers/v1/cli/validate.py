"""The validate entrypoint: `uv run validate <taskset-id> [--runtime.type subprocess] [options]`.

Registered as the `validate` console script — the model-free sibling of `eval`. Where `eval`
runs a model rollout per task, `validate` can either run each task's `validate` hook, run
setup only, or run both modes in independent runtimes. Each task is provisioned, set up,
checked, and torn down independently with bounded concurrency.

Fire-and-forget: progress is shown live (the `--rich` dashboard, or per-task log lines) and
nothing is written to disk.
"""

import asyncio
import contextlib
import logging
import random
import signal
import sys
import time
from typing import Any
from uuid import uuid4

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.dashboard import TaskProgress, validate_dashboard
from verifiers.v1.cli.resolve import (
    extract_id,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.validate import ValidateConfig
from verifiers.v1.env import resolve_runtime_config
from verifiers.v1.runtimes import make_runtime
from verifiers.v1.taskset import Taskset
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run validate [<taskset-id>] [--runtime.type subprocess] [options] [@ file.toml]\n"
    "       runs setup-only, apply-answer, or both validation modes (no model)"
)


def _narrow(argv: list[str]) -> type[ValidateConfig]:
    """`ValidateConfig` with `taskset` narrowed to the config type of the id on the CLI — so
    the single `cli()` parse stays typed and `-h` renders the taskset's fields. Absent an id
    (a `@ file.toml` may carry it) the base type is left for the validator to resolve."""
    taskset_id = extract_id(argv, "taskset")
    if not taskset_id:
        return ValidateConfig
    ftype = vf.taskset_config_type(taskset_id)
    return type(
        ValidateConfig.__name__,
        (ValidateConfig,),
        {"__annotations__": {"taskset": ftype}, "taskset": ftype(id=taskset_id)},
    )


ResultRow = dict[str, Any]


def _classify(valid: bool, exc: BaseException | None) -> str:
    """The outcome reason: `valid` (passed), `invalid` (returned False), `timeout` (a stage
    timed out), or `error` (raised) — the error's message carries the detail."""
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
        "index": task.idx,
        "name": task.name,
        "mode": mode,
        "valid": bool(valid),
        "reason": _classify(valid, exc),
        "elapsed": round(time.time() - start, 2),
        "error": str(exc) if exc is not None else None,
        "error_type": type(exc).__name__ if exc is not None else None,
    }


async def _run_apply_answer(
    taskset: Taskset, task, config: ValidateConfig
) -> ResultRow:
    """Provision one runtime, run setup + validate, tear it down, and return a row."""
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"validate-apply-answer-{task.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.setup_timeout if config.setup_timeout is not None else task.timeout.setup
    )
    valid, exc = False, None
    try:
        await runtime.start()
        await asyncio.wait_for(taskset.setup(task, runtime), setup_timeout)
        valid = await asyncio.wait_for(
            taskset.validate(task, runtime), config.validate_timeout
        )
    except Exception as e:
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning("runtime teardown failed (task %s)", task.idx, exc_info=True)
    return _row(task, "apply-answer", valid, exc, start)


async def _run_noop(taskset: Taskset, task, config: ValidateConfig) -> ResultRow:
    """Provision one runtime, run setup only, tear it down, and return a row."""
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"validate-noop-{task.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.setup_timeout if config.setup_timeout is not None else task.timeout.setup
    )
    valid, exc = False, None
    try:
        await runtime.start()
        await asyncio.wait_for(taskset.setup(task, runtime), setup_timeout)
        valid = True
    except Exception as e:
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning("runtime teardown failed (task %s)", task.idx, exc_info=True)
    return _row(task, "noop", valid, exc, start)


def _both_reason(apply_answer: ResultRow, noop: ResultRow) -> str:
    reasons = {str(apply_answer["reason"]), str(noop["reason"])}
    if apply_answer["valid"] and noop["valid"]:
        return "valid"
    if "error" in reasons:
        return "error"
    if "timeout" in reasons:
        return "timeout"
    return "invalid"


def _both_error(
    apply_answer: ResultRow, noop: ResultRow
) -> tuple[str | None, str | None]:
    failed = [
        ("apply_answer", apply_answer),
        ("noop", noop),
    ]
    parts = [
        f"{name}: {row['error'] or row['reason']}"
        for name, row in failed
        if not row["valid"]
    ]
    error_types = {
        str(row["error_type"])
        for _, row in failed
        if not row["valid"] and row["error_type"]
    }
    return ("; ".join(parts) or None, "+".join(sorted(error_types)) or None)


async def _run_both(taskset: Taskset, task, config: ValidateConfig) -> ResultRow:
    """Run apply-answer and noop as independent high-level validations."""
    start = time.time()
    apply_answer = await _run_apply_answer(taskset, task, config)
    noop = await _run_noop(taskset, task, config)
    valid = bool(apply_answer["valid"] and noop["valid"])
    error, error_type = _both_error(apply_answer, noop)
    return {
        "index": task.idx,
        "name": task.name,
        "mode": "both",
        "valid": valid,
        "reason": _both_reason(apply_answer, noop),
        "elapsed": round(time.time() - start, 2),
        "error": error,
        "error_type": error_type,
        "apply_answer": apply_answer,
        "noop": noop,
    }


async def _validate_task(taskset: Taskset, task, config: ValidateConfig) -> ResultRow:
    if config.mode == "apply-answer":
        return await _run_apply_answer(taskset, task, config)
    if config.mode == "noop":
        return await _run_noop(taskset, task, config)
    return await _run_both(taskset, task, config)


async def run_validate(config: ValidateConfig) -> list[dict]:
    """Run each task's `validate` hook with bounded concurrency, showing progress live. Returns
    the result rows in memory — nothing is persisted."""
    taskset = vf.load_taskset(config.taskset)
    tasks = taskset.load_tasks()
    if config.shuffle:
        random.Random(0).shuffle(tasks)
    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]
    if isinstance(config.runtime, vf.SubprocessConfig) and (
        taskset.NEEDS_CONTAINER or any(t.image for t in tasks)
    ):
        raise SystemExit(
            "taskset needs a container runtime to validate - pass --runtime.type docker (or prime)"
        )
    logger.info(
        "validating %d task(s) from %s on the %s runtime (mode=%s)",
        len(tasks),
        config.name,
        config.runtime.type,
        config.mode,
    )

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    states = [TaskProgress(idx=t.idx, name=t.name) for t in tasks]
    state_by_idx = {s.idx: s for s in states}

    async def _one(task) -> dict:
        st = state_by_idx[task.idx]
        async with sem or contextlib.nullcontext():
            st.start = time.time()
            st.state = "running"
            row = await _validate_task(taskset, task, config)
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
        return await asyncio.gather(*(_one(t) for t in tasks))


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(_narrow(argv))  # full option help, narrowed to the given taskset
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --taskset.id) or a @ file.toml

    config_type = _narrow(argv)
    sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
    config = cli(config_type)
    # Nothing is persisted, so logs are the whole output. Under `--rich` the dashboard owns the
    # screen, so keep logs off the console (else stray records print over the UI).
    setup_logging("DEBUG" if config.verbose else "INFO", console=not config.rich)
    if config.rich:
        logging.lastResort = None  # drop stdlib records that bypass loguru
    # Make SIGTERM behave like Ctrl-C so a killed run still runs each task's `finally`
    # (tears down containers/sandboxes) — and the atexit backstop catches the rest.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    asyncio.run(run_validate(config))


if __name__ == "__main__":
    main()
