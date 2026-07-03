"""The validate entrypoint: `uv run validate <taskset-id> [--runtime.type subprocess] [options]`.

Registered as the `validate` console script — the model-free sibling of `eval`. Where `eval`
runs a model rollout per task, `validate` runs each task's `validate` hook: a per-task check
that the ground truth holds (a SWE row's gold patch makes its tests pass, gsm8k's verifier
accepts the gold answer), in a runtime with the taskset's `setup` already applied. Each task
is provisioned, set up, validated, and torn down independently with bounded concurrency.

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

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.dashboard import TaskProgress, validate_dashboard
from verifiers.v1.utils.logging import setup_logging
from verifiers.v1.cli.resolve import (
    extract_id,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.validate import ValidateConfig
from verifiers.v1.decorators import invoke
from verifiers.v1.env import resolve_runtime_config
from verifiers.v1.runtimes import make_runtime
from verifiers.v1.state import state_cls
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run validate [<taskset-id>] [--runtime.type subprocess] [options] [@ file.toml]\n"
    "       runs each task's `validate` hook (per-task validation, no model)"
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


async def _validate_task(taskset: Taskset, task, config: ValidateConfig) -> dict:
    """Validate one task: provision its runtime, run `setup` then `validate` (each under its
    stage timeout), tear the runtime down, and return the result row. A raised error is
    captured onto the row (one bad task is data, not a crash) — never re-raised."""
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task), name=f"validate-{task.idx}"
    )
    setup_timeout = (
        config.setup_timeout if config.setup_timeout is not None else task.timeout.setup
    )
    valid, exc = False, None
    try:
        trace = Trace(task=task, state=state_cls(type(taskset))())
        await runtime.start()
        await asyncio.wait_for(
            invoke(taskset.setup, {"task": task, "trace": trace, "runtime": runtime}),
            setup_timeout,
        )
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
    return {
        "index": task.idx,
        "name": task.name,
        "valid": bool(valid),
        "reason": _classify(valid, exc),
        "elapsed": round(time.time() - start, 2),
        "error": str(exc) if exc is not None else None,
        "error_type": type(exc).__name__ if exc is not None else None,
    }


async def run_validate(config: ValidateConfig) -> list[dict]:
    """Run each task's `validate` hook with bounded concurrency, showing progress live. Returns
    the result rows in memory — nothing is persisted."""
    taskset = vf.load_taskset(config.taskset)
    tasks = taskset.load_tasks()
    if config.shuffle:
        random.Random(0).shuffle(tasks)
    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]
    # Lazy tasksets bind task content at request time (Taskset.resolve_task); resolve before
    # validating so hooks see real tasks, not stubs. Eager tasksets are unchanged.
    tasks = [await taskset.resolve_task(task) for task in tasks]
    if isinstance(config.runtime, vf.SubprocessConfig) and (
        taskset.NEEDS_CONTAINER or any(t.image for t in tasks)
    ):
        raise SystemExit(
            "taskset needs a container runtime to validate - pass --runtime.type docker (or prime)"
        )
    logger.info(
        "validating %d task(s) from %s on the %s runtime",
        len(tasks),
        config.name,
        config.runtime.type,
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
