"""The debug entrypoint: setup tasks, run one shell action, and persist traces."""

import asyncio
import contextlib
import logging
import random
import shlex
import signal
import sys
import time
import traceback
from collections.abc import Awaitable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.output import append_trace, save_config
from verifiers.v1.cli.resolve import (
    extract_id,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.debug import DebugConfig
from verifiers.v1.decorators import invoke
from verifiers.v1.resolve import resolve_runtime_config
from verifiers.v1.runtimes import ProgramResult, Runtime, make_runtime
from verifiers.v1.state import state_cls
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Error, Trace
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run debug [<taskset-id>] (--command <cmd> | --script-path <path>) "
    "[--runtime.type subprocess] [options] [@ file.toml]\n"
    "       runs setup, then one command or uploaded host script, and saves traces"
)


def _narrow(argv: list[str]) -> type[DebugConfig]:
    """`DebugConfig` with `taskset` narrowed to the config type of the id on the CLI."""
    taskset_id = extract_id(argv, "taskset")
    if not taskset_id:
        return DebugConfig
    ftype = vf.taskset_config_type(taskset_id)
    return type(
        DebugConfig.__name__,
        (DebugConfig,),
        {"__annotations__": {"taskset": ftype}, "taskset": ftype(id=taskset_id)},
    )


def output_path(config: DebugConfig) -> Path:
    if config.output_dir is not None:
        return config.output_dir
    return Path("outputs") / f"{config.taskset.name}--debug" / config.uuid


def task_info(task) -> dict[str, Any]:
    return {"idx": task.idx, "name": task.name, "workdir": task.workdir}


def runtime_info(runtime: Runtime) -> dict[str, Any]:
    return {
        "type": runtime.type,
        "name": runtime.name,
        "descriptor": runtime.descriptor,
    }


def result_info(result: ProgramResult, start: float) -> dict[str, Any]:
    ok = result.exit_code == 0
    return {
        "ok": ok,
        "reason": "pass" if ok else "nonzero_exit",
        "exit_code": result.exit_code,
        "elapsed": round(time.time() - start, 2),
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def error_info(
    error: BaseException,
    start: float,
    timeout: float | None,
    stage: str,
) -> dict[str, Any]:
    if isinstance(error, asyncio.TimeoutError):
        message = (
            f"{stage} timed out after {timeout}s"
            if timeout is not None
            else f"{stage} timed out"
        )
        reason = "timeout"
    elif isinstance(error, asyncio.CancelledError):
        message = f"{stage} cancelled"
        reason = "cancelled"
    else:
        message = str(error)
        reason = "error"
    return {
        "ok": False,
        "reason": reason,
        "exit_code": None,
        "elapsed": round(time.time() - start, 2),
        "error": message,
        "error_type": type(error).__name__,
        "stdout": "",
        "stderr": "",
    }


def capture_trace_error(trace: Trace, error: BaseException) -> None:
    if isinstance(error, Exception):
        trace.capture_error(error)
        return
    trace.errors.append(
        Error(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
    )
    trace.stop("error")


def record_debug_error(
    trace: Trace,
    debug: dict[str, Any],
    runtime: Runtime,
    error: BaseException,
    setup_timeout: float | None,
    action_timeout: float | None,
) -> None:
    if not trace.timing.setup.end:
        trace.timing.setup.end = time.time()
    if trace.timing.generation.start and not trace.timing.generation.end:
        trace.timing.generation.end = time.time()
    in_action = bool(trace.timing.generation.start)
    stage = "debug action" if in_action else "setup"
    timeout = action_timeout if in_action else setup_timeout
    error_start = (
        trace.timing.generation.start if in_action else trace.timing.setup.start
    )
    debug.update(error_info(error, error_start, timeout, stage))
    debug.setdefault("runtime", runtime_info(runtime))
    capture_trace_error(trace, error)


def record_action_failure(trace: Trace, debug: dict[str, Any]) -> None:
    message = debug.get("error") or f"debug action failed: {debug['reason']}"
    trace.errors.append(
        Error(
            type=debug.get("error_type") or "DebugActionError",
            message=str(message),
            traceback=None,
        )
    )


async def run_timed(
    action: Awaitable[ProgramResult], config: DebugConfig
) -> dict[str, Any]:
    """Run the whole debug action under one `--timeout.total` budget."""
    start = time.time()
    try:
        result = await asyncio.wait_for(action, config.timeout.total)
    except Exception as e:
        return error_info(e, start, config.timeout.total, "debug action")
    return result_info(result, start)


async def run_action(runtime: Runtime, config: DebugConfig) -> dict[str, Any]:
    if config.command is not None:
        return {
            "action": "command",
            "command": config.command,
            **(await run_timed(runtime.run(["sh", "-lc", config.command], {}), config)),
        }
    assert config.script_path is not None
    # /tmp keyed by the (run-unique) runtime name: outside the task workdir so the script
    # doesn't show up in the repo state being inspected, and never shared between runs on
    # the subprocess runtime, where an absolute path is a host path.
    remote_path = f"/tmp/{runtime.name}-script.sh"
    quoted = shlex.quote(remote_path)
    command = f"chmod +x {quoted} && {quoted}"

    async def upload_and_run() -> ProgramResult:
        await runtime.write(remote_path, config.script_path.read_bytes())
        return await runtime.run(["sh", "-lc", command], {})

    return {
        "action": "script",
        "script_path": str(config.script_path),
        "remote_script_path": remote_path,
        "command": command,
        **(await run_timed(upload_and_run(), config)),
    }


async def debug_task(taskset: Taskset, task, config: DebugConfig) -> tuple[Trace, bool]:
    trace = Trace(task=task, state=state_cls(type(taskset))())
    debug = {
        "task": task_info(task),
        "action": "command" if config.command is not None else "script",
        "ok": False,
        "reason": "not_run",
    }
    cancelled = False
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"debug-{task.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.timeout.setup if config.timeout.setup is not None else task.timeout.setup
    )
    try:
        trace.timing.setup.start = time.time()
        await runtime.start()
        debug["runtime"] = runtime_info(runtime)
        await asyncio.wait_for(
            invoke(taskset.setup, {"task": task, "trace": trace, "runtime": runtime}),
            setup_timeout,
        )
        trace.timing.setup.end = time.time()

        trace.timing.generation.start = time.time()
        debug.update(await run_action(runtime, config))
        trace.timing.generation.end = time.time()
        if not debug.get("ok"):
            record_action_failure(trace, debug)
        trace.stop(str(debug["reason"]))
    except asyncio.CancelledError as e:
        cancelled = True
        record_debug_error(
            trace, debug, runtime, e, setup_timeout, config.timeout.total
        )
    except Exception as e:
        record_debug_error(
            trace, debug, runtime, e, setup_timeout, config.timeout.total
        )
    finally:
        trace.info["debug"] = debug
        try:
            await runtime.stop()
        except asyncio.CancelledError:
            # a task cancellation delivered mid-stop would abort before the caller can
            # persist the trace — absorb it here; the caller re-raises after appending
            cancelled = True
        except Exception:
            logger.warning("runtime teardown failed (task %s)", task.idx, exc_info=True)
    return trace, cancelled


async def run_debug(config: DebugConfig) -> list[Trace]:
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
            "taskset needs a container runtime to debug - pass --runtime.type docker (or prime)"
        )

    out = output_path(config)
    save_config(config, out)
    logger.info(
        "debugging %d task(s) from %s on the %s runtime",
        len(tasks),
        config.name,
        config.runtime.type,
    )
    logger.info("results: %s", out)

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    write_lock = asyncio.Lock()

    async def one(task) -> Trace:
        async with sem or contextlib.nullcontext():
            trace, cancelled = await debug_task(taskset, task, config)
        await append_trace(out, trace, write_lock)
        info = trace.info["debug"]
        detail = f" - {info['error']}" if info.get("error") else ""
        logger.info(
            "idx=%s ok=%s reason=%s%s",
            task.idx,
            info["ok"],
            info["reason"],
            detail,
        )
        if cancelled:
            raise asyncio.CancelledError
        return trace

    return await asyncio.gather(*(one(task) for task in tasks))


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(_narrow(argv))
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(USAGE)

    config_type = _narrow(argv)
    sys.argv = [sys.argv[0], *argv]
    config = cli(config_type)
    setup_logging("DEBUG" if config.verbose else "INFO")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    asyncio.run(run_debug(config))


if __name__ == "__main__":
    main()
