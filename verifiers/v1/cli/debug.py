"""The debug entrypoint: setup tasks, run one shell action, and persist traces."""

import asyncio
import contextlib
import logging
import random
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.output import append_trace, save_config
from verifiers.v1.cli.resolve import (
    extract_id,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.debug import DebugConfig
from verifiers.v1.env import resolve_runtime_config
from verifiers.v1.runtimes import ProgramResult, Runtime, make_runtime
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace
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


def tail(text: str, chars: int) -> str:
    if not text or chars == 0:
        return ""
    return text[-chars:]


def task_info(task) -> dict[str, Any]:
    return {"idx": task.idx, "name": task.name, "workdir": task.workdir}


def runtime_info(runtime: Runtime) -> dict[str, Any]:
    return {
        "type": runtime.type,
        "name": runtime.name,
        "descriptor": runtime.descriptor,
    }


def result_info(
    result: ProgramResult,
    start: float,
    output_tail_chars: int,
) -> dict[str, Any]:
    ok = result.exit_code == 0
    return {
        "ok": ok,
        "reason": "pass" if ok else "nonzero_exit",
        "exit_code": result.exit_code,
        "elapsed": round(time.time() - start, 2),
        "stdout_tail": tail(result.stdout or "", output_tail_chars),
        "stderr_tail": tail(result.stderr or "", output_tail_chars),
    }


def error_info(
    error: Exception,
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
        "stdout_tail": "",
        "stderr_tail": "",
    }


async def run_command(
    runtime: Runtime,
    command: str,
    config: DebugConfig,
) -> dict[str, Any]:
    start = time.time()
    try:
        result = await asyncio.wait_for(
            runtime.run(["sh", "-lc", command], {}),
            config.timeout,
        )
    except Exception as e:
        return error_info(e, start, config.timeout, "debug action")
    return result_info(result, start, config.output_tail_chars)


async def run_action(runtime: Runtime, config: DebugConfig) -> dict[str, Any]:
    if config.command is not None:
        return {
            "action": "command",
            "command": config.command,
            **(await run_command(runtime, config.command, config)),
        }
    assert config.script_path is not None
    remote_path = config.remote_script_path
    await runtime.write(remote_path, config.script_path.read_bytes())
    command = f"chmod +x {shlex.quote(remote_path)} && {shlex.quote(remote_path)}"
    return {
        "action": "script",
        "script_path": str(config.script_path),
        "remote_script_path": remote_path,
        "command": command,
        **(await run_command(runtime, command, config)),
    }


async def debug_task(taskset: Taskset, task, config: DebugConfig) -> Trace:
    trace = Trace(task=task)
    debug = {
        "task": task_info(task),
        "action": "command" if config.command is not None else "script",
        "ok": False,
        "reason": "not_run",
    }
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task), name=f"debug-{task.idx}"
    )
    setup_timeout = (
        config.setup_timeout if config.setup_timeout is not None else task.timeout.setup
    )
    try:
        trace.timing.setup.start = time.time()
        await runtime.start()
        debug["runtime"] = runtime_info(runtime)
        await asyncio.wait_for(taskset.setup(task, runtime), setup_timeout)
        trace.timing.setup.end = time.time()

        trace.timing.generation.start = time.time()
        debug.update(await run_action(runtime, config))
        trace.timing.generation.end = time.time()
        trace.stop(str(debug["reason"]))
    except Exception as e:
        if not trace.timing.setup.end:
            trace.timing.setup.end = time.time()
        if trace.timing.generation.start and not trace.timing.generation.end:
            trace.timing.generation.end = time.time()
        stage = "debug action" if trace.timing.generation.start else "setup"
        timeout = config.timeout if stage == "debug action" else setup_timeout
        error_start = (
            trace.timing.generation.start
            if stage == "debug action"
            else trace.timing.setup.start
        )
        debug.update(error_info(e, error_start, timeout, stage))
        debug.setdefault("runtime", runtime_info(runtime))
        trace.capture_error(e)
    finally:
        trace.info["debug"] = debug
        try:
            await runtime.stop()
        except Exception:
            logger.warning("runtime teardown failed (task %s)", task.idx, exc_info=True)
    return trace


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
            trace = await debug_task(taskset, task, config)
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
