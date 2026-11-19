"""Regrade saved submissions without rerunning their solver agents."""

import asyncio
import json
import logging
import os
import sys
from collections import Counter
from typing import Any

from pydantic_config import cli
from pydantic_core import to_jsonable_python

from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_taskset_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.cli.regrade import RegradeConfig
from verifiers.v1.regrading import (
    RequiresRerun,
    load_saved_artifacts,
    prepare_replay,
    revision,
)
from verifiers.v1.runtimes import backend_cls
from verifiers.v1.task import Task, TaskData, task_key
from verifiers.v1.trace import Trace, WireTrace
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.loaders import load_taskset
from verifiers.v1.utils.logging import clip, get_logger, setup_logging
from verifiers.v1.utils.trace_store import read_episodes

logger = get_logger(__name__)
USAGE = "usage: uv run vf-regrade <taskset-id> --source <run-dir> [--runtime.type docker|prime] [-o <output-dir>] [options]"


def identity(data: TaskData) -> tuple[str, str] | None:
    if data.id is not None:
        return "id", data.id
    if data.name is not None:
        return "name", data.name
    return None


def _scores(trace: Trace) -> dict[str, Any]:
    return trace.model_dump(include={"reward", "rewards", "metrics", "dropped"})


async def regrade_trace(
    saved: Trace, task: Task, config: RegradeConfig
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trace_id": saved.id,
        "task_id": task.data.id,
        "task_name": task.data.name,
        "old_revision": revision(saved.task.data),
        "new_revision": revision(task.data),
        "old_task_config_sha256": saved.info.get("task_config_sha256"),
        "new_task_config_sha256": task_key(task.config.model_dump(mode="json")),
        "old": _scores(saved),
    }
    try:
        artifacts = load_saved_artifacts(config.source, saved)
        replay = prepare_replay(saved, task, config.runtime, artifacts)
        await replay.run(
            setup_timeout=config.timeout.setup, total_timeout=config.timeout.total
        )
        trace = replay.trace
        if reason := trace.info.get("scoring_skipped"):
            return {**row, "status": "skipped", "reason": reason}
        return {
            **row,
            "status": "regraded",
            "new": _scores(trace),
            "diagnostics": to_jsonable_python(
                trace.info.get("scoring_diagnostics", {}),
                fallback=str,
                bytes_mode="base64",
            ),
            "judge_calls": to_jsonable_python(
                trace.info.get("judge_calls", []), fallback=str, bytes_mode="base64"
            ),
            "usage": [usage.model_dump(mode="json") for usage in trace.extra_usage],
        }
    except RequiresRerun as exc:
        return {**row, "status": "requires_rerun", "reason": str(exc)}
    except Exception as exc:
        logger.warning(
            "Failed to regrade saved submission - %s: %s",
            type(exc).__name__,
            clip(str(exc)),
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return {
            **row,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


async def run_regrade(config: RegradeConfig) -> list[dict[str, Any]]:
    if not backend_cls(config.runtime).SUPPORTS_TASK_IMAGES:
        raise ValueError("Regrading requires a runtime with an isolated filesystem")
    taskset = load_taskset(config.taskset).select(config.select)
    if not taskset.bounded:
        raise ValueError("Select a finite taskset for regrading")
    tasks: dict[tuple[str, str], Task] = {}
    for task in taskset:
        key = identity(task.data)
        if key is None:
            raise ValueError("Regrading tasks need a durable id or name")
        if key in tasks:
            raise ValueError(f"Ambiguous current task identity: {key}")
        tasks[key] = task
    episodes = read_episodes(config.source, WireTrace)
    out = output_path(config)
    if out.resolve().is_relative_to(config.source.resolve()):
        raise ValueError("Regrading output must be outside the source run")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Regrading output already contains files: {out}")
    write_config(config, out, "regrade.json")
    results_path = out / "results.jsonl"
    results_path.touch()
    semaphore = asyncio.Semaphore(config.max_concurrent)
    rows = []
    seen = set()

    async def one(episode_id: str, trace: Trace) -> None:
        async with semaphore:
            task = tasks.get(identity(trace.task.data))
            if task is not None:
                row = await regrade_trace(trace, task, config)
            else:
                row = {
                    "trace_id": trace.id,
                    "status": "unmatched",
                    "reason": "No selected task has this durable id or name",
                    "old": _scores(trace),
                }
            row["episode_id"] = episode_id

            # No await between completion and persistence: cancellation cannot
            # interrupt a row or interleave writes from these event-loop tasks.
            with results_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)

    jobs = []
    try:
        for episode in episodes:
            for trace in episode.traces:
                # Judge/helper traces have their own minted task and are not submissions.
                if identity(trace.task.data) != identity(episode.task.data):
                    continue
                if trace.id in seen:
                    raise ValueError(f"Duplicate source trace id: {trace.id}")
                seen.add(trace.id)
                jobs.append(asyncio.create_task(one(episode.id, trace)))
        await asyncio.gather(*jobs)
    except BaseException:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        raise
    (out / "summary.json").write_text(
        json.dumps(dict(Counter(row["status"] for row in rows)), indent=2) + "\n"
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(
        list(sys.argv[1:]) if argv is None else list(argv), flag="--taskset.id"
    )
    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(narrow_taskset_config(RegradeConfig, extract_id(argv, "taskset")))
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(USAGE)
    with plugin_errors():
        config_type = narrow_taskset_config(RegradeConfig, extract_id(argv, "taskset"))
        sys.argv = [sys.argv[0], *argv]
        config = cli(config_type)
    setup_logging("DEBUG" if config.verbose else "INFO")
    os.environ.setdefault("VF_RUN_ID", config.run.id)
    install_interrupt()
    try:
        rows = asyncio.run(run_regrade(config))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"results: {output_path(config)}")
    print(json.dumps(dict(Counter(row["status"] for row in rows)), sort_keys=True))
    if any(row["status"] != "regraded" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
