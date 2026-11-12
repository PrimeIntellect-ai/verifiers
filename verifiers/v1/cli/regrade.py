"""Regrade saved submissions without rerunning their solver agents."""

import asyncio
import copy
import json
import logging
import os
import posixpath
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic_config import cli

from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_taskset_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.cli.regrade import RegradeConfig
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.grading import grade_task, verifier_runtime
from verifiers.v1.runtimes import RuntimeConfig, backend_cls, make_backend
from verifiers.v1.runtimes.deployment import Deployment
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task, TaskData, task_key
from verifiers.v1.trace import Trace, TraceTask, WireTrace
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import ARTIFACTS_DIR, load_artifacts
from verifiers.v1.utils.compile import resolve_runtime_config
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


def revision(data: TaskData) -> dict[str, Any]:
    return {
        field: getattr(data, field, None)
        for field in (
            "task_version",
            "source_digest",
            "solver_digest",
            "verifier_digest",
        )
    }


def _major(version: str | None) -> int | None:
    match = re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+].+)?", version or ""
    )
    return int(match[1]) if match else None


def rerun_reason(saved: Trace, task: Task) -> str | None:
    """Refuse to apply a new task to a solver that received a different problem."""
    old, new = saved.task.data, task.data
    before, after = revision(old), revision(new)
    old_major, new_major = _major(before["task_version"]), _major(after["task_version"])
    if old_major is not None and new_major is not None and old_major != new_major:
        return "Task major version changed"
    for field in (
        "prompt",
        "system_prompt",
        "image",
        "workdir",
        "agent_user",
        "agent_home",
        "network",
        "skills",
        "filesystem",
    ):
        if getattr(old, field, None) != getattr(new, field, None):
            return f"Solver input changed: {field}"
    if (
        not before["solver_digest"]
        or not after["solver_digest"]
        or before["solver_digest"] != after["solver_digest"]
    ):
        return "Solver source identity changed or is unavailable"
    if task.scoring.mode != "isolated":
        return "Current task requires shared runtime state"
    if resolve_scoring(old.scoring, saved.agent.config.scoring).mode != "isolated":
        return "Original task was scored with shared runtime state"
    if (
        not saved.is_completed
        or not saved.ok
        or saved.errors
        or saved.is_format_violation
        or saved.is_truncated
        or saved.stop_condition not in {None, "agent_completed", "scoring_timeout"}
        or (saved.is_timeout and saved.stop_condition != "scoring_timeout")
    ):
        return "Original solver did not complete"
    return None


def solver_runtime_config(
    saved: Trace, task: Task, base: RuntimeConfig
) -> RuntimeConfig:
    """Reconstruct the workspace's image and cwd without starting its runtime."""
    original = saved.agent.runtime
    if original is None:
        # An explicit task image can be resolved without runtime metadata. An
        # implicit image may have come from Compose or another workspace instead.
        if saved.task.data.image is None:
            raise ValueError("Saved submission has no resolved workspace image")
        original = resolve_runtime_config(
            saved.agent.config.runtime, Task(saved.task.data)
        )
    if original.type != base.type or getattr(original, "import_path", None) != getattr(
        base, "import_path", None
    ):
        raise ValueError(
            "Selected runtime is incompatible with the saved workspace runtime"
        )
    image = getattr(original, "image", None)
    if not image:
        raise ValueError("Saved submission has no resolved workspace image")
    workdir = getattr(original, "workdir", None) or saved.task.data.workdir or "/app"
    # Resolve current task policy with the native precedence, then supply the
    # recorded execution workspace to grading_task/verifier_runtime. A dedicated
    # current verifier image still takes precedence there.
    return resolve_runtime_config(base, task).model_copy(
        update={"image": image, "workdir": workdir}
    )


def _scores(trace: Trace) -> dict[str, Any]:
    return {
        "reward": trace.reward,
        "rewards": {
            name: value.model_dump() if value is not None else None
            for name, value in trace.rewards.items()
        },
        "metrics": dict(trace.metrics),
        "dropped": trace.dropped,
    }


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
    if reason := rerun_reason(saved, task):
        return {**row, "status": "requires_rerun", "reason": reason}
    try:
        if not saved.id or Path(saved.id).parts != (saved.id,) or saved.id == "..":
            raise ValueError("Trace id must be a single path component")
        digest = saved.info.get("artifact_manifest_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return {
                **row,
                "status": "requires_rerun",
                "reason": "Trace has no verified artifact manifest identity",
            }
        try:
            artifacts = load_artifacts(
                config.source / "artifacts" / saved.id,
                max_bytes=task.data.artifact_max_bytes,
                expected_manifest_sha256=digest,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return {
                **row,
                "status": "requires_rerun",
                "reason": f"Cannot reconstruct saved submission - {type(exc).__name__}: {exc}",
            }
        try:
            solver_config = solver_runtime_config(saved, task, config.runtime)
        except ValueError as exc:
            return {**row, "status": "requires_rerun", "reason": str(exc)}
        workdir = solver_config.workdir
        expected_roots = {
            posixpath.normpath(posixpath.join(workdir, entry.source))
            for entry in task.data.artifacts
        }
        expected_roots.add(ARTIFACTS_DIR)
        if missing := expected_roots - artifacts.keys():
            return {
                **row,
                "status": "requires_rerun",
                "reason": f"Saved submission is missing declared artifact roots: {sorted(missing)}",
            }
        absent = {
            posixpath.normpath(posixpath.join(workdir, entry.source))
            for entry in task.data.artifacts
            if entry.required
            and artifacts[posixpath.normpath(posixpath.join(workdir, entry.source))]
            is None
        }
        if absent:
            return {
                **row,
                "status": "requires_rerun",
                "reason": f"Required artifact roots were absent from the saved submission: {sorted(absent)}",
            }
        task.validate_artifacts(artifacts)

        # A borrowed config-only view lets the native grading lifecycle provision
        # just its fresh verifier. No solver runtime is started or reconstructed.
        backend = make_backend(solver_config)
        solver = backend.bind()
        policy = task.scoring
        grader = task.grading_task(solver, scoring=policy)
        grading_config = verifier_runtime(grader, solver, config.runtime, policy)
        trace = copy.deepcopy(saved)
        trace.task = TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        )
        trace.state = state_cls(type(task))(artifacts=artifacts)
        trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
        trace.errors, trace.dropped = [], None
        trace.is_timeout = False
        if trace.stop_condition == "scoring_timeout":
            trace.stop_condition = "agent_completed"
        trace.info.pop("judge_calls", None)
        trace.info.pop("scoring_diagnostics", None)
        timeouts = task.data.timeout
        if config.timeout.setup is not None:
            timeouts = timeouts.model_copy(update={"setup": config.timeout.setup})
        deployment = Deployment(backend, borrowed_runtime=solver)
        await deployment.start()
        try:
            async with asyncio.timeout(config.timeout.total):
                await grade_task(
                    grader, trace, grading_config, timeouts, policy, deployment
                )
        finally:
            await deployment.close()
        if not trace.rewards:
            raise ValueError("Verifier produced no rewards")
        return {
            **row,
            "status": "regraded",
            "new": _scores(trace),
            "diagnostics": trace.info.get("scoring_diagnostics", {}),
            "judge_calls": trace.info.get("judge_calls", []),
            "usage": [usage.model_dump() for usage in trace.extra_usage],
        }
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
    tasks: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for task in taskset:
        key = identity(task.data)
        if key is None:
            raise ValueError("Regrading tasks need a durable id or name")
        tasks[key].append(task)
    if duplicates := [key for key, values in tasks.items() if len(values) != 1]:
        raise ValueError(f"Ambiguous current task identities: {duplicates}")
    episodes = read_episodes(config.source, WireTrace)
    out = output_path(config)
    if out.resolve() == config.source.resolve() or out.resolve().is_relative_to(
        config.source.resolve()
    ):
        raise ValueError("Regrading output must be outside the source run")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Regrading output already contains files: {out}")
    write_config(config, out, "regrade.json")
    results_path = out / "results.jsonl"
    results_path.touch()
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(config.max_concurrent)
    rows = []
    seen = set()

    async def one(episode_id: str, trace: Trace) -> None:
        async with semaphore:
            candidates = tasks.get(identity(trace.task.data), [])
            if candidates:
                row = await regrade_trace(trace, candidates[0], config)
            else:
                row = {
                    "trace_id": trace.id,
                    "status": "unmatched",
                    "reason": "No selected task has this durable id or name",
                    "old": _scores(trace),
                }
            row["episode_id"] = episode_id

            async def persist():
                async with lock:
                    with results_path.open("a") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)

            await run_shielded(persist())

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
