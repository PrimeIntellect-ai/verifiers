"""Replay captured submissions using the ordinary isolated verifier lifecycle."""

import asyncio
import copy
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifiers.v1.configs.scoring import ScoringConfig, resolve_scoring
from verifiers.v1.grading import grade_task, verifier_runtime
from verifiers.v1.runtimes import RuntimeConfig
from verifiers.v1.runtimes.deployment import Deployment
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task, TaskData, TaskTimeout, task_key
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.utils.artifacts import ARTIFACTS_DIR, artifact_source, load_artifacts
from verifiers.v1.utils.compile import resolve_runtime_config


class RequiresRerun(ValueError):
    """The saved submission cannot be graded against the selected task."""


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


def _validate_solver(saved: Trace, task: Task, policy: ScoringConfig) -> None:
    """Refuse to apply a new task to a solver that received a different problem."""
    old, new = saved.task.data, task.data
    before, after = revision(old), revision(new)
    old_major, new_major = _major(before["task_version"]), _major(after["task_version"])
    if old_major is not None and new_major is not None and old_major != new_major:
        raise RequiresRerun("Task major version changed")
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
            raise RequiresRerun(f"Solver input changed: {field}")
    if not before["solver_digest"] or before["solver_digest"] != after["solver_digest"]:
        raise RequiresRerun("Solver source identity changed or is unavailable")
    if policy.mode != "isolated":
        raise RequiresRerun("Current task requires shared runtime state")
    if resolve_scoring(old.scoring, saved.agent.config.scoring).mode != "isolated":
        raise RequiresRerun("Original task was scored with shared runtime state")
    if (
        not saved.is_completed
        or not saved.ok
        or saved.errors
        or saved.is_format_violation
        or saved.stop_condition not in {None, "agent_completed", "scoring_timeout"}
        or (saved.is_timeout and saved.stop_condition != "scoring_timeout")
    ):
        raise RequiresRerun("Original solver did not complete")


def _solver_runtime_config(saved: Trace, base: RuntimeConfig) -> RuntimeConfig:
    """Reconstruct the workspace's image and cwd without starting its runtime."""
    original = saved.agent.runtime
    if original is None:
        # An explicit task image can be resolved without runtime metadata. An
        # implicit image may have come from Compose or another workspace instead.
        if saved.task.data.image is None:
            raise RequiresRerun("Saved submission has no resolved workspace image")
        try:
            original = resolve_runtime_config(
                saved.agent.config.runtime, Task(saved.task.data)
            )
        except ValueError as exc:
            raise RequiresRerun(str(exc)) from exc
    if original.type != base.type or getattr(original, "import_path", None) != getattr(
        base, "import_path", None
    ):
        raise RequiresRerun(
            "Selected runtime is incompatible with the saved workspace runtime"
        )
    image = getattr(original, "image", None)
    if not image:
        raise RequiresRerun("Saved submission has no resolved workspace image")
    workdir = getattr(original, "workdir", None) or saved.task.data.workdir or "/app"
    # Supply only the recorded workspace coordinates. The current verifier's
    # image, resources and network policy are resolved separately.
    return base.model_copy(update={"image": image, "workdir": workdir})


def load_saved_artifacts(source: Path, trace: Trace) -> dict[str, bytes | None]:
    """Read the canonical archives identified by this saved trace."""
    if Path(trace.id).parts != (trace.id,) or trace.id == "..":
        raise ValueError("Trace id must be a single path component")
    digest = trace.info.get("artifact_manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RequiresRerun("Trace has no verified artifact manifest identity")
    try:
        return load_artifacts(
            source / "artifacts" / trace.id,
            max_bytes=trace.task.data.artifact_max_bytes,
            expected_manifest_sha256=digest,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise RequiresRerun(
            f"Cannot reconstruct saved submission - {type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class VerifierReplay:
    trace: Trace
    grader: Task
    runtime: RuntimeConfig
    scoring: ScoringConfig
    timeouts: TaskTimeout

    async def run(
        self, *, setup_timeout: float | None = None, total_timeout: float | None = None
    ) -> None:
        trace = self.trace
        trace.timing.scoring.start, trace.timing.scoring.end = time.time(), 0.0
        artifacts = dict(trace.state.artifacts)
        try:
            if self.scoring.skip_on_truncated and trace.is_truncated:
                trace.info["scoring_diagnostics"] = {"scoring_skipped": "truncated"}
                return
            timeouts = self.timeouts
            if setup_timeout is not None:
                timeouts = timeouts.model_copy(update={"setup": setup_timeout})
            deployment = Deployment()
            try:
                async with asyncio.timeout(total_timeout):
                    await grade_task(
                        self.grader,
                        trace,
                        self.runtime,
                        timeouts,
                        self.scoring,
                        deployment,
                    )
            finally:
                await deployment.close()
            if not trace.rewards and not trace.is_dropped:
                raise ValueError("Verifier produced no rewards")
        except TimeoutError:
            trace.record_timeout("scoring")
            raise
        except Exception as error:
            trace.record_error(error)
            raise
        finally:
            trace.state.artifacts = artifacts
            trace.timing.scoring.end = time.time()


def prepare_replay(
    saved: Trace,
    task: Task,
    runtime: RuntimeConfig,
    artifacts: Mapping[str, bytes | None],
    *,
    scoring: ScoringConfig | None = None,
) -> VerifierReplay:
    """Check a captured submission and prepare a fresh verifier without provisioning.

    Pass artifacts from the live capture or from ``load_saved_artifacts``.
    """
    policy = scoring if scoring is not None else task.scoring
    _validate_solver(saved, task, policy)
    solver = _solver_runtime_config(saved, runtime)
    workdir = getattr(solver, "workdir", None) or "/app"
    workdirs = saved.info.get("harbor_artifact_workdirs", {})
    try:
        declared = [
            (artifact_source(entry, workdir, workdirs), entry.required)
            for entry in (*saved.task.data.artifacts, *task.data.artifacts)
        ]
        grader_artifacts = [
            entry
            if getattr(entry, "service", "main") == "main"
            else entry.model_copy(
                update={"source": artifact_source(entry, workdir, workdirs)}
            )
            for entry in task.data.artifacts
        ]
    except ValueError as exc:
        raise RequiresRerun(str(exc)) from exc
    if missing := ({root for root, _ in declared} | {ARTIFACTS_DIR}) - artifacts.keys():
        raise RequiresRerun(
            f"Saved submission is missing declared artifact roots: {sorted(missing)}"
        )
    if absent := {
        root for root, required in declared if required and artifacts[root] is None
    }:
        raise RequiresRerun(
            f"Required artifact roots were absent from the saved submission: {sorted(absent)}"
        )
    if (
        sum(len(archive) for archive in artifacts.values() if archive is not None)
        > task.data.artifact_max_bytes
    ):
        raise RequiresRerun("Saved artifacts exceed the task's byte limit")

    grader = task.with_data(artifacts=grader_artifacts).grading_task(
        solver, scoring=policy
    )
    grader.scoring_deferred = False
    grader.validate_artifacts(artifacts)
    config = verifier_runtime(grader, solver, runtime, policy)
    if any(not mount.read_only for mount in getattr(config, "mounts", {}).values()):
        raise RequiresRerun("Regrading requires read-only verifier mounts")

    trace = copy.deepcopy(saved)
    trace.id, trace.attempt, trace.run = uuid.uuid4().hex, 1, None
    trace.task = TraceTask(
        type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
    )
    trace.state = state_cls(type(task))(artifacts=dict(artifacts))
    trace.rewards, trace.metrics, trace.extra_usage = {}, {}, []
    trace.errors, trace.dropped = [], None
    trace.is_timeout = False
    if trace.stop_condition == "scoring_timeout":
        trace.stop_condition = "agent_completed"
    for key in (
        "artifact_manifest_sha256",
        "judge_calls",
        "scoring_diagnostics",
        "scoring_skipped",
    ):
        trace.info.pop(key, None)
    trace.info["task_config_sha256"] = task_key(task.config.model_dump(mode="json"))
    trace.info["regrading"] = {
        "source_trace_id": saved.id,
        "source_attempt": saved.attempt,
    }
    return VerifierReplay(trace, grader, config, policy, task.data.timeout)
