"""Compare authored verifiers on a captured submission using native regrading."""

import copy
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, SerializeAsAny

from verifiers.v1.configs.scoring import ScoringConfig, resolve_scoring
from verifiers.v1.grading import verifier_runtime
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.utils.artifacts import ARTIFACTS_DIR

from .taskset import HarborTask, make_tar

logger = logging.getLogger(__name__)


class VerifierComparison(BaseModel):
    """Original result and independent native traces, including judge evidence/usage."""

    original: SerializeAsAny[Trace]
    traces: dict[str, SerializeAsAny[Trace]]

    @property
    def scores(self) -> dict[str, dict[str, float]]:
        """Named aggregate rewards and metrics (including RewardKit criteria).

        Failed/dropped comparisons have no numeric verdict; their errors and billed
        attempts remain on their traces. Names unique to one design remain here.
        """
        return {
            label: {
                **{
                    f"reward/{name}": value.score
                    for name, value in trace.rewards.items()
                    if value is not None
                },
                **{
                    f"metric/{name}": value
                    for name, value in trace.metrics.items()
                    if value is not None
                },
            }
            if trace.ok and not trace.is_timeout and not trace.is_dropped
            else {}
            for label, trace in {"original": self.original, **self.traces}.items()
        }

    @property
    def comparison(self):
        """RewardKit's matching-name comparison; no statistical validation."""
        from rewardkit.compare import compare

        return compare(self.scores)

    def format(self) -> str:
        """Render upstream's table alongside trace IDs and failed-verifier attribution."""
        from rewardkit.compare import format_comparison

        lines = []
        for label, trace in {"original": self.original, **self.traces}.items():
            status = (
                trace.last_error.message
                if trace.last_error
                else trace.dropped or trace.stop_condition or "completed"
            )
            lines.append(f"{label}: {trace.id} ({status})")
        lines.append(format_comparison(self.scores))
        lines.append("diff = first available score minus last; descriptive only.")
        return "\n".join(lines)


async def compare_verifiers(
    trace: Trace,
    verifiers: Mapping[str, HarborTask],
    *,
    scoring: ScoringConfig | None = None,
) -> VerifierComparison:
    """Regrade one saved submission with named Harbor task/verifier definitions.

    Load with ``read_episodes(path, Trace[HarborData], with_artifacts=True)``;
    pass tasks such as ``{"a": task, "b": task.with_data(task_dir="/variant")}``.
    Each task uses its ordinary test.sh/judge/composite precedence. Scoring runs in
    fresh runtimes, including retries; only captured artifacts travel. This executes
    verifiers and may incur runtime/model costs. Agent judges need a bound executor.

    The original is retained unchanged. Result traces can be saved with the native
    trace store; each carries its source trace ID, verifier config and input hashes.
    """
    if not verifiers or "original" in verifiers or any(not name for name in verifiers):
        raise ValueError("provide named verifiers; 'original' is reserved")
    if not trace.is_completed or trace.has_error or trace.is_timeout:
        raise ValueError("comparison requires a completed, non-errored submission")
    solver = trace.agent.runtime
    if solver is None:
        raise ValueError("saved submission has no solver runtime metadata")
    original = copy.deepcopy(trace)
    if ARTIFACTS_DIR not in original.state.artifacts:
        raise ValueError("Load captured submission artifacts with with_artifacts=True")
    comparison = VerifierComparison(original=original, traces={})
    workdir = PurePosixPath(getattr(solver, "workdir", None) or "/")
    fingerprints = {
        source: hashlib.sha256(archive).hexdigest() if archive is not None else None
        for source, archive in original.state.artifacts.items()
    }
    # Validate every design before provisioning or making model calls.
    prepared = {}
    for label, definition in verifiers.items():
        task = copy.deepcopy(definition)
        task.scoring_deferred = False
        policy = resolve_scoring(task.scoring, original.agent.config.scoring)
        policy = resolve_scoring(policy, scoring or ScoringConfig()).model_copy(
            update={"mode": "isolated"}
        )
        artifacts = {}
        for artifact in [*original.task.data.artifacts, *task.data.artifacts]:
            source = PurePosixPath(artifact.source)
            if not source.is_absolute():
                service = getattr(artifact, "service", "main")
                directory = (
                    workdir
                    if service == "main"
                    else original.info.get("harbor_artifact_workdirs", {}).get(service)
                )
                if directory is None:
                    raise ValueError(
                        f"No saved artifact workdir for service {service!r}"
                    )
                source = PurePosixPath(directory) / source
            artifacts[str(source)] = artifact.model_copy(update={"source": str(source)})
        required = artifacts.keys()
        if missing := required - original.state.artifacts.keys():
            raise ValueError(f"{label}: submission artifacts were not saved: {missing}")
        if any(
            artifact.required and original.state.artifacts[source] is None
            for source, artifact in artifacts.items()
        ):
            raise ValueError(f"{label}: a required submission artifact is absent")
        task = task.with_data(artifacts=list(artifacts.values()))
        grader = task.grading_task(solver, scoring=policy)
        config = verifier_runtime(
            grader,
            solver,
            original.agent.config.runtime,
            policy,
        )
        if any(not mount.read_only for mount in getattr(config, "mounts", {}).values()):
            raise ValueError("comparison verifiers cannot share writable mounts")
        tests_hash = (
            hashlib.sha256(make_tar(Path(task.data.task_dir) / "tests")).hexdigest()
            if task.data.verifier_image is None
            else None
        )
        prepared[label] = task, policy, config, tests_hash
    for label, (task, policy, config, tests_hash) in prepared.items():
        judge_traces = []
        executor = task.judge_executor
        if executor is not None:

            async def capture_judge(
                request, runtime, executor=executor, judge_traces=judge_traces
            ):
                verdict = await executor(
                    replace(request, on_trace=judge_traces.append), runtime
                )
                if not any(verdict is child for child in judge_traces):
                    judge_traces.append(verdict)
                return verdict

            task.judge_executor = capture_judge
        result = copy.deepcopy(original)
        result.id, result.run = uuid.uuid4().hex, None
        result.task = TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        )
        for key in ("harbor_judge", "harbor_composite"):
            result.info.pop(key, None)
        provenance = {
            "source_trace_id": original.id,
            "source_task_hash": original.task.hash,
            "label": label,
            "config": task.config.model_dump(mode="json"),
            "scoring": policy.model_dump(mode="json"),
            "runtime": config.model_dump(mode="json"),
            "artifacts": fingerprints,
            "tests_sha256": tests_hash,
        }
        result.timing.scoring.start, result.timing.scoring.end = time.time(), 0.0
        try:
            await task.rescore(result, scoring=policy)
        except Exception as error:
            logger.warning("comparison verifier %s failed", label, exc_info=True)
            # Keep failed traces and billed attempts without aborting siblings.
            if result.ok and not result.is_timeout:
                result.record_error(error)
        finally:
            result.timing.scoring.end = time.time()
            result.info["agent_judge_traces"] = [
                json.loads(child.to_json()) for child in judge_traces
            ]
            for child in judge_traces:
                if child.usage is not None:
                    result.extra_usage.append(child.usage)
                result.extra_usage.extend(child.extra_usage)
        result.info["regrading"] = copy.deepcopy(provenance)
        result.state.artifacts = dict(original.state.artifacts)
        comparison.traces[label] = result
    return comparison
