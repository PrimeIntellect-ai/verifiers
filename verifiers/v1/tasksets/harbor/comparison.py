"""Compare authored verifiers by regrading captured submissions in native runtimes."""

import copy
import logging
import time
import uuid
from collections.abc import Mapping
from pathlib import PurePosixPath

from pydantic import BaseModel, SerializeAsAny

from verifiers.v1.configs.scoring import ScoringConfig, resolve_scoring
from verifiers.v1.episode import Episode
from verifiers.v1.grading import verifier_runtime
from verifiers.v1.judges.agent import collect_judge_traces
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.utils.artifacts import ARTIFACTS_DIR
from verifiers.v1.utils.logging import clip, get_logger

from .taskset import HarborTask

logger = get_logger(__name__)


class VerifierComparison(BaseModel):
    original: SerializeAsAny[Trace]
    episodes: dict[str, SerializeAsAny[Episode]]
    """Each episode starts with the regraded submission, followed by its judge attempts."""
    table: str = ""
    """RewardKit's matching-name score table; descriptive, not statistical validation."""

    @property
    def scores(self) -> dict[str, dict[str, float]]:
        """Inputs to RewardKit's ``compare`` and ``format_comparison`` helpers.

        Failed or dropped regrades have no numeric verdict. A dropped original
        retains its recorded scores for diagnosis; unique metric names remain here.
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
            if trace.ok
            and not trace.is_timeout
            and (label == "original" or not trace.is_dropped)
            else {}
            for label, trace in {
                "original": self.original,
                **{name: episode.traces[0] for name, episode in self.episodes.items()},
            }.items()
        }


async def compare_verifiers(
    trace: Trace,
    verifiers: Mapping[str, HarborTask],
    *,
    scoring: ScoringConfig | None = None,
) -> VerifierComparison:
    """Regrade a saved submission with named Harbor tasks in independent runtimes.

    Load with ``read_episodes(path, Trace[HarborData], with_artifacts=True)``.
    Agent judges need a bound native executor. Every returned episode can be saved
    with ``write_episode``; judge attempts retain their own evidence and usage.
    ``table`` uses RewardKit's formatter; ``scores`` retains all named metrics.
    """
    from rewardkit.compare import format_comparison

    if not verifiers or "original" in verifiers or any(not name for name in verifiers):
        raise ValueError("provide named verifiers; 'original' is reserved")
    if not trace.is_completed or trace.has_error or trace.is_timeout:
        raise ValueError("comparison requires a completed, non-errored submission")
    solver = trace.agent.runtime
    if solver is None:
        raise ValueError("saved submission has no solver runtime metadata")
    if ARTIFACTS_DIR not in trace.state.artifacts:
        raise ValueError("Load captured submission artifacts with with_artifacts=True")
    comparison = VerifierComparison(original=copy.deepcopy(trace), episodes={})
    workdir = PurePosixPath(getattr(solver, "workdir", None) or "/")
    prepared = {}
    # Check every variant's inputs before provisioning or making model calls.
    for label, definition in verifiers.items():
        task = copy.deepcopy(definition)
        task.scoring_deferred = False
        policy = resolve_scoring(task.scoring, trace.agent.config.scoring)
        policy = resolve_scoring(policy, scoring or ScoringConfig()).model_copy(
            update={"mode": "isolated"}
        )
        artifacts = {}
        for artifact in [*trace.task.data.artifacts, *task.data.artifacts]:
            source = PurePosixPath(artifact.source)
            if not source.is_absolute():
                service = getattr(artifact, "service", "main")
                directory = (
                    workdir
                    if service == "main"
                    else trace.info.get("harbor_artifact_workdirs", {}).get(service)
                )
                if directory is None:
                    raise ValueError(
                        f"No saved artifact workdir for service {service!r}"
                    )
                source = PurePosixPath(directory) / source
            artifacts[str(source)] = artifact.model_copy(update={"source": str(source)})
        if missing := artifacts.keys() - trace.state.artifacts.keys():
            raise ValueError(f"{label}: submission artifacts were not saved: {missing}")
        if any(
            artifact.required and trace.state.artifacts[source] is None
            for source, artifact in artifacts.items()
        ):
            raise ValueError(f"{label}: a required submission artifact is absent")
        task = task.with_data(artifacts=list(artifacts.values()))
        config = verifier_runtime(
            task.grading_task(solver, scoring=policy),
            solver,
            trace.agent.config.runtime,
            policy,
        )
        if any(not mount.read_only for mount in getattr(config, "mounts", {}).values()):
            raise ValueError("comparison verifiers cannot share writable mounts")
        prepared[label] = task, policy
    for label, (task, policy) in prepared.items():
        result = copy.deepcopy(comparison.original)
        result.id, result.attempt, result.run = uuid.uuid4().hex, 1, None
        result.task = TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        )
        result.info["regrading"] = {
            "source_trace_id": trace.id,
            "source_attempt": trace.attempt,
            "label": label,
        }
        result.timing.scoring.start, result.timing.scoring.end = time.time(), 0.0
        with collect_judge_traces(task) as judges:
            try:
                await task.rescore(result, scoring=policy)
            except Exception as error:
                logger.warning(
                    "Failed comparison verifier label=%s - %s: %s",
                    label,
                    type(error).__name__,
                    clip(str(error)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                if result.ok and not result.is_timeout:
                    result.record_error(error)
            finally:
                result.timing.scoring.end = time.time()
        result.state.artifacts = dict(comparison.original.state.artifacts)
        comparison.episodes[label] = Episode(
            task=result.task, traces=[result, *judges], ok=result.ok
        )
    comparison.table = format_comparison(comparison.scores)
    return comparison
