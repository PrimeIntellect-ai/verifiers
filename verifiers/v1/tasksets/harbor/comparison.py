"""Compare authored verifiers by regrading captured submissions in native runtimes."""

import copy
import logging
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, SerializeAsAny

from verifiers.v1.configs.scoring import ScoringConfig, resolve_scoring
from verifiers.v1.episode import Episode
from verifiers.v1.judges.agent import collect_judge_traces
from verifiers.v1.regrading import RequiresRerun, load_saved_artifacts, prepare_replay
from verifiers.v1.trace import Trace
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
    source: Path | None = None,
) -> VerifierComparison:
    """Regrade a submission with named Harbor tasks in independent runtimes.

    For a trace loaded with ``read_episodes``, pass its run directory as ``source``
    to verify and read the captured archives. A live trace can supply its own
    capture. Agent judges need a bound native executor. Each result episode
    contains the regraded submission and its native judge attempts.
    """
    from rewardkit.compare import format_comparison

    if not verifiers or "original" in verifiers or any(not name for name in verifiers):
        raise ValueError("provide named verifiers; 'original' is reserved")
    artifacts = (
        load_saved_artifacts(source, trace)
        if source is not None
        else trace.state.artifacts
    )
    comparison = VerifierComparison(original=copy.deepcopy(trace), episodes={})
    comparison.original.state.artifacts = dict(artifacts)
    prepared = {}
    for label, task in verifiers.items():
        policy = resolve_scoring(task.scoring, trace.agent.config.scoring)
        if scoring is not None:
            policy = resolve_scoring(policy, scoring)
        policy = policy.model_copy(update={"mode": "isolated"})
        try:
            prepared[label] = prepare_replay(
                trace, task, trace.agent.config.runtime, artifacts, scoring=policy
            )
        except RequiresRerun as error:
            raise RequiresRerun(f"{label}: {error}") from error

    for label, replay in prepared.items():
        result = replay.trace
        result.info["regrading"]["label"] = label
        with collect_judge_traces(replay.grader) as judges:
            try:
                await replay.run()
            except Exception as error:
                logger.warning(
                    "Failed comparison verifier label=%s - %s: %s",
                    label,
                    type(error).__name__,
                    clip(str(error)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        comparison.episodes[label] = Episode(
            task=result.task,
            traces=[result, *judges],
            ok=result.ok and not result.is_timeout,
        )
    comparison.table = format_comparison(comparison.scores)
    return comparison
