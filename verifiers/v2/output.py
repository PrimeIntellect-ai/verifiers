"""On-disk output: results.jsonl (one full transcript per line) + metadata.json.

The transcript is itself the full data dump — written verbatim and consumed by
the platform (visualization) and prime-rl (training). There is no curated
intermediate record; saving is just `transcript.model_dump_json()`.
"""

from pathlib import Path

from verifiers.v2.transcript import Transcript
from verifiers.v2.types import SamplingConfig, StrictBaseModel


class EvalMetadata(StrictBaseModel):
    """metadata.json — one summary object per eval run."""

    env_id: str
    model: str
    base_url: str
    num_tasks: int
    num_rollouts: int
    sampling: SamplingConfig
    date: str
    duration: float
    avg_reward: float
    avg_rewards: dict[str, float]
    avg_metrics: dict[str, float]


def save_results(
    transcripts: list[Transcript], metadata: EvalMetadata, results_dir: Path
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.jsonl").open("w") as f:
        for transcript in transcripts:
            f.write(transcript.model_dump_json(exclude_none=True) + "\n")
    (results_dir / "metadata.json").write_text(
        metadata.model_dump_json(indent=2, exclude_none=True)
    )
    return results_dir
