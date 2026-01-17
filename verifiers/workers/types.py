"""
Request/Response types for EnvWorker IPC.

These dataclasses are designed to be serializable for cross-process communication.
"""

from dataclasses import dataclass, field

from verifiers.types import RolloutInput, RolloutResult, SamplingArgs


@dataclass
class Shutdown:
    """Sentinel to signal worker shutdown."""

    pass


@dataclass
class RolloutRequest:
    """Request to run a group of rollouts.

    The orchestrator (vf-eval or prime-rl) is responsible for constructing
    group_inputs with the desired duplication.
    """

    group_inputs: list[RolloutInput]
    example_id: int
    model_name: str
    sampling_args: SamplingArgs = field(default_factory=dict)
    independent_scoring: bool = False
    state_columns: list[str] = field(default_factory=list)


@dataclass
class RolloutResponse:
    """Response containing rollout results."""

    example_id: int
    results: list[RolloutResult]


@dataclass
class MetadataRequest:
    """Request for environment metadata and dataset."""

    num_examples: int = -1


@dataclass
class MetadataResponse:
    """Response with environment metadata and dataset."""

    dataset: list[dict]
    sampling_args: SamplingArgs
    max_seq_len: int | None
