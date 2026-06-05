"""The per-rollout collaborators, bundled so agents hold no rollout state."""

from dataclasses import dataclass

from verifiers.nano.clients import Client
from verifiers.nano.toolset import Toolset
from verifiers.nano.types import SamplingConfig
from verifiers.nano.user import User


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs. Built by the Environment."""

    client: Client
    model: str
    sampling: SamplingConfig
    user: User | None = None
    toolset: Toolset | None = None
