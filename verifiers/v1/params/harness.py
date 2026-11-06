"""What a `Harness` is built from, and what wires it into its rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from verifiers.v1.clients.client import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.task import TaskData

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


@dataclass(frozen=True)
class HarnessParams(Generic[ConfigT]):
    """One rollout's harness: its config (the task's skills merged in) and the model
    it serves."""

    config: ConfigT
    model: ModelContext


@dataclass(frozen=True)
class HarnessWiring:
    """How a started harness reaches its rollout, known once the rollout's
    interception and tool servers are up."""

    endpoint: str
    """The interception's model route (`{base_url}/v1`)."""
    secret: str
    """Bearer token for `endpoint`."""
    mcp_urls: dict[str, str]
    """The task's tool servers, by name."""
    data: TaskData
    """The task as the harness sees it: network notice and prompt rewrites applied."""
    tool_interception_url: str | None = None
    """The `/tool` gate, when task interceptors or stops must see each tool call."""
