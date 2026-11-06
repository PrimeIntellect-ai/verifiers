"""What a `Harness` is built from, and what wires it into its rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from verifiers.v1.clients.client import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.task import TaskData
from verifiers.v1.types import SystemMessage

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


def with_network_notice(data: TaskData, policy: NetworkPolicyConfig) -> TaskData:
    """`data` with the box's network restrictions appended to its system prompt, or
    `data` itself when the box is unrestricted."""
    if not policy.network_restricted:
        return data
    if not policy.allow:
        notice = "External network access is disabled."
    elif policy.block:
        notice = (
            "External network access is allowed except for these blocked "
            f"destinations: {', '.join(policy.block)}. "
            "Do not circumvent this selection of domains."
        )
    else:
        notice = (
            "External network access is limited to these destinations: "
            f"{', '.join(policy.allow)}. "
            "Do not circumvent this selection of domains."
        )
    system_prompt = "\n\n".join(text for text in (data.system_prompt, notice) if text)
    # Keep existing transcript system messages intact on resume; user-only lists
    # must stay user-only for ACP harnesses.
    if (
        data.system_prompt is None
        and isinstance(data.prompt, list)
        and any(m.role == "system" for m in data.prompt)
    ):
        prompt = [SystemMessage(content=system_prompt), *data.prompt]
        return data.model_copy(update={"prompt": prompt})
    return data.model_copy(update={"system_prompt": system_prompt})
