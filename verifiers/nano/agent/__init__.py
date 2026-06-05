"""Agents: programs that run in a runtime and drive the conversation.

`DefaultAgent` stages a tiny openai chat-loop (with a bash tool); `RLMAgent`
installs and runs the rlm CLI. Each agent owns its own provisioning (so the
runtime is just the box). `AgentConfig` is the discriminated config union and
`make_agent` builds the agent for a config.
"""

from typing import Annotated

from pydantic import Field

from verifiers.nano.agent.base import Agent
from verifiers.nano.agent.default import DefaultAgent, DefaultAgentConfig
from verifiers.nano.agent.rlm import RLMAgent, RLMAgentConfig

AgentConfig = Annotated[
    DefaultAgentConfig | RLMAgentConfig, Field(discriminator="kind")
]


def make_agent(config: AgentConfig) -> Agent:
    if isinstance(config, RLMAgentConfig):
        return RLMAgent(config)
    return DefaultAgent(config)


__all__ = [
    "Agent",
    "AgentConfig",
    "make_agent",
    "DefaultAgent",
    "DefaultAgentConfig",
    "RLMAgent",
    "RLMAgentConfig",
]
