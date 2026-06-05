"""The built-in default agent: its `agent.py` (class + config) and the `program.py`
script it stages into the runtime."""

from verifiers.nano.agent.default.agent import DefaultAgent, DefaultAgentConfig

__all__ = ["DefaultAgent", "DefaultAgentConfig"]
