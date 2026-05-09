"""Experimental Braintrust tracing variants of the core environment classes.

Usage::

    from verifiers.envs.experimental.braintrust_tracing.stateful_tool_env import StatefulToolEnv

These classes are drop-in replacements for their non-tracing counterparts.
Set ``BRAINTRUST_API_KEY`` and optionally ``VF_BRAINTRUST_PROJECT`` to enable
trace logging to Braintrust.
"""

from verifiers.envs.experimental.braintrust_tracing.environment import Environment
from verifiers.envs.experimental.braintrust_tracing.multiturn_env import MultiTurnEnv
from verifiers.envs.experimental.braintrust_tracing.stateful_tool_env import (
    StatefulToolEnv,
)
from verifiers.envs.experimental.braintrust_tracing.tool_env import ToolEnv

__all__ = [
    "Environment",
    "MultiTurnEnv",
    "ToolEnv",
    "StatefulToolEnv",
]
