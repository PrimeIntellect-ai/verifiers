"""Experimental Braintrust tracing variants of the core environment classes.

**Option A — drop-in replacement classes**::

    from verifiers.envs.experimental.braintrust_tracing.stateful_tool_env import StatefulToolEnv

These classes are drop-in replacements for their non-tracing counterparts.

**Option B — monkey-patching integration (recommended)**::

    from verifiers.envs.experimental.braintrust_tracing.integration import setup_verifiers_tracing
    setup_verifiers_tracing()

This patches the *core* classes in-place so all subclasses automatically
produce Braintrust traces.  No import changes needed in your environment.

Both options require ``BRAINTRUST_API_KEY`` (and optionally
``VF_BRAINTRUST_PROJECT``) to be set.
"""

from verifiers.envs.experimental.braintrust_tracing.environment import Environment
from verifiers.envs.experimental.braintrust_tracing.integration import (
    setup_verifiers_tracing,
)
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
    "setup_verifiers_tracing",
]
