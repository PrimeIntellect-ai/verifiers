"""Shared types for the solver taskset and its toolset (kept here to avoid an import cycle)."""

from __future__ import annotations

import verifiers.v1 as vf


class GeneralAgentState(vf.State):
    """Per-rollout state shared between the toolset and scoring. The toolset writes the agent's
    live task DB here after each tool call (as a plain dict); the taskset's reward reads it back
    off `trace.state` to hash against the gold solution."""

    db: dict | None = None


class GeneralAgentToolsetConfig(vf.ToolsetConfig):
    """Placement for the per-task tool server (defaults to its own host runtime, where the task
    cache lives). The task to serve is fetched per rollout over the `/task` channel."""
