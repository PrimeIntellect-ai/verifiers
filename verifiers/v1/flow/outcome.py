"""The one platform tool: `submit_outcome(name, summary)` with a closed allowlist.

Installed as an MCP toolset on any harness that supports MCP. The submitted name
lands in the rollout's `OutcomeState` (mirrored onto `trace.state`) and a `@vf.stop`
hook ends the rollout. Harnesses without MCP fall back to an `Outcome: <name>`
first line in the final reply, as qx does.
"""

from __future__ import annotations

import re

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.task import DataT

OUTCOME_LINE = re.compile(r"^Outcome:\s*([A-Za-z0-9_.-]+)\s*$", re.MULTILINE)


class OutcomeState(vf.State):
    outcome: str | None = None
    summary: str = ""


class OutcomeToolsConfig(vf.ToolsetConfig):
    allowed: list[str] = Field(default_factory=list)


class OutcomeTools(vf.Toolset[OutcomeToolsConfig, OutcomeState]):
    TOOL_PREFIX = None

    @vf.tool
    async def submit_outcome(self, outcome: str, summary: str = "") -> str:
        """Finish this stage with one of the allowed outcome names and a short summary."""
        if outcome not in self.config.allowed:
            return f"refused: outcome must be one of {self.config.allowed}"
        self.state.outcome = outcome
        self.state.summary = summary
        return "recorded"


class FlowTaskConfig(vf.TaskConfig):
    outcomes: list[str] = Field(default_factory=list)
    """Set by the engine from the node's declared outcomes."""


class FlowTask(vf.Task[DataT, OutcomeState, FlowTaskConfig]):
    """Base for tasks run by agent nodes: carries the outcome tool and its stop."""

    @classmethod
    def toolsets(cls, config: FlowTaskConfig) -> list[vf.Toolset]:
        if not config.outcomes:
            return []
        return [OutcomeTools(OutcomeToolsConfig(allowed=list(config.outcomes)))]

    @vf.stop
    async def outcome_submitted(self, trace: vf.Trace) -> bool:
        return bool(getattr(trace.state, "outcome", None))


def outcome_of(trace: vf.Trace, allowed: list[str]) -> tuple[str | None, str]:
    """The trace's submitted outcome, else the `Outcome:` line of its last reply."""
    state = trace.state
    outcome = getattr(state, "outcome", None)
    if outcome in allowed:
        return outcome, getattr(state, "summary", "")
    match = OUTCOME_LINE.search(trace.last_reply or "")
    if match and match.group(1) in allowed:
        return match.group(1), (trace.last_reply or "")[match.end() :].strip()
    return None, ""


if __name__ == "__main__":
    OutcomeTools.run()
