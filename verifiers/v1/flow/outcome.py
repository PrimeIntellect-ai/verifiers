"""The one routing convention: a node emits an outcome name.

An agent calls the `submit_outcome` MCP tool (served by the engine for every agent
node that declares `outcomes`); the name lands on the rollout's state when the task's
`State` carries an `outcome` field (`OutcomeState`). Any node, agent or command, may
instead end its text with an `Outcome: <name>` line.
"""

from __future__ import annotations

import re

from pydantic import Field

import verifiers.v1 as vf

OUTCOME_LINE = re.compile(r"^Outcome:\s*([A-Za-z0-9_.-]+)\s*$", re.MULTILINE)


class OutcomeState(vf.State):
    """Give a Task this state (or a subclass) to receive `submit_outcome` results."""

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


def parse_outcome(text: str | None) -> tuple[str | None, str]:
    """The last `Outcome: <name>` line in `text` and whatever follows it."""
    matches = list(OUTCOME_LINE.finditer(text or ""))
    if not matches:
        return None, ""
    last = matches[-1]
    return last.group(1), (text or "")[last.end() :].strip()


def outcome_of(trace: vf.Trace) -> tuple[str | None, str]:
    """A trace's submitted outcome, else the `Outcome:` line of its last reply."""
    outcome = getattr(trace.state, "outcome", None)
    if outcome:
        return outcome, getattr(trace.state, "summary", "")
    return parse_outcome(trace.last_reply)


if __name__ == "__main__":
    OutcomeTools.run()
