"""The submit tool of `submit_tool.py`, in the module its tool server runs as."""

from typing import Literal

from pydantic import BaseModel, Field

import verifiers.v1 as vf


class Verdict(BaseModel):
    """What the agent hands in."""

    decision: Literal["accept", "reject"]
    reason: str = Field(min_length=1)


class SubmitState(vf.State):
    submission: Verdict | None = None


class SubmitTools(vf.Toolset[vf.ToolsetConfig, SubmitState]):
    TOOL_PREFIX = None  # the agent sees `submit`, not `submit_tools_submit`

    @vf.tool
    async def submit(self, verdict: Verdict) -> str:
        """Hand in your verdict. Call it once, when your work is done: it ends your run."""
        self.state.submission = verdict
        return "Submitted."


if __name__ == "__main__":
    SubmitTools.run()
