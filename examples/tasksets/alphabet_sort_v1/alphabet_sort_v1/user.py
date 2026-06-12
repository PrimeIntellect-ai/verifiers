"""alphabet-sort-v1 user simulator: replays the episode's pre-generated follow-up turns.

Launched by the framework as a per-rollout subprocess (a `vf.User`). It holds the task's
follow-up prompts and turn count (from `ALPHABET_SORT_INFO`) and serves a single `respond`
tool: after each assistant turn it injects the next follow-up as a user message, until all
turns are done. This colocates v0's `MultiTurnEnv.env_response` with the agent — the harness
drives it, never the model.
"""

import json
import os

from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf

INFO = json.loads(os.environ["ALPHABET_SORT_INFO"])
FOLLOW_UPS = INFO["follow_ups"]
NUM_TURNS = INFO["num_turns"]

mcp = FastMCP("user")

# One `respond` call per assistant turn; track how many turns the model has taken.
_turns = 0


@mcp.tool()
def respond(message: str) -> str:
    """Inject the next follow-up turn, or end the episode once all turns are done."""
    global _turns
    _turns += 1
    if _turns >= NUM_TURNS:
        return json.dumps({"messages": [], "done": True})
    return json.dumps(
        {
            "messages": [{"role": "user", "content": FOLLOW_UPS[_turns - 1]}],
            "done": False,
        }
    )


vf.run_mcp_server(mcp)
