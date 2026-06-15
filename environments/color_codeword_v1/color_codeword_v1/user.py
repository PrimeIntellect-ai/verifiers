"""color-codeword-v1 user simulator: reveals each turn's colored squares after the prior answer.

Launched per rollout as a `vf.User` subprocess. Holds the episode's per-turn colors (from
`COLOR_CODEWORD_INFO`) and serves a single `respond` tool: after each assistant turn it injects the
next turn's squares as a user message (image_url parts), until every `max_turns` turn has been
answered. This colocates v0's per-turn image presentation with the agent — the framework drives it,
never the model.
"""

import json
import os

from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf

from color_codeword_v1 import image_content, turn_text

INFO = json.loads(os.environ["COLOR_CODEWORD_INFO"])
COLORS_PER_TURN = INFO["colors_per_turn"]
MAX_TURNS = INFO["max_turns"]

mcp = FastMCP("user")

_turns = 0  # assistant turns taken so far (one respond call per turn)


@mcp.tool()
def respond(message: str) -> str:
    """Show the next turn's squares, or end the episode once every turn has been answered."""
    global _turns
    _turns += 1
    if _turns >= MAX_TURNS:
        return json.dumps({"messages": [], "done": True})
    colors = COLORS_PER_TURN[_turns]
    total = sum(len(COLORS_PER_TURN[t]) for t in range(_turns + 1))
    text = turn_text(_turns, len(colors), MAX_TURNS, total)
    return json.dumps(
        {
            "messages": [{"role": "user", "content": image_content(colors, text)}],
            "done": False,
        }
    )


vf.run_mcp_server(mcp)
