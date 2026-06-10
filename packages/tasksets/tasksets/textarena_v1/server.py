"""TextArena user simulator: the game engine as a framework-driven conversation partner.

Launched by the framework as a host subprocess per rollout (a `vf.User`). It holds one
TextArena game in memory, set up from the `TEXTARENA_INFO` the taskset injects (game id +
RNG seed), and serves a single `respond` tool: given the model's last
message it steps the game and returns the next observation as a user message, plus whether
the episode is over. When the game ends it writes the game's own outcome
(`env.state.rewards`) to `OUTCOME_FILE` in the runtime workspace, where the taskset's reward
reads it back.

Feedback is trimmed to the latest block (the text after `Feedback:` in the last `[GAME]`
message) so each injected user turn stays small and doesn't duplicate the running history.
"""

import json
import os
import random

from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf

from textarena_v1 import OUTCOME_FILE

import nltk

# textarena pulls nltk corpora on import; keep it quiet and ensure they're present first.
_orig_download = nltk.download
nltk.download = lambda *a, **k: _orig_download(*a, **{**k, "quiet": True})  # type: ignore[assignment]
nltk.download("words")
nltk.download("averaged_perceptron_tagger_eng")

import textarena as ta  # noqa: E402

INFO = json.loads(os.environ["TEXTARENA_INFO"])

# Generic seed: the game derives its whole setup from the global RNG at reset, so seeding it
# reproduces the exact episode the taskset built the instruction from — no per-game keys.
env = ta.make(env_id=INFO["game"])
random.seed(INFO["seed"])
env.reset(num_players=1)

mcp = FastMCP("user")


def latest_feedback(observation: str) -> str:
    latest = observation.split("[GAME]")[-1].strip()
    return latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest


@mcp.tool()
def respond(message: str) -> str:
    """Step the game with the model's move; return the next user message + done."""
    env.step(message)  # TextArena parses the bracketed move out of the message itself
    if env.state.done:
        reward = float((env.state.rewards or {}).get(0, 0.0))
        reason = str(env.state.game_info[0]["reason"])
        with open(OUTCOME_FILE, "w") as f:
            json.dump({"reward": reward, "reason": reason}, f)
        return json.dumps(
            {"messages": [{"role": "user", "content": reason}], "done": True}
        )
    _, observation = env.get_observation()
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": latest_feedback(str(observation))}
            ],
            "done": False,
        }
    )


vf.run_mcp_server(mcp)
