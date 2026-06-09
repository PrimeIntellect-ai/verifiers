"""TextArena user simulator: the game engine as a framework-driven conversation partner.

Launched by the framework as a host subprocess per rollout (a `vf.User`, structurally a
tool server). It holds one TextArena game in memory, seeded with the rollout's secret word
from the env the taskset injects, and serves a single `respond` tool: given the model's
last message (a guess), it steps the game and returns the game's feedback as the next user
message, plus whether the game is over. The interception server drives it — the harness and
its program never see it.

Feedback is trimmed to the latest block (the text after `Feedback:` in the last `[GAME]`
message) so each injected user turn stays small and doesn't duplicate the running history.
"""

import json
import os
import re

from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf

import nltk

# textarena pulls nltk corpora on import; keep it quiet and ensure they're present first.
_orig_download = nltk.download
nltk.download = lambda *a, **k: _orig_download(*a, **{**k, "quiet": True})  # type: ignore[assignment]
nltk.download("words")
nltk.download("averaged_perceptron_tagger_eng")

import textarena as ta  # noqa: E402

GAME = os.environ["TEXTARENA_GAME"]
ANSWER = os.environ["TEXTARENA_ANSWER"]

env = ta.make(env_id=GAME)
env.reset(num_players=1)
env.state.game_state["secret_word"] = ANSWER

mcp = FastMCP("user")


def parse_guess(message: str) -> str:
    """The model's latest `<guess>...</guess>` (else the whole message), passed to the game
    verbatim — TextArena owns its own action format (Wordle expects a bracketed `[word]`)."""
    matches = re.findall(r"<guess>(.*?)</guess>", message, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else message.strip()


def latest_feedback(observation: str) -> str:
    latest = observation.split("[GAME]")[-1].strip()
    return latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest


@mcp.tool()
def respond(message: str) -> str:
    """Step the game with the model's latest guess; return the next user message + done."""
    env.step(parse_guess(message))
    if env.state.done:
        reason = str(env.state.game_info[0]["reason"])
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
