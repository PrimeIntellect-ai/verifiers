"""echo (v1, multi-turn): echo a phrase per turn, driven by a container-safe user simulator.

The v1 multi-turn fixture for the e2e matrix. The user simulator is a self-contained uv
`script` (staged into the runtime + run via `uv run`, like a tool server), so it works on
every runtime — unlike a host-bound `command=[sys.executable, ...]` user-sim, which can't run
inside a container. A user-sim is a task tool, so this needs a tool-supporting harness.
"""

import json

import verifiers.v1 as vf

PHRASES = ["hello world", "goodbye world"]
SYSTEM = "Repeat the user's message back to them exactly, with no extra words."

# A self-contained uv user simulator: one `respond` tool that injects the next phrase per turn.
USER_SCRIPT = b"""# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp"]
# ///
import json, os
from mcp.server.fastmcp import FastMCP

PHRASES = json.loads(os.environ["ECHO_PHRASES"])
mcp = FastMCP("user", host="127.0.0.1", port=int(os.environ["MCP_PORT"]))
_turns = 0


@mcp.tool()
def respond(message: str) -> str:
    global _turns
    _turns += 1
    if _turns >= len(PHRASES):
        return json.dumps({"messages": [], "done": True})
    return json.dumps(
        {"messages": [{"role": "user", "content": PHRASES[_turns]}], "done": False}
    )


mcp.run(transport="streamable-http")
"""


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoMultiTask(vf.Task):
    phrases: list[str]


class EchoMultiConfig(vf.TasksetConfig):
    phrases: list[str] = PHRASES


class EchoMultiTaskset(vf.Taskset[EchoMultiTask, EchoMultiConfig]):
    def load_tasks(self) -> list[EchoMultiTask]:
        return [
            EchoMultiTask(
                idx=0,
                instruction=self.config.phrases[0],
                system_prompt=SYSTEM,
                phrases=self.config.phrases,
            )
        ]

    def user(self, task: EchoMultiTask) -> vf.User:
        return vf.User(
            name="user",
            script=USER_SCRIPT,
            env={"ECHO_PHRASES": json.dumps(task.phrases)},
        )

    @vf.reward(weight=1.0)
    async def echoed(self, task: EchoMultiTask, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        phrases = task.phrases
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(_key(p) in _key(r or "") for r, p in zip(replies, phrases))
        return matched / len(phrases)


def load_taskset(config: EchoMultiConfig) -> EchoMultiTaskset:
    return EchoMultiTaskset(config)
