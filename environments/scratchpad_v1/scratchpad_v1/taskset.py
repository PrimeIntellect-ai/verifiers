"""scratchpad-v1 — a per-rollout isolation test for a SHARED, writable tool server.

Each rollout is assigned a unique word and asked to round-trip it through the shared scratchpad
server, then report what came back. The reward is 1 iff the model reports its own word. The server
is `shared` (one instance for the whole eval, simulating an expensive build) yet writable, so this
is a direct test of per-rollout isolation via `self.state`: `uv run eval scratchpad-v1 -n 8 -r 1`
scores a mean reward of 1.0 because each rollout's write to `self.state` stays isolated even though
every rollout shares the one server process.
"""

import verifiers.v1 as vf

from scratchpad_v1.servers.scratchpad import (
    ScratchpadState,
    ScratchpadToolset,
    ScratchpadToolsetConfig,
)

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango",
]  # fmt: skip

INSTRUCTION = (
    'Call the `scratchpad_roundtrip` tool with word="{word}". It returns a single word. '
    "Then reply with that returned word verbatim as your final answer — nothing else."
)


class ScratchpadTask(vf.Task[ScratchpadState]):
    word: str
    tools_config: ScratchpadToolsetConfig = ScratchpadToolsetConfig(shared=True)
    """How the scratchpad toolset is placed (baked from the taskset config at load)."""

    def tools(self) -> list[vf.Toolset]:
        return [ScratchpadToolset(self.tools_config)]

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        # A tool call then a final answer; cap turns so a chatty model still terminates.
        return trace.num_turns >= 4

    @vf.reward(weight=1.0)
    async def isolated(self, trace: vf.Trace) -> float:
        answer = trace.last_reply
        return float(self.word in (answer or ""))


class ScratchpadConfig(vf.TasksetConfig):
    # SHARED + writable: one instance for the whole eval (a stand-in for an expensive build), reused
    # across rollouts. Each rollout's writes are isolated via `self.state` (the per-rollout
    # shared-state channel) — the per-rollout isolation a writable shared server needs.
    tools: ScratchpadToolsetConfig = ScratchpadToolsetConfig(shared=True)


class ScratchpadTaskset(vf.Taskset[ScratchpadTask, ScratchpadConfig]):
    def load_tasks(self) -> list[ScratchpadTask]:
        return [
            ScratchpadTask(
                idx=i,
                word=w,
                prompt=INSTRUCTION.format(word=w),
                tools_config=self.config.tools,
            )
            for i, w in enumerate(WORDS)
        ]
