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


class ScratchpadTaskData(vf.TaskData):
    word: str


class ScratchpadTask(vf.Task[ScratchpadTaskData, ScratchpadState]):
    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        # A tool call then a final answer; cap turns so a chatty model still terminates.
        return trace.num_turns >= 4

    @vf.reward(weight=1.0)
    async def isolated(self, trace: vf.Trace) -> float:
        answer = trace.last_reply
        return float(self.data.word in (answer or ""))


class ScratchpadConfig(vf.TasksetConfig):
    # SHARED + writable: one instance for the whole eval (a stand-in for an expensive build),
    # reused across rollouts. Each rollout's writes are isolated via `self.state` (the
    # per-rollout shared-state channel) — the per-rollout isolation a writable shared server
    # needs. Knobs at the taskset level: `--taskset.tools.runtime.type docker`.
    tools: ScratchpadToolsetConfig = ScratchpadToolsetConfig()


class ScratchpadTaskset(vf.Taskset[ScratchpadTask, ScratchpadConfig]):
    tools = (ScratchpadToolset,)
    # Declared on the TASKSET: shared scope is structural (one eval-level server), built
    # with `ScratchpadConfig.tools` (exact-type match) via `Taskset.server_config`.

    def load(self) -> list[ScratchpadTaskData]:
        return [
            ScratchpadTaskData(
                idx=i,
                word=w,
                prompt=INSTRUCTION.format(word=w),
            )
            for i, w in enumerate(WORDS)
        ]
