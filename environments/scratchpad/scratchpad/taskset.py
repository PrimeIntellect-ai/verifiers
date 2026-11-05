"""scratchpad — a per-rollout isolation test for a SHARED, writable tool server.

Each rollout is assigned a unique word and asked to round-trip it through the shared scratchpad
server, then report what came back. The reward is 1 iff the model reports its own word. The server
is shared across rollouts yet writable, so this
is a direct test of per-rollout isolation via `self.state`: a run of 8 tasks
scores a mean reward of 1.0 because each rollout's write to `self.state` stays isolated even though
every rollout shares its server process.
"""

import verifiers.v1 as vf
from scratchpad.servers.scratchpad import (
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


class ScratchpadTaskConfig(vf.TaskConfig):
    tools: ScratchpadToolsetConfig = ScratchpadToolsetConfig(shared=True)


class ScratchpadTask(
    vf.Task[ScratchpadTaskData, ScratchpadState, ScratchpadTaskConfig]
):
    @classmethod
    def toolsets(cls, config: ScratchpadTaskConfig) -> list[vf.Toolset]:
        return [ScratchpadToolset(config.tools)]

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        # A tool call then a final answer; cap turns so a chatty model still terminates.
        return trace.num_turns >= 4

    @vf.reward(weight=1.0)
    async def isolated(self, trace: vf.Trace) -> float:
        answer = trace.last_reply
        return float(self.data.word in (answer or ""))


class ScratchpadConfig(vf.TasksetConfig):
    task: ScratchpadTaskConfig = ScratchpadTaskConfig()


class ScratchpadTaskset(vf.Taskset[ScratchpadTask, ScratchpadConfig]):
    def load(self) -> list[ScratchpadTask]:
        return [
            ScratchpadTask(
                ScratchpadTaskData(
                    word=w,
                    prompt=INSTRUCTION.format(word=w),
                ),
                self.config.task,
            )
            for w in WORDS
        ]
