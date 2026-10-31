"""scratchpad — a per-rollout isolation test for a SHARED, writable tool server.

Each rollout is assigned a unique word and asked to round-trip it through the shared scratchpad
server, then report what came back. The reward is 1 iff the server returned the rollout's own word
and the model reported it. The server is taskset-scoped (one instance per environment worker) yet
writable, so this is a direct test of per-rollout isolation via `self.state`:
`uv run vf-eval scratchpad -n 8 -r 1` scores a mean reward of 1.0 because each rollout's write to
`self.state` stays isolated even though every rollout handled by that worker shares its server
process. The word is in the prompt, so the reward reads the tool result, not just the reply: a
reply that skips the tool, or reports its own word over a leaked one, scores 0.
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


def roundtrip_results(trace: vf.Trace) -> list[str]:
    """The text of every `scratchpad_roundtrip` result in the trace."""
    call_ids = {
        call.id
        for message in trace.assistant_messages
        for call in message.tool_calls or []
        if call.name == "scratchpad_roundtrip"
    }
    return [
        message.content.strip()
        if isinstance(message.content, str)
        else "\n".join(
            part.text
            for part in message.content
            if isinstance(part, vf.TextContentPart)
        ).strip()
        for message in trace.tool_messages
        if message.tool_call_id in call_ids
    ]


class ScratchpadTask(vf.Task[ScratchpadTaskData, ScratchpadState]):
    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        # A tool call then a final answer; cap turns so a chatty model still terminates.
        return trace.num_turns >= 4

    @vf.reward(weight=1.0)
    async def isolated(self, trace: vf.Trace) -> float:
        results = roundtrip_results(trace)
        return float(
            bool(results)
            and all(result == self.data.word for result in results)
            and self.data.word in trace.last_reply
        )


class ScratchpadConfig(vf.TasksetConfig):
    tools: ScratchpadToolsetConfig = ScratchpadToolsetConfig()


class ScratchpadTaskset(vf.Taskset[ScratchpadTask, ScratchpadConfig]):
    @classmethod
    def toolsets(cls, config: ScratchpadConfig) -> list[vf.Toolset]:
        return [ScratchpadToolset(config.tools)]

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
