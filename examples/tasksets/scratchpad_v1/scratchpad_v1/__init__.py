"""scratchpad-v1 — a contrived taskset that tests per-rollout isolation of a SHARED, writable
tool server.

Each rollout is assigned a unique word and asked to round-trip it through the shared scratchpad
tool, then report what came back. The reward is 1 iff the model reports its own word. Without
per-rollout isolation, concurrent rollouts clobber the shared slot and report the wrong word —
so running `uv run eval scratchpad-v1 -n 5 -c 5` is a direct test of the multiplexing.
"""

import sys

import verifiers.v1 as vf

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango",
]

INSTRUCTION = (
    'Call the `scratchpad_roundtrip` tool with word="{word}". It returns a string like '
    "`X|Y`. Then reply with that returned string verbatim as your final answer — nothing else."
)


class ScratchpadTask(vf.Task):
    word: str


class ScratchpadConfig(vf.TasksetConfig):
    # SHARED: one server for the whole eval (simulated expensive setup), reused across
    # rollouts. It is writable + stateful — exactly the per-rollout isolation under test.
    tools: vf.ToolsConfig = vf.ToolsConfig(colocated=False, shared=True)


class ScratchpadTaskset(vf.Taskset[ScratchpadTask, ScratchpadConfig]):
    def load_tasks(self) -> list[ScratchpadTask]:
        return [
            ScratchpadTask(idx=i, word=w, instruction=INSTRUCTION.format(word=w))
            for i, w in enumerate(WORDS)
        ]

    def tools(self, task: ScratchpadTask) -> list[vf.Tools]:
        return [
            vf.Tools(
                name="scratchpad", command=[sys.executable, "-m", "scratchpad_v1.server"]
            )
        ]

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        # A tool call then a final answer; cap turns so a chatty model still terminates.
        return trace.num_turns >= 4

    @vf.reward(weight=1.0)
    async def isolated(self, task: ScratchpadTask, trace: vf.Trace) -> float:
        answer = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.word in (answer or ""))


def load_taskset(config: ScratchpadConfig) -> ScratchpadTaskset:
    return ScratchpadTaskset(config)
