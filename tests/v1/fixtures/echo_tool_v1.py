"""echo (v1, MCP tool): retrieve a stamped echo from a `vf.Toolset`, then report it.

The v1 tool fixture for the e2e matrix. The task constructs an `EchoToolset`
(`vf.Toolset`) in `Task.toolsets`, with one `@vf.tool` method whose placement is
CLI-tunable (`--taskset.task.tools.colocated`, `--taskset.task.tools.runtime.type`):
it runs colocated in the harness's runtime or in its own runtime, and the harness
must reach it wherever it lives. The tool stamps its output with a token the prompt
never reveals, so the reward is 1.0 only if the model actually called the tool —
trivial when the infra works, impossible when it doesn't. The tool is task-agnostic,
so it would also serve taskset-scoped (`Taskset.toolsets`).
"""

from bare_submit_tool_v1 import BareSubmitToolset

import verifiers.v1 as vf
from verifiers.v1.types import content_text

PHRASE = "hello world"
ECHO_TOKEN = "ok-7f3"  # the tool stamps this; only a real tool call can surface it


class EchoToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "echo"  # the model sees `echo_back` (matches the prompt)

    @vf.tool
    def back(self, message: str) -> str:
        """Echo the message back, stamped so the caller can prove the tool ran."""
        return f"{message} [{ECHO_TOKEN}]"


class EchoToolTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()
    bare: bool = False


class EchoToolTask(vf.Task[vf.TaskData, vf.State, EchoToolTaskConfig]):
    @classmethod
    def toolsets(cls, config: EchoToolTaskConfig) -> list[vf.Toolset]:
        toolset = BareSubmitToolset if config.bare else EchoToolset
        return [toolset(config.tools)]

    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        # A stamped TOOL result proves the tool really ran with the phrase.
        results = (content_text(m.content).lower() for m in trace.tool_messages)
        return float(any(PHRASE in r and ECHO_TOKEN in r for r in results))


class EchoToolConfig(vf.TasksetConfig):
    task: EchoToolTaskConfig = EchoToolTaskConfig()


class EchoToolTaskset(vf.Taskset[EchoToolTask, EchoToolConfig]):
    def load(self) -> list[EchoToolTask]:
        if self.config.task.bare:
            prompt = (
                "Call the `submit__now` tool, then reply with exactly what it returns "
                "inside <answer></answer> tags."
            )
        else:
            prompt = (
                f'Call the `echo_back` tool with the message "{PHRASE}", then reply '
                "with exactly what it returns inside <answer></answer> tags."
            )
        return [
            EchoToolTask(
                vf.TaskData(
                    idx=0,
                    prompt=prompt,
                ),
                self.config.task,
            )
        ]


__all__ = ["EchoToolTaskset"]


if __name__ == "__main__":
    EchoToolset.run()
