"""echo (v1, MCP tool): retrieve a stamped echo from a `vf.Toolset`, then report it.

The v1 tool fixture for the e2e matrix. The taskset declares an `EchoToolset` (`vf.Toolset`)
with one `@vf.tool` method whose placement is CLI-tunable (`--taskset.task.tools.colocated`,
`--taskset.task.tools.shared`, `--taskset.task.tools.runtime.type`): it runs colocated in the harness's
runtime, shared once per eval, or in its own runtime, and the harness must reach it wherever it
lives. The tool stamps its output with a token the prompt never reveals, so the reward is
1.0 only if the model actually called the tool — trivial when the infra works, impossible when
it doesn't. The tool is task-agnostic, so it works in `shared` placement too.
"""

import verifiers.v1 as vf

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


class EchoToolTask(vf.Task[vf.TaskData, vf.State, EchoToolTaskConfig]):
    tools = (EchoToolset,)
    # Built with the task config's `tools` field (placement stays CLI-tunable via
    # --taskset.task.tools.*), resolved by `Task.server_config`.

    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        # The stamped token reaches the answer only if the model called the MCP tool.
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        last = (last or "").lower()
        return float(PHRASE in last and ECHO_TOKEN in last)


class EchoToolConfig(vf.TasksetConfig):
    task: EchoToolTaskConfig = EchoToolTaskConfig()


class EchoToolTaskset(vf.Taskset[EchoToolTask, EchoToolConfig]):
    def load(self) -> list[vf.TaskData]:
        return [
            vf.TaskData(
                idx=0,
                prompt=(
                    f'Call the `echo_back` tool with the message "{PHRASE}", then reply '
                    "with exactly what it returns inside <answer></answer> tags."
                ),
            )
        ]


__all__ = ["EchoToolTaskset"]


if __name__ == "__main__":
    EchoToolset.run()
