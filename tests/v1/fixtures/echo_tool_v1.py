"""echo (v1, MCP tool): retrieve a stamped echo from a `vf.Toolset`, then report it.

The v1 tool fixture for the e2e matrix. The task declares an `EchoToolset` (`vf.Toolset`)
with one `@vf.tool` method whose placement is CLI-tunable (`--taskset.task.tools.colocated`,
`--taskset.task.tools.runtime.type`): it runs colocated in the harness's runtime or in its
own runtime, and the harness must reach it wherever it lives. The tool stamps its output
with a token the prompt never reveals, so the reward is 1.0 only if the model actually
called the tool — trivial when the infra works, impossible when it doesn't. The tool is
task-agnostic, so it would also serve taskset-scoped (`Taskset.tools`).
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

    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        # The stamped token surfaces in an ASSISTANT message only if the model called
        # the MCP tool and relayed its result — wherever in the exchange that happened
        # (in a conversation the last turn may be a closing pleasantry).
        replies = ((m.content or "").lower() for m in trace.assistant_messages)
        return float(any(PHRASE in r and ECHO_TOKEN in r for r in replies))


class EchoToolConfig(vf.TasksetConfig):
    task: EchoToolTaskConfig = EchoToolTaskConfig()


class EchoToolTaskset(vf.Taskset[EchoToolTask, EchoToolConfig]):
    def load(self) -> list[EchoToolTask]:
        return [
            EchoToolTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        f'Call the `echo_back` tool with the message "{PHRASE}", then reply '
                        "with exactly what it returns inside <answer></answer> tags."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["EchoToolTaskset"]


if __name__ == "__main__":
    EchoToolset.run()
