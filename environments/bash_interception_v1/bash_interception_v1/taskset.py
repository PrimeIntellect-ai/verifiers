import json

import verifiers.v1 as vf

BLOCK_SENTINEL = "bash-tool-executed"
BLOCK_COMMAND = f"touch {BLOCK_SENTINEL}"
GIT_SENTINEL = "git-tool-executed"
GIT_COMMAND = f"git --version && touch {GIT_SENTINEL}"


class BashInterceptionTask(vf.Task):
    # Replace the matching tool call before the harness can execute it.
    @vf.on_response
    def block_bash(self, response: vf.Response) -> vf.AssistantMessage | None:
        for call in response.message.tool_calls or []:
            try:
                arguments = json.loads(call.arguments)
            except ValueError:
                continue
            if (
                call.name.lower() in {"bash", "exec_command", "shell_command"}
                and isinstance(arguments, dict)
                and arguments.get("command", arguments.get("cmd")) == BLOCK_COMMAND
            ):
                return vf.AssistantMessage(
                    content="Blocked before Bash harness execution."
                )

    # Or return a synthetic tool result so the agent can continue safely.
    @vf.on_response
    def block_git(
        self, trace: vf.Trace, response: vf.Response
    ) -> vf.ToolMessage | None:
        if trace.task.data.idx != 1:
            return None
        call = next(
            (
                call
                for call in response.message.tool_calls or []
                if call.name.lower() in {"bash", "exec_command", "shell_command"}
                and (arguments := json.loads(call.arguments)).get(
                    "command", arguments.get("cmd")
                )
                == GIT_COMMAND
            ),
            None,
        )
        if call:
            return vf.ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content="This request is blocked. You should answer with something.",
            )

    # Check the filesystem to prove that neither intercepted command ran.
    @vf.reward
    async def intercepted(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        sentinel = BLOCK_SENTINEL if self.data.idx == 0 else GIT_SENTINEL
        executed = (await runtime.run(["test", "-e", sentinel], {})).exit_code == 0
        if self.data.idx == 0:
            return float(bool(trace.interceptions) and not executed)
        return float(
            not executed
            and bool(trace.tool_messages)
            and "This request is blocked. You should answer with something."
            in str(trace.tool_messages[-1].content)
            and trace.num_turns == 2
            and bool(trace.last_reply)
        )


class BashInterceptionTaskset(vf.Taskset[BashInterceptionTask]):
    def load(self) -> list[BashInterceptionTask]:
        return [
            BashInterceptionTask(
                vf.TaskData(
                    idx=0,
                    prompt=f"Use the bash tool once to run `{BLOCK_COMMAND}`, then stop.",
                ),
                self.config.task,
            ),
            BashInterceptionTask(
                vf.TaskData(
                    idx=1,
                    prompt=(
                        f"Use the bash tool once to run `{GIT_COMMAND}`, then follow "
                        "the tool result's instructions."
                    ),
                ),
                self.config.task,
            ),
        ]
