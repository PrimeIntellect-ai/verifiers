import json

import verifiers.v1 as vf

STOP_SENTINEL = "bash-tool-executed"
STOP_COMMAND = f"touch {STOP_SENTINEL}"
REWRITE_SENTINEL = "git-tool-executed"
REWRITE_COMMAND = f"git --version && touch {REWRITE_SENTINEL}"


def command(message: vf.AssistantMessage, snippet: str) -> vf.ToolCall | None:
    for call in message.tool_calls or []:
        if call.name != "bash":
            continue
        try:
            arguments = json.loads(call.arguments)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(arguments, dict) and snippet in arguments.get("command", ""):
            return call
    return None


class BashInterceptionTask(vf.Task):
    @vf.stop
    def stop_bash(self, response: vf.Response) -> bool:
        # The response is still buffered here. Stopping prevents Bash from
        # receiving the proposed call, so the command cannot execute.
        return (
            self.data.idx == 0 and command(response.message, STOP_COMMAND) is not None
        )

    @vf.intercept
    def rewrite_git(self, request: vf.Request) -> vf.Request | None:
        if self.data.idx != 1 or not request.messages:
            return None
        result = request.messages[-1]
        if not isinstance(result, vf.ToolMessage) or result.content:
            return None
        assistant = next(
            (
                message
                for message in reversed(request.messages[:-1])
                if isinstance(message, vf.AssistantMessage)
                and any(
                    call.id == result.tool_call_id for call in message.tool_calls or []
                )
            ),
            None,
        )
        call = command(assistant, REWRITE_SENTINEL) if assistant is not None else None
        if call is None or call.id != result.tool_call_id:
            return None
        # This fills the pre-execution result. Bash returns it to the model instead
        # of running the command, and the rollout continues normally.
        replacement = result.model_copy(
            update={
                "content": "This request is blocked. You should answer with something."
            }
        )
        return request.model_copy(
            update={"messages": [*request.messages[:-1], replacement]}
        )

    @vf.reward
    async def rewritten(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        sentinel = STOP_SENTINEL if self.data.idx == 0 else REWRITE_SENTINEL
        executed = (await runtime.run(["test", "-e", sentinel], {})).exit_code == 0
        if self.data.idx == 0:
            return float(not executed and trace.stop_condition == "stop_bash")
        return float(
            not executed
            and bool(trace.tool_messages)
            and "This request is blocked" in str(trace.tool_messages[-1].content)
            and trace.num_turns == 2
        )


class BashInterceptionTaskset(vf.Taskset[BashInterceptionTask]):
    def load(self) -> list[BashInterceptionTask]:
        prompts = (
            f"Use bash once to run `{STOP_COMMAND}`, then stop.",
            (
                f"Use bash exactly once to run `{REWRITE_COMMAND}`. Whatever the tool "
                "returns, do not call another tool; answer with something."
            ),
        )
        return [
            BashInterceptionTask(vf.TaskData(idx=i, prompt=prompt), self.config.task)
            for i, prompt in enumerate(prompts)
        ]
