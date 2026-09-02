"""tool-interception: deny a tool call before it runs, rewrite a completed result, and
observe a failed one — in one deterministic three-step shell sequence.

The request hook sees each proposed call first (an empty result for it, before the
harness's gate lets it run) and every completed result before the model does. Needs a
harness that asks the rollout's gate (`SUPPORTS_TOOL_INTERCEPTION`), e.g. `bash`.
"""

import json

import verifiers.v1 as vf
from verifiers.v1.types import content_text

BLOCKED_SENTINEL = ".vf-tool-interception-blocked"
BLOCKED_COMMAND = f"touch {BLOCKED_SENTINEL}"
SUCCESS_MARKER = "vf-native-tool-result"
SUCCESS_COMMAND = f"printf {SUCCESS_MARKER}"
FAILURE_MARKER = "vf-native-tool-failure"
FAILURE_COMMAND = f"sh -c 'printf {FAILURE_MARKER} >&2; exit 1'"
EXPECTED_COMMANDS = [BLOCKED_COMMAND, SUCCESS_COMMAND, FAILURE_COMMAND]
BLOCKED_RESULT = (
    "Step 1 completed: the command was blocked before execution. You have not finished "
    f"the task. Continue with step 2 by using the shell exactly once to run `{SUCCESS_COMMAND}`."
)
SUCCESS_RESULT = (
    "Step 2 completed: the command ran, but this text replaced its result before the "
    f"agent saw it. Continue with step 3 by using the shell exactly once to run "
    f"`{FAILURE_COMMAND}`. After that tool returns its expected error, finish without "
    "using any more tools."
)


def shell_command(call: vf.ToolCall) -> str | None:
    """The shell command a call runs, across the agents' shell tool shapes."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    command = arguments.get("command", arguments.get("cmd"))
    if isinstance(command, list):
        command = command[-1] if command else None
    return command if isinstance(command, str) else None


def issuing_call(request: vf.Request, result: vf.ToolMessage) -> vf.ToolCall | None:
    for message in reversed(request.messages[:-1]):
        if not isinstance(message, vf.AssistantMessage):
            continue
        return next(
            (
                call
                for call in message.tool_calls or []
                if call.id == result.tool_call_id
            ),
            None,
        )
    return None


class ToolInterceptionTask(vf.Task):
    @vf.intercept
    def rewrite_tool_result(
        self, request: vf.Request, trace: vf.Trace
    ) -> vf.Request | None:
        if not request.messages or not isinstance(request.messages[-1], vf.ToolMessage):
            return None
        result = request.messages[-1]
        call = issuing_call(request, result)
        command = shell_command(call) if call is not None else None
        output = content_text(result.content)
        if command == FAILURE_COMMAND and FAILURE_MARKER in output:
            trace.info["native_failure_observed"] = True
            return None
        if command == BLOCKED_COMMAND and not output:
            replacement = BLOCKED_RESULT
        elif command == SUCCESS_COMMAND and SUCCESS_MARKER in output:
            replacement = SUCCESS_RESULT
        else:
            return None
        return request.model_copy(
            update={
                "messages": [
                    *request.messages[:-1],
                    result.model_copy(update={"content": replacement}),
                ]
            }
        )

    @vf.reward
    async def intercepted(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        blocked = (
            await runtime.run(["test", "-e", BLOCKED_SENTINEL], {})
        ).exit_code == 0
        messages = trace.messages
        calls = [
            call
            for message in messages
            if isinstance(message, vf.AssistantMessage)
            for call in message.tool_calls or []
        ]
        results = [
            message for message in messages if isinstance(message, vf.ToolMessage)
        ]
        result_text = [content_text(message.content) for message in results]
        checks = {
            "ordered_tool_calls": [shell_command(call) for call in calls]
            == EXPECTED_COMMANDS,
            "ordered_tool_results": len(results) == 3
            and [message.tool_call_id for message in results]
            == [call.id for call in calls],
            "blocked_before_execution": not blocked,
            "expected_results": len(result_text) == 3
            and result_text[0] == BLOCKED_RESULT
            and result_text[1] == SUCCESS_RESULT
            and FAILURE_MARKER in result_text[2],
            "rewrites": [record.handler for record in trace.request_rewrites]
            == ["rewrite_tool_result", "rewrite_tool_result"],
            "failure_observed": trace.info.get("native_failure_observed") is True,
        }
        trace.info["tool_interception_checks"] = checks
        return float(all(checks.values()))


class ToolInterceptionTaskset(vf.Taskset[ToolInterceptionTask]):
    def load(self) -> list[ToolInterceptionTask]:
        prompt = (
            "Complete a three-step shell-tool sequence, one call at a time. Begin by using "
            f"the shell exactly once to run `{BLOCKED_COMMAND}`. Wait for that result and "
            "follow its next instruction exactly; each result reveals only the next step. "
            "The blocked first call is expected. The third step is complete when its "
            "instructed command returns the expected error; finish then without using "
            "another tool."
        )
        return [
            ToolInterceptionTask(
                vf.TaskData(
                    idx=0,
                    prompt=prompt,
                    system_prompt=(
                        "Follow the requested shell-tool sequence exactly and never issue its "
                        "calls in parallel. Continue after the blocked first call, then finish "
                        "after the instructed third command returns its expected error."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["ToolInterceptionTaskset"]
