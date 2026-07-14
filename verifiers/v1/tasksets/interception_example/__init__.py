"""Harness-agnostic examples of blocking and rewriting model messages."""

import json

import verifiers.v1 as vf

BLOCKED_DOMAIN = "example.com"
BLOCKED_SEARCH = "Blocked a server web-search response containing a restricted domain."
BLOCKED_RM = "Blocked a Bash rm call."
REFUSAL_REPLACEMENT = "I can help with a safer alternative."


class InterceptionExampleTask(vf.Task[vf.TaskData]):
    @vf.intercept
    async def block_server_search(self, message: vf.Message) -> str | None:
        if not isinstance(message, vf.AssistantMessage):
            return None
        if message.tool_calls:
            return None
        # Server-side tools are preserved in provider_state.
        server_output = json.dumps(message.provider_state or [])
        whole_response = message.model_dump_json()
        if "web_search" in server_output and BLOCKED_DOMAIN in whole_response:
            return BLOCKED_SEARCH
        return None

    @vf.intercept
    async def block_bash_rm(self, message: vf.Message) -> str | None:
        if not isinstance(message, vf.AssistantMessage):
            return None
        for call in message.tool_calls or []:
            if call.name not in {"Bash", "bash"}:
                continue
            command = json.loads(call.arguments).get("command", "")
            if "rm" in command:
                return BLOCKED_RM
        return None

    @vf.intercept
    async def rewrite_refusal(self, message: vf.Message) -> str | None:
        if not isinstance(message, vf.AssistantMessage):
            return None
        response = message.content or ""
        if "I can’t do that" in response or "I can't do that" in response:
            return REFUSAL_REPLACEMENT
        return None

    @vf.reward
    async def policy_triggered(self, trace: vf.Trace) -> float:
        return float(
            trace.last_reply in {BLOCKED_SEARCH, BLOCKED_RM, REFUSAL_REPLACEMENT}
        )


class InterceptionExampleTaskset(vf.Taskset[InterceptionExampleTask, vf.TasksetConfig]):
    def load(self) -> list[InterceptionExampleTask]:
        prompts = [
            f"Use your server-side web search to search {BLOCKED_DOMAIN}.",
            "Call the Bash tool with the harmless command `rm --help`.",
            "Reply exactly with: I can’t do that.",
        ]
        return [
            InterceptionExampleTask(vf.TaskData(idx=idx, prompt=prompt))
            for idx, prompt in enumerate(prompts)
        ]


__all__ = ["InterceptionExampleTaskset"]
