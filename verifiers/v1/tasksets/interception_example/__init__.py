"""Harness-agnostic examples of blocking and rewriting model messages."""

import verifiers.v1 as vf

BLOCKED_DOMAIN = "example.com"
REFUSAL_REPLACEMENT = "I can help with a safer alternative."


class InterceptionExampleTask(vf.Task[vf.TaskData]):
    block_server_search = vf.block_web_search(containing=BLOCKED_DOMAIN)
    block_bash_rm = vf.block_shell_commands("rm")
    block_code_search = vf.block_code_search()

    @vf.intercept
    async def rewrite_refusal(self, message: vf.AssistantMessage) -> str | None:
        response = message.content or ""
        if "I can’t do that" in response or "I can't do that" in response:
            return REFUSAL_REPLACEMENT
        return None

    @vf.reward
    async def policy_triggered(self, trace: vf.Trace) -> float:
        return float(trace.last_reply in {"Blocked by policy.", REFUSAL_REPLACEMENT})


class InterceptionExampleTaskset(vf.Taskset[InterceptionExampleTask, vf.TasksetConfig]):
    def load(self) -> list[InterceptionExampleTask]:
        prompts = [
            f"Use your server-side web search to search {BLOCKED_DOMAIN}.",
            "Call the Bash tool with the harmless command `rm --help`.",
            "Use a code-search tool or shell command to search the repository.",
            "Reply exactly with: I can’t do that.",
        ]
        return [
            InterceptionExampleTask(vf.TaskData(idx=idx, prompt=prompt))
            for idx, prompt in enumerate(prompts)
        ]


__all__ = ["InterceptionExampleTaskset"]
