from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    CompactionConfig,
    compaction_args,
    launch_chat_program,
)
from verifiers.v1.harnesses.utils.sandbox_tools import RemoteToolsSpec
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

# Frames the model as a coding agent and names its local tools (a pure-text chat loop gets no
# harness-injected prompt). The edit clause is appended only when the `edit` tool is enabled.
BASH_SYSTEM_PROMPT = (
    "You are a coding agent. You have access to a bash tool for running shell commands."
)
EDIT_SYSTEM_PROMPT = (
    "You also have an edit tool for single-occurrence string replacement in a file."
)


class BashHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--env.agent.harness.edit false` for a bash-only agent."""

    compaction: CompactionConfig | None = CompactionConfig()
    """Context compaction. Auto-compaction by default; set `summarize_at_tokens`
    for a fixed threshold, and `null` (`"None"` in TOML) to disable it."""


class BashHarness(Harness[BashHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = False

    async def remote_tools(self, runtime: Runtime, data: TaskData):
        from verifiers.v1.harnesses.bash.remote import BashRemoteTools

        return RemoteToolsSpec(
            BashRemoteTools,
            {"edit": self.config.edit},
            native_tools=frozenset({"bash", "edit"} if self.config.edit else {"bash"}),
        )

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(CHAT_PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        if not tool_interception_url:
            raise HarnessError("bash harness requires a remote tool gate")
        system_prompt, prompt = self.resolve_prompt(data)
        fragments = [BASH_SYSTEM_PROMPT]
        if self.config.edit:
            fragments.append(EDIT_SYSTEM_PROMPT)
        harness_prompt = " ".join(fragments)
        system_prompt = "\n\n".join(p for p in (harness_prompt, system_prompt) if p)
        # Default resume replays the transcript. If there was no task system
        # prompt, that transcript already contains this harness prompt.
        if (
            data.system_prompt is None
            and prompt is not None
            and not isinstance(prompt, str)
            and any(
                message.role == "system" and message.content == harness_prompt
                for message in prompt
            )
        ):
            system_prompt = None
        args = [
            "--bash",
            f"--tool-interception-url={tool_interception_url}",
            *compaction_args(self.config.compaction),
        ]
        if self.config.edit:
            args.append("--edit")
        return await launch_chat_program(
            CHAT_PROGRAM_SOURCE,
            self.config,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            system_prompt,
            prompt,
            extra_args=args,
            env={**self.config.resolved_env},
            activate=False,
        )
