import json
import os
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    CompactionConfig,
    compaction_args,
    has_internal_mcp,
    launch_chat_program,
    run_in_process_chat,
)
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
# Appended when search is enabled, so the model knows the extra tool exists.
SEARCH_PROMPT = (
    "You also have a search tool that returns Google results (title, URL, snippet) for a query; "
    "use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)


class BashHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--env.agent.harness.edit false` for a bash-only agent."""

    search: bool = False
    """Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the
    eval environment; the key is handed to the program over argv (like the interception secret) so
    the agent's `bash` subprocesses don't inherit it. Results matching the runtime's
    network blocklist are excluded."""

    compaction: CompactionConfig | None = CompactionConfig()
    """Context compaction. Auto-compaction by default; set `summarize_at_tokens`
    for a fixed threshold, and `null` (`"None"` in TOML) to disable it."""


class BashHarness(Harness[BashHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = False
    RUNS_ON_HOST = True

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
        system_prompt, prompt = self.resolve_prompt(data)
        fragments = [BASH_SYSTEM_PROMPT]
        if self.config.edit:
            fragments.append(EDIT_SYSTEM_PROMPT)
        if self.config.search:
            fragments.append(SEARCH_PROMPT)
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
        env = {**self.config.resolved_env}
        serper_key = ""
        search_block: list[str] | None = None
        if self.config.search:
            # Pop SERPER_API_KEY from the sandbox env so `bash` commands never inherit it.
            popped = env.pop("SERPER_API_KEY", None)
            if popped is None:
                popped = os.environ.get("SERPER_API_KEY")
            if not popped:
                raise ValueError(
                    "bash search=true requires SERPER_API_KEY in the eval environment "
                    "(the host env or the harness config's env)"
                )
            serper_key = popped
            if isinstance(runtime.config, NetworkPolicyConfig) and runtime.config.block:
                search_block = list(runtime.config.block)

        if has_internal_mcp(mcp_urls):
            args = ["--bash"]
            if tool_interception_url:
                args.append(
                    f"--tool-interception-url={runtime.host_url(tool_interception_url)}"
                )
            args += compaction_args(self.config.compaction)
            if self.config.edit:
                args.append("--edit")
            if self.config.search:
                args += ["--search", f"--serper-key={serper_key}"]
                if search_block:
                    args.append(f"--search-block={json.dumps(search_block)}")
            return await launch_chat_program(
                CHAT_PROGRAM_SOURCE,
                self.config,
                ctx,
                trace,
                runtime,
                runtime.host_url(endpoint),
                secret,
                mcp_urls,
                system_prompt,
                prompt,
                extra_args=args,
                env=env,
                activate=False,
            )

        return await run_in_process_chat(
            self.config,
            ctx,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            system_prompt,
            prompt,
            bash=True,
            edit=self.config.edit,
            search=self.config.search,
            serper_key=serper_key,
            search_block=search_block,
            compaction_cfg=self.config.compaction,
            tool_interception_url=tool_interception_url,
            env=env,
        )
