import json
import os
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import (
    SEARCH_API_KEY_ENV,
    HarnessConfig,
    SearchProvider,
)
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    CompactionConfig,
    compaction_args,
    launch_chat_program,
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
    "You also have a search tool that returns web results (title, URL, snippet) for a query; "
    "use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)


class BashHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--env.agent.harness.edit false` for a bash-only agent."""

    search: bool = False
    """Offer a `search` tool (web results from `search_provider`). Requires the provider's key
    (`SERPER_API_KEY` or `ECO_API_KEY`) in the eval environment; the key is handed to the program
    over argv (like the interception secret) so the agent's `bash` subprocesses don't inherit it.
    Results matching the runtime's network blocklist are excluded."""

    search_provider: SearchProvider = "serper"
    """Backend of the `search` tool: `serper` (Google results via serper.dev) or `eco`
    (You.com Eco Search)."""

    compaction: CompactionConfig | None = CompactionConfig()
    """Context compaction. Auto-compaction by default; set `summarize_at_tokens`
    for a fixed threshold, and `null` (`"None"` in TOML) to disable it."""


class BashHarness(Harness[BashHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = False

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
        args = ["--bash"]
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        args += compaction_args(self.config.compaction)
        if self.config.edit:
            args.append("--edit")
        if self.config.search:
            # Resolve the key and keep it OUT of the program env: it's handed to the program over
            # argv (--search-key), so popping it here stops the agent's `bash` subprocesses from
            # inheriting it via the environment / /proc/self/environ. Prefer a key set in the harness
            # env (harness config env / forward_env); fall back to the host env only when the key is
            # *absent* (None), not present-but-empty — a rollout setting the key to "" is
            # deliberately masking the host secret, so honor that (the check below then fails loudly
            # rather than leaking the host key). The pop is scoped to search=true and the chosen
            # provider, so an unrelated key forwarded for the agent's own bash-side use is left
            # untouched.
            key_env = SEARCH_API_KEY_ENV[self.config.search_provider]
            search_key = env.pop(key_env, None)
            if search_key is None:
                search_key = os.environ.get(key_env)
            if not search_key:
                raise ValueError(
                    f"bash search=true requires {key_env} in the eval environment "
                    "(the host env or the harness config's env)"
                )
            args += [
                "--search",
                f"--search-provider={self.config.search_provider}",
                f"--search-key={search_key}",
            ]
            if isinstance(runtime.config, NetworkPolicyConfig) and runtime.config.block:
                args.append(f"--search-block={json.dumps(runtime.config.block)}")
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
            env=env,
            activate=False,
        )
