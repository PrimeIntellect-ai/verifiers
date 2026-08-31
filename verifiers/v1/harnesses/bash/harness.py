import os

from pydantic import PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
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
    "You also have a search tool that returns Google results (title, URL, snippet) for a query; "
    "use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)


class CompactionConfig(BaseConfig):
    """Context compaction policy for the bash agent loop."""

    summarize_at_tokens: PositiveInt | None = None
    """Compact at this token count. When unset, compact when 16k tokens remain below the
    model context window when the provider advertises it."""


class BashHarnessConfig(HarnessConfig):
    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--env.agent.harness.edit false` for a bash-only agent."""

    search: bool = False
    """Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the
    eval environment; the key is handed to the program over argv (like the interception secret) so
    the agent's `bash` subprocesses don't inherit it."""

    compaction: CompactionConfig | None = None
    """Context compaction policy. Set an empty config to use automatic thresholds."""


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
        system_prompt = "\n\n".join(
            p for p in (" ".join(fragments), system_prompt) if p
        )
        env = {**self.config.resolved_env}
        args = ["--bash"]
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        if self.config.compaction is not None:
            args.append("--compaction")
            threshold = self.config.compaction.summarize_at_tokens
            if threshold is not None:
                args.append(f"--summarize-at-tokens={threshold}")
        if self.config.edit:
            args.append("--edit")
        if self.config.search:
            # Resolve the key and keep it OUT of the program env: it's handed to the program over
            # argv (--serper-key), so popping it here stops the agent's `bash` subprocesses from
            # inheriting it via $SERPER_API_KEY / /proc/self/environ. Prefer a key set in the harness
            # env (harness config env / forward_env); fall back to the host env only when the key is
            # *absent* (None), not present-but-empty — a rollout setting SERPER_API_KEY="" is
            # deliberately masking the host secret, so honor that (the check below then fails loudly
            # rather than leaking the host key). The pop is scoped to search=true, so an unrelated
            # key forwarded for the agent's own bash-side use is left untouched.
            serper_key = env.pop("SERPER_API_KEY", None)
            if serper_key is None:
                serper_key = os.environ.get("SERPER_API_KEY")
            if not serper_key:
                raise ValueError(
                    "bash search=true requires SERPER_API_KEY in the eval environment "
                    "(the host env or the harness config's env)"
                )
            args += ["--search", f"--serper-key={serper_key}"]
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
