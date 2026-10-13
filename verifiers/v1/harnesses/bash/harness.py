import os
import secrets
import shlex

from pydantic import NonNegativeInt, PositiveInt
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

# Appended when the harness has skills; the program has no skill discovery of its own.
SKILLS_PROMPT = (
    "Skills are folders under {root}. Each holds a SKILL.md card and the resources the "
    "card names. Read a card with bash before you use its skill."
)


def skill_card_fields(card: str, folder: str) -> tuple[str, str]:
    """A SKILL.md card's `name` and `description` from its front matter: single-line
    `key: value` entries, quotes stripped. The name defaults to the card's folder."""
    fields: dict[str, str] = {}
    lines = card.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip("\"'")
    return fields.get("name") or folder, fields.get("description", "")


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

    model_retries: NonNegativeInt = 10
    """Retries of a failed model call (429, 5xx, a dropped connection or stream), with
    the OpenAI SDK's backoff capped at 8 s: 10 rides out about a minute of provider
    errors. The interception relays a failure before its stream commits for the program
    to retry, so this is the retry budget for those. With a request gate on the client,
    the gate alone retries a 429/529: the interception marks one it relays as final
    (`x-should-retry: false`), and the program does not retry it."""

    command_timeout: PositiveInt = 3600
    """Seconds one `bash` command may run. On expiry its whole process group is killed
    and the model gets the output so far with a note."""

    compaction: CompactionConfig | None = CompactionConfig()
    """Context compaction. Auto-compaction by default; set `summarize_at_tokens`
    for a fixed threshold, and `null` (`"None"` in TOML) to disable it."""


class BashHarness(Harness[BashHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = False
    SUPPORTS_SKILLS = True

    def __init__(self, config: BashHarnessConfig) -> None:
        super().__init__(config)
        # Per instance: two sessions sharing one box never mix their skill sets.
        self.skills_dir = f"/tmp/vf-bash-skills-{secrets.token_hex(8)}"

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(CHAT_PROGRAM_SOURCE, self.config.resolved_env)
        if self.config.skills:
            await self.install_skills(runtime, self.skills_dir)

    async def skills_listing(self, runtime: Runtime) -> str | None:
        """The system-prompt section naming each installed skill: its card's name,
        path and description."""
        if not self.config.skills:
            return None
        root = shlex.quote(self.skills_dir)
        found = await runtime.run(
            ["sh", "-c", f"find {root} -name SKILL.md 2>/dev/null | sort"], {}
        )
        lines = [SKILLS_PROMPT.format(root=self.skills_dir)]
        for path in filter(None, found.stdout.splitlines()):
            card = (await runtime.read(path)).decode(errors="replace")
            name, description = skill_card_fields(card, path.split("/")[-2])
            lines.append(
                f"- `{name}` ({path})" + (f": {description}" if description else "")
            )
        return "\n".join(lines)

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
        skills = await self.skills_listing(runtime)
        system_prompt = "\n\n".join(
            p for p in (" ".join(fragments), system_prompt, skills) if p
        )
        env = {**self.config.resolved_env}
        args = [
            "--bash",
            f"--max-retries={self.config.model_retries}",
            f"--command-timeout={self.config.command_timeout}",
        ]
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
