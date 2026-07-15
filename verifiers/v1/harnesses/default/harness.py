import json
import os
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

# Frames the model as a coding agent and names its local tools (a pure-text chat loop gets no
# harness-injected prompt). The edit clause is appended only when the `edit` tool is enabled.
BASH_SYSTEM_PROMPT = "You are a coding agent. You have access to a bash tool for running shell commands."
EDIT_SYSTEM_PROMPT = "You also have an edit tool for single-occurrence string replacement in a file."
# Appended when search is enabled, so the model knows the extra tool exists.
SEARCH_PROMPT = (
    "You also have a search tool that returns Google results (title, URL, snippet) for a query; "
    "use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)


class DefaultHarnessConfig(HarnessConfig):
    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--harness.edit false` for a bash-only agent."""

    search: bool = False
    """Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the
    eval environment; the key is handed to the program over argv (like the interception secret) so
    the agent's `bash` subprocesses don't inherit it."""


class DefaultHarness(Harness[DefaultHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        fragments = [BASH_SYSTEM_PROMPT]
        if self.config.edit:
            fragments.append(EDIT_SYSTEM_PROMPT)
        if self.config.search:
            fragments.append(SEARCH_PROMPT)
        system_prompt = "\n\n".join(p for p in (" ".join(fragments), system_prompt) if p)
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt}",
        ]
        if self.config.edit:
            args.append("--edit")
        if self.config.search:
            # Resolve the key and keep it OUT of the program env: it's handed to the program over
            # argv (--serper-key), so popping it here stops the agent's `bash` subprocesses from
            # inheriting it via $SERPER_API_KEY / /proc/self/environ. Prefer a key set in the harness
            # env (--harness.env / forward_env); fall back to the host env only when the key is
            # *absent* (None), not present-but-empty — a rollout setting SERPER_API_KEY="" is
            # deliberately masking the host secret, so honor that (the check below then fails loudly
            # rather than leaking the host key). The pop is scoped to search=true, so an unrelated
            # key forwarded for the agent's own bash-side use is left untouched.
            serper_key = env.pop("SERPER_API_KEY", None)
            if serper_key is None:
                serper_key = os.environ.get("SERPER_API_KEY")
            if not serper_key:
                raise ValueError(
                    "default search=true requires SERPER_API_KEY in the eval environment "
                    "(the host env or --harness.env)"
                )
            args += ["--search", f"--serper-key={serper_key}"]
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            args.append(
                "--mcp-config=" + json.dumps({"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}})
            )
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            # Base64 images can exceed exec limits, so hand Messages off through a file.
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(path, json.dumps([message_to_wire(m) for m in prompt]).encode())
            args.append(f"--initial-messages-file={path}")
        args += self.extra_program_args()
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        return await runtime.run_program([*program, *args], env)

    def extra_program_args(self) -> list[str]:
        """Extra argv a subclass appends to the program invocation (e.g. the compacting
        harness's `--compact-at-tokens`). The base harness appends nothing."""
        return []
