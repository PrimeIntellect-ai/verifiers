"""The built-in default harness: a chat loop with a local `bash` tool, plus optional `edit`/`web_search`.

A growing-message-list chat loop with a local `bash` tool that runs shell commands in the runtime,
the taskset's MCP tools, and two optional local tools: `edit` (single-occurrence string replacement
in a file, ported from the rlm `edit` skill; on by default — a model handles it more reliably than
hand-built `sed`/heredoc shell) and `web_search` (Google results via serper.dev; off by default,
needs `SERPER_API_KEY`). This is the fallback harness when no `--harness.id` is given. Its uv script
(deps: openai, mcp) is prepared during setup, then launched as the harness program. For a pure chat
loop with no local tools, use the `null` harness.
"""

import json
import os
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

# Frames the model as a coding agent and names its local tools (a pure-text chat loop gets no
# harness-injected prompt). The edit clause is appended only when the `edit` tool is enabled.
BASH_SYSTEM_PROMPT = (
    "You are a coding agent. You have access to a bash tool for running shell commands."
)
EDIT_SYSTEM_PROMPT = (
    "You also have an edit tool for single-occurrence string replacement in a file."
)
# Appended when web_search is enabled, so the model knows the extra tool exists.
WEB_SEARCH_PROMPT = (
    "You also have a web_search tool that returns Google results (title, URL, snippet) for a "
    "query; use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)


class DefaultHarnessConfig(HarnessConfig):
    """The built-in default harness. A uv script (deps: openai, mcp), so it runs in any runtime
    that has `uv` (the harness bootstraps it) with no other setup."""

    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--harness.edit false` for a bash-only agent."""

    web_search: bool = False
    """Offer a `web_search` tool (Google results via serper.dev). Requires `SERPER_API_KEY` in the
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
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task)
        fragments = [BASH_SYSTEM_PROMPT]
        if self.config.edit:
            fragments.append(EDIT_SYSTEM_PROMPT)
        if self.config.web_search:
            fragments.append(WEB_SEARCH_PROMPT)
        system_prompt = "\n\n".join(
            p for p in (" ".join(fragments), system_prompt) if p
        )
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt}",
        ]
        if self.config.edit:
            args.append("--edit")
        if self.config.web_search:
            key = os.environ.get("SERPER_API_KEY")
            if not key:
                raise ValueError(
                    "default web_search=true requires SERPER_API_KEY in the eval environment"
                )
            args += ["--web-search", f"--serper-key={key}"]
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            args.append(
                "--mcp-config="
                + json.dumps(
                    {
                        "mcpServers": {
                            name: {"url": url} for name, url in mcp_urls.items()
                        }
                    }
                )
            )
        # A Messages prompt (e.g. an image-bearing prompt) seeds the chat loop directly via env
        # (it can be large multimodal content that overflows argv); a plain string is the single
        # first user message; None means the task has no prompt and the user simulator opens it.
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            env["INITIAL_MESSAGES"] = json.dumps([message_to_wire(m) for m in prompt])
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], env)
