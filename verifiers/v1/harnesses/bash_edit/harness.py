"""The built-in bash_edit harness: the `bash` chat loop plus a local `edit` tool.

Same growing-message-list chat loop as the `bash` harness — a local `bash` tool that runs shell
commands in the runtime, plus the taskset's MCP tools — with one extra local tool: `edit`, a
single-occurrence string replacement in a file (ported from the rlm `edit` skill). For agentic
tasks where targeted file edits are common and a model handles `edit` more reliably than
hand-built `sed`/heredoc shell. Its uv script (deps: openai, mcp) is prepared during setup, then
launched as the harness program.
"""

import json
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

# Frames the model as a coding agent and names its local tools (a pure-text chat loop gets no
# harness-injected prompt).
BASH_EDIT_SYSTEM_PROMPT = (
    "You are a coding agent. You have access to a bash tool for running shell commands and "
    "an edit tool for single-occurrence string replacement in a file."
)


class BashEditHarnessConfig(HarnessConfig):
    """The built-in bash_edit harness. A uv script (deps: openai, mcp), so it runs in any runtime
    that has `uv` (the harness bootstraps it) with no other setup."""


class BashEditHarness(Harness[BashEditHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)

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
        system_prompt = "\n\n".join(
            p for p in (BASH_EDIT_SYSTEM_PROMPT, system_prompt) if p
        )
        env = {**self.config.env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt}",
        ]
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
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)
        return await runtime.run_program([*program, *args], env)
