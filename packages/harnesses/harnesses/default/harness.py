"""The built-in default harness: runs a small chat-loop program as a uv script.

A growing-message-list chat loop with the taskset's MCP tools (host-side, resolved to URLs by
the Environment) — and no tools of its own. The program is a uv script (deps: openai, mcp),
launched via `runtime.run_uv_script` — so it works on any runtime with `uv` (the harness
bootstraps it), with no runtime-specific setup. For a shell-driving agent, use a dedicated
agentic harness (e.g. `mini-swe-agent`).
"""

import json
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import EvalClient, RolloutContext
from verifiers.v1.dialects import chat, responses
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class DefaultHarnessConfig(HarnessConfig):
    """The built-in harness. A uv script (deps: openai, mcp), so it runs in any runtime that
    has `uv` (the harness bootstraps it) with no other setup."""

    id: str = "default"


class DefaultHarness(Harness[DefaultHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

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
        client = getattr(ctx.client, "inner", ctx.client)
        use_responses = isinstance(client, EvalClient) and (
            ctx.model.startswith("openai/")
            or client.base_url.startswith("https://api.openai.com")
        )
        env = {
            **self.config.env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
            "APPEND_SYSTEM_PROMPT": system_prompt or "",
        }
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            env["MCP_CONFIG"] = json.dumps(
                {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
            )
        # A Messages prompt (e.g. an image-bearing prompt) seeds the chat loop directly;
        # a plain string is the single first user message; None means the task has no prompt and
        # the framework's user simulator opens the conversation (no opening user message here).
        if prompt is None:
            args = [""]
        elif isinstance(prompt, str):
            args = [prompt]
        else:
            env["INITIAL_MESSAGES"] = json.dumps(
                responses.messages_to_wire(prompt)
                if use_responses
                else [chat.message_to_wire(message) for message in prompt]
            )
            args = [""]
        if use_responses:
            args.append("--responses")
        return await runtime.run_uv_script(PROGRAM_SOURCE, args=args, env=env)
