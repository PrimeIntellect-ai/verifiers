"""Run an external program (python script / bash executable) as the agent.

The program makes OpenAI-style API calls; a small per-rollout `InterceptionServer`
catches them, routes each to our `Client`, records a `Turn`, and returns the
result in OpenAI shape. We inject `OPENAI_BASE_URL`/`OPENAI_API_KEY` so the
program's SDK talks to us. A much simpler take on v1's interception machinery:
one ephemeral localhost server per rollout, chat-completions only, no streaming.
"""

import asyncio
import os
import secrets

from aiohttp import web
from pydantic import Field

from verifiers.nano.errors import ProgramError
from verifiers.nano.harness import Harness, HarnessConfig, RolloutContext
from verifiers.nano.transcript import Transcript, Turn
from verifiers.nano.types import (
    AssistantMessage,
    Message,
    Messages,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

# Host provider credentials are scrubbed from the subprocess so the program can
# only reach our interception endpoint (not a real provider). A program with its
# own provider precedence (e.g. rlm prefers PRIME_API_KEY) would otherwise bypass
# interception entirely.
PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "PRIME_API_KEY",
    "PRIME_TEAM_ID",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "RLM_API_KEY",
    "RLM_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def parse_message(raw: dict) -> Message:
    """An OpenAI request message dict -> a typed Message."""
    role = raw.get("role")
    content = raw.get("content")
    if isinstance(content, list):  # multimodal parts -> joined text
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = content or ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(tool_call_id=raw.get("tool_call_id", ""), content=content)
    if role == "assistant":
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"]["arguments"],
            )
            for c in (raw.get("tool_calls") or [])
        ] or None
        return AssistantMessage(content=raw.get("content"), tool_calls=calls)
    return UserMessage(content=content)


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
    if not raw:
        return None
    return [
        Tool(
            name=t["function"]["name"],
            description=t["function"].get("description", ""),
            parameters=t["function"].get("parameters", {}),
            strict=t["function"].get("strict"),
        )
        for t in raw
        if t.get("type", "function") == "function"
    ]


def serialize_completion(response: Response, model: str) -> dict:
    """A `Response` -> an OpenAI chat.completion dict the program's SDK expects."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage = (
        {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage
        else None
    )
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


class InterceptionServer:
    """A localhost server that proxies one rollout's chat-completions calls."""

    def __init__(self, ctx: RolloutContext, transcript: Transcript) -> None:
        self.ctx = ctx
        self.transcript = transcript
        self.secret = secrets.token_urlsafe(16)
        self.base_url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # actual ephemeral port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *exc) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def handle_chat(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.secret}":
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        prompt: Messages = [parse_message(m) for m in body.get("messages", [])]
        tools = parse_tools(body.get("tools"))
        try:
            response = await self.ctx.client.get_response(
                prompt, self.ctx.model, self.ctx.sampling, tools
            )
        except Exception as e:  # surface to the program as an API error
            return web.json_response({"error": str(e)}, status=502)
        self.transcript.trajectory.append(Turn(prompt=prompt, response=response))
        self.transcript.messages.append(response.message)
        return web.json_response(serialize_completion(response, self.ctx.model))


class ProgramConfig(HarnessConfig):
    command: list[str] = Field(default_factory=list)
    """Program argv; the task instruction is appended as the final argument."""
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class ProgramHarness(Harness):
    """Runs `config.command <instruction>` with its model calls intercepted."""

    config: ProgramConfig

    async def run_turns(self, ctx: RolloutContext, transcript: Transcript) -> None:
        async with InterceptionServer(ctx, transcript) as server:
            env = {k: v for k, v in os.environ.items() if k not in PROVIDER_ENV_VARS}
            env.update(
                {
                    "OPENAI_BASE_URL": server.base_url,
                    "OPENAI_API_KEY": server.secret,
                    "OPENAI_MODEL": ctx.model,
                    "RLM_MODEL": ctx.model,
                    "VF_INSTRUCTION": transcript.task.instruction,
                    **self.config.env,
                }
            )
            argv = [*self.config.command, transcript.task.instruction]
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                cwd=self.config.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ProgramError(
                f"program exited {proc.returncode}: {stderr.decode(errors='replace')[:1000]}"
            )
        transcript.stop("program_completed")
