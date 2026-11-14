"""inline-chat: the `null` harness's chat call, made in-process.

The `null` harness runs a stateless chat program in the runtime and relaunches it
for every turn of an exchange, so each turn pays a new Python interpreter and its
imports for one model call. For env-driven exchanges with many short turns (text
games: hundreds of one-reply turns per episode) that dominates the turn and starves
the inference engine.

This harness makes the same call - one streamed chat completion through the
rollout's interception ``endpoint`` with its ``secret`` - from the env server's
own process, on a shared HTTP client, and returns a synthetic success result (the
contract `Harness.launch` allows: the interception is the record, not the
process). Tool-less chat only: no tools, MCP or compaction.
"""

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import MODEL_TRANSPORT_ERROR_EXIT_CODE
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.compaction import is_context_overflow
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

_HTTP: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    """One connection pool per env-server process, shared by every rollout."""
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=5.0),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
        )
    return _HTTP


class InlineChatHarnessConfig(HarnessConfig):
    pass


class InlineChatHarness(Harness[InlineChatHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_RESUME = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError(
                "the inline-chat harness has no tools; use the null harness for MCP servers"
            )
        system, prompt = self.resolve_prompt(data)
        messages = [{"role": "system", "content": system}] if system else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(message) for message in prompt)
        client = AsyncOpenAI(
            base_url=endpoint, api_key=secret, http_client=_http(), max_retries=2
        )
        try:
            # Streamed like the chat program, so the interception server sees the same request.
            stream = await client.chat.completions.create(
                model=ctx.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            finished = False
            async for chunk in stream:
                finished = finished or any(
                    choice.finish_reason is not None for choice in chunk.choices
                )
            if not finished:
                return ProgramResult(
                    MODEL_TRANSPORT_ERROR_EXIT_CODE,
                    "",
                    "model stream ended before a completion finished",
                )
        except APIStatusError as error:
            # As in the chat program: running out of context ends the run, it is not a crash.
            if is_context_overflow(error):
                return ProgramResult(0, "", "")
            return ProgramResult(1, "", f"{error.status_code}: {error}")
        except (APIConnectionError, httpx.TransportError) as error:
            return ProgramResult(MODEL_TRANSPORT_ERROR_EXIT_CODE, "", repr(error))
        return ProgramResult(0, "", "")
