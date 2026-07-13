"""The built-in direct harness: an in-process chat loop — no subprocess, no uv script, no tools.

The cheapest possible episode: the chat loop runs inside the eval process itself, POSTing to
its interception endpoint like any program would — so the trace, stops, limits, and per-agent
routing all work unmodified — but with nothing to provision or launch. Cost per episode is
essentially the model call itself, which makes an agent-as-judge (the `llm-judge`
topology's judge) as cheap as a plain judge call while still producing a real, inspectable
trace.

Tool-less by design (`null` is the subprocess chat loop with MCP tools); a model that calls a
tool anyway gets an error tool-result and one chance to answer directly."""

from openai import AsyncOpenAI

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace


class DirectHarnessConfig(HarnessConfig):
    """The built-in direct harness: an in-process chat loop. Nothing runs in the runtime, so
    it works on any runtime with zero setup — but the *episode's* code (tool servers, scoring
    scripts) still uses the runtime normally."""


class DirectHarness(Harness[DirectHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

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
        messages: list[dict] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(m) for m in prompt)
        # No SDK retries: a re-sent request whose turn interception already committed
        # re-samples and forks the trace (same contract as the default harness).
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        bounced = False
        try:
            while True:
                try:
                    completion = await client.chat.completions.create(
                        model=ctx.model, messages=messages
                    )
                except Exception:
                    # A refused turn (a @stop fired, a budget hit) surfaces as an HTTP error;
                    # unlike a subprocess program we can see the live trace, so treat a stopped
                    # rollout as the clean exit it is. A real failure re-raises (classified by
                    # `Harness.run`, with `RolloutSession.error` carrying the true cause).
                    if trace.stop_condition is not None:
                        break
                    raise
                message = completion.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                if (
                    bounced
                ):  # exactly one chance to answer directly — never an open loop
                    break
                bounced = True
                # Tool-less by design: bounce the hallucinated call back with an error result.
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "error: no tools are available; answer directly",
                    }
                    for call in message.tool_calls
                )
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


__all__ = ["DirectHarness", "DirectHarnessConfig"]
