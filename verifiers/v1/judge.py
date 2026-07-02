"""A reusable per-task LLM judge for v1 tasksets.

Predates `vf.JudgeSpec` (`verifiers.v1.agent`), which subsumes this: a `JudgeSpec` with the
`null` harness is the same single-call judge, with the model bound through the env's shared
model table, its budget enforced, and the run recorded on `trace.agents` with provenance.
Prefer `JudgeSpec` for new tasksets; this class is kept for existing downstream users.

Most tasksets that can't grade deterministically reach for the same shape: an OpenAI-compatible
endpoint, a prompt built from `(question, answer, response)`, one chat call, and a verdict parsed
out of the reply. `Judge` centralizes that — the client construction (incl. the Prime key/team
fallback), the call, and usage/cost capture — and leaves the two things that actually differ as
hooks: `build_messages` (prompt setup) and `parse`
(verdict parsing). Set `schema` to use OpenAI structured outputs (where the provider supports it),
in which case `JudgeResponse.parsed` is the validated pydantic object.

    class CorrectnessJudge(vf.Judge[bool]):
        prompt = "Question: {question}\\nAnswer: {answer}\\nResponse: {response}\\nCorrect? yes/no"

        def parse(self, response: vf.JudgeResponse[bool]) -> bool:
            return response.text.strip().lower().startswith("yes")

    self.judge = CorrectnessJudge(self.config.judge)  # self.config.judge: vf.JudgeConfig

    @vf.reward
    async def correct(self, task, trace) -> float:
        result = await self.judge.evaluate(
            trace=trace, question=task.question, answer=task.answer, response=...
        )
        return float(result.parsed)

Passing `trace=` records the call onto it — a typed record appended to `trace.info["judge"]` and
the call's tokens + cost added to `trace.extra_usage` (kept separate from the agent's `trace.usage`),
so judge behaviour and spend are no longer invisible. The record lands even if the judge refuses, an
empty structured output comes back, or `parse` raises (the request was already billed). Omit `trace`
for a pure call (e.g. in tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import BaseModel

from verifiers.v1.clients.config import BaseClientConfig, build_async_openai
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.types import Messages, SamplingConfig, StrictBaseModel, Usage

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace

ParsedT = TypeVar("ParsedT")


class JudgeSamplingConfig(SamplingConfig):
    """Sampling knobs for a judge call (`temperature` / `top_p` / `reasoning_effort` /
    `max_tokens`, plus provider-specific keys via `extra='allow'`). Same shape as the rollout's
    `SamplingConfig` — set e.g. `judge.sampling.max_tokens`."""


class JudgeConfig(BaseClientConfig):
    """An LLM-judge endpoint. Inherits `base_url` / `api_key_var` / `headers` (with the Prime
    auto-config) from `BaseClientConfig`; adds the model and sampling. Subclass to add
    taskset-specific fields."""

    model: str = "openai/gpt-5-mini"
    sampling: JudgeSamplingConfig = JudgeSamplingConfig()


class JudgeResponse(StrictBaseModel, Generic[ParsedT]):
    """One judge call's result — returned to the caller and (JSON-serialized) appended to
    `trace.info["judge"]` for debugging, including the provider-reported `usage` (tokens + cost)."""

    text: str
    """The judge's raw reply."""
    parsed: ParsedT | None = None
    """The verdict the taskset acts on (`parse`'s output, or the structured object for `schema`)."""
    usage: Usage | None = None


class Judge(Generic[ParsedT]):
    """A per-task LLM judge over an OpenAI-compatible endpoint.

    Override `build_messages` (prompt setup) and `parse` (verdict parsing) — or just set the
    `prompt` template and use the defaults — then call `evaluate(**fields)`. Set `schema` to opt
    into structured outputs.
    """

    prompt: str | None = None
    """Default template for `build_messages`, formatted with the `evaluate` kwargs and sent as a
    single user message. Override `build_messages` for system+user or non-template prompts."""
    schema: type[BaseModel] | None = None
    """Pydantic schema for OpenAI structured outputs. When set, the call uses
    `response_format=schema` and `JudgeResponse.parsed` is the validated object (provider must
    support structured outputs)."""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self.client: AsyncOpenAI = build_async_openai(self.config)

    def build_messages(self, **fields: Any) -> str | Messages:
        """Prompt-setup hook: turn the `evaluate` fields into the messages to send (a single user
        message as a plain `str`, or a `vf.Messages` list). The default formats the `prompt`
        template with the fields; override it for a system+user / non-template prompt."""
        if self.prompt is None:
            raise ValueError(
                f"{type(self).__name__} has no `prompt`; set it or override build_messages"
            )
        return self.prompt.format(**fields)

    def parse(self, response: JudgeResponse[ParsedT]) -> ParsedT:
        """Parsing hook: turn a `JudgeResponse` into the verdict. The default returns the
        structured object when `schema` is set, otherwise the raw reply text."""
        if self.schema is not None:
            return cast(ParsedT, response.parsed)
        return cast(ParsedT, response.text)

    async def complete(
        self,
        messages: str | Messages,
        *,
        trace: "Trace | None" = None,
        schema: type[BaseModel] | None = None,
        parse: Callable[[JudgeResponse[Any]], Any] | None = None,
        **sampling: Any,
    ) -> JudgeResponse[Any]:
        """Call the judge once: send `messages`, build the `JudgeResponse`, and — when a `trace` is
        given — record it (`Trace.record_judge`) in a `finally`, so the call's tokens + cost are
        captured even when a refusal / empty structured output / `parse` hook raises after the
        billed request. `schema` opts into structured outputs; `parse` sets `JudgeResponse.parsed`."""
        wire = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else [message_to_wire(m) for m in messages]
        )
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": wire}
        kwargs.update(self.config.sampling.model_dump(exclude_none=True))
        kwargs.update(sampling)

        response: JudgeResponse[Any] | None = None
        try:
            if schema is not None:
                completion = await self.client.beta.chat.completions.parse(
                    response_format=schema, **kwargs
                )
                choice = completion.choices[0]
                response = JudgeResponse(
                    text=choice.message.content or "",
                    parsed=choice.message.parsed,
                    usage=Usage.from_openai(completion.usage),
                )
                if choice.message.refusal is not None:
                    raise RuntimeError(
                        f"judge refused structured output: {choice.message.refusal}"
                    )
                if response.parsed is None:
                    # No refusal but no object either — e.g. a truncated/malformed reply. Surface it
                    # rather than returning a None verdict callers read as a (falsy) failure.
                    raise RuntimeError(
                        f"judge returned no parseable structured output "
                        f"(finish_reason={choice.finish_reason})"
                    )
            else:
                completion = await self.client.chat.completions.create(**kwargs)
                response = JudgeResponse(
                    text=completion.choices[0].message.content or "",
                    usage=Usage.from_openai(completion.usage),
                )
            if parse is not None:
                response.parsed = parse(response)
            return response
        finally:
            if trace is not None and response is not None:
                trace.record_judge(response)

    async def evaluate(
        self, *, trace: "Trace | None" = None, **fields: Any
    ) -> JudgeResponse[ParsedT]:
        """Render the prompt (`build_messages(**fields)`), call the judge, and parse the verdict
        (`parse`). When a `trace` is given the call is recorded onto it (`trace.info["judge"]` +
        usage on `trace.extra_usage`), even if a refusal / empty output / parse raises."""
        messages = self.build_messages(**fields)
        return await self.complete(
            messages, trace=trace, schema=self.schema, parse=self.parse
        )
