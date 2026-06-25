"""A reusable per-task LLM judge for v1 tasksets.

Most tasksets that can't grade deterministically reach for the same shape: an OpenAI-compatible
endpoint, a prompt built from `(question, answer, response)`, one chat call, and a verdict parsed
out of the reply. `Judge` centralizes that — the client construction (incl. the Prime key/team
fallback), the call, usage/cost capture, and recording each call to `trace.info["judge"]` — and
leaves the two things that actually differ as hooks: `build_messages` (prompt setup) and `parse`
(verdict parsing). Set `schema` to use OpenAI structured outputs (where the provider supports it),
in which case `JudgeResponse.parsed` is the validated pydantic object.

    class CorrectnessJudge(vf.BinaryJudge):
        prompt = "Question: {question}\\nAnswer: {answer}\\nResponse: {response}\\nCorrect? yes/no"

    self.judge = CorrectnessJudge(self.config.judge)  # self.config.judge: vf.JudgeConfig

    @vf.reward
    async def correct(self, task, trace) -> float:
        result = await self.judge.evaluate(
            trace=trace, question=task.question, answer=task.answer, response=...
        )
        return float(result.parsed)

The judge's tokens + cost land on `trace.extra_usage` (folded into `trace.usage`), and a typed
record of every call is appended to `trace.info["judge"]`, so judge behaviour and spend are no
longer invisible.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, cast
from urllib.parse import urlparse

from openai import AsyncOpenAI
from pydantic import BaseModel

from verifiers.utils.client_utils import load_prime_config
from verifiers.v1.clients.config import (
    PRIME_INFERENCE_HOST,
    BaseClientConfig,
)
from verifiers.v1.types import StrictBaseModel, Usage

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace

ParsedT = TypeVar("ParsedT")

_YES_NO = re.compile(r"\b(yes|no)\b")


class JudgeConfig(BaseClientConfig):
    """An LLM-judge endpoint. Inherits `base_url` / `api_key_var` / `headers` (with the Prime
    auto-config) from `BaseClientConfig`; adds the model and sampling knobs. Subclass to add
    taskset-specific fields (e.g. a `prompt` override)."""

    model: str = "openai/gpt-4.1-mini"
    temperature: float | None = None
    max_tokens: int | None = None


class JudgeResponse(StrictBaseModel, Generic[ParsedT]):
    """One judge call's result — returned to the caller and (JSON-serialized) appended to
    `trace.info["judge"]` for debugging, including the provider-reported `usage` (tokens + cost)."""

    model: str
    text: str
    """The judge's raw reply."""
    parsed: ParsedT | None = None
    """The verdict the taskset acts on (`parse`'s output, or the structured object for `schema`)."""
    usage: Usage | None = None


class Judge(Generic[ParsedT]):
    """A per-task LLM judge over an OpenAI-compatible endpoint.

    Override `build_messages` (prompt setup) and `parse` (verdict parsing) — or just set the
    `prompt` template and use the defaults — then call `evaluate(trace=..., **fields)`. Set
    `schema` to opt into structured outputs.
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
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            cfg = self.config
            api_key = os.environ.get(cfg.api_key_var)
            host = urlparse(cfg.base_url).hostname or ""
            if (
                not api_key
                and cfg.api_key_var == "PRIME_API_KEY"
                and (
                    host == PRIME_INFERENCE_HOST
                    or host.endswith(f".{PRIME_INFERENCE_HOST}")
                )
            ):
                api_key = load_prime_config().get("api_key")
            self._client = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=api_key or "EMPTY",
                default_headers=cfg.headers or None,
            )
        return self._client

    def build_messages(self, **fields: Any) -> str | list[dict[str, Any]]:
        """Prompt-setup hook: turn the `evaluate` fields into the messages to send. The default
        formats `prompt` into a single user message."""
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
        messages: str | list[dict[str, Any]],
        *,
        trace: "Trace | None" = None,
        schema: type[BaseModel] | None = None,
        parse: Callable[[JudgeResponse[Any]], Any] | None = None,
        **sampling: Any,
    ) -> JudgeResponse[Any]:
        """Call the judge once: send `messages`, capture usage/cost, record to the trace, and
        return the `JudgeResponse`. `schema` opts into structured outputs; `parse` sets
        `JudgeResponse.parsed` from the reply (the structured object, if any, is set first)."""
        wire = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else list(messages)
        )
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": wire}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        kwargs.update(sampling)

        parsed: Any = None
        if schema is not None:
            completion = await self.client.beta.chat.completions.parse(
                response_format=schema, **kwargs
            )
            parsed = completion.choices[0].message.parsed
        else:
            completion = await self.client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""

        response: JudgeResponse[Any] = JudgeResponse(
            model=self.config.model,
            text=text,
            parsed=parsed,
            usage=Usage.from_openai(completion.usage),
        )
        if parse is not None:
            response.parsed = parse(response)
        if trace is not None:
            trace.info.setdefault("judge", []).append(response.model_dump(mode="json"))
            if response.usage is not None:
                trace.extra_usage.append(response.usage)
        return response

    async def evaluate(
        self, *, trace: "Trace | None" = None, **fields: Any
    ) -> JudgeResponse[ParsedT]:
        """Render the prompt (`build_messages`), call the judge, and parse the verdict (`parse`).
        Returns the `JudgeResponse` with `.parsed` set to the verdict."""
        messages = self.build_messages(**fields)
        return await self.complete(
            messages, trace=trace, schema=self.schema, parse=self.parse
        )


class BinaryJudge(Judge[bool]):
    """A yes/no judge. `parse` reads the first standalone `yes`/`no` word in the reply, so a
    verbose verdict ("No, the response mentions yes but ...") is graded on its verdict, not on
    any `yes` it happens to contain. Set `prompt` (asking the model to answer "yes" or "no")."""

    def parse(self, response: JudgeResponse[bool]) -> bool:
        match = _YES_NO.search(response.text.lower())
        return bool(match and match.group(1) == "yes")
