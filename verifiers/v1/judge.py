"""The judge: one direct model call for a verdict — cheap, light, single-API-call grading.

Most tasks that can't grade deterministically reach for the same shape: an OpenAI-compatible
endpoint, a prompt built from `(question, answer, response)`, one chat call, and a verdict
parsed out of the reply. `Judge` centralizes that — the client construction (incl. the Prime
key/team fallback), the call, and usage/cost capture — and leaves the two things that differ
as hooks: `build_messages` (prompt setup) and `parse` (verdict parsing). Set `schema` to use
OpenAI structured outputs (where the provider supports it), in which case
`JudgeResponse.parsed` is the validated pydantic object.

It is a utility, not an abstraction: owned by whoever's reward calls it — a task's `@reward`
when the verdict is part of the env's own grading, a topology's `@reward(agent=...)` when it
crosses agents. (For judging plugged from eval config alone use the `llm-judge` topology; for
a judge that investigates with tools, its own model, its own runtime, the `agentic-judge`
topology — the judge as a full agent.)

    class CorrectnessJudge(vf.Judge[bool]):
        prompt = "Question: {question}\\nAnswer: {answer}\\nResponse: {response}\\nCorrect? yes/no"

        def parse(self, response: vf.JudgeResponse[bool]) -> bool:
            return response.text.strip().lower().startswith("yes")

    class MathTask(vf.Task):
        answer: str

        @vf.reward
        async def correct(self, trace) -> float:
            result = await CorrectnessJudge(JUDGE_CONFIG).evaluate(
                trace=trace, question=self.prompt_text, answer=self.answer,
                response=trace.last_reply,
            )
            return float(result.parsed)

Passing `trace=` records the call onto it — a typed record appended to `trace.info["judge"]`
and the call's tokens + cost added to `trace.extra_usage` (kept separate from the agent's
`trace.usage`), so judge behaviour and spend are no longer invisible. The record lands even
if the judge refuses, an empty structured output comes back, or `parse` raises (the request
was already billed). Omit `trace` for a pure call (e.g. in tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, cast, get_args

from openai import AsyncOpenAI
from pydantic import BaseModel, model_validator
from typing_extensions import TypeVar

from verifiers.v1.clients.config import BaseClientConfig, build_async_openai
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.scoring import parse_judge_choice
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
    judge-specific fields."""

    model: str = "openai/gpt-5.4-nano"
    sampling: JudgeSamplingConfig = JudgeSamplingConfig()
    prompt: str | None = None
    """Prompt-template override for `build_messages` (None = the judge class's own `prompt`),
    so a judge's prompt is tunable from config alone."""
    prompt_file: Path | None = None
    """Load the `prompt` template from a text file instead of inlining it (mutually exclusive
    with `prompt`; the same `{field}` placeholders work). Read once at judge construction, so
    a bad path fails the eval up front. A relative path resolves against the process working
    directory."""

    @model_validator(mode="after")
    def check_prompt_source(self) -> "JudgeConfig":
        if self.prompt is not None and self.prompt_file is not None:
            raise ValueError("set `prompt` or `prompt_file`, not both")
        return self


class JudgeResponse(StrictBaseModel, Generic[ParsedT]):
    """One judge call's result — returned to the caller and (JSON-serialized) appended to
    `trace.info["judge"]` for debugging, including the provider-reported `usage` (tokens + cost)."""

    text: str
    """The judge's raw reply."""
    parsed: ParsedT | None = None
    """The verdict the caller acts on (`parse`'s output, or the structured object for `schema`)."""
    usage: Usage | None = None


def judge_verdict(text: str, choices: Sequence[str]) -> str:
    """The verdict label in `text` (via `parse_judge_choice`), raising when none is found.

    This is the error-attribution contract for judge rewards: a *model* failure (empty,
    wrong, or non-committal reply) scores 0, but a *judge* failure — and an unparseable
    verdict is one — must not be scored against the model. Raising errors the rollout
    (recorded on the trace, excluded/retried in training) instead of hiding a broken judge
    behind silent 0s."""
    verdict = parse_judge_choice(text, choices)
    if verdict is None:
        raise ValueError(f"judge returned no {'/'.join(choices)} verdict: {text!r}")
    return verdict


ConfigT = TypeVar("ConfigT", bound=JudgeConfig, default=JudgeConfig)


def judge_config_type(cls: type) -> type[JudgeConfig]:
    """The `JudgeConfig` subclass a judge parameterizes — `Judge[ParsedT, MyJudgeConfig]` — read
    off its generic bases, walking the MRO so a further subclass inherits it. Falls back to the
    base `JudgeConfig` when none is given (the common case: a judge written without the extra
    generic param). Mirrors `taskset_config_type` / `harness_config_type`."""
    for klass in getattr(cls, "__mro__", [cls]):
        for base in getattr(klass, "__orig_bases__", ()):
            for arg in get_args(base):
                if isinstance(arg, type) and issubclass(arg, JudgeConfig):
                    return arg
    return JudgeConfig


class Judge(Generic[ParsedT, ConfigT]):
    """A per-task LLM judge over an OpenAI-compatible endpoint.

    Override `build_messages` (prompt setup) and `parse` (verdict parsing) — or just set the
    `prompt` template and use the defaults — then call `evaluate(**fields)`. Set `schema` to opt
    into structured outputs. Generic over the verdict and (optionally) the config type —
    `Judge[bool]`, or `Judge[float, MyJudgeConfig]` to also narrow `self.config`
    (see `judge_config_type`).
    """

    prompt: str | None = None
    """Default template for `build_messages`, formatted with the `evaluate` kwargs and sent as a
    single user message. Override `build_messages` for system+user or non-template prompts;
    `config.prompt` (inline) or `config.prompt_file` (a text file) overrides it from config."""
    schema: type[BaseModel] | None = None
    """Pydantic schema for OpenAI structured outputs. When set, the call uses
    `response_format=schema` and `JudgeResponse.parsed` is the validated object (provider must
    support structured outputs)."""

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config = cast(ConfigT, config or judge_config_type(type(self))())
        self.client: AsyncOpenAI = build_async_openai(self.config)
        if self.config.prompt_file is not None:
            # Read eagerly so a bad path fails at construction, not mid-eval; shadows the
            # class `prompt` for this instance (`config.prompt` is None — they're exclusive).
            self.prompt = self.config.prompt_file.read_text(encoding="utf-8")

    def build_messages(self, **fields: Any) -> str | Messages:
        """Prompt-setup hook: turn the `evaluate` fields into the messages to send (a single user
        message as a plain `str`, or a `vf.Messages` list). The default formats the `prompt`
        template (the config's, else the class's) with the fields; override it for a
        system+user / non-template prompt."""
        template = self.config.prompt or self.prompt
        if template is None:
            raise ValueError(
                f"{type(self).__name__} has no `prompt`; set it or override build_messages"
            )
        return template.format(**fields)

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
