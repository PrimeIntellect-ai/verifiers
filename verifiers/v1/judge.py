"""A reusable per-task LLM judge for v1 tasksets.

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

A judge can also be *plugged* rather than called from taskset code: a judge with an `id` and a
`score` implementation is a plugin (like a taskset or harness — see `verifiers.v1.judges` for the
built-ins and `verifiers.v1.loaders` for resolution), attached to any eval via the base
`TasksetConfig.judges` and run by `Taskset.score` after the taskset's own `@reward`s.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, cast, get_args

from openai import AsyncOpenAI
from pydantic import BaseModel, SerializeAsAny
from typing_extensions import TypeVar

from verifiers.v1.clients.config import BaseClientConfig, build_async_openai
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.utils.install import env_name
from verifiers.v1.types import ID, Messages, SamplingConfig, StrictBaseModel, Usage

if TYPE_CHECKING:
    from verifiers.v1.task import Task
    from verifiers.v1.trace import Trace

ParsedT = TypeVar("ParsedT")


class JudgeSamplingConfig(SamplingConfig):
    """Sampling knobs for a judge call (`temperature` / `top_p` / `reasoning_effort` /
    `max_tokens`, plus provider-specific keys via `extra='allow'`). Same shape as the rollout's
    `SamplingConfig` — set e.g. `judge.sampling.max_tokens`."""


class JudgeConfig(BaseClientConfig):
    """An LLM-judge endpoint. Inherits `base_url` / `api_key_var` / `headers` (with the Prime
    auto-config) from `BaseClientConfig`; adds the model and sampling. Subclass to add
    judge-specific fields (see `verifiers.v1.judges.rubric.RubricJudgeConfig`)."""

    id: ID = ""
    """The judge id, which selects a judge plugin for a config-plugged judge (see
    `TasksetConfig.judges`): a built-in (`binary`, `rubric`), a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see `ID`).
    Empty for a judge the taskset builds and calls itself."""
    name: str = ""
    """The reward key this judge's verdict records under when plugged (see `Judge.reward_name`);
    defaults to the id's package name. Set it to disambiguate two plugged judges sharing an id."""
    weight: float = 1.0
    """How a plugged judge's verdict weighs into `trace.reward` (like `@vf.reward(weight=...)`)."""
    model: str = "deepseek/deepseek-v4-flash"
    sampling: JudgeSamplingConfig = JudgeSamplingConfig()
    prompt: str | None = None
    """Prompt-template override for `build_messages` (None = the judge class's own `prompt`),
    so a plugged judge's prompt is tunable from config alone."""


Judges = list[SerializeAsAny[JudgeConfig]]
"""The type of `TasksetConfig.judges` — a list of plugged-judge configs, each resolved by its
`id`. `SerializeAsAny` keeps the resolved subclasses' fields through `model_dump` (the
env-server wire). Use it to give a taskset config a default judge:
`judges: vf.Judges = [vf.BinaryJudgeConfig(id="binary")]`."""


class JudgeResponse(StrictBaseModel, Generic[ParsedT]):
    """One judge call's result — returned to the caller and (JSON-serialized) appended to
    `trace.info["judge"]` for debugging, including the provider-reported `usage` (tokens + cost)."""

    text: str
    """The judge's raw reply."""
    parsed: ParsedT | None = None
    """The verdict the taskset acts on (`parse`'s output, or the structured object for `schema`)."""
    usage: Usage | None = None


ConfigT = TypeVar("ConfigT", bound=JudgeConfig, default=JudgeConfig)


def judge_config_cls(cls: type) -> type[JudgeConfig]:
    """The `JudgeConfig` subclass a judge parameterizes — `Judge[ParsedT, MyJudgeConfig]` — read
    off its generic bases, walking the MRO so a further subclass inherits it. Falls back to the
    base `JudgeConfig` when none is given (the common case: a code-level judge written without
    the extra generic param). Mirrors `state_cls` / `taskset_config_type`."""
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
    `Judge[bool]` for a code-level judge, `Judge[float, MyJudgeConfig]` to also narrow
    `self.config` (which is how a plugged judge declares its config for `--taskset.judges`
    narrowing; see `judge_config_cls`). A pluggable judge additionally implements `score`.
    """

    prompt: str | None = None
    """Default template for `build_messages`, formatted with the `evaluate` kwargs and sent as a
    single user message. Override `build_messages` for system+user or non-template prompts;
    `config.prompt` overrides it from config."""
    schema: type[BaseModel] | None = None
    """Pydantic schema for OpenAI structured outputs. When set, the call uses
    `response_format=schema` and `JudgeResponse.parsed` is the validated object (provider must
    support structured outputs)."""

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config = cast(ConfigT, config or judge_config_cls(type(self))())
        self.client: AsyncOpenAI = build_async_openai(self.config)

    @property
    def reward_name(self) -> str:
        """The reward key this judge records under (and the built-in rubric judge's metric
        prefix): the config `name`, else the id's package name, else the snake-cased class
        name (a code-level judge with neither)."""
        fallback = re.sub(
            r"(?<!^)(?=[A-Z])", "_", type(self).__name__.removesuffix("Judge")
        ).lower()
        return self.config.name or env_name(self.config.id) or fallback or "judge"

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

    async def score(self, task: "Task", trace: "Trace") -> float | Mapping[str, float]:
        """The plugged-judge contract: grade one finished rollout, returning a verdict that
        `Taskset.score` records into `trace.rewards` under `config.reward_name` with
        `config.weight` (a mapping records each entry under its own key, like a `@reward`).
        Like the scoring hooks, an override declares the inputs it needs *by parameter name*
        and the framework injects them: any subset of `task`, `trace`, `runtime`. Not
        implemented on the base class — only a judge that implements `score` can be plugged
        via `TasksetConfig.judges`; code-level judges just call `evaluate` from a `@reward`."""
        raise NotImplementedError(
            f"{type(self).__name__} implements no `score`, so it can't be plugged via "
            "`taskset.judges`; implement `score` (see verifiers.v1.judges for examples) or "
            "call it from a taskset `@reward` instead."
        )

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
