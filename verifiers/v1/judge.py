from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, cast

from pydantic import BaseModel, SerializeAsAny, model_validator
from typing_extensions import TypeVar

from verifiers.v1.clients.config import BaseClientConfig, build_async_openai
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.scoring import parse_judge_choice
from verifiers.v1.utils.install import env_name
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.types import ID, Messages, SamplingConfig, StrictBaseModel, Usage

if TYPE_CHECKING:
    from verifiers.v1.task import TaskData
    from verifiers.v1.trace import Trace

ParsedT = TypeVar("ParsedT")


class JudgeSamplingConfig(SamplingConfig):
    pass


class JudgeConfig(BaseClientConfig):
    id: ID = ""
    """Plugin id; empty for a judge called directly by task code."""
    name: str = ""
    """Reward key override for a plugged judge."""
    weight: float = 1.0
    model: str = "openai/gpt-5.4-nano"
    sampling: JudgeSamplingConfig = JudgeSamplingConfig()
    prompt: str | None = None
    prompt_file: Path | None = None
    """Prompt file override, mutually exclusive with `prompt`."""

    @model_validator(mode="after")
    def check_prompt_source(self) -> "JudgeConfig":
        if self.prompt is not None and self.prompt_file is not None:
            raise ValueError("set `prompt` or `prompt_file`, not both")
        return self


Judges = list[SerializeAsAny[JudgeConfig]]
"""Config-plugged judges, resolved by id and serialized as their concrete types."""


def judge_key(config: JudgeConfig) -> str:
    return config.name or env_name(config.id)


def resolve_judges(entries: Sequence[Any]) -> list[JudgeConfig]:
    from verifiers.v1.loaders import judge_config_type

    resolved = []
    for entry in entries:
        raw = entry.model_dump() if isinstance(entry, BaseModel) else dict(entry)
        if not raw.get("id"):
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, a local package, or a hub `org/name` package)"
            )
        resolved.append(judge_config_type(raw["id"]).model_validate(raw))
    return resolved


def check_judges(entries: Sequence[JudgeConfig]) -> None:
    for entry in entries:
        if not entry.id:
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, a local package, or a hub `org/name` package)"
            )
    keys = [judge_key(entry) for entry in entries]
    if duplicates := {key for key in keys if keys.count(key) > 1}:
        raise ValueError(
            f"`judges` entries share a reward key {sorted(duplicates)}; set a "
            "distinct `name` on each to keep both verdicts"
        )


class JudgeResponse(StrictBaseModel, Generic[ParsedT]):
    text: str
    parsed: ParsedT | None = None
    usage: Usage | None = None


JudgeView = Literal["last_reply", "full_trace"]


def judge_question(task: "TaskData", question_field: str) -> str:
    if not question_field:
        return task.prompt_text
    question = getattr(task, question_field, None)
    if question is None:
        raise ValueError(
            f"judge found no {question_field!r} field on the task; point "
            "`question_field` at the task's question field, or leave it empty for the "
            "task prompt"
        )
    return str(question)


def judge_response(trace: "Trace", view: JudgeView) -> str:
    return trace.transcript if view == "full_trace" else trace.last_reply


def judge_verdict(text: str, choices: Sequence[str]) -> str:
    """Parse a verdict, raising so judge failures are not scored against the model."""
    verdict = parse_judge_choice(text, choices)
    if verdict is None:
        raise ValueError(f"judge returned no {'/'.join(choices)} verdict: {text!r}")
    return verdict


ConfigT = TypeVar("ConfigT", bound=JudgeConfig, default=JudgeConfig)


def judge_config_cls(cls: type) -> type[JudgeConfig]:
    """Resolve a judge's config specialization through its MRO, else `JudgeConfig`."""
    return generic_type(cls, JudgeConfig) or JudgeConfig


class Judge(Generic[ParsedT, ConfigT]):
    prompt: str | None = None
    """Default prompt template, overridden by config."""
    schema: type[BaseModel] | None = None

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config = cast(ConfigT, config or judge_config_cls(type(self))())
        if self.config.prompt_file is not None:
            self.prompt = self.config.prompt_file.read_text(encoding="utf-8")

    @property
    def reward_name(self) -> str:
        fallback = re.sub(
            r"(?<!^)(?=[A-Z])", "_", type(self).__name__.removesuffix("Judge")
        ).lower()
        return judge_key(self.config) or fallback or "judge"

    def build_messages(self, **fields: Any) -> str | Messages:
        template = self.config.prompt or self.prompt
        if template is None:
            raise ValueError(
                f"{type(self).__name__} has no `prompt`; set it or override build_messages"
            )
        return template.format(**fields)

    async def score(
        self, task: "TaskData", trace: "Trace"
    ) -> float | Mapping[str, float]:
        raise NotImplementedError(
            f"{type(self).__name__} implements no `score`, so it can't be plugged via "
            "`taskset.task.judges`; implement `score` (see verifiers.v1.judges for examples) or "
            "call it from a task `@reward` instead."
        )

    def parse(self, response: JudgeResponse[ParsedT]) -> ParsedT:
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
        """Call the judge and record billed usage even when parsing fails."""
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
            async with build_async_openai(self.config) as client:
                if schema is not None:
                    completion = await client.beta.chat.completions.parse(
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
                        raise RuntimeError(
                            f"judge returned no parseable structured output "
                            f"(finish_reason={choice.finish_reason})"
                        )
                else:
                    completion = await client.chat.completions.create(**kwargs)
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
        messages = self.build_messages(**fields)
        return await self.complete(
            messages, trace=trace, schema=self.schema, parse=self.parse
        )
