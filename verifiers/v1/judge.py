"""A reusable per-task LLM judge for v1 tasksets.

`Judge` centralizes the common shape — client construction, one chat call, usage/cost
capture — and leaves the two things that differ as hooks: `build_messages` (prompt
setup) and `parse` (verdict parsing). Set `schema` to use OpenAI structured outputs,
in which case `JudgeResponse.parsed` is the validated pydantic object.

    class CorrectnessJudge(vf.Judge[bool]):
        prompt = "Question: {question}\\nAnswer: {answer}\\nResponse: {response}\\nCorrect? yes/no"

        def parse(self, response: vf.JudgeResponse[bool]) -> bool:
            return response.text.strip().lower().startswith("yes")

A judge is cheap to construct (the HTTP client opens per call, inside `complete`), so
build it where you use it.

Passing `trace=` records the call onto it: a typed record in `trace.info["judge"]`
plus tokens/cost in `trace.extra_usage` (separate from the agent's `trace.usage`).
The record lands even if the judge refuses or `parse` raises — the request was
already billed. Omit `trace` for a pure call.

A judge with an `id` and a `score` implementation is also *pluggable*: its config
lives on `TaskConfig.judges` (`--env.taskset.task.judges` — judges are config, never
row data), and `Task.score` runs it after the task's own `@reward`s.
"""

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
                "each `judges` entry needs an `id` (a judge plugin: `score`, `reference`, "
                "`rubric`, a local package, or a hub `org/name` package)"
            )
        resolved.append(judge_config_type(raw["id"]).model_validate(raw))
    return resolved


def check_judges(entries: Sequence[JudgeConfig]) -> None:
    for entry in entries:
        if not entry.id:
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `score`, `reference`, "
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
        # Substitute only this judge's documented placeholders, in one pass over the
        # original template — str.format would crash on any literal brace in a custom
        # prompt (a JSON-shaped instruction), and sequential replaces would re-scan
        # substituted values (a question containing a literal "{answer}" must not
        # pull in the answer). An unknown placeholder stays as written.
        if not fields:
            return template
        pattern = re.compile(r"\{(" + "|".join(map(re.escape, fields)) + r")\}")
        return pattern.sub(lambda m: str(fields[m.group(1)]), template)

    async def score(
        self, task: "TaskData", trace: "Trace"
    ) -> float | Mapping[str, float]:
        """The plugged tier of the spec: one bare call, `verdict(complete(render(...)))`.
        Implement `render` + `verdict` and both tiers work; override `score` only
        when one call isn't the right shape (batching, a schema'd parse)."""
        if type(self).render is Judge.render:
            raise NotImplementedError(
                f"{type(self).__name__} implements no `render`, so it can't be "
                "plugged via `taskset.task.judges`; implement `render` + `verdict` "
                "(one spec — plugged and agent-executable), or override `score`."
            )
        response = await self.complete(self.render(task, trace), trace=trace)
        return self.verdict(task, trace, response.text)

    def render(self, task: "TaskData", trace: "Trace") -> str | Messages:
        """The complete judging prompt for one finished trace. `score` is
        `verdict(complete(render(...)))` — implement `render` + `verdict` and the
        same spec is also agent-executable: the `solver-grader` env runs `render`'s
        prompt as its grader's task and hands the final reply to `verdict`."""
        raise NotImplementedError(
            f"{type(self).__name__} implements no `render`, so it can't drive a judge "
            "agent (the `solver-grader` env); implement `render` + `verdict`, or plug it via "
            "`taskset.task.judges` instead."
        )

    def verdict(
        self, task: "TaskData", trace: "Trace", reply: str
    ) -> float | Mapping[str, float]:
        """Parse a judge `reply` into the verdict `score` would have produced for
        `trace`, raising on a malformed one — a judge failure must error the
        rollout, never score the model."""
        raise NotImplementedError(
            f"{type(self).__name__} implements no `verdict`; implement `render` + "
            "`verdict` to make it agent-executable (the `solver-grader` env)."
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
