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

    class MyData(vf.TaskData):
        answer: str

    class MyTask(vf.Task[MyData, vf.State, MyConfig]):
        @vf.reward
        async def correct(self, trace) -> float:
            judge = CorrectnessJudge(self.config.judge)  # config.judge: vf.JudgeConfig
            result = await judge.evaluate(
                trace=trace,
                question=self.data.prompt_text,
                answer=self.data.answer,
                response=...,
            )
            return float(result.parsed)

A judge is cheap to construct (the HTTP client is opened per call, inside `complete`, and
closed when the call returns), so build it where you use it.

Passing `trace=` records the call onto it — a request/response record appended to
`trace.info["judge_calls"]` and
the call's tokens + cost added to `trace.extra_usage` (kept separate from the agent's `trace.usage`),
so judge behaviour and spend are no longer invisible. The record lands even if the judge refuses, an
empty structured output comes back, or `parse` raises (the request was already billed). Omit `trace`
for a pure call (e.g. in tests).

A judge call the provider fails transiently (a 5xx, 408 or 429, a dropped connection or a
timeout) is sent again under the config's `retry` budget (`ProviderRetryConfig`); the failure
it gives up on raises, so a judge error never becomes a score. With `max_input_tokens` set, a
call whose reported input tokens reach it raises `JudgeInputTooLarge` instead of returning a
grade: some judge endpoints cut longer input silently and report exactly their cap. A request
too large by a wide margin (`input_token_floor`) raises before the call, unbilled.

A judge can also be *plugged* rather than called from task code: a judge with an `id` and a
`score` implementation is a plugin (like a taskset or harness — see `verifiers.v1.judges` for the
built-ins and `verifiers.v1.utils.loaders` for resolution). Its config lives on `TaskConfig.judges`
only — judges are config, never row data (`--taskset.task.judges`; a taskset config may
pre-plug them as class defaults) — and `Task.score` builds and runs it after the task's own
`@reward`s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from openai import APIConnectionError, APIStatusError
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)
from pydantic import BaseModel
from typing_extensions import TypeVar

from verifiers.v1.clients.base import build_async_openai
from verifiers.v1.configs.judge import (
    JudgeConfig,
    judge_key,
)
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import model_error
from verifiers.v1.types import Messages, SamplingConfig, Usage
from verifiers.v1.utils.generic import concrete_type, merge_defaults
from verifiers.v1.utils.retries import provider_retry_wait
from verifiers.v1.utils.score import parse_judge_choice

if TYPE_CHECKING:
    from verifiers.v1.task import TaskData
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

ParsedT = TypeVar("ParsedT")
T = TypeVar("T")

RETRY_BACKOFF_SECONDS = 1.0
"""First wait before a judge call the provider failed transiently is sent again; it
doubles per failure up to `ProviderRetryConfig.max_backoff`."""


BYTES_PER_TOKEN = 4.5
"""The most UTF-8 bytes of JSON request content one token covers in ordinary text
(English prose and code measure 3.8-3.9). Only the pre-call shortcut uses it: text with
long whitespace runs can take more bytes per token, so it never decides alone."""


class JudgeInputTooLarge(ValueError):
    """A judge request reaches the judge's `max_input_tokens`: the endpoint's reported
    input count is at the limit (``reported``), or the request is far over it before the
    call. The input is the task's to bound, and an endpoint that cuts it would grade
    part of it."""

    def __init__(self, tokens: int, limit: int, *, reported: bool) -> None:
        super().__init__(
            f"judge read {tokens} input tokens, at the judge's max_input_tokens of "
            f"{limit}: the endpoint may have cut the input"
            if reported
            else f"judge input needs at least {tokens} tokens, over the judge's "
            f"max_input_tokens of {limit}"
        )
        self.tokens = tokens
        self.limit = limit
        self.reported = reported


def input_token_floor(payload: object) -> int:
    """About the fewest tokens `payload` (JSON-serializable request content) can take:
    its UTF-8 size as JSON over `BYTES_PER_TOKEN`. A heuristic for refusing a clearly
    oversized request before the call; the endpoint's own count is the authority."""
    return math.ceil(
        len(json.dumps(payload, ensure_ascii=False).encode()) / BYTES_PER_TOKEN
    )


def _retry_after(error: APIStatusError) -> float | None:
    """The wait in seconds a provider's `Retry-After` header asks for, if any."""
    value = error.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class JudgeResponse(BaseModel, Generic[ParsedT]):
    text: str
    parsed: ParsedT | None = None
    usage: Usage | None = None


JudgeView = Literal["last_reply", "full_trace"]


def judge_question(task: TaskData, question_field: str) -> str:
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


def judge_response(trace: Trace, view: JudgeView) -> str:
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
    return concrete_type(cls, JudgeConfig) or JudgeConfig


class Judge(Generic[ParsedT, ConfigT]):
    prompt: str | None = None
    """Default prompt template, overridden by a config `prompt` file."""
    schema: type[BaseModel] | None = None

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config = cast(ConfigT, config or judge_config_cls(type(self))())
        if self.config.prompt is not None:
            self.prompt = self.config.prompt.read_text()

    @property
    def reward_name(self) -> str:
        fallback = re.sub(
            r"(?<!^)(?=[A-Z])", "_", type(self).__name__.removesuffix("Judge")
        ).lower()
        return judge_key(self.config) or fallback or "judge"

    def build_messages(self, **fields: Any) -> str | Messages:
        template = self.prompt
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

    async def score(self, task: TaskData, trace: Trace) -> float | Mapping[str, float]:
        raise NotImplementedError(
            f"{type(self).__name__} implements no `score`, so it can't be plugged via "
            "`taskset.task.judges`; implement `score` (see verifiers.v1.judges for examples) or "
            "call it from a task `@reward` instead."
        )

    def check_input(self, payload: object) -> None:
        """Raise `JudgeInputTooLarge` before the call when `payload`, the request content
        the model reads, is far over the config's `max_input_tokens`
        (`input_token_floor`)."""
        limit = self.config.max_input_tokens
        if limit is not None and (tokens := input_token_floor(payload)) > limit:
            raise JudgeInputTooLarge(tokens, limit, reported=False)

    def check_usage(self, usage: Usage | None) -> None:
        """Raise `JudgeInputTooLarge` when the endpoint reports input tokens at or over
        the config's `max_input_tokens`: an endpoint that cuts input reports exactly its
        cap, so a count at the limit means the model may have read only part of it. A
        response without a count leaves only the pre-call check."""
        limit = self.config.max_input_tokens
        if limit is not None and usage is not None and usage.input_tokens >= limit:
            raise JudgeInputTooLarge(usage.input_tokens, limit, reported=True)

    async def send(self, call: Callable[[], Awaitable[T]]) -> T:
        """Run one provider call, sending it again while it fails transiently and the
        config's `retry` budget lasts (`provider_retry_wait`). The failure it gives up
        on raises unchanged."""
        loop = asyncio.get_running_loop()
        failures = 0
        first_failure = 0.0
        while True:
            try:
                return await call()
            except (APIStatusError, APIConnectionError) as error:
                now = loop.time()
                if not failures:
                    first_failure = now
                retry_after = (
                    _retry_after(error) if isinstance(error, APIStatusError) else None
                )
                wait = provider_retry_wait(
                    model_error(error, retry_after=retry_after),
                    failures,
                    now - first_failure,
                    self.config.retry,
                    base=RETRY_BACKOFF_SECONDS,
                )
                if wait is None:
                    raise
                failures += 1
                logger.warning(
                    "judge %s call failed (%s), retry %d in %.1fs",
                    self.reward_name,
                    type(error).__name__,
                    failures,
                    wait,
                )
                await asyncio.sleep(wait)

    def parse(self, response: JudgeResponse[ParsedT]) -> ParsedT:
        if self.schema is not None:
            return cast(ParsedT, response.parsed)
        return cast(ParsedT, response.text)

    async def complete(
        self,
        messages: str | Messages,
        *,
        trace: Trace | None = None,
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
        # Transport options are SDK arguments, not model sampling parameters.
        request_options = {
            key: sampling.pop(key)
            for key in ("extra_headers", "extra_query", "timeout")
            if key in sampling
        }
        params = SamplingConfig.model_validate(
            merge_defaults(self.config.sampling, sampling)
        ).model_dump(exclude_none=True)
        sdk_fields = CompletionCreateParamsNonStreaming.__annotations__
        kwargs.update(
            {key: value for key, value in params.items() if key in sdk_fields}
        )
        # The SDK accepts provider-specific parameters only through extra_body.
        kwargs["extra_body"] = {
            key: value for key, value in params.items() if key not in sdk_fields
        }
        kwargs.update(request_options)
        self.check_input(kwargs["messages"])

        response: JudgeResponse[Any] | None = None
        try:
            async with build_async_openai(self.config) as client:
                if schema is not None:
                    completion = await self.send(
                        lambda: client.beta.chat.completions.parse(
                            response_format=schema, **kwargs
                        )
                    )
                    choice = completion.choices[0]
                    response = JudgeResponse(
                        text=choice.message.content or "",
                        parsed=choice.message.parsed,
                        usage=Usage.from_openai(completion.usage),
                    )
                    self.check_usage(response.usage)
                    if choice.message.refusal is not None:
                        raise RuntimeError(
                            f"judge refused structured output: {choice.message.refusal}"
                        )
                    if response.parsed is None:
                        raise RuntimeError(
                            f"judge returned no parseable structured output (finish_reason={choice.finish_reason})"
                        )
                else:
                    completion = await self.send(
                        lambda: client.chat.completions.create(**kwargs)
                    )
                    response = JudgeResponse(
                        text=completion.choices[0].message.content or "",
                        usage=Usage.from_openai(completion.usage),
                    )
                    self.check_usage(response.usage)
            if parse is not None:
                response.parsed = parse(response)
            return response
        finally:
            if trace is not None and response is not None:
                trace.record_judge_call(
                    name=self.reward_name,
                    request={
                        "model": kwargs["model"],
                        "messages": kwargs["messages"],
                    },
                    response=response,
                )

    async def evaluate(
        self, *, trace: Trace | None = None, **fields: Any
    ) -> JudgeResponse[ParsedT]:
        messages = self.build_messages(**fields)
        return await self.complete(
            messages, trace=trace, schema=self.schema, parse=self.parse
        )
