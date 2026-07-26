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

Passing `trace=` appends a small typed `JudgeCall` before the first provider attempt. Like an
agentic judge trace, it uses `ModelCall` for every provider exchange; opaque digests identify the
request and config without copying gold-bearing prompts into the trace. `JudgeResponse.call`
exposes the same evidence when no trace is supplied. On failure, the raised exception exposes it
as `error.judge_call`.

A judge can also be *plugged* rather than called from task code: a judge with an `id` and a
`score` implementation is a plugin (like a taskset or harness — see `verifiers.v1.judges` for the
built-ins and `verifiers.v1.loaders` for resolution). Its config lives on `TaskConfig.judges`
only — judges are config, never row data (`--taskset.task.judges`; a taskset config may
pre-plug them as class defaults) — and `Task.score` builds and runs it after the task's own
`@reward`s.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from pydantic import BaseModel, Field
from pydantic_core import to_json
from openai.lib._parsing import type_to_response_format_param
from typing_extensions import TypeVar

from verifiers.v1.clients.config import resolve_api_key
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.judge import (
    JudgeConfig,
    judge_key,
)
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.errors import ProviderError
from verifiers.v1.retries import backoff
from verifiers.v1.scoring import parse_judge_choice
from verifiers.v1.trace import (
    Error,
    JudgeCall,
    ModelCall,
    TimeSpan,
)
from verifiers.v1.types import Messages, StrictBaseModel, Usage
from verifiers.v1.utils.generic import generic_type

if TYPE_CHECKING:
    from verifiers.v1.task import TaskData
    from verifiers.v1.trace import Trace

ParsedT = TypeVar("ParsedT")


class JudgeResponse(StrictBaseModel, Generic[ParsedT]):
    text: str
    parsed: ParsedT | None = None
    usage: Usage | None = None
    call: JudgeCall | None = Field(default=None, exclude=True, repr=False)


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

    async def score(self, task: TaskData, trace: Trace) -> float | Mapping[str, float]:
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
        trace: Trace | None = None,
        schema: type[BaseModel] | None = None,
        parse: Callable[[JudgeResponse[Any]], Any] | None = None,
        **sampling: Any,
    ) -> JudgeResponse[Any]:
        """Call the judge and preserve its full attempt/result lifecycle."""
        wire = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else [message_to_wire(m) for m in messages]
        )
        effective = type(self.config.sampling).model_validate(
            {**self.config.sampling.model_dump(exclude_none=True), **sampling}
        )
        body: dict[str, Any] = {"messages": wire}
        if schema is not None:
            body["response_format"] = type_to_response_format_param(schema)
        dialect = ChatDialect()
        request = dialect.apply_overrides(body, self.config.model, effective)
        request_digest = hashlib.sha256(
            to_json(request, inf_nan_mode="null")
        ).hexdigest()
        config_digest = hashlib.sha256(
            to_json(self.config.model_dump(mode="json"), inf_nan_mode="null")
        ).hexdigest()
        call = JudgeCall(
            judge=f"{type(self).__module__}.{type(self).__qualname__}",
            config_digest=config_digest,
            request_digest=request_digest,
        )
        if trace is not None:
            trace.judge_calls.append(call)

        try:
            client = EvalClient(
                self.config.base_url,
                resolve_api_key(self.config),
                self.config.headers,
            )
            try:
                provider_response = None
                for attempt in range(self.config.max_retries + 1):
                    provider_call = ModelCall(
                        model=self.config.model,
                        sampling=dialect.parse_sampling(request),
                        endpoint=dialect.upstream_path,
                        time=TimeSpan(start=time.time()),
                    )
                    call.calls.append(provider_call)
                    try:
                        async with asyncio.timeout(self.config.timeout):
                            provider_response = await client.get_response(
                                dialect, body, self.config.model, effective
                            )
                    except BaseException as error:
                        provider_call.error = Error.from_exception(error)
                        if not isinstance(error, (ProviderError, TimeoutError)):
                            raise
                        transient = isinstance(error, TimeoutError)
                        delay = backoff(attempt)
                        if isinstance(error, ProviderError):
                            transient = (
                                error.should_retry
                                if error.should_retry is not None
                                else error.status_code in (408, 409, 429)
                                or error.status_code >= 500
                            )
                            if (
                                error.retry_after is not None
                                and 0 < error.retry_after <= 60
                            ):
                                delay = error.retry_after
                        if attempt == self.config.max_retries or not transient:
                            raise
                    else:
                        provider_call.finish_reason = provider_response.finish_reason
                        provider_call.usage = provider_response.usage
                        break
                    finally:
                        provider_call.time.end = time.time()
                    await asyncio.sleep(delay)
            finally:
                await client.close()

            assert provider_response is not None
            raw = cast(dict, provider_response.raw)
            refusal = raw["choices"][0]["message"].get("refusal")
            response = JudgeResponse[Any](
                text=provider_response.message.content or "",
                usage=provider_response.usage,
                call=call,
            )
            call.text = response.text
            if refusal is not None:
                call.outcome = "refusal"
                raise ValueError(f"judge refused output: {refusal}")
            try:
                if schema is not None:
                    response.parsed = schema.model_validate_json(response.text)
                if parse is not None:
                    response.parsed = parse(response)
            except Exception:
                call.outcome = "parse_error"
                raise
            call.outcome = "success"
            return response
        except BaseException as error:
            call.outcome = call.outcome or (
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "provider_error"
            )
            call.error = Error.from_exception(error)
            setattr(error, "judge_call", call)
            raise

    async def evaluate(
        self, *, trace: Trace | None = None, **fields: Any
    ) -> JudgeResponse[ParsedT]:
        messages = self.build_messages(**fields)
        return await self.complete(
            messages, trace=trace, schema=self.schema, parse=self.parse
        )
