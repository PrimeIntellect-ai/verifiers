from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from pydantic import (
    BaseModel,
    computed_field,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from verifiers.types import (
    ErrorData,
    FinishReason,
    Messages,
    ResponseTokens,
    ToolCall,
    ToolMessage,
    Usage,
)
from verifiers.utils.error_utils import error_data, validate_error_data
from verifiers.utils.save_utils import serialize_messages_for_output

from .types import JsonData, ModelConfig
from .utils.json_utils import json_data
from .utils.task_freeze_utils import assert_serializable

if TYPE_CHECKING:
    from .task import Task


class TimeSpan(BaseModel, extra="forbid"):
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start) if self.end else 0.0

    def begin(self) -> None:
        self.start = time.time()

    def finish(self) -> None:
        self.end = time.time()


class Timing(BaseModel, extra="forbid"):
    start_time: float = Field(default_factory=time.time)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)
    cleanup: TimeSpan = Field(default_factory=TimeSpan)
    model: list[TimeSpan] = Field(default_factory=list)
    runtime: list[TimeSpan] = Field(default_factory=list)

    @property
    def total(self) -> float:
        end = max(
            self.setup.end,
            self.generation.end,
            self.scoring.end,
            self.cleanup.end,
            self.start_time,
        )
        return max(0.0, end - self.start_time)


class TurnUsage(BaseModel, extra="forbid"):
    prompt_tokens: StrictInt = 0
    reasoning_tokens: StrictInt = 0
    completion_tokens: StrictInt = 0
    total_tokens: StrictInt = 0

    @classmethod
    def from_usage(cls, usage: Usage | None) -> "TurnUsage | None":
        if usage is None:
            return None
        return cls(
            prompt_tokens=usage.prompt_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


class TurnTokens(BaseModel, extra="forbid"):
    prompt_ids: list[StrictInt] = Field(default_factory=list)
    prompt_mask: list[StrictInt] = Field(default_factory=list)
    prompt_advantages: list[StrictFloat] | None = None
    completion_ids: list[StrictInt] = Field(default_factory=list)
    completion_mask: list[StrictInt] = Field(default_factory=list)
    completion_logprobs: list[StrictFloat] = Field(default_factory=list)
    completion_advantages: list[StrictFloat] | None = None
    overlong_prompt: StrictBool = False
    is_truncated: StrictBool = False

    @classmethod
    def from_response(
        cls, tokens: ResponseTokens | None, *, is_truncated: bool = False
    ) -> "TurnTokens | None":
        if tokens is None:
            return None
        return cls(
            prompt_ids=list(tokens.prompt_ids),
            prompt_mask=list(tokens.prompt_mask),
            completion_ids=list(tokens.completion_ids),
            completion_mask=list(tokens.completion_mask),
            completion_logprobs=list(tokens.completion_logprobs),
            is_truncated=is_truncated,
        )

    @model_validator(mode="after")
    def validate_lengths(self) -> "TurnTokens":
        if len(self.prompt_ids) != len(self.prompt_mask):
            raise ValueError("TurnTokens prompt_ids and prompt_mask lengths differ.")
        if self.prompt_advantages is not None and len(self.prompt_ids) != len(
            self.prompt_advantages
        ):
            raise ValueError(
                "TurnTokens prompt_ids and prompt_advantages lengths differ."
            )
        if len(self.completion_ids) != len(self.completion_mask):
            raise ValueError(
                "TurnTokens completion_ids and completion_mask lengths differ."
            )
        if len(self.completion_ids) != len(self.completion_logprobs):
            raise ValueError(
                "TurnTokens completion_ids and completion_logprobs lengths differ."
            )
        if self.completion_advantages is not None and len(self.completion_ids) != len(
            self.completion_advantages
        ):
            raise ValueError(
                "TurnTokens completion_ids and completion_advantages lengths differ."
            )
        return self


class Turn(BaseModel, extra="forbid"):
    """One model request/response boundary in a rollout transcript."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    prompt: Messages
    completion: Messages = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolMessage] = Field(default_factory=list)
    response_id: str | None = None
    model: str | None = None
    created: StrictInt | None = None
    finish_reason: FinishReason = None
    usage: TurnUsage | None = None
    tokens: TurnTokens | None = None
    reward: float | None = None
    is_truncated: bool = False
    timing: TimeSpan = Field(default_factory=TimeSpan)


class Extras(BaseModel, extra="forbid"):
    @staticmethod
    def schema_for(extras: "Extras | None") -> type["Extras"] | None:
        if extras is None:
            return None
        schema = type(extras)
        if not issubclass(schema, Extras):
            raise TypeError("extras config must be a vf.Extras object.")
        return schema

    @staticmethod
    def defaults_for(extras: "Extras | None") -> JsonData:
        if extras is None:
            return {}
        return json_data(extras.model_dump(mode="json", exclude_none=True))

    @staticmethod
    def merge_defaults(
        taskset_defaults: JsonData, harness_defaults: JsonData
    ) -> JsonData:
        conflicts = sorted(set(taskset_defaults) & set(harness_defaults))
        if conflicts:
            raise ValueError(
                f"Extras config keys are defined twice: {', '.join(conflicts)}."
            )
        return {**deepcopy(taskset_defaults), **deepcopy(harness_defaults)}

    @staticmethod
    def realize_schema(
        taskset_schema: type["Extras"] | None,
        harness_schema: type["Extras"] | None,
    ) -> type["Extras"] | None:
        schemas = [schema for schema in (taskset_schema, harness_schema) if schema]
        if not schemas:
            return None
        if len(schemas) == 1:
            return schemas[0]
        seen: dict[str, type[BaseModel]] = {}
        for schema in schemas:
            for field_name in schema.model_fields:
                if field_name in seen:
                    raise ValueError(
                        f"Extras field {field_name!r} is defined by both "
                        f"{seen[field_name].__name__} and {schema.__name__}."
                    )
                seen[field_name] = schema
        return cast(
            type[Extras],
            type(
                "RealizedExtras",
                tuple(schemas),
                {"__module__": __name__},
            ),
        )


class State(BaseModel, extra="forbid"):
    """Strict serializable v1 rollout state."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str | None = None
    transcript: list[Turn] = Field(default_factory=list)
    extras: JsonData = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    reward: float = 0.0
    artifacts: JsonData = Field(default_factory=dict)
    usage: dict[str, float] = Field(default_factory=dict)
    timing: Timing = Field(default_factory=Timing)
    is_completed: bool = False
    is_truncated: bool = False
    stop_condition: str | None = None
    error: ErrorData | None = None
    group_id: str | None = None
    metadata: JsonData = Field(default_factory=dict)
    model: ModelConfig | None = None
    teacher: ModelConfig | None = None

    @computed_field
    @property
    def prompt(self) -> Messages:
        if not self.transcript:
            return []
        return self.transcript[-1].prompt

    @computed_field
    @property
    def completion(self) -> Messages:
        if not self.transcript:
            return []
        return self.transcript[-1].completion

    @computed_field
    @property
    def messages(self) -> Messages:
        if not self.transcript:
            return []
        latest = self.transcript[-1]
        return [*latest.prompt, *latest.completion]

    def stop(self, condition: str = "state_done") -> None:
        if not condition:
            raise ValueError("State.stop condition must be non-empty.")
        self.is_completed = True
        self.stop_condition = condition

    def capture_error(self, error: BaseException) -> None:
        self.error = error_data(error)
        self.stop("has_error")

    def assert_serializable(self) -> None:
        assert_serializable(self.model_dump(mode="json", exclude_none=True))

    @staticmethod
    def serialized_messages(messages: object) -> list[JsonData]:
        serialized: list[JsonData] = []
        for index, message in enumerate(serialize_messages_for_output(messages)):
            serialized.append(json_data(message, context=f"message[{index}]"))
        return serialized

    @staticmethod
    def turn_record(turn: Turn) -> dict[str, object]:
        return {
            "id": turn.id,
            "prompt": State.serialized_messages(turn.prompt),
            "completion": State.serialized_messages(turn.completion),
            "tool_calls": [
                json_data(tool_call, context="tool_call")
                for tool_call in turn.tool_calls
            ],
            "tool_results": State.serialized_messages(turn.tool_results),
            "response_id": turn.response_id,
            "model": turn.model,
            "created": turn.created,
            "finish_reason": turn.finish_reason,
            "usage": turn.usage.model_dump(mode="json", exclude_none=True)
            if turn.usage is not None
            else None,
            "tokens": turn.tokens.model_dump(mode="json", exclude_none=True)
            if turn.tokens is not None
            else None,
            "reward": turn.reward,
            "is_truncated": turn.is_truncated,
            "timing": turn.timing.model_dump(mode="json", exclude_none=True),
        }

    def to_output(
        self, task: "Task", state_columns: list[str] | None = None
    ) -> dict[str, object]:
        prompt = self.prompt if self.transcript else task.prompt
        serialize_messages = type(self).serialized_messages
        turn_record = type(self).turn_record
        output: dict[str, object] = {
            "example_id": task.row_id,
            "prompt": serialize_messages(prompt),
            "completion": serialize_messages(self.completion),
            "reward": float(self.reward),
            "timing": self.timing.model_dump(mode="json"),
            "is_completed": self.is_completed,
            "is_truncated": self.is_truncated,
            "metrics": dict(self.metrics),
            "extras": deepcopy(self.extras),
            "stop_condition": self.stop_condition,
            "transcript": [turn_record(turn) for turn in self.transcript],
        }
        answer = getattr(task, "answer", None)
        if answer is not None:
            output["answer"] = str(answer)
        info = getattr(task, "info", None)
        if isinstance(info, dict) and info:
            output["info"] = deepcopy(info)
        if self.error is not None:
            output["error"] = validate_error_data(self.error)
            output["error_chain"] = self.error["error_chain_repr"]
            output["long_error_chain"] = self.error["error_chain_str"]
        usage = dict(self.usage)
        for turn in self.transcript:
            if turn.usage is None:
                continue
            usage["input_tokens"] = usage.get("input_tokens", 0.0) + float(
                turn.usage.prompt_tokens
            )
            usage["output_tokens"] = usage.get("output_tokens", 0.0) + float(
                turn.usage.completion_tokens
            )
        if usage:
            output["token_usage"] = {
                "input_tokens": float(usage.get("input_tokens", 0.0)),
                "output_tokens": float(usage.get("output_tokens", 0.0)),
            }
        reserved_output_fields = set(output)
        for key, value in self.metrics.items():
            if key in output:
                raise ValueError(
                    f"Metric name {key!r} conflicts with a reserved output field."
                )
            output[key] = value
        for column in state_columns or []:
            if column in output:
                if column in reserved_output_fields:
                    continue
                raise ValueError(
                    f"State column {column!r} conflicts with an existing output field."
                )
            if column == "prompt":
                output[column] = serialize_messages(prompt)
            elif column == "completion":
                output[column] = serialize_messages(self.completion)
            elif column == "messages":
                output[column] = serialize_messages(self.messages)
            elif column == "extras":
                output[column] = deepcopy(self.extras)
            elif column == "transcript":
                output[column] = [turn_record(turn) for turn in self.transcript]
            elif column in type(self).model_fields:
                value = getattr(self, column)
                model_dump = getattr(value, "model_dump", None)
                if callable(model_dump):
                    output[column] = model_dump(mode="json", exclude_none=True)
                else:
                    output[column] = deepcopy(value)
            elif column in self.extras:
                output[column] = deepcopy(self.extras[column])
            else:
                output[column] = None
        json.dumps(output)
        return output
