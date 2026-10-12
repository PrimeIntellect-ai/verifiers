"""A durable, serializable handle to one agent's exchange: `Session`.

A session is what it takes to continue an agent's conversation somewhere else — a later
stage, another process, a replacement box. It holds the box's filesystem checkpoint (None
for a box the agent borrowed), the conversation so far, the task it was solving, and the
model calls it had made. `agent.snapshot(interaction)` takes one; `agent.run(session=...)`
and `agent.interaction(session=...)` continue one the way `Harness.resume` continues an
exchange: the conversation becomes the opening prompt, so training samples see the prior
turns as prompt context. With `AgentConfig.save = "per_call"` a rollout hands its caller
one after every model response (`on_save`).

(`verifiers.v1.session` is the interception layer's per-rollout `RolloutSession`; this
module is the agent-level handle.)
"""

import importlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProviderError,
    SandboxError,
    TaskError,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, UserMessage

__all__ = [
    "MODEL_RETRY_NOTE",
    "RETRYABLE_SANDBOX_KINDS",
    "RETRY_NOTE",
    "Session",
    "resumable_conversation",
    "retryable_error",
]

RETRY_NOTE = (
    "The host reopened this session after an infrastructure failure. The conversation "
    "above is intact. Files are as they were at the last saved point of this "
    "conversation; background processes you started are gone, so restart any you "
    "still need."
)
"""The user message an interaction retry continues with (`AgentConfig.interaction_retries`)."""

MODEL_RETRY_NOTE = (
    "The host retried this turn after a model call failed. Nothing on the box "
    "changed; continue where you left off."
)
"""The user message an interaction retry continues with on the same box, after a
model-side failure that left part of the turn committed."""

RETRYABLE_SANDBOX_KINDS = frozenset({"box_lost", "io"})
"""`SandboxError.kind`s an interaction retry recovers from on a fresh box."""


def retryable_error(error: BaseException) -> bool:
    """Whether an interaction retry recovers from `error`: an infrastructure failure a
    fresh box (or a fresh model call) can get past — a lost box or failed transfer, a
    provider, harness or interception failure. Never a task's own failure."""
    if isinstance(error, TaskError):
        return False
    if isinstance(error, SandboxError):
        return getattr(error, "kind", None) in RETRYABLE_SANDBOX_KINDS
    return isinstance(error, (ProviderError, HarnessError, InterceptionError))


def resumable_conversation(trace: Trace) -> Messages:
    """`trace`'s current conversation (its last branch) as a session holds it: system
    messages dropped (the task re-emits its own system prompt) and a trailing assistant
    message with tool calls but no results trimmed (a turn cut off mid-command leaves
    one, and providers reject it)."""
    messages = [
        message
        for message in (trace.branches[-1].messages if trace.branches else [])
        if message.role != "system"
    ]
    while messages and messages[-1].role == "assistant" and messages[-1].tool_calls:
        messages.pop()
    return messages


def prompt_messages(prompt: Any) -> Messages:
    """A task prompt as conversation messages, system messages dropped."""
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [UserMessage(content=prompt)]
    return [message for message in prompt if message.role != "system"]


def task_type_name(task: Task) -> str:
    return f"{type(task).__module__}:{type(task).__qualname__}"


class Session(BaseModel):
    """One agent's exchange, durable: enough to continue it on a box restored from
    `checkpoint`. JSON-serializable (`model_dump_json` / `model_validate_json`)."""

    model_config = ConfigDict(frozen=True)

    v: Literal[1] = 1
    """Format version."""
    checkpoint: str | None
    """The box's filesystem checkpoint (`Runtime.checkpoint`); None for a borrowed box."""
    messages: Messages
    """The conversation, system messages dropped, trailing tool calls without results
    trimmed. The box matches it: files written by its tool calls are in the checkpoint."""
    task_type: str
    """`module:QualName` of the task class, to rebuild the task from."""
    task_data: dict
    """`task.data` dumped (JSON mode), without its prompt."""
    task_config: dict
    """`task.config` dumped (JSON mode)."""
    calls: int
    """Model calls the agent has made in this exchange so far, across reopenings."""

    @classmethod
    def of(
        cls,
        task: Task,
        *,
        checkpoint: str | None,
        messages: Messages,
        calls: int,
    ) -> "Session":
        """The session of `task`'s exchange at `messages`, on `checkpoint`."""
        return cls(
            checkpoint=checkpoint,
            messages=messages,
            task_type=task_type_name(task),
            task_data=task.data.model_dump(mode="json", exclude={"prompt"}),
            task_config=task.config.model_dump(mode="json"),
            calls=calls,
        )

    def rebuild_task(self) -> Task:
        """The session's task, rebuilt from its recorded class, data and config (with no
        prompt: a reopening sets the conversation as the prompt)."""
        module_name, _, qualname = self.task_type.partition(":")
        if "<locals>" in qualname or not module_name or not qualname:
            raise ValueError(
                f"session task type {self.task_type!r} cannot be imported; pass the "
                "task explicitly (task=...)"
            )
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        if not (isinstance(target, type) and issubclass(target, Task)):
            raise TypeError(f"session task type {self.task_type!r} is not a Task class")
        data = target.data_type().model_validate({**self.task_data, "prompt": None})
        config = target.config_type().model_validate(self.task_config)
        return target(data, config)
