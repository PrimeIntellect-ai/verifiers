"""Small public helpers for constructing traces in tests.

The production graph records whether an assistant message came from a model call with
``MessageNode.sampled``.  This helper supplies that provenance for synthetic traces so
tests can focus on scoring or trace consumers instead of repeating envelope and parent
construction.
"""

from collections.abc import Collection, Iterable
from typing import Any

from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.state import State
from verifiers.v1.task import TaskData
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import AssistantMessage, Message


def make_trace(
    messages: str | Iterable[Message] = (),
    *,
    task: TraceTask[Any] | None = None,
    sampled: Collection[int] | None = None,
) -> Trace[Any, State, AgentConfig]:
    """Build a linear synthetic trace for tests.

    A string is shorthand for one sampled assistant reply, including an empty reply.
    Otherwise, ``messages`` are stored in order and linked to their immediate
    predecessor.  By default every assistant message is treated as a model response;
    pass the message indexes in ``sampled`` to preserve prompt-supplied assistant
    context as non-sampled.  Only assistant messages may be sampled, matching the
    production graph invariant.  The default is an empty trace.

    A caller can represent tool use with normal public message types, for example an
    :class:`~verifiers.v1.types.AssistantMessage` containing ``tool_calls`` followed by
    a :class:`~verifiers.v1.types.ToolMessage`.  The helper deliberately does not add
    model calls or token attribution: those details belong to tests that exercise them.
    """
    if isinstance(messages, str):
        message_list = [AssistantMessage(content=messages)]
    else:
        message_list = list(messages)

    sampled_indexes = (
        {
            index
            for index, message in enumerate(message_list)
            if isinstance(message, AssistantMessage)
        }
        if sampled is None
        else set(sampled)
    )
    invalid_indexes = sampled_indexes.difference(range(len(message_list)))
    if invalid_indexes:
        raise ValueError(f"sampled indexes are out of range: {sorted(invalid_indexes)}")
    non_assistant_indexes = {
        index
        for index in sampled_indexes
        if not isinstance(message_list[index], AssistantMessage)
    }
    if non_assistant_indexes:
        raise ValueError(
            f"only assistant messages may be sampled: {sorted(non_assistant_indexes)}"
        )

    if task is None:
        task = TraceTask(type="Task", data=TaskData(idx=0))

    return Trace(
        agent=AgentInfo(config=AgentConfig()),
        task=task,
        nodes=[
            MessageNode(
                parent=index - 1 if index else None,
                message=message,
                sampled=index in sampled_indexes,
            )
            for index, message in enumerate(message_list)
        ],
    )
