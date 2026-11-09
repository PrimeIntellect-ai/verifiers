"""Trace deltas: a trace's changes leave once each and reassemble into the trace that sent them."""

import asyncio
import copy

import numpy as np
import pytest

import verifiers.v1 as vf
from verifiers.v1.delta import (
    HEADER_FIELDS,
    LIST_FIELDS,
    SCALAR_FIELDS,
    DeltaStreamer,
    EpisodeAssembly,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.semantic import ParentLink
from verifiers.v1.trace import ModelCall, TimeSpan, TraceTask
from verifiers.v1.types import AssistantMessage, Usage, UserMessage


class MyTask(vf.TaskData):
    answer: str


async def settle() -> None:
    """Let a notified change reach `send`: the flush is a callback that starts a task."""
    for _ in range(3):
        await asyncio.sleep(0)


def make_trace(**kwargs) -> vf.Trace:
    data = MyTask(idx=0, prompt="q", answer="a")
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=TraceTask(type="MyTask", data=data),
        **kwargs,
    )


def add_turn(trace: vf.Trace, reply: str) -> None:
    if not trace.nodes:
        trace.nodes.append(MessageNode(parent=None, message=UserMessage(content="q")))
    trace.nodes.append(
        MessageNode(
            parent=len(trace.nodes) - 1,
            message=AssistantMessage(content=reply),
            sampled=True,
            token_ids=[1, 2, 3],
            mask=[False, True, True],
            logprobs=[-0.1, -0.2],
        )
    )
    trace.calls.append(
        ModelCall(
            node=len(trace.nodes) - 1,
            usage=Usage(prompt_tokens=3, completion_tokens=2),
            time=TimeSpan(start=1.0, end=2.0),
        )
    )
    trace.notify()


def recorder(traces: list[vf.Trace]):
    """A streamer over `traces` whose `send` keeps a copy of each delta, as a wire would."""
    sent: list[dict] = []

    async def send(delta: dict) -> None:
        sent.append(copy.deepcopy(delta))

    return DeltaStreamer(lambda: traces, send), sent


def assemble(deltas: list[dict]) -> EpisodeAssembly:
    assembly = EpisodeAssembly()
    for delta in deltas:
        assembly.apply(delta)
    return assembly


def test_every_serialized_trace_field_is_streamed():
    """A new Trace field streams without anyone listing it; a renamed header or list field fails here."""
    serialized = {
        name for name, info in vf.Trace.model_fields.items() if not info.exclude
    }
    serialized |= set(vf.Trace.model_computed_fields)
    assert set(HEADER_FIELDS) | set(LIST_FIELDS) | set(SCALAR_FIELDS) == serialized
    assert {"attempt", "reward", "last_reply"} <= set(SCALAR_FIELDS)


@pytest.mark.asyncio
async def test_deltas_stream_once_and_reassemble_the_trace():
    traces: list[vf.Trace] = []
    streamer, sent = recorder(traces)
    async with streamer:
        trace = make_trace()
        traces.append(trace)
        streamer.watch(trace)
        trace.timing.boot.start = 1.0
        trace.notify()
        await settle()
        assert len(sent) == 1, "the first sight and the boot span share one delta"
        add_turn(trace, "a1")
        trace.nodes[1].routed_experts = np.arange(6, dtype=np.uint8).reshape(3, 2, 1)
        await settle()
        add_turn(trace, "a2")
        trace.nodes[0].semantic_parents.append(ParentLink(node=1, type="reply"))
        await settle()
        trace.extra_usage.append(Usage(prompt_tokens=1, completion_tokens=1))
        trace.notify()
        await asyncio.sleep(0)  # a flush is in flight
        copy.deepcopy(trace)  # isolated grading copies the live trace mid-stream
        await settle()
        trace.extra_usage = [Usage(prompt_tokens=9, completion_tokens=9)]  # a rescore
        trace.record_reward("match", 1.0)
        trace.stop("agent_completed")
    # The exit flushed the tail (reward, stop) without re-sending any node.
    assert sum(len(delta.get("nodes", [])) for delta in sent) == 3
    assert sum("open" in delta for delta in sent) == 1
    assert sent[-1]["set"]["stop_condition"] == "agent_completed"

    handed = copy.deepcopy(sent)
    raw = assemble(sent).traces[trace.id]
    assert sent == handed, "assembling never mutates a delta the caller holds"
    raw.pop("pending")
    rebuilt = vf.Trace.model_validate(raw)
    assert rebuilt.messages == trace.messages
    assert rebuilt.nodes[1].routed_experts is None, (
        "training arrays stay with the sender"
    )
    assert rebuilt.nodes[0].semantic_parents == trace.nodes[0].semantic_parents
    assert rebuilt.rewards["match"].value == 1.0
    assert rebuilt.extra_usage == trace.extra_usage
    assert raw["reward"] == trace.reward and raw["num_turns"] == trace.num_turns


@pytest.mark.asyncio
async def test_a_retry_under_the_same_id_restarts_the_trace():
    """Retries keep the trace id but start a fresh trace: its nodes replace the abandoned attempt's."""
    traces: list[vf.Trace] = []
    streamer, sent = recorder(traces)
    async with streamer:
        first = make_trace()
        traces.append(first)
        streamer.watch(first)
        add_turn(first, "a1")
        add_turn(first, "a2")
        await settle()
        second = make_trace(id=first.id, attempt=2)
        traces[:] = [second]
        streamer.watch(second)
        add_turn(second, "b1")
        second.errors.append(vf.Error(type="E", message="b"))
        second.notify()
        await settle()
        # the agent prepends the abandoned attempts' errors once the retry finishes
        second.errors = [vf.Error(type="E", message="a")] + second.errors
    (raw,) = assemble(sent).traces.values()
    assert [error["message"] for error in raw["errors"]] == ["a", "b"]
    assert [node["message"]["content"] for node in raw["nodes"]] == ["q", "b1"]
    assert raw["attempt"] == 2
