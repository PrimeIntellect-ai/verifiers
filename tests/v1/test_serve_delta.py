"""The env-serve delta stream: a trace's changes leave once each and reassemble into
the episode the worker finished with."""

import asyncio
import copy

import numpy as np
import pytest

import verifiers.v1 as vf
from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
from verifiers.v1.graph import MessageNode
from verifiers.v1.semantic import ParentLink
from verifiers.v1.serve.delta import (
    DeltaStreamer,
    EpisodeAssembly,
    TraceSummary,
    dump,
    pack,
    unpack,
)
from verifiers.v1.trace import ModelCall, TimeSpan, TraceTask
from verifiers.v1.types import AssistantMessage, Usage, UserMessage


class MyTask(vf.TaskData):
    prompt: str
    answer: str


async def settle() -> None:
    """Let a notified change reach the wire: the flush is a callback that starts a task."""
    for _ in range(3):
        await asyncio.sleep(0)


def make_trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="a")),
    )


def add_turn(trace: vf.Trace, reply: str) -> None:
    parent = len(trace.nodes) - 1 if trace.nodes else None
    if parent is None:
        trace.nodes.append(MessageNode(parent=None, message=UserMessage(content="q")))
        parent = 0
    trace.nodes.append(
        MessageNode(
            parent=parent,
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


def test_delta_fields_cover_every_serialized_trace_field():
    """A new Trace field must be routed through the stream, or it silently vanishes."""
    from verifiers.v1.serve.delta import HEADER_FIELDS, LIST_FIELDS, SCALAR_FIELDS

    streamed = set(HEADER_FIELDS) | set(LIST_FIELDS) | set(SCALAR_FIELDS)
    serialized = {
        name for name, info in vf.Trace.model_fields.items() if not info.exclude
    } | set(vf.Trace.model_computed_fields)
    assert streamed == serialized


@pytest.mark.asyncio
async def test_failed_send_is_diffed_again():
    """A delta the wire refused is not lost: its cursor stays and the next flush
    carries the same content, so the client still assembles the whole trace."""
    traces: list[vf.Trace] = []
    frames: list[bytes] = []
    fail = {"on": False}

    async def send(data: dict) -> None:
        if fail["on"]:
            raise OSError("host unreachable")
        frames.append(pack(data))

    async with DeltaStreamer(lambda: traces, send) as streamer:
        trace = make_trace()
        traces.append(trace)
        streamer.watch(trace)
        await settle()
        fail["on"] = True
        add_turn(trace, "a1")
        await settle()
        fail["on"] = False
        add_turn(trace, "a2")
        await settle()
        traces = [trace]
    assembly = EpisodeAssembly()
    for frame in frames:
        assembly.apply(unpack(frame))
    assert len(assembly.traces[trace.id]["nodes"]) == 3
    assert len(assembly.traces[trace.id]["calls"]) == 2


@pytest.mark.asyncio
async def test_pending_preview_streams_and_clears_on_commit():
    traces: list[vf.Trace] = []
    frames: list[bytes] = []

    async def send(data: dict) -> None:
        frames.append(pack(data))

    async with DeltaStreamer(lambda: traces, send) as streamer:
        trace = make_trace()
        traces.append(trace)
        streamer.watch(trace)
        add_turn(trace, "a1")
        await settle()
        # the harness sends its next request: the tool result is previewed at once
        request, subagent = ["request"], ["sub-agent"]  # unhashable, like a PendingTurn
        trace.preview(request, [UserMessage(content="tool says 42")])
        await settle()
        preview = unpack(frames[-1])
        assert preview["pending"][0]["content"] == "tool says 42"
        assert "nodes" not in preview
        # a concurrent request (a sub-agent) previews under its own key; its failure
        # takes only its own messages back
        trace.preview(subagent, [UserMessage(content="sub-agent asks")])
        await settle()
        assert [m["content"] for m in unpack(frames[-1])["pending"]] == [
            "tool says 42",
            "sub-agent asks",
        ]
        trace.clear_preview(subagent)
        trace.notify()
        await settle()
        assert [m["content"] for m in unpack(frames[-1])["pending"]] == ["tool says 42"]
        # the model answers: the turn commits and the preview goes with it
        trace.nodes.append(
            MessageNode(parent=1, message=UserMessage(content="tool says 42"))
        )
        trace.clear_preview(request)
        add_turn(trace, "a2")
        await settle()
        committed = unpack(frames[-1])
        assert len(committed["nodes"]) == 2 and "pending" not in committed
        trace.stop("agent_completed")
        trace.ok = True
        episode = Episode(
            env=EnvInfo(id="my-env"), task=trace.task, ok=True, traces=[trace]
        )
        traces = list(episode.traces)
    assembly = EpisodeAssembly()
    for frame in frames:
        assembly.apply(unpack(frame))
        if unpack(frame).get("pending"):
            assert assembly.traces[trace.id]["pending"][0]["content"] == "tool says 42"
    assert assembly.traces[trace.id]["pending"] == []
    summaries = [
        TraceSummary(id=trace.id, nodes=len(trace.nodes), calls=len(trace.calls))
    ]
    record = assembly.finish(dump(episode, exclude={"traces"}), summaries)
    assert "pending" not in record["traces"][0]
    assert WireEpisode.model_validate(unpack(pack(record))).traces[0].id == trace.id


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_dtype", [np.uint8, np.uint16])
async def test_deltas_stream_once_and_reassemble_the_episode(routing_dtype):
    traces: list[vf.Trace] = []
    frames: list[bytes] = []

    async def send(data: dict) -> None:
        frames.append(pack(data))

    async with DeltaStreamer(lambda: traces, send) as streamer:
        trace = make_trace()
        traces.append(trace)
        streamer.watch(trace)
        trace.timing.boot.start = 1.0
        trace.notify()
        await settle()
        assert len(frames) == 1, "the mint and the boot span coalesce into one delta"
        add_turn(trace, "a1")
        trace.nodes[1].routed_experts = np.arange(6, dtype=np.uint8).reshape(3, 2, 1)
        await settle()
        add_turn(trace, "a2")
        # The next prefill corrects the previous turn's last routing row, potentially
        # widening its dtype. The previous node has already crossed the wire.
        trace.nodes[1].routed_experts = np.array(
            [[[0], [1]], [[2], [3]], [[100], [np.iinfo(routing_dtype).max]]],
            dtype=routing_dtype,
        )
        trace.nodes[0].semantic_parents.append(ParentLink(node=1, type="reply"))
        await settle()
        trace.record_reward("match", 1.0)
        trace.stop("agent_completed")
        trace.ok = True
        episode = Episode(
            env=EnvInfo(id="my-env"), task=trace.task, ok=True, traces=[trace]
        )
        traces = list(episode.traces)
    # The exit flushed the tail (reward, stop, ok) without re-sending any node.
    deltas = [unpack(frame) for frame in frames]
    sent_nodes = sum(len(delta.get("nodes", [])) for delta in deltas)
    sent_calls = sum(len(delta.get("calls", [])) for delta in deltas)
    assert (sent_nodes, sent_calls) == (3, 2)
    assert sum("open" in delta for delta in deltas) == 1
    assert any("links" in delta for delta in deltas)
    assert deltas[-1]["set"]["stop_condition"] == "agent_completed"

    handed = copy.deepcopy(deltas)
    assembly = EpisodeAssembly()
    for delta in deltas:
        assembly.apply(delta)
    assert deltas == handed, "assembling never mutates a delta the caller holds"
    head = dump(episode, exclude={"traces"})
    summaries = [
        TraceSummary(id=trace.id, nodes=len(trace.nodes), calls=len(trace.calls))
    ]
    rebuilt = WireEpisode.model_validate(unpack(pack(assembly.finish(head, summaries))))
    assert rebuilt.ok and rebuilt.traces[0].id == trace.id
    assert rebuilt.traces[0].messages == trace.messages
    np.testing.assert_array_equal(
        rebuilt.traces[0].nodes[1].routed_experts, trace.nodes[1].routed_experts
    )
    assert rebuilt.traces[0].nodes[1].routed_experts.dtype == routing_dtype
    assert (
        rebuilt.traces[0].nodes[0].semantic_parents == trace.nodes[0].semantic_parents
    )
    assert rebuilt.traces[0].rewards["match"].value == 1.0
    assert rebuilt.traces[0].stop_condition == "agent_completed"
    assert len(rebuilt.traces[0].calls) == 2


@pytest.mark.asyncio
async def test_discarded_attempt_drops_its_trace():
    traces: list[vf.Trace] = []
    frames: list[bytes] = []

    async def send(data: dict) -> None:
        frames.append(pack(data))

    async with DeltaStreamer(lambda: traces, send) as streamer:
        first = make_trace()
        traces.append(first)
        streamer.watch(first)
        add_turn(first, "a1")
        await settle()
        traces = []  # the attempt is retried
        second = make_trace()
        traces.append(second)
        streamer.watch(second)
        await settle()
    deltas = [unpack(frame) for frame in frames]
    assert {"trace": first.id, "discard": True} in deltas
    assembly = EpisodeAssembly()
    for delta in deltas:
        assembly.apply(delta)
    assert list(assembly.traces) == [second.id]


def test_finish_refuses_a_gap():
    assembly = EpisodeAssembly()
    assembly.apply({"trace": "t", "open": {"id": "t"}, "nodes": [{}]})
    with pytest.raises(RuntimeError, match="assembled 1 nodes"):
        assembly.finish({}, [TraceSummary(id="t", nodes=2, calls=0)])
