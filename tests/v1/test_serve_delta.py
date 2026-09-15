"""The env-serve delta stream: a trace's changes leave once each and reassemble into
the episode the worker finished with."""

import asyncio

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


class Slot:
    def __init__(self) -> None:
        self.traces: list[vf.Trace] = []


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


@pytest.mark.asyncio
async def test_pending_preview_streams_and_clears_on_commit():
    slot = Slot()
    frames: list[bytes] = []

    async def send(data: bytes) -> None:
        frames.append(data)

    async with DeltaStreamer(slot, send) as streamer:
        trace = make_trace()
        slot.traces.append(trace)
        streamer.watch(trace)
        add_turn(trace, "a1")
        await settle()
        # the harness sends its next request: the tool result is previewed at once
        trace.preview([UserMessage(content="tool says 42")])
        await settle()
        preview = unpack(frames[-1])
        assert preview["pending"][0]["content"] == "tool says 42"
        assert "nodes" not in preview
        # the model answers: the turn commits and the preview goes with it
        trace.nodes.append(
            MessageNode(parent=1, message=UserMessage(content="tool says 42"))
        )
        trace._pending = []
        add_turn(trace, "a2")
        await settle()
        committed = unpack(frames[-1])
        assert len(committed["nodes"]) == 2 and "pending" not in committed
        trace.stop("agent_completed")
        trace.ok = True
        episode = Episode(
            env=EnvInfo(id="my-env"), task=trace.task, ok=True, traces=[trace]
        )
        slot.traces = list(episode.traces)
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
async def test_deltas_stream_once_and_reassemble_the_episode():
    slot = Slot()
    frames: list[bytes] = []

    async def send(data: bytes) -> None:
        frames.append(data)

    async with DeltaStreamer(slot, send) as streamer:
        trace = make_trace()
        slot.traces.append(trace)
        streamer.watch(trace)
        trace.timing.boot.start = 1.0
        trace.notify()
        await settle()
        assert len(frames) == 1, "the mint and the boot span coalesce into one delta"
        add_turn(trace, "a1")
        await settle()
        add_turn(trace, "a2")
        trace.nodes[0].semantic_parents.append(ParentLink(node=1, type="reply"))
        await settle()
        trace.record_reward("match", 1.0)
        trace.stop("agent_completed")
        trace.ok = True
        episode = Episode(
            env=EnvInfo(id="my-env"), task=trace.task, ok=True, traces=[trace]
        )
        slot.traces = list(episode.traces)
    # The exit flushed the tail (reward, stop, ok) without re-sending any node.
    deltas = [unpack(frame) for frame in frames]
    sent_nodes = sum(len(delta.get("nodes", [])) for delta in deltas)
    sent_calls = sum(len(delta.get("calls", [])) for delta in deltas)
    assert (sent_nodes, sent_calls) == (3, 2)
    assert sum("open" in delta for delta in deltas) == 1
    assert any("links" in delta for delta in deltas)
    assert deltas[-1]["set"]["stop_condition"] == "agent_completed"

    assembly = EpisodeAssembly()
    for delta in deltas:
        assembly.apply(delta)
    head = dump(episode, exclude={"traces"})
    summaries = [
        TraceSummary(id=trace.id, nodes=len(trace.nodes), calls=len(trace.calls))
    ]
    rebuilt = WireEpisode.model_validate(unpack(pack(assembly.finish(head, summaries))))
    assert rebuilt.ok and rebuilt.traces[0].id == trace.id
    assert rebuilt.traces[0].messages == trace.messages
    assert (
        rebuilt.traces[0].nodes[0].semantic_parents == trace.nodes[0].semantic_parents
    )
    assert rebuilt.traces[0].rewards["match"].value == 1.0
    assert rebuilt.traces[0].stop_condition == "agent_completed"
    assert len(rebuilt.traces[0].calls) == 2


@pytest.mark.asyncio
async def test_discarded_attempt_drops_its_trace():
    slot = Slot()
    frames: list[bytes] = []

    async def send(data: bytes) -> None:
        frames.append(data)

    async with DeltaStreamer(slot, send) as streamer:
        first = make_trace()
        slot.traces.append(first)
        streamer.watch(first)
        add_turn(first, "a1")
        await settle()
        slot.traces = []  # the attempt is retried
        second = make_trace()
        slot.traces.append(second)
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
