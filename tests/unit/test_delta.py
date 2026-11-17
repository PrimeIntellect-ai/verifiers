"""Trace deltas: a trace's changes leave once each and reassemble into the trace that sent them."""

import asyncio
import copy

import numpy as np
import pytest

import verifiers.v1 as vf
from verifiers.v1.delta import TRAINING_ARRAY_FIELDS, DeltaStreamer, EpisodeAssembly
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


def recorder(traces: list[vf.Trace], kind: type[DeltaStreamer] = DeltaStreamer):
    """A streamer over `traces` whose `send` keeps a copy of each delta, as a wire would."""
    sent: list[dict] = []

    async def send(delta: dict) -> None:
        sent.append(copy.deepcopy(delta))

    return kind(lambda: traces, send), sent


def assemble(deltas: list[dict]) -> EpisodeAssembly:
    assembly = EpisodeAssembly()
    for delta in deltas:
        assembly.apply(delta)
    return assembly


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
        trace.preview("request", [UserMessage(content="tool output")])
        await settle()
        pending = assemble(sent).traces[trace.id]["pending"]
        assert [m["content"] for m in pending] == ["tool output"], "the preview streams"
        trace.clear_preview("request")
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
    assert raw.pop("pending") == [], "a committed turn clears the preview"
    rebuilt = vf.Trace[MyTask, vf.State, vf.AgentConfig].model_validate(raw)
    assert rebuilt.nodes[1].routed_experts is None, (
        "training arrays stay with the sender"
    )
    # Every field the run set arrives, including ones no list in delta.py names.
    sent_fields = {"nodes": {"__all__": TRAINING_ARRAY_FIELDS}}
    assert rebuilt.model_dump(exclude=sent_fields) == trace.model_dump(
        exclude=sent_fields
    )
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


@pytest.mark.asyncio
async def test_an_array_in_info_survives_a_grading_copy():
    trace = make_trace(info={"arr": np.zeros(2)})
    streamer, sent = recorder([trace])
    async with streamer:
        streamer.watch(trace)
        await settle()
        # isolated grading: trace.info = result.info
        trace.info = copy.deepcopy(trace.info)
        trace.info["arr"][0] = 7.0
        trace.notify()
    assert assemble(sent).traces[trace.id]["info"]["arr"].tolist() == [7.0, 0.0]


class EveryTrace(DeltaStreamer):
    """Diffs every watched trace at each flush, as the streamer did before it kept marks."""

    def diff(self, everything: bool = False):
        return super().diff(everything=True)


async def three_traces(streamer: DeltaStreamer, traces: list[vf.Trace]) -> None:
    async with streamer:
        for trace in traces:
            streamer.watch(trace)
        await settle()
        for turn in range(4):
            for index, trace in enumerate(traces):
                if (turn + index) % 2:
                    add_turn(trace, f"{index}.{turn}")
            await settle()
        traces[1].nodes[1].semantic_parents.append(
            ParentLink(node=0, type="compaction")
        )
        traces[1].notify()
        traces[2].record_reward("match", 1.0)
        traces[2].notify()
        await settle()
        traces[0].info["late"] = True  # no notify: rides the exit's flush


def untimed(value):
    """`value` without wall-clock stamps, which differ between two runs of one scenario."""
    if isinstance(value, dict):
        return {
            k: untimed(v) for k, v in value.items() if k not in ("timestamp", "start")
        }
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


@pytest.mark.asyncio
async def test_a_flush_sends_what_diffing_every_trace_sends():
    """Diffing only the notified traces sends the same deltas, in the same order, as diffing
    every trace when every change notifies; the exit's flush catches an unannounced change."""
    runs = []
    for kind in (DeltaStreamer, EveryTrace):
        traces = [make_trace(id=f"t{index}") for index in range(3)]
        streamer, sent = recorder(traces, kind)
        await three_traces(streamer, traces)
        runs.append(sent)
    assert untimed(runs[0]) == untimed(runs[1])
    assert {"late": True}.items() <= runs[0][-1]["set"]["info"].items()


@pytest.mark.asyncio
async def test_a_flush_diffs_only_the_traces_that_notified():
    """A notify costs its own trace's diff: other watched traces are not diffed, and one
    whose delta failed to send is diffed again at the next flush, whoever triggers it."""
    traces = [make_trace(), make_trace()]
    first, second = traces
    sent: list[str] = []
    fail = False

    async def send(delta: dict) -> None:
        if fail:
            raise OSError("host unreachable")
        sent.append(delta["trace"])

    async with DeltaStreamer(lambda: traces, send) as streamer:
        for trace in traces:
            streamer.watch(trace)
        await settle()
        sent.clear()
        diffed: list[str] = []
        dump_trace = streamer.diff

        def diff(everything: bool = False):
            deltas = dump_trace(everything)
            diffed.extend(trace_id for trace_id, _, _ in deltas)
            return deltas

        streamer.diff = diff
        second.info["note"] = "no notify yet"
        add_turn(first, "a1")
        await settle()
        assert sent == diffed == [first.id]
        fail = True
        add_turn(first, "a2")
        await settle()
        fail = False
        second.notify()
        await settle()
        assert sent == [first.id, first.id, second.id]
        sent.clear()
        second.record_reward("match", 1.0)
    assert sent == [second.id], "the exit flushes what no notify announced"
