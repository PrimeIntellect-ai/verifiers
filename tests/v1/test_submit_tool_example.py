"""`examples/submit_tool.py`, the documented submit-tool pattern: the payload schema
reaches the agent and rejects bad calls, a submission stops the run, `finalize` tells
"never submitted" from "ran out of time", and reminders are bounded."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import verifiers.v1 as vf

EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.fixture(scope="module")
def example():
    """`submit_tool` with its toolset module importable, as `PYTHONPATH=examples` runs it;
    the toolset's names are re-exported on the module for the tests."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        module = importlib.import_module("submit_tool")
        tools = importlib.import_module("submit_tools")
        for name in ("Verdict", "SubmitState", "SubmitTools"):
            setattr(module, name, getattr(tools, name))
        yield module
    finally:
        sys.path.remove(str(EXAMPLES))
        sys.modules.pop("submit_tool", None)
        sys.modules.pop("submit_tools", None)


def trace_for(example, submission=None) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="ReviewTask", data=vf.TaskData(prompt="q")),
        state=example.SubmitState(submission=submission),
    )


async def test_the_payload_schema_reaches_the_agent_and_rejects_bad_calls(example):
    mcp = MCPServer("submit")
    example.SubmitTools(vf.ToolsetConfig()).register(mcp)
    (tool,) = await mcp.list_tools()
    assert tool.name == "submit"
    verdict = tool.input_schema["$defs"]["Verdict"]
    assert verdict["properties"]["decision"]["enum"] == ["accept", "reject"]
    with pytest.raises(ToolError, match="literal_error"):
        await mcp.call_tool("submit", {"verdict": {"decision": "maybe", "reason": "x"}})


async def test_submit_records_the_verdict_and_the_stop_fires(example):
    tools = example.SubmitTools(vf.ToolsetConfig())
    verdict = example.Verdict(decision="accept", reason="it is prime")
    assert await tools.submit(verdict) == "Submitted."
    assert tools.state.submission == verdict
    task = example.ReviewTask(vf.TaskData(prompt="q"))
    assert task.submitted(trace_for(example, verdict)) is True
    assert task.submitted(trace_for(example)) is False


async def test_finalize_keeps_the_submission(example):
    task = example.ReviewTask(vf.TaskData(prompt="q"))
    trace = trace_for(example, example.Verdict(decision="reject", reason="composite"))
    await task.finalize(trace, None)
    assert trace.info["submission"] == {"decision": "reject", "reason": "composite"}


async def test_finalize_tells_out_of_time_from_undecided(example):
    task = example.ReviewTask(vf.TaskData(prompt="q"))
    with pytest.raises(example.Undecided) as undecided:
        await task.finalize(trace_for(example), None)
    assert not isinstance(undecided.value, example.OutOfTime)
    timed_out = trace_for(example)
    timed_out.record_timeout("agent")
    assert timed_out.is_timeout and timed_out.stop_condition == "agent_timeout"
    with pytest.raises(example.OutOfTime):
        await task.finalize(timed_out, None)


class ScriptedInteraction:
    """`agent.interaction(task)`'s surface: each turn plays the next scripted step,
    which may submit or end the run before returning its segment."""

    def __init__(self, example, steps):
        self.trace = trace_for(example)
        self.steps = list(steps)
        self.sent: list[str | None] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def turn(self, message=None):
        self.sent.append(message)
        return self.steps.pop(0)(self.trace)


def scripted_agent(interaction):
    return SimpleNamespace(interaction=lambda task: interaction)


def idle(trace):
    return vf.Segment(messages=[])


def submits(example):
    def step(trace):
        trace.state.submission = example.Verdict(decision="accept", reason="x")
        return vf.Segment(messages=[])

    return step


async def test_an_agent_that_stops_without_submitting_is_reminded_a_bounded_number_of_times(
    example,
):
    ix = ScriptedInteraction(example, [idle, idle, idle])
    trace = await example.run_until_submitted(scripted_agent(ix), None, reminders=2)
    assert ix.sent == [None, example.REMINDER, example.REMINDER]
    assert trace.state.submission is None


async def test_a_reminded_agent_that_submits_gets_no_further_reminder(example):
    ix = ScriptedInteraction(example, [idle, submits(example), idle])
    trace = await example.run_until_submitted(scripted_agent(ix), None, reminders=2)
    assert ix.sent == [None, example.REMINDER]
    assert trace.state.submission is not None


async def test_a_run_that_ended_gets_no_reminder(example):
    def times_out(trace):
        trace.record_timeout("agent")
        return vf.Segment(messages=[], terminated=True)

    ix = ScriptedInteraction(example, [times_out])
    await example.run_until_submitted(scripted_agent(ix), None, reminders=2)
    assert ix.sent == [None]
