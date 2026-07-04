"""Seed construction for the replay taskset: resume points recovered from trace records."""

from random import Random

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.tasksets.replay import compaction_seeds, recheck_seed, tool_call_seed

SYSTEM = vf.SystemMessage(content="be helpful")
TASK_USER = vf.UserMessage(content="solve the task")
RESTART_USER = vf.UserMessage(content="Another model produced a summary: progress so far ...")


def _reply(message: vf.AssistantMessage) -> vf.Response:
    return vf.Response(id="", created=0, model="test", message=message, finish_reason="stop")


def _assistant(content: str, *call_ids: str) -> vf.AssistantMessage:
    calls = [vf.ToolCall(id=call_id, name="run", arguments="{}") for call_id in call_ids]
    return vf.AssistantMessage(content=content or None, tool_calls=calls or None)


def _tool(call_id: str) -> vf.ToolMessage:
    return vf.ToolMessage(tool_call_id=call_id, content=f"result of {call_id}", name="run")


def _compacted_trace() -> vf.Trace:
    """The rlm compaction shape: a tool-using branch that ends in a handoff summary, then a
    second branch re-rooted at the shared system node with the summary as a fresh user turn."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("", "c1")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    summary_request = vf.UserMessage(content="perform a checkpoint compaction")
    a2 = _assistant("handoff summary")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, a1, _tool("c1"), summary_request]).commit(_reply(a2))
    a3 = _assistant("", "c2")
    graph.prepare_turn(trace, [SYSTEM, RESTART_USER]).commit(_reply(a3))
    a4 = _assistant("final answer")
    graph.prepare_turn(trace, [SYSTEM, RESTART_USER, a3, _tool("c2")]).commit(_reply(a4))
    return trace


def test_compaction_seeds_recover_the_restart_prompt():
    seeds = compaction_seeds(_compacted_trace())
    assert [seed.prompt for seed in seeds] == [RESTART_USER.content]
    assert seeds[0].name.endswith(":compaction0")
    assert seeds[0].tokens > 0


def test_compaction_seeds_ignore_linear_traces():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(_assistant("done")))
    assert compaction_seeds(trace) == []


def test_compaction_seeds_handle_any_restart_shape():
    """Detection is structural, so a harness that keeps the task message alongside its own
    handoff turn still qualifies — the seed is then the restart context as Messages."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(_assistant("working...")))
    notes = vf.UserMessage(content="notes from the previous attempt")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, notes]).commit(_reply(_assistant("resumed")))
    seeds = compaction_seeds(trace)
    assert len(seeds) == 1
    assert [m.content for m in seeds[0].prompt] == [TASK_USER.content, notes.content]


def test_compaction_seeds_ignore_retokenization_forks():
    """A fork whose context carries assistant/tool copies is a renderer-level split of the
    same conversation, not a context restart."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(_assistant("draft")))
    rewritten = vf.AssistantMessage(content="draft (rewritten)")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, rewritten, vf.UserMessage(content="go on")]).commit(
        _reply(_assistant("done"))
    )
    assert compaction_seeds(trace) == []


def test_seeds_carry_their_anchor_snapshot():
    trace = _compacted_trace()
    restart = next(i for i, node in enumerate(trace.nodes) if node.message.content == RESTART_USER.content)
    trace.info["snapshots"] = {str(restart): "snap-restart", str(len(trace.nodes) - 1): "snap-final"}
    assert compaction_seeds(trace)[0].snapshot == "snap-restart"
    assert recheck_seed(trace, "check").snapshot == "snap-final"
    # With snapshots recorded, only snapshotted tool nodes are valid resume points — none here.
    assert tool_call_seed(trace, Random(0)) is None


def test_tool_call_seed_resumes_after_a_complete_tool_run():
    seed = tool_call_seed(_compacted_trace(), Random(0))
    assert seed is not None
    messages = seed.prompt
    assert messages[0].role == "user"  # leading system stripped; the harness re-emits it
    assert messages[-1].role == "tool"
    assert messages[-2].role == "assistant"
    assert {call.id for call in messages[-2].tool_calls} == {messages[-1].tool_call_id}


def test_tool_call_seed_is_deterministic():
    trace = _compacted_trace()
    assert tool_call_seed(trace, Random("salt")) == tool_call_seed(trace, Random("salt"))


def test_tool_call_seed_skips_partial_tool_runs():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("", "c1", "c2")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    # Only c1's result ever arrived: resuming here would leave c2 dangling in the context.
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, a1, _tool("c1")]).commit(_reply(_assistant("gave up")))
    assert tool_call_seed(trace, Random(0)) is None


def test_recheck_seed_appends_the_verification_turn():
    seed = recheck_seed(_compacted_trace(), "check your work")
    assert seed is not None
    messages = seed.prompt
    # The final branch (post-compaction), system stripped, plus the new user turn.
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant", "user"]
    assert messages[0].content == RESTART_USER.content
    assert messages[-1].content == "check your work"
    assert messages[-2].content == "final answer"


def test_recheck_seed_strips_truncation_artifacts():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("partial work")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    # The rollout was cut right after issuing a tool call: its results never arrived.
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, a1, vf.UserMessage(content="continue")]).commit(
        _reply(_assistant("", "c9"))
    )
    seed = recheck_seed(trace, "check your work")
    assert seed is not None
    assert [m.role for m in seed.prompt] == ["user", "assistant", "user"]
    assert seed.prompt[-2].content == "partial work"
