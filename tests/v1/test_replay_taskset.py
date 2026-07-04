"""The replay taskset: seed construction from trace records, config resolution, delegation."""

import json
from random import Random

import pytest
from pydantic import ValidationError

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.loaders import load_taskset, taskset_config_type
from verifiers.v1.tasksets.replay import compaction_seeds, recheck_seed, tool_call_seed

SYSTEM = vf.SystemMessage(content="be helpful")
TASK_USER = vf.UserMessage(content="solve the task")
RESTART_USER = vf.UserMessage(
    content="Another model produced a summary: progress so far ..."
)


def _reply(message: vf.AssistantMessage) -> vf.Response:
    return vf.Response(
        id="", created=0, model="test", message=message, finish_reason="stop"
    )


def _assistant(content: str, *call_ids: str) -> vf.AssistantMessage:
    calls = [
        vf.ToolCall(id=call_id, name="run", arguments="{}") for call_id in call_ids
    ]
    return vf.AssistantMessage(content=content or None, tool_calls=calls or None)


def _tool(call_id: str) -> vf.ToolMessage:
    return vf.ToolMessage(
        tool_call_id=call_id, content=f"result of {call_id}", name="run"
    )


def _compacted_trace() -> vf.Trace:
    """The rlm compaction shape: a tool-using branch that ends in a handoff summary, then a
    second branch re-rooted at the shared system node with the summary as a fresh user turn."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("", "c1")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    summary_request = vf.UserMessage(content="perform a checkpoint compaction")
    a2 = _assistant("handoff summary")
    graph.prepare_turn(
        trace, [SYSTEM, TASK_USER, a1, _tool("c1"), summary_request]
    ).commit(_reply(a2))
    a3 = _assistant("", "c2")
    graph.prepare_turn(trace, [SYSTEM, RESTART_USER]).commit(_reply(a3))
    a4 = _assistant("final answer")
    graph.prepare_turn(trace, [SYSTEM, RESTART_USER, a3, _tool("c2")]).commit(
        _reply(a4)
    )
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
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(
        _reply(_assistant("working..."))
    )
    notes = vf.UserMessage(content="notes from the previous attempt")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, notes]).commit(
        _reply(_assistant("resumed"))
    )
    seeds = compaction_seeds(trace)
    assert len(seeds) == 1
    assert [m.content for m in seeds[0].prompt] == [TASK_USER.content, notes.content]


def test_compaction_seeds_ignore_retokenization_forks():
    """A fork whose context carries assistant/tool copies is a renderer-level split of the
    same conversation, not a context restart."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(_assistant("draft")))
    rewritten = vf.AssistantMessage(content="draft (rewritten)")
    graph.prepare_turn(
        trace, [SYSTEM, TASK_USER, rewritten, vf.UserMessage(content="go on")]
    ).commit(_reply(_assistant("done")))
    assert compaction_seeds(trace) == []


def test_seeds_carry_their_anchor_snapshot():
    trace = _compacted_trace()
    restart = next(
        i
        for i, node in enumerate(trace.nodes)
        if node.message.content == RESTART_USER.content
    )
    trace.info["snapshots"] = {
        str(restart): "snap-restart",
        str(len(trace.nodes) - 1): "snap-final",
    }
    assert compaction_seeds(trace)[0].snapshot == "snap-restart"
    assert recheck_seed(trace, "check").snapshot == "snap-final"
    # With snapshots recorded, only snapshotted tool nodes are valid resume points — none here.
    assert tool_call_seed(trace, Random(0)) is None


def test_tool_call_seed_resumes_after_a_complete_tool_run():
    seed = tool_call_seed(_compacted_trace(), Random(0))
    assert seed is not None
    messages = seed.prompt
    assert (
        messages[0].role == "user"
    )  # leading system stripped; the harness re-emits it
    assert messages[-1].role == "tool"
    assert messages[-2].role == "assistant"
    assert {call.id for call in messages[-2].tool_calls} == {messages[-1].tool_call_id}


def test_tool_call_seed_is_deterministic():
    trace = _compacted_trace()
    assert tool_call_seed(trace, Random("salt")) == tool_call_seed(
        trace, Random("salt")
    )


def test_tool_call_seed_skips_partial_tool_runs():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("", "c1", "c2")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    # Only c1's result ever arrived: resuming here would leave c2 dangling in the context.
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, a1, _tool("c1")]).commit(
        _reply(_assistant("gave up"))
    )
    assert tool_call_seed(trace, Random(0)) is None


def test_recheck_seed_appends_the_verification_turn():
    seed = recheck_seed(_compacted_trace(), "check your work")
    assert seed is not None
    messages = seed.prompt
    # The final branch (post-compaction), system stripped, plus the new user turn.
    assert [m.role for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[0].content == RESTART_USER.content
    assert messages[-1].content == "check your work"
    assert messages[-2].content == "final answer"


def test_recheck_seed_strips_truncation_artifacts():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("partial work")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    # The rollout was cut right after issuing a tool call: its results never arrived.
    graph.prepare_turn(
        trace, [SYSTEM, TASK_USER, a1, vf.UserMessage(content="continue")]
    ).commit(_reply(_assistant("", "c9")))
    seed = recheck_seed(trace, "check your work")
    assert seed is not None
    assert [m.role for m in seed.prompt] == ["user", "assistant", "user"]
    assert seed.prompt[-2].content == "partial work"


def test_recheck_seed_drops_attempts_with_mid_branch_dangling_tool_runs():
    """A partially answered tool run makes everything after it malformed context — with no
    model-produced turn left before it, there is nothing to check."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("", "c1", "c2")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    # Only c1's result was ever recorded, and the model kept going anyway.
    graph.prepare_turn(trace, [SYSTEM, TASK_USER, a1, _tool("c1")]).commit(
        _reply(_assistant("gave up"))
    )
    assert recheck_seed(trace, "check") is None


def test_compaction_seeds_skip_the_tasks_own_launch_context():
    """Branch order follows leaf order, so an auxiliary same-system conversation can sort
    before the main one — the main branch's fork on the task's own launch is not a restart."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="solve the task"))
    a1 = _assistant("working")
    graph.prepare_turn(trace, [SYSTEM, TASK_USER]).commit(_reply(a1))
    graph.prepare_turn(trace, [SYSTEM, vf.UserMessage(content="aux question")]).commit(
        _reply(_assistant("aux answer"))
    )
    graph.prepare_turn(
        trace, [SYSTEM, TASK_USER, a1, vf.UserMessage(content="go on")]
    ).commit(_reply(_assistant("done")))
    assert compaction_seeds(trace) == []


def test_snapshotted_records_offer_only_snapshotted_anchors():
    trace = _compacted_trace()
    trace.info["snapshots"] = {"0": "ref-at-the-system-node"}  # no anchor has a ref
    assert compaction_seeds(trace) == []
    assert recheck_seed(trace, "check") is None
    assert tool_call_seed(trace, Random(0)) is None


def test_replay_config_narrows_source_and_survives_dump_round_trip():
    config_cls = taskset_config_type("replay")
    config = config_cls.model_validate(
        {
            "id": "replay",
            "records": "does-not-exist-*.jsonl",
            "mode": "recheck",
            "source": {"id": "echo-v1", "phrases": ["ping"]},
        }
    )
    assert type(config.source).__name__ == "EchoConfig"
    # The env-server pool dumps and re-validates the resolved config in every worker.
    again = config_cls.model_validate(config.model_dump(mode="json"))
    assert again == config
    with pytest.raises(ValidationError, match="anchor"):
        config_cls.model_validate(
            {
                "id": "replay",
                "records": "x.jsonl",
                "mode": "recheck",
                "anchor": "tool-call",
                "source": {"id": "echo-v1"},
            }
        )


def test_replay_config_freezes_record_globs(tmp_path):
    for name in ("a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text("")
    config_cls = taskset_config_type("replay")
    config = config_cls.model_validate(
        {
            "id": "replay",
            "records": str(tmp_path / "*.jsonl"),
            "source": {"id": "echo-v1"},
        }
    )
    assert config.records == [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    # A file appearing later must not shift task indices on a worker's re-validation.
    (tmp_path / "c.jsonl").write_text("")
    again = config_cls.model_validate(config.model_dump(mode="json"))
    assert again.records == config.records


def _replay_taskset(tmp_path, source: dict, trace: vf.Trace):
    (tmp_path / "records.jsonl").write_text(json.dumps(trace.to_record()) + "\n")
    return load_taskset(
        taskset_config_type("replay").model_validate(
            {
                "id": "replay",
                "records": str(tmp_path / "records.jsonl"),
                "mode": "recheck",
                "source": source,
            }
        )
    )


def test_capabilities_state_and_typed_tasks_delegate_to_the_source(tmp_path):
    from echo_tool_v1 import EchoToolTask
    from echo_user_sim_v1 import EchoUserSimState, EchoUserSimTaskset
    from echo_v1 import EchoTask

    trace = vf.Trace(task=EchoTask(idx=0, prompt="say ping", answer="ping"))
    graph.prepare_turn(trace, [SYSTEM, vf.UserMessage(content="say ping")]).commit(
        _reply(_assistant("ping"))
    )
    taskset = _replay_taskset(tmp_path, {"id": "echo-v1"}, trace)
    # The harness-capability gate and the rollout's state type must see the source's, not the
    # wrapper's delegating stubs.
    assert taskset.defines_tools is False
    assert taskset.defines_user is False
    assert taskset.state_type() is vf.State
    tasks = taskset.load_tasks()
    assert isinstance(tasks[0], EchoTask) and tasks[0].answer == "ping"
    assert tasks[0].idx == 0 and tasks[0].name.startswith("recheck:")

    tool_trace = vf.Trace(task=EchoToolTask(idx=0, prompt="say ping"))
    graph.prepare_turn(tool_trace, [SYSTEM, vf.UserMessage(content="say ping")]).commit(
        _reply(_assistant("ping"))
    )
    tool_replay = _replay_taskset(tmp_path, {"id": "echo-tool-v1"}, tool_trace)
    assert tool_replay.defines_tools is True

    user_sim_replay = _replay_taskset(tmp_path, {"id": "echo-user-sim-v1"}, trace)
    assert user_sim_replay.defines_user is True
    assert user_sim_replay.state_type() is EchoUserSimState
    assert EchoUserSimTaskset  # imported for the state assertion's provenance
