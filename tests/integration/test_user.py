"""Multi-turn exchanges with a user: a scripted user drives the harness across the
harness x runtime grid (echo-user-sim-v1), the caller drives an agent turn by turn
(`agent.interaction`), an exchange resumes with MCP access intact
(echo-acp-resume-v1), and a modeled user composes with the assistant (`user-sim`)."""

import pytest
from grid import grid

import verifiers.v1 as vf


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("resume",)), indirect=True
)
async def test_user(run_v1, harness, harness_runtime, tmp_path):
    """Multi-turn, driven by a scripted user — an interaction loop in the env's
    `run()`. The task is prompt-less, so one run covers the whole exchange shape:
    the caller opens (the user speaks first), each later turn resumes the harness
    onto the conversation, and leaving the loop ends the exchange (`user_closed`).
    The user runs in the eval process itself, so there is no placement axis."""
    (trace,) = await run_v1(
        "echo-user-sim-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=6,
        max_tokens=8192,
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.stop_condition == "user_closed"  # leaving the interaction ended it
    assert trace.reward == 1.0


async def test_interaction(live_ctx):
    """Drive an agent turn-by-turn through `agent.interaction()` — the caller IS
    the run's user. Runs on the tool-less `null` harness: nothing but the exchange
    itself, yet a real rollout — trace, scoring, and the `user_closed` stop all
    apply."""
    from verifiers.v1.harnesses.null import NullHarnessConfig

    agent = vf.make_agent(
        vf.AgentConfig(
            harness=NullHarnessConfig(id="null"),
            model=live_ctx.model,
            sampling=live_ctx.sampling,
            client=live_ctx.client,
        ),
    )
    task = vf.Task(
        vf.TaskData(
            idx=0,
            prompt=None,  # the interaction's caller opens the conversation
            system_prompt="Repeat the user's message back exactly, no extra words.",
        )
    )
    async with agent.interaction(task) as interaction:
        first = await interaction.turn("hello world")
        assert isinstance(first, vf.Segment)
        assert not first.terminated
        assert [message.role for message in first.messages] == ["assistant"]
        assert "hello world" in first.last_reply.lower()
        second = await interaction.turn("goodbye world")
        assert not second.terminated
        assert [message.role for message in second.messages] == ["assistant"]
        assert "goodbye world" in second.last_reply.lower()
    trace = interaction.trace
    assert trace is not None and trace.errors == []
    assert trace.stop_condition == "user_closed"  # closing the interaction ended it
    assert trace.num_turns == 2


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("resume", "mcp")), indirect=True
)
async def test_resume_with_tool(run_v1, harness, harness_runtime, tmp_path):
    """Each harness preserves context and MCP access across two segments."""
    (trace,) = await run_v1(
        "echo-acp-resume-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=8,
        max_tokens=8192,
        rollout_timeout=600,
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.stop_condition == "user_closed"
    assert trace.rewards["resumed"].score == 1.0
    segments = trace.info["acp_segments"]
    assert len(segments) == 2
    assert segments[0]["terminated"] is False
    assert segments[1]["terminated"] is False
    assert trace.last_reply == segments[1]["last_reply"]
    # Kimi Code is broken upstream: its Responses adapter drops message `phase` on replay.
    if harness.id != "kimi-code":
        assert trace.num_branches == 1
    assert "tool" in segments[1]["roles"]
    assert segments[1]["tool_outputs"]
    if harness.id == "pi":
        recall = next(tool for tool in trace.tools if tool.name == "resume_recall")
        assert recall.parameters["properties"]["codeword"]["type"] == "string"
        assert "codeword" in recall.parameters["required"]
    if harness.id == "codex":
        recall = next(tool for tool in trace.tools if tool.name.endswith("recall"))
        assert recall.namespace
        assert "codeword" in recall.parameters["properties"]
        assert any(
            call.namespace == recall.namespace and call.name == recall.name
            for message in trace.assistant_messages
            for call in message.tool_calls or []
        )
    if harness.id == "rlm":
        assert "turns_since_last_compaction" in trace.metrics
        assert all(call.acp is not None for call in trace.calls)
        assert all(
            trace.nodes[parent.node].sampled and node.sampled
            for node in trace.nodes
            for parent in node.semantic_parents
        )
    if harness.id == "prime-agent":
        lifecycle = trace.info["acp_lifecycle"]["ai.primeintellect.prime-agent"]
        assert len(lifecycle) == 2
        for status in lifecycle:
            assert status["infrastructure_status"] == "ok"
            assert status["terminal_quiescence_observed"] is True
            assert (
                status["terminal_quiescence"]["quiescence"]["outstandingSubagents"] == 0
            )


async def test_env_id_user_sim(run_v1, tmp_path):
    """The user-sim env over the echo taskset: a modeled user (null harness) opens
    the conversation from the task's prompt-as-scenario; the assistant's trace is
    judged by the task's own reward; both sides land agent-stamped on one episode."""
    traces = await run_v1(
        "echo-v1",
        harness=None,  # multi-agent: each seat pins its own
        env={"id": "user-sim", "assistant": {"harness": {"id": "null"}}},
        output_dir=tmp_path,
        max_turns=6,
        rollout_timeout=300,
    )
    assert sorted(t.agent.name for t in traces) == ["assistant", "user"]
    (assistant,) = [t for t in traces if t.agent.name == "assistant"]
    (user,) = [t for t in traces if t.agent.name == "user"]
    assert assistant.ok and user.ok
    assert user.agent.trainable is False
    assert user.num_turns >= 1  # the modeled user actually spoke
    assert assistant.metrics["user_turns"] >= 1
    # The env nulls the assistant task's prompt: the scenario is hidden from the
    # assistant's harness while the task's rewards score off non-prompt fields
    # (`answer`). The nulled row is what persists (provenance is the row's idx);
    # both sides land as ONE durable episode.
    assert assistant.task.data.prompt is None
    assert "echoed" in assistant.rewards
    from verifiers.v1.trace import WireTrace
    from verifiers.v1.utils.trace_store import read_episodes

    (record,) = read_episodes(tmp_path, WireTrace)
    assert {t.agent.name for t in record.traces} == {"assistant", "user"}
    assert record.id  # both traces are persisted under one durable episode identity


async def test_env_id_user_sim_with_tools(run_v1, tmp_path):
    """THE tau-bench shape — a tool-using assistant composed with a modeled user.
    The assistant's MCP tool loop runs entirely inside a harness segment and the
    user exchange advances between segments, so the two can never race or amputate
    each other (the failure mode of injecting user turns at the model boundary).
    Reward 1.0 proves the tool actually ran mid-conversation: its token never
    appears in any prompt."""
    traces = await run_v1(
        "echo-tool-v1",
        harness=None,  # multi-agent: each seat pins its own
        env={"id": "user-sim", "assistant": {"harness": {"id": "null"}}},
        output_dir=tmp_path,
        max_turns=8,
        # Reasoning models can spend thousands of tokens on a turn; the default
        # 2048 truncates the reply mid-relay, cutting the stamp out of the text.
        max_tokens=8192,
        rollout_timeout=300,
    )
    (assistant,) = [t for t in traces if t.agent.name == "assistant"]
    (user,) = [t for t in traces if t.agent.name == "user"]
    assert assistant.ok and user.ok
    assert assistant.task.data.prompt is None  # the scenario stayed off the wire
    assert user.num_turns >= 1  # the modeled user actually drove the exchange
    assert assistant.rewards["echoed"].score == 1.0  # the tool ran, mid-conversation
    # The tool was advertised to the masked chat, regardless of harness qualification.
    assert assistant.tools
    assert any(tool.name.endswith("back") for tool in assistant.tools)
