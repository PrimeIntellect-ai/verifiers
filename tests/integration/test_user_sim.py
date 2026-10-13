"""The `user-sim` env: a modeled user opens the conversation from the task's
prompt-as-scenario and the assistant is judged by the task's own reward, over the plain
echo task and the tool task (the tau-bench shape)."""

from verifiers.v1.trace import WireTrace
from verifiers.v1.utils.trace_store import read_episodes


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
