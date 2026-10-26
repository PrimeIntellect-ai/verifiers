"""Trace construction, serialization round trips, and the judge-facing transcript."""

import json

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import resolve_agent
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.dialects.responses import ResponsesDialect, fold_assistant
from verifiers.v1.graph import MessageNode, prepare_turn
from verifiers.v1.session import RolloutLimits
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils.trace_store import write_episode


class MyTask(vf.TaskData):
    answer: str = ""  # a task-specific field WireTaskData must absorb


class MyState(vf.State):
    score: int = 0


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        {},
        {"reasoning_effort": "high"},
        {"temperature": 0.7, "max_tokens": 128, "reasoning_effort": "high"},
        {"temperature": None},
        {"max_completion_tokens": None},
        {"extra_body": {"top_k": None}},
        {"extra_body": None},
        {"top_k": 10, "extra_body": {"frequency_penalty": 0.5}},
        {"extra_body": {"top_k": 20, "max_completion_tokens": None}},
    ],
)
def test_bare_trace_round_trip(sampling):
    spec = vf.AgentConfig(sampling=sampling)
    defaults = vf.SamplingConfig(temperature=0.7, max_tokens=128, top_k=40)
    # Resolve the live overrides before serializing the effective config on a trace.
    resolved = resolve_agent(spec, sampling=defaults)
    assert set(vf.SamplingConfig.model_fields) <= resolved.sampling.model_fields_set
    # The minimal trace: a base task, no nodes, no extras — dump and back into a plain Trace.
    tr = vf.Trace(
        agent=vf.AgentInfo(config=resolved),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=3, prompt="hello"),
            key="dataset/example-3",
            hash="content-digest",
        ),
    )
    episode = vf.Episode(task=tr.task, traces=[tr], ok=True)
    for rt in (
        vf.Trace.model_validate(tr.model_dump()),
        vf.Trace.model_validate_json(tr.model_dump_json(exclude_none=True)),
        vf.Trace.model_validate(tr.to_record()),
        vf.Episode.model_validate_json(
            episode.model_dump_json(exclude_none=True)
        ).traces[0],
        vf.Episode.model_validate(episode.to_record()).traces[0],
    ):
        assert rt.id == tr.id
        assert rt.task.type == "Task"
        assert rt.task.data.idx == 3 and rt.task.data.prompt == "hello"
        assert rt.task.key == "dataset/example-3" and rt.task.hash == "content-digest"
        assert rt.num_turns == 0 and rt.num_branches == 0
        assert rt.reward == 0.0 and rt.errors == []
        assert rt.agent.config == tr.agent.config
        assert rt.agent.config.sampling.model_dump() == resolved.sampling.model_dump()


def test_custom_task_state_round_trip(tmp_path):
    # Custom data and state round-trip into the same parameterization. Data fields are
    # typed (not just `model_extra`); `state` is runtime-only and never crosses the wire.
    tr = vf.Trace[MyTask, MyState](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="gold")),
        state=MyState(score=7, artifacts={"/artifact.bin": b"\xff\x00\xfe"}),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    tr.record_reward("r", 0.5)
    wire = tr.model_dump()
    assert "state" not in wire  # transient state is excluded from the dump

    rt = vf.Trace[MyTask, MyState].model_validate(wire)
    assert (
        isinstance(rt.task.data, MyTask) and rt.task.data.answer == "gold"
    )  # typed custom field
    assert rt.task.type == "MyTask"  # the producing class's name survives the wire
    assert rt.num_turns == 1 and rt.num_branches == 1
    assert rt.reward == 0.5  # property recomputed from `rewards`
    write_episode(tmp_path, vf.Episode(task=tr.task, traces=[tr], ok=True))
    saved = json.loads((tmp_path / "traces.jsonl").read_bytes())
    assert "state" not in saved["traces"][0]
    assert tr.state.artifacts["/artifact.bin"] == b"\xff\x00\xfe"


@pytest.mark.parametrize("history_type", ["additional_tools", "tool_search_output"])
def test_wire_trace_round_trip(history_type):
    mcp = {
        "type": "mcp",
        "server_label": "alpha",
        "server_url": "https://example.invalid/mcp",
        "authorization": "trace-test-oauth-token",
        "headers": {"Authorization": "Bearer trace-test-header-token"},
        "Authorization": "trace-test-case-oauth-token",
        "Headers": {"X-Api-Key": "trace-test-case-header-token"},
    }
    function = {
        "type": "function",
        "name": "echo",
        "description": "Echo the supplied text.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        "strict": False,
    }
    stale = function | {"parameters": {"type": "object"}, "strict": True}
    request = ResponsesDialect().parse_request(
        {
            "input": [
                {
                    "type": history_type,
                    "call_id": "search",
                    "tools": [
                        stale,
                        {"type": "namespace", "name": "mcp__world", "tools": [stale]},
                    ],
                }
            ],
            "tools": [
                function,
                {"type": "namespace", "name": "mcp__world", "tools": [function]},
                {"type": "namespace", "name": "mcp__other", "tools": [function]},
                {"type": "custom", "name": "patch", "format": {"type": "text"}},
                {"type": "web_search", "search_context_size": "low"},
                mcp,
                mcp | {"server_label": "beta"},
            ],
        }
    )
    assert request.tools is not None and len(request.tools) == 7
    assert all(tool.parameters == function["parameters"] for tool in request.tools[:3])
    assert all(tool.strict is False for tool in request.tools[:3])
    assistant = fold_assistant(
        [
            {
                "type": "function_call",
                "call_id": "call_0",
                "name": "echo",
                "namespace": "mcp__world",
                "arguments": '{"text":"hello"}',
            }
        ]
    )
    # Two leaves off one root → 2 branches (a compaction-shaped trace), so the round-trip has to
    # carry node `parent` links for `num_branches` to survive.
    tr = vf.Trace[MyTask, vf.State](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="a")),
        tools=request.tools,
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=assistant, sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    tr.record_reward("r", 1.0)
    tr.rewards.setdefault("solved", None)  # seeded: expected but never scored
    tr.metrics.setdefault("acc", None)
    tr.info = {"build": "ok"}
    tr.root_reply = "root answer"
    tr.stop("done")
    prepare_turn(tr, [], request.tools).commit_prompt()

    # the dump is plain pydantic — derived values are properties, so they're not serialized
    data = json.loads(tr.model_dump_json(exclude_none=True))
    assert "trace-test-" not in json.dumps(data)
    assert mcp["authorization"] == "trace-test-oauth-token"
    assert mcp["headers"] == {"Authorization": "Bearer trace-test-header-token"}
    assert mcp["Authorization"] == "trace-test-case-oauth-token"
    assert mcp["Headers"] == {"X-Api-Key": "trace-test-case-header-token"}
    chat_request = ChatDialect().parse_request(
        {"tools": [mcp, mcp | {"server_label": "beta"}]}
    )
    chat_trace = tr.model_copy(update={"tools": []})
    prepare_turn(chat_trace, [], chat_request.tools).commit_prompt()
    assert [tool.name for tool in chat_trace.tools] == ["alpha", "beta"]
    assert "trace-test-" not in chat_trace.model_dump_json()
    assert (
        vf.WireTrace.model_validate_json(chat_trace.model_dump_json()).tools
        == chat_request.tools
    )
    assert "reward" not in data and "is_truncated" not in data
    # exclude_none drops None FIELDS, not None dict values — unscored seeds survive
    assert data["rewards"]["solved"] is None and data["metrics"]["acc"] is None

    rt = vf.WireTrace.model_validate(data)
    assert rt.num_branches == tr.num_branches == 2  # branch topology survived
    assert rt.num_turns == tr.num_turns == 2
    assert rt.reward == 1.0  # property recomputed from `rewards`, seeds contribute 0
    assert rt.rewards["solved"] is None
    assert rt.stop_condition == "done"
    assert rt.info == {"build": "ok"}
    assert rt.root_reply == "root answer"
    assert rt.last_reply == "root answer"
    rt.root_reply = ""
    assert rt.last_reply == ""
    assert (
        rt.tools == tr.tools
    )  # the advertised tools persist (tool-use SFT reads them)
    assert [tool.namespace for tool in rt.tools[:3]] == [
        None,
        "mcp__world",
        "mcp__other",
    ]
    assert all(tool.parameters == function["parameters"] for tool in rt.tools[:3])
    assert all(tool.strict is False for tool in rt.tools[:3])
    assert rt.tools[3].type == "custom"
    assert rt.tools[3].model_extra == {"format": {"type": "text"}}
    assert rt.tools[4].type == "web_search"
    assert rt.tools[4].model_extra == {"search_context_size": "low"}
    assert [tool.name for tool in rt.tools[5:]] == ["alpha", "beta"]
    assert rt.nodes[1].message.tool_calls[0].namespace == "mcp__world"
    assert rt.task.data.model_extra == {
        "answer": "a"
    }  # taskset extras preserved on WireTaskData

    # the env-server wire form (a plain model_dump) loads too
    assert vf.WireTrace.model_validate(tr.model_dump()).num_branches == 2


def test_transcript(make_full_trace):
    transcript = make_full_trace().transcript
    assert "[user]\nCapital of France?" in transcript
    assert '[tool_call search({"q": "france"})]' in transcript
    assert "[tool search]\nTOOL RESULT: Paris is the capital." in transcript
    assert "It is Paris." in transcript
    assert "SECRET REASONING" not in transcript  # reasoning is excluded


def test_sampling_mask_from_packed_matches_lists():
    import base64

    import numpy as np

    from verifiers.v1.types import SamplingMask

    rows = [[4, 7], [], [9]]
    ids, counts = np.array([4, 7, 9], np.int32), np.array([2, 0, 1], np.int32)
    packed = {
        name: {
            "data": base64.b64encode(a.tobytes()).decode(),
            "shape": list(a.shape),
            "dtype": "int32",
        }
        for name, a in (("ids", ids), ("counts", counts))
    }
    for mask in (
        SamplingMask.from_sampling_mask(rows),
        SamplingMask.from_sampling_mask(packed),
    ):
        np.testing.assert_array_equal(mask.ids, ids)
        np.testing.assert_array_equal(mask.counts, counts)


def test_max_turn_tokens_limit():
    limits = RolloutLimits(max_turn_tokens=16)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
    )
    trace.calls.append(vf.ModelCall(finish_reason="stop"))
    assert limits.reached(trace) is None
    # A length finish below the cap ran out of context: not this limit.
    usage = vf.Usage(prompt_tokens=100, completion_tokens=8)
    trace.calls.append(vf.ModelCall(finish_reason="length", usage=usage))
    assert limits.reached(trace) is None
    usage = vf.Usage(prompt_tokens=100, completion_tokens=16)
    trace.calls.append(vf.ModelCall(finish_reason="length", usage=usage))
    assert limits.reached(trace) == "max_turn_tokens"
    trace.stop("max_turn_tokens")
    assert trace.is_truncated


@pytest.mark.parametrize("max_tokens,expected", [(None, 16), (8, 8), (64, 16)])
def test_max_turn_tokens_caps_sampling(max_tokens, expected):
    spec = vf.AgentConfig(max_turn_tokens=16)
    sampling = vf.SamplingConfig(max_tokens=max_tokens)
    assert resolve_agent(spec, sampling=sampling).sampling.max_tokens == expected


@pytest.mark.parametrize("pinned", [False, True])
def test_run_sampling_stays_with_run_client(pinned):
    run_sampling = vf.SamplingConfig(temperature=0.8, top_k=20)
    client = vf.EvalClientConfig(base_url="http://judge.invalid/v1") if pinned else None
    spec = vf.AgentConfig(client=client, sampling={"reasoning_effort": "high"})
    resolved = resolve_agent(
        spec, client=vf.TrainClientConfig(), sampling=run_sampling
    ).sampling
    expected = {"reasoning_effort": "high"}
    if not pinned:
        expected |= {"temperature": 0.8, "top_k": 20}
    assert resolved.model_dump(exclude_none=True) == expected
