import base64
import json

import msgpack
import numpy as np
import pytest
from renderers.client import parse_generate_response

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients.train import response_from_generate
from verifiers.v1.routing import RoutingData
from verifiers.v1.types import TurnTokens


def _response(message: vf.AssistantMessage) -> vf.Response:
    return vf.Response(
        id="",
        created=0,
        model="test",
        message=message,
        finish_reason="stop",
    )


def _routed_payload(
    num_tokens: int, start: int, base: int, layers: int = 2, top_k: int = 1
):
    """A fake `generate` router-replay sidecar (uint8 `[num_tokens, layers, top_k]`, base64)."""
    arr = (
        np.arange(num_tokens * layers * top_k)
        .reshape(num_tokens, layers, top_k)
        .astype(np.uint8)
        + base
    )
    return {
        "data": base64.b64encode(arr.tobytes()).decode(),
        "shape": list(arr.shape),
        "start": start,
    }


def _full_routing_response(rows=4, *, prompt=None, completion=None):
    payload = _routed_payload(rows, 0, 11, top_k=2)
    weights = np.tile(np.array([0.12345679, 0.8765432], dtype="<f4"), (rows, 2, 1))
    payload.update(
        dtype="uint8",
        format_version=1,
        weights={
            "data": base64.b64encode(weights.tobytes()).decode(),
            "dtype": "float32",
        },
    )
    parsed = parse_generate_response(
        json.dumps(
            {"choices": [{"routed_experts": payload}]}, separators=(",", ":")
        ).encode()
    )
    return response_from_generate(
        {
            "prompt_ids": [10, 11, 12] if prompt is None else prompt,
            "completion_ids": [20, 21] if completion is None else completion,
            "content": "a1",
            "routed_experts": parsed["choices"][0]["routed_experts"],
        },
        "test",
    )


@pytest.mark.parametrize(
    "rows,completion", [(4, [20, 21]), (3, [20]), (5, [20, 21]), (3, [])]
)
def test_full_routing_train_response_graph_and_msgpack_round_trip(rows, completion):
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    response = _full_routing_response(rows, completion=completion)
    # Exercise the validated carrier as well as the train client's construct-only path.
    response.tokens = TurnTokens(
        prompt_ids=response.tokens.prompt_ids,
        completion_ids=response.tokens.completion_ids,
        routed_experts=response.tokens.routed_experts,
    )
    source = response.tokens.routed_experts
    graph.prepare_turn(trace, [vf.UserMessage(content="u1")]).commit(response)
    data = trace.branches[0].routed_experts
    assert isinstance(data, RoutingData)
    assert len(data) == len(trace.branches[0].token_ids) == 3 + len(completion)
    assert data.ids[:rows].tobytes() == base64.b64decode(source["data"])
    assert data.weights[:rows].tobytes() == base64.b64decode(source["weights"]["data"])
    assert data.valid[:rows].all()
    if rows < len(data):
        assert not data.valid[-1] and not data.weights[-1].any()
        assert not data.ids[-1].any()
    dumped = trace.model_dump(mode="python")
    assert all(
        node["routed_experts"]["__routing__"]
        for node in dumped["nodes"]
        if node["token_ids"]
    )
    restored = vf.Trace.model_validate(
        msgpack.unpackb(msgpack.packb(dumped), raw=False)
    )
    for name in ("ids", "weights", "valid"):
        assert (
            getattr(restored.branches[0].routed_experts, name).tobytes()
            == getattr(data, name).tobytes()
        )
        for node in restored.nodes:
            if node.token_ids:
                with pytest.raises(ValueError):
                    getattr(node.routed_experts, name).flags.writeable = True


@pytest.mark.parametrize(
    "rows,completion,spans",
    [(2, [20, 21], None), (6, [20, 21], None), (2, [], None), (4, [20, 21], [(0, 4)])],
)
def test_full_routing_invalid_coverage_or_spans_does_not_mutate_graph(
    rows, completion, spans
):
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    response = _full_routing_response(rows, completion=completion)
    response.tokens.message_spans = spans
    before = trace.model_dump(mode="python")
    with pytest.raises(ValueError, match="capture must cover|message spans"):
        graph.prepare_turn(trace, [vf.UserMessage(content="u1")]).commit(response)
    assert trace.model_dump(mode="python") == before


@pytest.mark.parametrize("next_mode", ["full", "legacy", "missing", "parallel"])
def test_full_routing_reused_prefix_is_rejected_atomically(next_mode):
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    user = vf.UserMessage(content="u1")
    first = graph.prepare_turn(trace, [user])
    parallel = graph.prepare_turn(trace, [user])
    first.commit(_full_routing_response())
    before = trace.model_dump(mode="python")
    if next_mode == "parallel":
        pending, response = parallel, _full_routing_response()
    else:
        pending = graph.prepare_turn(
            trace,
            [user, vf.AssistantMessage(content="a1"), vf.UserMessage(content="u2")],
        )
        response = _full_routing_response(
            8, prompt=[10, 11, 12, 20, 21, 30, 31], completion=[40, 41]
        )
        response.tokens.message_spans = [(0, 3), None, (5, 7)]
        if next_mode == "legacy":
            response.tokens.routed_experts = _routed_payload(8, 0, 0)
        elif next_mode == "missing":
            response.tokens.routed_experts = None
    with pytest.raises(ValueError, match="reused-prefix"):
        pending.commit(response)
    assert trace.model_dump(mode="python") == before


def test_routed_experts_attributed_and_aligned_across_turns():
    """Each turn's full routing (start=0) is attributed to the nodes it created; the new turn's
    nodes get this turn's slice and reused nodes keep theirs, so `Branch.routed_experts`
    concatenates back to a `[tokens, layers, top_k]` array aligned 1:1 with `branch.token_ids` —
    and survives the base64 wire round-trip."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    user = vf.UserMessage(content="u1")
    graph.prepare_turn(trace, [user]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a1"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11, 12],
                completion_ids=[20, 21],
                message_spans=[(0, 2)],
                routed_experts=_routed_payload(5, 0, 0),
            ),
        )
    )
    graph.prepare_turn(
        trace,
        [user, vf.AssistantMessage(content="a1"), vf.UserMessage(content="u2")],
    ).commit(
        vf.Response(
            id="b",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a2"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11, 12, 20, 21, 30, 31],
                completion_ids=[40, 41],
                message_spans=[(0, 2), None, (5, 7)],
                routed_experts=_routed_payload(9, 0, 100),
            ),
        )
    )
    branch = trace.branches[-1]
    re = branch.routed_experts
    assert re is not None
    assert re.shape[0] == len(branch.token_ids)

    restored = type(trace).model_validate(trace.model_dump())
    re2 = restored.branches[-1].routed_experts
    assert re2 is not None and re2.shape == re.shape and bool((re2 == re).all())
    assert all(
        node.routed_experts is None or node.routed_experts.flags.owndata
        for node in trace.nodes
    )


def test_routed_experts_none_when_absent():
    """No routing captured (engine ran without `enable_return_routed_experts`) -> the branch
    reports None and the trainer simply skips replay."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    graph.prepare_turn(trace, [vf.UserMessage(content="u1")]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a1"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1, 2], completion_ids=[3], message_spans=[(0, 2)]
            ),
        )
    )
    assert trace.branches[-1].routed_experts is None


def test_tool_call_hash_matches_v0_content_and_arguments_normalization():
    left = vf.AssistantMessage(
        content=None,
        tool_calls=[
            vf.ToolCall(id="call_0", name="lookup", arguments='{"b": 2, "a": 1}')
        ],
    )
    right = vf.AssistantMessage(
        content="",
        tool_calls=[vf.ToolCall(id="call_0", name="lookup", arguments='{"a":1,"b":2}')],
    )

    assert graph.message_hash(left) == graph.message_hash(right)


def test_reasoning_content_participates_in_graph_prefix_matching():
    task = vf.TaskData(idx=0, prompt="use a tool")
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=task),
    )
    user = vf.UserMessage(content="use a tool")
    call = vf.ToolCall(id="call_0", name="lookup", arguments="{}")

    graph.prepare_turn(trace, [user]).commit(
        _response(
            vf.AssistantMessage(
                content=None,
                reasoning_content="plan A",
                tool_calls=[call],
            )
        )
    )
    graph.prepare_turn(
        trace,
        [
            user,
            vf.AssistantMessage(
                content=None,
                reasoning_content="plan B",
                tool_calls=[call],
            ),
            vf.ToolMessage(content="result", tool_call_id="call_0"),
        ],
    ).commit(_response(vf.AssistantMessage(content="done")))

    tool_call_nodes = [
        node
        for node in trace.nodes
        if isinstance(node.message, vf.AssistantMessage) and node.message.tool_calls
    ]
    assert len(tool_call_nodes) == 2


def test_parallel_commits_reconcile_shared_prompt_prefix():
    """Two requests prepared from the same graph snapshot share any common prompt prefix that
    the first response commits while the second is in flight. A later turn must keep following
    its original child path rather than re-rooting through the sibling and stranding an orphan."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="root")),
    )
    system = vf.SystemMessage(content="shared")
    user_a = vf.UserMessage(content="child A")
    user_b = vf.UserMessage(content="child B")
    assistant_a = vf.AssistantMessage(content="A1")

    # Both model requests leave before either response has committed its prompt.
    pending_a = graph.prepare_turn(trace, [system, user_a])
    pending_b = graph.prepare_turn(trace, [system, user_b])
    assistant_a_id = pending_a.commit(_response(assistant_a))
    pending_b.commit(_response(vf.AssistantMessage(content="B1")))

    graph.prepare_turn(
        trace,
        [
            system,
            user_a,
            assistant_a,
            vf.ToolMessage(content="tool A", tool_call_id="call_a"),
        ],
    ).commit(_response(vf.AssistantMessage(content="A2")))

    roots = [node for node in trace.nodes if node.parent is None]
    identities = [
        (node.parent, graph.message_hash(node.message)) for node in trace.nodes
    ]
    assert len(roots) == 1
    assert len(identities) == len(set(identities))
    assert trace.num_branches == 2
    assert assistant_a_id not in graph.leaves(trace)


def test_parallel_commit_reconciles_only_token_identical_prefixes():
    """Content-equivalent prompt nodes share exact physical variants and retain token forks."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="root")),
    )
    system = vf.SystemMessage(content="shared")

    first = graph.prepare_turn(trace, [system])
    second = graph.prepare_turn(trace, [system])
    third = graph.prepare_turn(trace, [system])
    first.commit(
        vf.Response(
            id="a",
            created=0,
            model="test",
            message=vf.AssistantMessage(content="A"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1, 2], completion_ids=[3], message_spans=[(0, 2)]
            ),
        )
    )
    second.commit(
        vf.Response(
            id="b",
            created=0,
            model="test",
            message=vf.AssistantMessage(content="B"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1, 2], completion_ids=[4], message_spans=[(0, 2)]
            ),
        )
    )
    third.commit(
        vf.Response(
            id="c",
            created=0,
            model="test",
            message=vf.AssistantMessage(content="C"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1, 99], completion_ids=[5], message_spans=[(0, 2)]
            ),
        )
    )

    roots = [node for node in trace.nodes if node.parent is None]
    assert [node.token_ids for node in roots] == [[1, 2], [1, 99]]
    assert trace.num_branches == 3


def test_parallel_commit_reconciles_unspanned_assistant_tokens():
    """A concurrently committed sampled assistant can be unspanned when it reappears in a
    prompt. Its stored physical tokens still delimit that message and must not shift onto the
    following input node."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="root")),
    )
    user = vf.UserMessage(content="question")
    assistant = vf.AssistantMessage(content="answer")
    follow_up = vf.UserMessage(content="follow up")

    pending = graph.prepare_turn(trace, [user, assistant, follow_up])
    graph.prepare_turn(trace, [user]).commit(
        vf.Response(
            id="first",
            created=0,
            model="test",
            message=assistant,
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1], completion_ids=[2, 3], message_spans=[(0, 1)]
            ),
        )
    )
    pending.commit(
        vf.Response(
            id="second",
            created=0,
            model="test",
            message=vf.AssistantMessage(content="done"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1, 2, 3, 4],
                completion_ids=[5],
                message_spans=[(0, 1), None, (3, 4)],
            ),
        )
    )

    assert trace.num_branches == 1
    assert [node.token_ids for node in trace.nodes] == [[1], [2, 3], [4], [5]]


def test_unspanned_reconciliation_stops_at_next_message_boundary():
    """A longer sampled variant may share the prompt's token prefix only by consuming the next
    message. Reconciliation must choose the variant ending before that attributed boundary."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="root")),
    )
    user = vf.UserMessage(content="question")
    assistant = vf.AssistantMessage(content="answer")
    follow_up = vf.UserMessage(content="follow up")

    pending = graph.prepare_turn(trace, [user, assistant, follow_up])
    first = graph.prepare_turn(trace, [user])
    second = graph.prepare_turn(trace, [user])
    first.commit(
        vf.Response(
            id="short",
            created=0,
            model="test",
            message=assistant,
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1], completion_ids=[2, 3], message_spans=[(0, 1)]
            ),
        )
    )
    second.commit(
        vf.Response(
            id="long",
            created=0,
            model="test",
            message=assistant,
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1], completion_ids=[2, 3, 4], message_spans=[(0, 1)]
            ),
        )
    )
    pending.commit(
        vf.Response(
            id="continued",
            created=0,
            model="test",
            message=vf.AssistantMessage(content="done"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1, 2, 3, 4],
                completion_ids=[5],
                message_spans=[(0, 1), None, (3, 4)],
            ),
        )
    )

    follow_up_node = next(node for node in trace.nodes if node.message == follow_up)
    assert follow_up_node.token_ids == [4]
    assert trace.nodes[follow_up_node.parent].token_ids == [2, 3]


def test_renderer_level_break_forks_by_token_id():
    """Two turns with the *same* message sequence and identical message hashes, but the prior
    assistant turn is retokenized (renderer drift — e.g. a chat template dropping a `<think>`
    block on re-render): the stored prefix tokens no longer match this turn's `prompt_ids`.
    Message-hash dedup alone would silently reuse the stale prefix; token-identity prefix reuse
    must fork at the diverging node. Each branch's leaf→root token concatenation still equals
    its own `prompt_ids + completion_ids`."""
    user = vf.UserMessage(content="u1")
    a1 = vf.AssistantMessage(content="a1")
    u2 = vf.UserMessage(content="u2")

    def first_turn(trace):
        graph.prepare_turn(trace, [user]).commit(
            vf.Response(
                id="a",
                created=0,
                model="t",
                message=a1,
                finish_reason="stop",
                tokens=TurnTokens(
                    prompt_ids=[1, 2, 3], completion_ids=[4, 5], message_spans=[(0, 2)]
                ),
            )
        )

    def second_turn(trace, prompt_ids):
        graph.prepare_turn(trace, [user, a1, u2]).commit(
            vf.Response(
                id="b",
                created=0,
                model="t",
                message=vf.AssistantMessage(content="a2"),
                finish_reason="stop",
                tokens=TurnTokens(
                    prompt_ids=prompt_ids,
                    completion_ids=[8],
                    message_spans=[(0, 2), (2, 5), (5, 7)],
                ),
            )
        )

    # Control: the prior turn re-renders to the same tokens -> stays one linear branch.
    linear = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    first_turn(linear)
    second_turn(linear, [1, 2, 3, 4, 5, 6, 7])
    assert linear.num_branches == 1
    assert linear.branches[0].token_ids == [1, 2, 3, 4, 5, 6, 7, 8]

    # Break: the assistant turn retokenizes (4 -> 99), so prompt_ids diverge at that node.
    broken = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    first_turn(broken)
    second_turn(broken, [1, 2, 3, 99, 5, 6, 7])
    assert broken.num_branches == 2
    assert sorted(b.token_ids for b in broken.branches) == [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 99, 5, 6, 7, 8],
    ]


def test_prompt_supplied_assistant_messages_are_not_sampled_turns():
    task = vf.TaskData(idx=0, prompt="few-shot")
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=task),
    )
    fabricated = vf.AssistantMessage(
        content=None,
        tool_calls=[vf.ToolCall(id="call_0", name="lookup", arguments="{}")],
    )
    response = vf.AssistantMessage(content="real answer")

    graph.prepare_turn(
        trace,
        [
            vf.UserMessage(content="question"),
            fabricated,
            vf.ToolMessage(content="fabricated result", tool_call_id="call_0"),
        ],
    ).commit(_response(response))

    assert [n.sampled for n in trace.nodes] == [False, False, False, True]
    assert trace.num_turns == 1
    assert trace.assistant_messages == [response]
