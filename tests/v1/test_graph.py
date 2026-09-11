import base64

import numpy as np

import verifiers.v1 as vf
from verifiers.v1 import graph
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


def test_bridged_turn_recovers_the_prior_turn_unforwarded_row():
    """A turn never forwards its own final sampled token, so the engine returns one routing row
    fewer than the turn has prompt plus completion positions, and that final position is filled
    with a copy of its predecessor: a placeholder, effectively. The next turn's prefill does
    forward it and reports it as row 0 (the `- 1` in `routed_experts_prompt_start`), so the
    placeholder must be replaced by that real row rather than discarded."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    user = vf.UserMessage(content="u1")

    # Start first turn
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
                routed_experts=_routed_payload(4, 0, 0),
            ),
        )
    )
    sampled_node = trace.nodes[-1]
    assert sampled_node.routed_experts is not None
    # First turn has the placeholder routing assignments:
    assert bool(
        (sampled_node.routed_experts[-1] == sampled_node.routed_experts[-2]).all()
    )
    forwarded_rows = sampled_node.routed_experts[:-1].copy()

    # Start second turn
    bridged = _routed_payload(4, 4, 100)
    true_row = np.frombuffer(base64.b64decode(bridged["data"]), dtype=np.uint8).reshape(
        bridged["shape"]
    )[0]
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
                routed_experts=bridged,
            ),
        )
    )

    recovered = sampled_node.routed_experts
    assert recovered is not None
    # Did the first turn's final placeholder get corrected?
    assert bool((recovered[-1] == true_row).all()), (
        f"expected turn 1's final position to hold {true_row.tolist()}, "
        f"got {recovered[-1].tolist()}"
    )
    assert bool((recovered[:-1] == forwarded_rows).all()), (
        "only the fabricated final row may change"
    )
    assert recovered.flags.owndata
    branch = trace.branches[-1]
    assert branch.routed_experts is not None
    assert branch.routed_experts.shape[0] == len(branch.token_ids)


def _routed_payload_uint16(
    num_tokens: int, start: int, base: int, layers: int = 2, top_k: int = 1
):
    """As `_routed_payload`, but self-describing as uint16, which the engine emits once a
    response's expert ids exceed 255."""
    arr = (
        np.arange(num_tokens * layers * top_k).reshape(num_tokens, layers, top_k) + base
    ).astype(np.uint16)
    return {
        "data": base64.b64encode(arr.tobytes()).decode(),
        "shape": list(arr.shape),
        "start": start,
        "dtype": "uint16",
    }


def _new_trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )


def _turn_response(content, prompt_ids, completion_ids, spans, routed=None):
    """A `generate` response, built the way `train.py` does: unvalidated, so a payload's `dtype`
    key survives (`RoutedExperts` does not declare one)."""
    return vf.Response(
        id=content,
        created=0,
        model="t",
        message=vf.AssistantMessage(content=content),
        finish_reason="stop",
        tokens=TurnTokens.model_construct(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            message_spans=spans,
            routed_experts=routed,
        ),
    )


_U1 = vf.UserMessage(content="u1")
_A1 = vf.AssistantMessage(content="a1")
_U2 = vf.UserMessage(content="u2")
_A2 = vf.AssistantMessage(content="a2")
_U3 = vf.UserMessage(content="u3")
_T1 = vf.ToolMessage(content="t1", tool_call_id="c1")

_TURN1 = {
    "prompt": [_U1],
    "prompt_ids": [10, 11, 12],
    "completion_ids": [20, 21],
    "spans": [(0, 2)],
}
_TURN2 = {
    "prompt": [_U1, _A1, _U2],
    "prompt_ids": [10, 11, 12, 20, 21, 30, 31],
    "completion_ids": [40, 41],
    "spans": [(0, 2), None, (5, 7)],
}
_TURN3 = {
    "prompt": [_U1, _A1, _U2, _A2, _U3],
    "prompt_ids": [10, 11, 12, 20, 21, 30, 31, 40, 41, 50, 51],
    "completion_ids": [60, 61],
    "spans": [(0, 2), None, (5, 7), None, (9, 11)],
}


def _bridged_start(previous) -> int:
    """`routed_experts_prompt_start` as `train.py` computes it, reaching back one position so the
    turn that could not forward it supplies it now. None models a turn that could not bridge."""
    if previous is None:
        return 0
    return max(
        len(previous["prompt_ids"]) + len(previous["completion_ids"]) - 1,
        0,
    )


def _engine_payload(turn, previous, base, **kwargs):
    """What a real engine returns for `turn`: rows from the bridged start up to the last position
    it forwarded, which excludes the token it sampled last. Deriving it keeps a fixture from
    describing a response the engine could not produce."""
    start = _bridged_start(previous)
    positions = len(turn["prompt_ids"]) + len(turn["completion_ids"])
    return _routed_payload(positions - 1 - start, start, base, **kwargs)


def _commit(trace, turn, routed, content):
    graph.prepare_turn(trace, turn["prompt"]).commit(
        _turn_response(
            content,
            turn["prompt_ids"],
            turn["completion_ids"],
            turn["spans"],
            routed,
        )
    )


def test_only_the_last_turn_of_three_keeps_a_placeholder():
    """Each turn repairs the previous turn's placeholder, so after three turns only the third is
    left, unrepairable because no later prefill ever forwards its final token. Pinning the whole
    branch at once also pins that no neighbouring row was disturbed."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    _commit(trace, _TURN2, _engine_payload(_TURN2, _TURN1, 100), "a2")
    _commit(trace, _TURN3, _engine_payload(_TURN3, _TURN2, 200), "a3")

    branch = trace.branches[-1]
    routing = branch.routed_experts
    assert routing is not None
    assert routing.shape[0] == len(branch.token_ids)
    assert routing.reshape(routing.shape[0], -1).tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [100, 101],
        [102, 103],
        [104, 105],
        [106, 107],
        [200, 201],
        [202, 203],
        [204, 205],
        [206, 207],
        [206, 207],
    ]
    assert all(
        node.routed_experts is None or node.routed_experts.flags.owndata
        for node in trace.nodes
    )


def test_unbridged_turn_takes_the_row_at_the_offset_not_row_zero():
    """A turn that could not bridge sends `start = 0`, so the prefix's final position sits deep in
    the array rather than at row 0, and only `arr[off - 1]` finds it."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    sampled_node = trace.nodes[1]
    forwarded_rows = sampled_node.routed_experts[:-1].copy()

    _commit(trace, _TURN2, _engine_payload(_TURN2, None, 100), "a2")

    rows = sampled_node.routed_experts
    assert rows.shape[0] == len(sampled_node.token_ids)
    assert rows[-1].tolist() == [[108], [109]]
    assert bool((rows[:-1] == forwarded_rows).all())


def test_placeholder_replaced_when_the_next_turn_widens_to_uint16():
    """The engine picks uint8 or uint16 per response, so the real row can be wider than the node
    holding the placeholder; the node must widen rather than truncate or decline."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    sampled_node = trace.nodes[1]
    forwarded_rows = sampled_node.routed_experts[:-1].copy()

    _commit(trace, _TURN2, _routed_payload_uint16(4, 4, 300), "a2")

    rows = sampled_node.routed_experts
    assert rows.dtype == np.uint16
    assert rows[-1].tolist() == [[300], [301]]
    assert bool((rows[:-1] == forwarded_rows).all())
    branch = trace.branches[-1]
    assert branch.routed_experts is not None
    assert branch.routed_experts.shape[0] == len(branch.token_ids)


def test_placeholder_kept_when_routing_starts_at_the_first_new_token():
    """With `start == path_len` the array begins after the prefix, so `off` is 0 and there is no
    real row to install. Writing `arr[-1]` here would drop a row and silently disable replay."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    sampled_node = trace.nodes[1]
    before = sampled_node.routed_experts.copy()

    _commit(trace, _TURN2, _routed_payload(3, 5, 100), "a2")

    rows = sampled_node.routed_experts
    assert rows.shape == before.shape
    assert rows.shape[0] == len(sampled_node.token_ids)
    assert bool((rows == before).all())
    branch = trace.branches[-1]
    assert branch.routed_experts.shape[0] == len(branch.token_ids)


def test_placeholder_kept_when_the_routing_array_is_too_short():
    """A synthetic short array, one row shy of reaching the prefix's final position. No engine
    sends this, but the bound is what stops an off-by-one from splicing an empty slice."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    sampled_node = trace.nodes[1]
    before = sampled_node.routed_experts.copy()

    _commit(trace, _TURN2, _routed_payload(4, 0, 100), "a2")

    rows = sampled_node.routed_experts
    assert rows.shape == before.shape
    assert rows.shape[0] == len(sampled_node.token_ids)
    assert bool((rows == before).all())


def test_placeholder_kept_when_the_routing_axes_disagree():
    """A payload whose `[layers, top_k]` axes differ cannot be spliced into the node at all, so
    the node must be left alone rather than raising out of `commit`."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    sampled_node = trace.nodes[1]
    before = sampled_node.routed_experts.copy()

    _commit(trace, _TURN2, _engine_payload(_TURN2, _TURN1, 100, top_k=2), "a2")

    rows = sampled_node.routed_experts
    assert rows.shape == before.shape
    assert bool((rows == before).all())


def test_placeholder_kept_when_the_owner_has_no_routing():
    """A turn that ran without router replay leaves its nodes unset, so a later turn has no
    placeholder to replace and must not write one row into an otherwise empty node."""
    trace = _new_trace()
    _commit(trace, _TURN1, None, "a1")
    assert trace.nodes[1].routed_experts is None

    _commit(trace, _TURN2, _engine_payload(_TURN2, _TURN1, 100), "a2")

    assert trace.nodes[1].routed_experts is None
    assert trace.branches[-1].routed_experts is None


def test_placeholder_kept_when_retokenization_narrows_the_prefix():
    """Token-identity narrowing forks the prefix at the first divergent node, so the surviving
    prefix is shorter than the one `prepare_turn` resolved and its final position is no longer the
    one the array describes. Using the un-narrowed prefix here would corrupt a committed branch."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    sampled_node = trace.nodes[1]
    before = sampled_node.routed_experts.copy()

    drifted = {**_TURN2, "prompt_ids": [10, 11, 12, 20, 99, 30, 31]}
    _commit(trace, drifted, _engine_payload(drifted, _TURN1, 100), "a2")

    rows = sampled_node.routed_experts
    assert rows.shape == before.shape
    assert bool((rows == before).all())


def test_placeholder_replaced_on_a_deserialized_trace():
    """A trace off the msgpack wire holds `np.frombuffer` views over immutable bytes, so the
    replacement must not assume the node's array is writeable."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    restored = type(trace).model_validate(trace.model_dump())
    restored_node = restored.nodes[1]
    assert not restored_node.routed_experts.flags.writeable
    forwarded_rows = np.array(restored_node.routed_experts[:-1])

    _commit(restored, _TURN2, _engine_payload(_TURN2, _TURN1, 100), "a2")

    rows = restored_node.routed_experts
    assert rows.shape[0] == len(restored_node.token_ids)
    assert rows[-1].tolist() == [[100], [101]]
    assert bool((rows[:-1] == forwarded_rows).all())
    assert rows.flags.owndata


def _commit_with_stale_prefix(tool_span, parallel_routed, stale_routed):
    """A parallel request commits the tool message after `prepare_turn` resolved, so the stale
    turn's reconciliation appends that tool node to its prefix, past the assistant. The stale turn
    still bridges against turn 1, which is all it knew about when it was prepared."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    prompt = [_U1, _A1, _T1]
    pending = graph.prepare_turn(trace, prompt)
    assert pending.prefix_node_ids == [0, 1]

    prompt_ids = [10, 11, 12, 20, 21, 30, 31] if tool_span else [10, 11, 12, 20, 21]
    spans = [(0, 2), None, tool_span]
    graph.prepare_turn(trace, prompt).commit(
        _turn_response("a2", prompt_ids, [40, 41], spans, parallel_routed)
    )
    pending.commit(_turn_response("a3", prompt_ids, [50, 51], spans, stale_routed))
    assert trace.nodes[4].parent == 2
    return trace


def test_placeholder_kept_when_the_prefix_ends_in_a_tool_message():
    """The prefix's final position then belongs to a forwarded tool token whose row is already
    real, so nothing is this turn's to replace: not the tool row it points at, and not the
    assistant placeholder further back, which belongs to an earlier position."""
    trace = _new_trace()
    _commit(trace, _TURN1, _engine_payload(_TURN1, None, 0), "a1")
    prompt = [_U1, _A1, _T1]
    pending = graph.prepare_turn(trace, prompt)
    prompt_ids = [10, 11, 12, 20, 21, 30, 31]
    spans = [(0, 2), None, (5, 7)]
    graph.prepare_turn(trace, prompt).commit(
        _turn_response("a2", prompt_ids, [40, 41], spans, _routed_payload(4, 4, 200))
    )
    tool_rows = trace.nodes[2].routed_experts.copy()
    assistant_rows = trace.nodes[1].routed_experts.copy()

    pending.commit(
        _turn_response("a3", prompt_ids, [50, 51], spans, _routed_payload(4, 4, 100))
    )

    assert trace.nodes[4].parent == 2
    assert not trace.nodes[2].sampled
    assert bool((trace.nodes[2].routed_experts == tool_rows).all())
    assert bool((trace.nodes[1].routed_experts == assistant_rows).all())


def test_placeholder_replaced_past_a_tokenless_prefix_node():
    """A tool node the renderer gave no span to owns no sequence position, so the walk must skip
    it and repair the assistant node that really owns the prefix's final position."""
    trace = _commit_with_stale_prefix(
        tool_span=None,
        parallel_routed=None,
        stale_routed=_routed_payload(2, 4, 100),
    )

    assert trace.nodes[2].token_ids == []
    assert trace.nodes[1].routed_experts[-1].tolist() == [[100], [101]]


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
