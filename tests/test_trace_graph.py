"""The message-graph trace (`verifiers/v1/graph.py`, `Trace.nodes`).

Pins the load-bearing invariants: the message hash mirrors `branching.same_message`; the
per-node token deltas concat back to each turn's exact `prompt_ids + completion_ids`;
branches fall out of the graph walk (and match the old `branching.segment`); and a resample
forks into two branches.
"""

import verifiers.v1 as vf
from verifiers.v1 import branching, graph
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SystemMessage,
    ToolCall,
    ToolMessage,
    TurnTokens,
    UserMessage,
    Usage,
)


def _resp(content, prompt_ids, completion_ids, spans, tool_calls=None):
    msg = AssistantMessage(content=content, tool_calls=tool_calls)
    return Response(
        id="r",
        created=0,
        model="m",
        message=msg,
        finish_reason="stop",
        usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=len(completion_ids)),
        tokens=TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=[0.0] * len(completion_ids),
            message_spans=spans,
        ),
    )


def _trace():
    return Trace[vf.Task](task=vf.Task(idx=0, instruction="go"))


def test_message_hash_mirrors_same_message():
    pairs = [
        (UserMessage(content="a"), UserMessage(content="a"), True),
        (UserMessage(content="a"), UserMessage(content="b"), False),
        (UserMessage(content="a"), SystemMessage(content="a"), False),  # role differs
        (AssistantMessage(content=None), AssistantMessage(content=""), True),  # None == ""
        (
            AssistantMessage(content="x", reasoning_content="r1"),
            AssistantMessage(content="x", reasoning_content="r2"),
            True,  # reasoning_content ignored
        ),
        (
            AssistantMessage(tool_calls=[ToolCall(id="1", name="f", arguments="{}")]),
            AssistantMessage(tool_calls=[ToolCall(id="1", name="f", arguments="{}")]),
            True,
        ),
        (
            AssistantMessage(tool_calls=[ToolCall(id="1", name="f", arguments="{}")]),
            AssistantMessage(tool_calls=[ToolCall(id="1", name="g", arguments="{}")]),
            False,
        ),
        (ToolMessage(tool_call_id="1", content="o"), ToolMessage(tool_call_id="1", content="o"), True),
        (ToolMessage(tool_call_id="1", content="o"), ToolMessage(tool_call_id="2", content="o"), False),
    ]
    for a, b, same in pairs:
        assert (graph.message_hash(a) == graph.message_hash(b)) == same == branching.same_message(a, b)


def test_token_deltas_concat_back_to_each_turn():
    S, U = SystemMessage(content="s"), UserMessage(content="u0")
    t = _trace()
    p0 = [0, 1, 2, 3, 4, 5, 6]
    graph.add_turn(t, [S, U], _resp("a0", p0, [100, 101], [(0, 2), (2, 5)]))
    A0, U1 = AssistantMessage(content="a0"), UserMessage(content="u1")
    p1 = [0, 1, 2, 3, 4, 5, 6, 100, 101, 102, 7, 8, 9, 10, 11]  # bridge: starts with p0+completion0
    graph.add_turn(t, [S, U, A0, U1], _resp("a1", p1, [200, 201], [(0, 2), (2, 5), (5, 10), (10, 13)]))

    turns = t.branches[0].turns
    assert list(turns[0].tokens.prompt_ids) == p0
    assert list(turns[0].tokens.completion_ids) == [100, 101]
    # each turn's reconstructed prompt+completion equals exactly what the model saw
    assert list(turns[1].tokens.prompt_ids) + list(turns[1].tokens.completion_ids) == p1 + [200, 201]


def test_branches_match_segment_and_resample_forks():
    S, U = SystemMessage(content="s"), UserMessage(content="u0")
    t = _trace()
    graph.add_turn(t, [S, U], _resp("a0", [0, 1, 2, 3, 4, 5, 6], [100, 101], [(0, 2), (2, 5)]))
    A0, U1 = AssistantMessage(content="a0"), UserMessage(content="u1")
    p1 = [0, 1, 2, 3, 4, 5, 6, 100, 101, 102, 7, 8, 9, 10, 11]
    graph.add_turn(t, [S, U, A0, U1], _resp("a1", p1, [200, 201], [(0, 2), (2, 5), (5, 10), (10, 13)]))
    assert t.num_branches == 1 and t.num_turns == 2

    # graph branches agree with the old segment() over the reconstructed turns
    groups = branching.segment(t.trajectory)
    assert len(groups) == t.num_branches
    assert sum(len(g) for g in groups) == t.num_turns

    # a resample of [S, U] forks a second branch at the shared prefix
    graph.add_turn(t, [S, U], _resp("a0b", [0, 1, 2, 3, 4, 5, 6], [300, 301], [(0, 2), (2, 5)]))
    assert t.num_branches == 2
