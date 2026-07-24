import base64

import numpy as np

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.dialects.responses import OpenAIResponse, ResponsesDialect
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
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x"))
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
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x"))
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
    trace = vf.Trace(task=vf.TraceTask(type="Task", data=task))
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


def test_responses_custom_tool_calls_are_typed():
    dialect = ResponsesDialect()
    patch = "*** Begin Patch\n*** End Patch"
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Applying the fix."}],
        },
        {
            "type": "custom_tool_call",
            "call_id": "call_0",
            "name": "apply_patch",
            "input": patch,
        },
    ]

    response = dialect.parse_response(OpenAIResponse.model_validate({"output": output}))
    prompt, _ = dialect.parse_request(
        {
            "input": [
                {"role": "user", "content": "Fix it"},
                *output,
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_0",
                    "output": "Done!",
                },
            ]
        }
    )

    call = vf.ToolCall(id="call_0", name="apply_patch", arguments=patch)
    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls == [call]
    assert prompt[1].tool_calls == [call]
    assert prompt[2] == vf.ToolMessage(tool_call_id="call_0", content="Done!")


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
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x"))
    )
    first_turn(linear)
    second_turn(linear, [1, 2, 3, 4, 5, 6, 7])
    assert linear.num_branches == 1
    assert linear.branches[0].token_ids == [1, 2, 3, 4, 5, 6, 7, 8]

    # Break: the assistant turn retokenizes (4 -> 99), so prompt_ids diverge at that node.
    broken = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x"))
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
    trace = vf.Trace(task=vf.TraceTask(type="Task", data=task))
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
