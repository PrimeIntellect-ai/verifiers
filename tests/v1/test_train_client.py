import verifiers.v1 as vf
from renderers import RenderedTokens

from verifiers.v1 import graph
from verifiers.v1.clients.train import TrainClient
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.types import TurnTokens, content_to_parts


class _FakeRenderer:
    supports_tools = True

    def get_stop_token_ids(self):
        return []

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        assert previous_prompt_ids == [10, 11, 12]
        assert previous_completion_ids == [20, 21]
        assert [m["role"] for m in new_messages] == ["user"]
        return RenderedTokens(
            token_ids=[10, 11, 12, 20, 21, 30, 31, 32],
            message_indices=[-1, -1, -1, -1, -1, 0, 0, -1],
            sampled_mask=[False] * 8,
            is_content=[False, False, False, False, False, True, True, False],
            message_roles=["user"],
        )


def test_chat_dialect_recovers_tool_message_name_from_assistant_call():
    messages, _ = ChatDialect().parse_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "result"},
            ]
        }
    )

    assert isinstance(messages[1], vf.ToolMessage)
    assert messages[1].name == "lookup"
    assert message_to_wire(messages[1])["name"] == "lookup"


async def test_train_client_uses_prepared_turn_for_renderer_bridge(monkeypatch):
    trace = vf.Trace(task=vf.Task(idx=0, instruction="x"))
    user = vf.UserMessage(content="u1")
    assistant = vf.AssistantMessage(content="a1")
    graph.add_turn(
        trace,
        [user],
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=assistant,
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11, 12],
                completion_ids=[20, 21],
                message_spans=[(0, 2)],
            ),
        ),
    )
    user2 = vf.UserMessage(content="u2")
    prompt = [user, assistant, user2]
    turn = graph.prepare_turn(trace, prompt)
    captured = {}

    async def fake_maybe_offload(renderer, fn):
        return fn()

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "request_id": "r",
            "prompt_ids": kwargs["prompt_ids"],
            "completion_ids": [40],
            "completion_logprobs": [-0.1],
            "content": "a2",
            "finish_reason": "stop",
            "prompt_attribution": kwargs["prompt_attribution"],
        }

    monkeypatch.setattr("renderers.client._maybe_offload", fake_maybe_offload)
    monkeypatch.setattr("renderers.client.generate", fake_generate)

    client = TrainClient(openai=None)  # type: ignore[arg-type]
    monkeypatch.setattr(client, "_renderer_pool", lambda model: _FakeRenderer())

    response = await client.get_response(
        ChatDialect(),
        {"messages": [message_to_wire(m) for m in prompt]},
        "model",
        vf.SamplingConfig(),
        session_id="trace-id",
        turn=turn,
    )

    assert response.message.content == "a2"
    assert captured["prompt_ids"] == [10, 11, 12, 20, 21, 30, 31, 32]
    assert captured["sampling_params"]["routed_experts_prompt_start"] == 4
    assert response.tokens.message_spans == [
        None,
        None,
        (5, 7),
    ]


async def test_train_client_does_not_bridge_multimodal_prompts(monkeypatch):
    trace = vf.Trace(task=vf.Task(idx=0, instruction="x"))
    user = vf.UserMessage(
        content=content_to_parts(
            [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}},
            ]
        )
    )
    assistant = vf.AssistantMessage(content="seen")
    graph.add_turn(
        trace,
        [user],
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=assistant,
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11, 12],
                completion_ids=[20, 21],
                message_spans=[(0, 2)],
            ),
        ),
    )
    prompt = [user, assistant, vf.UserMessage(content="next")]
    turn = graph.prepare_turn(trace, prompt)
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "request_id": "r",
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4],
            "completion_logprobs": [-0.1],
            "content": "done",
            "finish_reason": "stop",
        }

    class NoBridgeRenderer(_FakeRenderer):
        def bridge_to_next_turn(self, *args, **kwargs):
            raise AssertionError("multimodal prompts should not bridge in this PR")

    monkeypatch.setattr("renderers.client.generate", fake_generate)
    client = TrainClient(openai=None)  # type: ignore[arg-type]
    monkeypatch.setattr(client, "_renderer_pool", lambda model: NoBridgeRenderer())

    response = await client.get_response(
        ChatDialect(),
        {"messages": [message_to_wire(m) for m in prompt]},
        "model",
        vf.SamplingConfig(),
        turn=turn,
    )

    assert response.message.content == "done"
    assert captured["prompt_ids"] is None
    assert captured["prompt_attribution"] is None
