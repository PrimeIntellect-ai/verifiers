from typing import Any

from renderers.base import ParsedResponse
from renderers.proxy import RenderingProxy


class _FakeToolCallRenderer:
    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
        assert completion_ids == [101, 102]
        return ParsedResponse(
            content="",
            reasoning_content="I should use a tool.",
            tool_calls=[
                {
                    "function": {
                        "name": "echo",
                        "arguments": {"text": "hello"},
                    }
                }
            ],
        )


def test_proxy_converts_parsed_tool_call_to_openai_chat_response():
    proxy = object.__new__(RenderingProxy)
    proxy._renderer = _FakeToolCallRenderer()

    response = RenderingProxy._to_chat_response(
        proxy,
        {
            "id": "gen_1",
            "model": "test-model",
            "choices": [
                {
                    "token_ids": [101, 102],
                    "logprobs": [-0.1, -0.2],
                    "finish_reason": "tool_calls",
                    "routed_experts": [[[1]], [[2]]],
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        [1, 2, 3],
    )

    choice: dict[str, Any] = response["choices"][0]
    assert response["prompt_token_ids"] == [1, 2, 3]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["token_ids"] == [101, 102]
    assert choice["routed_experts"] == [[[1]], [[2]]]
    assert choice["message"]["reasoning_content"] == "I should use a tool."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text": "hello"}'},
        }
    ]
    assert choice["logprobs"] == {
        "content": [
            {"token": "", "logprob": -0.1},
            {"token": "", "logprob": -0.2},
        ]
    }
