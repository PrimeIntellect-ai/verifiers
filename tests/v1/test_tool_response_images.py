import verifiers.v1 as vf
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.legacy import _to_v1_messages


def test_chat_tool_message_preserves_image_content_parts():
    body = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "screenshot"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]
    }

    prompt, _ = ChatDialect().parse_request(body)
    tool = prompt[0]

    assert isinstance(tool, vf.ToolMessage)
    assert isinstance(tool.content, list)
    assert tool.content[1].image_url.url == "data:image/png;base64,abc"
    assert message_to_wire(tool)["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,abc"
    )


def test_anthropic_tool_result_preserves_images():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "screenshot"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "abc",
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    }

    prompt, _ = AnthropicDialect().parse_request(body)
    tool = prompt[0]

    assert isinstance(tool, vf.ToolMessage)
    assert isinstance(tool.content, list)
    assert tool.content[1].image_url.url == "data:image/png;base64,abc"


def test_responses_function_call_output_preserves_images():
    body = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "output_text", "text": "screenshot"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc",
                    },
                ],
            }
        ]
    }

    prompt, _ = ResponsesDialect().parse_request(body)
    tool = prompt[0]

    assert isinstance(tool, vf.ToolMessage)
    assert isinstance(tool.content, list)
    assert tool.content[1].image_url.url == "data:image/png;base64,abc"


def test_legacy_tool_message_preserves_images():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "screenshot"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        }
    ]

    tool = _to_v1_messages(messages)[0]

    assert isinstance(tool, vf.ToolMessage)
    assert isinstance(tool.content, list)
    assert tool.content[1].image_url.url == "data:image/png;base64,abc"
