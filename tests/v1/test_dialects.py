"""Dialect wire parsing of finish labels: providers emit labels outside the SDKs' closed
enums; widened wire fields plus the existing unknown -> None coercion absorb them."""

import logging
from collections.abc import Iterator

import pytest

from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.chat import ChatDialect, ChatStreamParser

CHAT_HEAD = (
    b'data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,'
    b'"model":"test-model","choices":[{"index":0,'
    b'"delta":{"role":"assistant","content":"Hello"}}]}\n\n'
)
CHAT_TERMINAL_ERROR = (
    b'data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,'
    b'"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"error"}],'
    b'"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
)


@pytest.fixture
def chat_warnings() -> Iterator[list[logging.LogRecord]]:
    """Capture records from the chat dialect's own logger. A handler directly on that
    logger: v1's logging setup turns off library propagation, so root-attached capture
    (caplog) would miss these once any other test has configured logging."""
    logger = logging.getLogger("verifiers.v1.dialects.chat")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_chat_dialect_absorbs_out_of_enum_finish_reason(
    chat_warnings: list[logging.LogRecord],
):
    """A stream ending with a provider-specific finish label still finishes: the
    assembled response parses, the unknown label records as None, and the weirdness
    is warned about."""
    parser = ChatStreamParser()
    parser.feed(CHAT_HEAD)
    parser.feed(CHAT_TERMINAL_ERROR)
    response = parser.finish()

    assert response.finish_reason is None
    assert response.message.content == "Hello"
    assert any("'error'" in r.getMessage() for r in chat_warnings)

    # The non-streamed path validates against the same response model.
    dialect = ChatDialect()
    completion = dialect.validate_response(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "error",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    assert dialect.parse_response(completion).finish_reason is None


def test_anthropic_dialect_absorbs_out_of_enum_stop_reason():
    """The Anthropic SDK closes `stop_reason` to a `Literal` too; the same validation
    boundary must not reject a gateway-specific label."""
    dialect = AnthropicDialect()
    message = dialect.validate_response(
        {
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "error",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )
    response = dialect.parse_response(message)
    assert response.finish_reason is None
    assert response.message.content == "Hello"


def test_stream_parser_keeps_enum_finish_reasons() -> None:
    """Known labels round-trip unchanged: the widening doesn't alter valid values."""
    parser = ChatStreamParser()
    parser.feed(
        b'data: {"id": "cmpl-2", "object": "chat.completion.chunk", "created": 1,'
        b' "model": "m", "choices": [{"index": 0, "delta": {"content": "done"},'
        b' "finish_reason": "stop"}]}\n\n'
    )

    assert parser.finish().finish_reason == "stop"
