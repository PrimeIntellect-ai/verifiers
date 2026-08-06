"""Chat dialect wire parsing."""

from verifiers.v1.dialects.chat import ChatStreamParser


def test_stream_parser_tolerates_out_of_enum_finish_reason() -> None:
    """Providers surface upstream failures as `finish_reason: "error"`, outside the SDK's
    Literal; the assembled completion must still validate, with the out-of-enum value
    normalized to None like the non-stream path."""
    parser = ChatStreamParser()
    parser.feed(
        b'data: {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 1,'
        b' "model": "m", "choices": [{"index": 0, "delta": {"content": "partial"}}]}\n\n'
    )
    parser.feed(
        b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}\n\n'
    )

    response = parser.finish()

    assert response.finish_reason is None
    assert response.message.content == "partial"


def test_stream_parser_keeps_enum_finish_reasons() -> None:
    parser = ChatStreamParser()
    parser.feed(
        b'data: {"id": "cmpl-2", "object": "chat.completion.chunk", "created": 1,'
        b' "model": "m", "choices": [{"index": 0, "delta": {"content": "done"},'
        b' "finish_reason": "stop"}]}\n\n'
    )

    assert parser.finish().finish_reason == "stop"
