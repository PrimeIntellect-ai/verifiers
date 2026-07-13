"""Chat dialect response parsing."""

from verifiers.v1.dialects.chat import ChatDialect


def _completion(service_tier: str) -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "service_tier": service_tier,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_response_tolerates_unknown_service_tier():
    """Providers return service tiers outside OpenAI's fixed Literal (e.g. Prime's
    `provisioned`); the chat dialect must parse the completion rather than reject it, and keep
    the label rather than drop it."""
    dialect = ChatDialect()
    response = dialect.validate_response(_completion("provisioned"))
    assert response.service_tier == "provisioned"
    assert dialect.parse_response(response).message.content == "hi"
