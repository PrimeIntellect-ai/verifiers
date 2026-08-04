from unittest.mock import AsyncMock

import httpx
import pytest

from verifiers.utils.client_utils import (
    post_chat_completion_with_routed_experts_sidecar,
)


@pytest.mark.asyncio
async def test_chat_completion_propagates_upstream_http_error() -> None:
    """Provider error bodies must not be parsed as successful completions."""
    response = httpx.Response(
        400,
        json={"error": {"message": "invalid request", "code": 400}},
        request=httpx.Request("POST", "https://provider.test/v1/chat/completions"),
    )
    client = type("Client", (), {"post": AsyncMock(return_value=response)})()

    with pytest.raises(httpx.HTTPStatusError, match="400 Bad Request"):
        await post_chat_completion_with_routed_experts_sidecar(
            client,
            "/chat/completions",
            body={"model": "test", "messages": []},
        )
