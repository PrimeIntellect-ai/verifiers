from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError

from verifiers.clients.openai.openai_clients import (
    OAIChatCompletionsClient,
    OAIChatCompletionsTokenClient,
)
from verifiers.errors import OverlongPromptError


@pytest.mark.asyncio
async def test_chat_completions_client_maps_context_length_bad_request_to_overlong():
    raw_client = MagicMock()
    raw_client.chat = MagicMock()
    raw_client.chat.completions = MagicMock()
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    raw_client.chat.completions.create = AsyncMock(
        side_effect=BadRequestError(
            "bad request",
            response=httpx.Response(
                400,
                text="This model's maximum context length is 8192 tokens.",
                request=request,
            ),
            body=None,
        )
    )

    client = OAIChatCompletionsClient(raw_client)

    with pytest.raises(OverlongPromptError):
        await client.get_native_response(
            prompt=[{"role": "user", "content": "hello"}],
            model="test-model",
            sampling_args={},
            tools=None,
        )


@pytest.mark.asyncio
async def test_chat_completions_token_client_does_not_swallow_overlong_prompt():
    raw_client = MagicMock()
    client = OAIChatCompletionsTokenClient(raw_client)
    state = {"trajectory": []}

    with patch.object(
        OAIChatCompletionsClient,
        "get_native_response",
        AsyncMock(side_effect=OverlongPromptError()),
    ):
        with pytest.raises(OverlongPromptError):
            await client.get_native_response(
                prompt=[{"role": "user", "content": "hello"}],
                model="test-model",
                sampling_args={},
                tools=None,
                state=state,
            )
