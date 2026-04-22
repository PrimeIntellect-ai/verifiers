import pytest
from openai.types.chat import ChatCompletion

import verifiers as vf
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient


@pytest.mark.asyncio
async def test_error_envelope_without_choices_raises_provider_message():
    client = OpenAIChatCompletionsClient(object())
    response = ChatCompletion.model_construct(
        id="chatcmpl-error",
        model="openai/gpt-5.4",
        choices=None,
        error={
            "message": "Request too large for gpt-5.4",
            "code": 429,
        },
    )

    with pytest.raises(vf.InvalidModelResponseError, match="Request too large"):
        await client.raise_from_native_response(response)
