import logging
from typing import Optional

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletionToolParam

import verifiers as vf
from verifiers.types import Messages

_TOKENS_CLIENT: AsyncOpenAI | None = None

logger = logging.getLogger(__name__)


async def tokenize_vllm(
    client: AsyncOpenAI,
    messages: Messages,
    tools: list[ChatCompletionToolParam] | None,
    model: str,
    extra_kwargs: dict = {},
    **kwargs,
) -> list[int]:
    """Tokenize messages using the vLLM /tokenize API."""

    global _TOKENS_CLIENT
    if _TOKENS_CLIENT is None:
        logger.debug("Lazily copying OpenAI client for requests to /tokenize API")
        url_without_v1 = str(client.base_url).replace("/v1/", "")
        _TOKENS_CLIENT = client.copy(base_url=url_without_v1)
    tokens_client = _TOKENS_CLIENT

    # Copy from vllm/entrypoints/openai/protocol.py
    class TokenizeResponse(BaseModel):
        count: int
        max_model_len: int
        tokens: list[int]
        token_strs: Optional[list[str]] = None

    try:
        if isinstance(messages, str):
            body = dict(
                model=model,
                prompt=messages,
                **extra_kwargs,
            )
            tokenize_response = await tokens_client.post(
                "/tokenize", body=body, cast_to=TokenizeResponse
            )
        else:
            body = dict(
                model=model,
                messages=messages,
                tools=tools,
                **extra_kwargs,
            )
            tokenize_response = await tokens_client.post(
                "/tokenize", body=body, cast_to=TokenizeResponse
            )
        return tokenize_response.tokens
    except Exception as e:
        raise vf.ModelError(e)
