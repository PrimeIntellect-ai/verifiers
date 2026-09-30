"""An unaccounted, empty Responses completion must be a provider failure."""

import asyncio
import json

import httpx
import pytest

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.errors import ProviderError
from verifiers.v1.types import SamplingConfig


def test_empty_unaccounted_responses_reply_is_retryable() -> None:
    async def check() -> None:
        empty = {
            "id": "resp-empty",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [],
        }
        accounted = {
            **empty,
            "id": "resp-accounted",
            "usage": {"input_tokens": 12, "output_tokens": 0, "total_tokens": 12},
        }
        tool = {
            **empty,
            "id": "resp-tool",
            "output": [
                {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"}
            ],
        }
        replies = iter([empty, accounted, tool])

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(replies), request=request)

        client = EvalClient(EvalClientConfig(base_url="https://example.test/v1", api_key_var="R01_TEST_KEY"))
        await client.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            client.client = http_client
            dialect = ResponsesDialect()
            sampling = SamplingConfig()
            with pytest.raises(ProviderError, match="response_id=resp-empty") as caught:
                await client.get_response(dialect, {"model": "test-model", "input": "hi"}, sampling)
            assert caught.value.status_code == 502

            valid_empty = await client.get_response(
                dialect, {"model": "test-model", "input": "hi"}, sampling
            )
            assert valid_empty.finish_reason == "stop"
            assert valid_empty.message.content is None
            assert valid_empty.usage is not None
            tool_reply = await client.get_response(
                dialect, {"model": "test-model", "input": "hi"}, sampling
            )
            assert [call.name for call in tool_reply.message.tool_calls or []] == ["lookup"]

            parser = dialect.stream_parser()
            parser.feed(
                f"data: {json.dumps({'type': 'response.completed', 'response': empty})}\n\n".encode()
            )
            with pytest.raises(ProviderError, match="response_id=resp-empty"):
                parser.finish()

    asyncio.run(check())
