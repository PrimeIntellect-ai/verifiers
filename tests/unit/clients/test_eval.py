import logging

import httpx
import pytest

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.client import BaseClientConfig
from verifiers.v1.errors import ProviderError


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_upstream_not_found_preserves_route_model_and_provider_error(
    monkeypatch, caplog, stream
):
    monkeypatch.setenv("TEST_VERIFIERS_API_KEY", "client-secret")
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            404,
            request=request,
            text="provider missing expected-model; echoed prompt-secret client-secret auth-secret",
            headers={"X-Request-ID": "provider-request", "X-Secret": "header-secret"},
        )

    config = BaseClientConfig(
        base_url="https://example.com/api/v1", api_key_var="TEST_VERIFIERS_API_KEY"
    )
    client = EvalClient(config)
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = {
        "model": "expected-model",
        "messages": [{"role": "user", "content": "prompt-secret"}],
    }
    headers = httpx.Headers(
        {"Authorization": "Bearer auth-secret", "X-Session-ID": "session-1"}
    )
    try:
        with (
            caplog.at_level(logging.WARNING, logger="verifiers.v1.clients.eval"),
            pytest.raises(ProviderError) as caught,
        ):
            await client._request(
                "https://user:password@example.com/api/v1/chat/completions?key=url-secret",
                body,
                headers,
                stream=stream,
            )
    finally:
        await client.close()
    error = caught.value
    assert error.status_code == 404
    assert "url=https://example.com/api/v1/chat/completions" in str(error)
    assert "model='expected-model'" in str(error)
    assert "provider missing expected-model" in str(error)
    assert calls == 1
    assert "request_id=session-1" in caplog.text
    assert "provider-request" in caplog.text
    assert "body_preview=" in caplog.text
    for secret in (
        "prompt-secret",
        "client-secret",
        "auth-secret",
        "header-secret",
        "url-secret",
        "password",
    ):
        assert secret not in caplog.text
