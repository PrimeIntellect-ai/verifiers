import json
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
    assert (
        "provider missing expected-model; echoed prompt-secret client-secret auth-secret"
        in str(error)
    )
    assert calls == 1
    log_text = caplog.text
    assert "url=https://example.com/api/v1/chat/completions" in log_text
    assert "model='expected-model'" in log_text
    assert "status=404" in log_text
    assert "request_id=session-1" in log_text
    assert "provider-request" in log_text
    assert "body_preview=" not in log_text
    assert "provider missing" not in log_text
    for secret in (
        "prompt-secret",
        "client-secret",
        "auth-secret",
        "header-secret",
        "url-secret",
        "password",
    ):
        assert secret not in log_text


async def test_upstream_error_log_omits_echoed_url_and_json_escaped_payload(
    monkeypatch, caplog
):
    monkeypatch.setenv("TEST_VERIFIERS_API_KEY", "client-secret")
    payload = 'quoted "payload"\\path\nnext client-secret auth-secret'
    echoed_url = (
        "https://url-user:url-password@example.com/callback?token=url-token#fragment"
    )
    provider_error = json.dumps(
        {"error": f"invalid payload: {payload}; echoed URL: {echoed_url}"}
    )

    def upstream(request):
        return httpx.Response(400, request=request, text=provider_error)

    config = BaseClientConfig(
        base_url="https://example.com/api/v1", api_key_var="TEST_VERIFIERS_API_KEY"
    )
    client = EvalClient(config)
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = {
        "model": "expected-model",
        "messages": [{"role": "user", "content": payload}],
    }
    headers = httpx.Headers({"Authorization": "Bearer auth-secret"})
    try:
        with (
            caplog.at_level(logging.WARNING, logger="verifiers.v1.clients.eval"),
            pytest.raises(ProviderError) as caught,
        ):
            await client._request(
                "https://example.com/api/v1/chat/completions?key=request-url-secret",
                body,
                headers,
            )
    finally:
        await client.close()

    assert f"error={provider_error}" in str(caught.value)
    log_text = caplog.text
    assert "url=https://example.com/api/v1/chat/completions" in log_text
    assert "model='expected-model'" in log_text
    assert "status=400" in log_text
    assert "body_preview=" not in log_text
    assert "invalid payload" not in log_text
    for secret in (
        "client-secret",
        "auth-secret",
        "url-user",
        "url-password",
        "url-token",
        "request-url-secret",
        "quoted",
        "\\nnext",
    ):
        assert secret not in log_text


async def test_upstream_error_log_ignores_many_distinct_request_strings(
    monkeypatch, caplog
):
    monkeypatch.setenv("TEST_VERIFIERS_API_KEY", "client-secret")
    secrets = [f"prompt-secret-{index:04d}" for index in range(4096)]

    def upstream(request):
        return httpx.Response(
            429, request=request, text="provider rejected request without echoing it"
        )

    config = BaseClientConfig(
        base_url="https://example.com/api/v1", api_key_var="TEST_VERIFIERS_API_KEY"
    )
    client = EvalClient(config)
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = {
        "model": "expected-model",
        "messages": [{"role": "user", "content": secret} for secret in secrets],
    }
    try:
        with (
            caplog.at_level(logging.WARNING, logger="verifiers.v1.clients.eval"),
            pytest.raises(ProviderError) as caught,
        ):
            await client._request(
                "https://example.com/api/v1/chat/completions", body, httpx.Headers()
            )
    finally:
        await client.close()

    assert caught.value.status_code == 429
    assert "error=provider rejected request without echoing it" in str(caught.value)
    log_text = caplog.text
    assert "model='expected-model'" in log_text
    assert "status=429" in log_text
    assert "body_preview=" not in log_text
    assert "provider rejected request" not in log_text
    assert "client-secret" not in log_text
    assert all(secret not in log_text for secret in secrets)
