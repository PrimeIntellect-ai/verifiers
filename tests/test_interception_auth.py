"""Tests for per-rollout authentication on the interception server.

Verifies that:
- Requests with valid tokens are accepted
- Requests with invalid/missing tokens are rejected (401)
- Unregistered rollout IDs are rejected (404)
- Graceful fallback: rollouts registered without a token skip auth
"""

import asyncio
from typing import Any

import pytest
from aiohttp import ClientSession

from verifiers.utils.interception_utils import (
    InterceptionServer,
    generate_interception_token,
)


@pytest.fixture
async def server():
    srv = InterceptionServer(port=0)
    await srv.start()
    yield srv
    await srv.stop()


def _chat_payload(content: str = "hello") -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
    }


async def _post(
    base: str,
    rollout_id: str,
    token: str | None = None,
    timeout: float = 0.5,
    payload: Any | None = None,
):
    """POST to a rollout endpoint, return (status, body) or 'timeout'."""
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with ClientSession() as session:
            async with session.post(
                f"{base}/rollout/{rollout_id}/v1/chat/completions",
                json=_chat_payload() if payload is None else payload,
                headers=headers,
                timeout=__import__("aiohttp").ClientTimeout(total=timeout),
            ) as resp:
                body = await resp.json()
                return resp.status, body
    except asyncio.TimeoutError:
        return "accepted", None


@pytest.mark.asyncio
async def test_valid_token_accepted(server: InterceptionServer):
    """Request with the correct bearer token is accepted."""
    token = generate_interception_token()
    server.register_rollout("rollout_auth_ok", auth_token=token)
    base = f"http://127.0.0.1:{server.port}"

    result = await _post(base, "rollout_auth_ok", token=token)
    # "accepted" means the server didn't reject — it's waiting for a model response
    assert result[0] == "accepted" or result[0] == 200

    server.unregister_rollout("rollout_auth_ok")


@pytest.mark.asyncio
async def test_missing_token_rejected(server: InterceptionServer):
    """Request with no Authorization header is rejected when auth is configured."""
    token = generate_interception_token()
    server.register_rollout("rollout_no_token", auth_token=token)
    base = f"http://127.0.0.1:{server.port}"

    status, body = await _post(base, "rollout_no_token", token=None)
    assert status == 401
    assert body["error"] == "Unauthorized"

    server.unregister_rollout("rollout_no_token")


@pytest.mark.asyncio
async def test_wrong_token_rejected(server: InterceptionServer):
    """Request with an incorrect bearer token is rejected."""
    token = generate_interception_token()
    server.register_rollout("rollout_bad_token", auth_token=token)
    base = f"http://127.0.0.1:{server.port}"

    status, body = await _post(base, "rollout_bad_token", token="wrong-token")
    assert status == 401
    assert body["error"] == "Unauthorized"

    server.unregister_rollout("rollout_bad_token")


@pytest.mark.asyncio
async def test_unknown_rollout_404(server: InterceptionServer):
    """Request to a non-existent rollout ID returns 404."""
    base = f"http://127.0.0.1:{server.port}"

    status, body = await _post(base, "rollout_nonexistent", token=None)
    assert status == 404


@pytest.mark.asyncio
async def test_no_token_graceful_fallback(server: InterceptionServer):
    """Rollout registered without a token accepts any request (backwards compat)."""
    server.register_rollout("rollout_no_auth")
    base = f"http://127.0.0.1:{server.port}"

    result = await _post(base, "rollout_no_auth", token=None)
    # Should be accepted (not 401), waiting for model response
    assert result[0] == "accepted" or result[0] == 200

    server.unregister_rollout("rollout_no_auth")


@pytest.mark.asyncio
async def test_cross_rollout_blocked(server: InterceptionServer):
    """Token for rollout A cannot be used to access rollout B."""
    token_a = generate_interception_token()
    token_b = generate_interception_token()
    server.register_rollout("rollout_a", auth_token=token_a)
    server.register_rollout("rollout_b", auth_token=token_b)
    base = f"http://127.0.0.1:{server.port}"

    # Use A's token to access B's endpoint
    status, body = await _post(base, "rollout_b", token=token_a)
    assert status == 401, "Cross-rollout access should be rejected"

    server.unregister_rollout("rollout_a")
    server.unregister_rollout("rollout_b")


@pytest.mark.asyncio
async def test_missing_messages_rejected(server: InterceptionServer):
    """Authenticated requests without messages return a 400 instead of crashing."""
    token = generate_interception_token()
    server.register_rollout("rollout_missing_messages", auth_token=token)
    base = f"http://127.0.0.1:{server.port}"

    status, body = await _post(
        base,
        "rollout_missing_messages",
        token=token,
        payload={"model": "test-model"},
    )
    assert status == 400
    assert body["error"] == "Request body must include 'messages'"

    server.unregister_rollout("rollout_missing_messages")


@pytest.mark.asyncio
async def test_non_object_body_rejected(server: InterceptionServer):
    """Authenticated requests must send a JSON object."""
    token = generate_interception_token()
    server.register_rollout("rollout_non_object", auth_token=token)
    base = f"http://127.0.0.1:{server.port}"

    status, body = await _post(
        base,
        "rollout_non_object",
        token=token,
        payload=["not", "an", "object"],
    )
    assert status == 400
    assert body["error"] == "Request body must be a JSON object"

    server.unregister_rollout("rollout_non_object")
