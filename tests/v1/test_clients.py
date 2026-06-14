import pytest

from verifiers.v1.clients.config import EvalClientConfig, resolve_client
from verifiers.v1.clients.eval import EvalClient


@pytest.mark.asyncio
async def test_resolve_client_uses_prime_config(monkeypatch):
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setattr(
        "verifiers.v1.clients.config.load_prime_config",
        lambda: {"api_key": "prime-key", "team_id": "prime-team"},
    )

    client = resolve_client(EvalClientConfig())

    assert isinstance(client, EvalClient)
    assert client.api_key == "prime-key"
    assert client.http.headers["X-Prime-Team-ID"] == "prime-team"
    await client.close()
