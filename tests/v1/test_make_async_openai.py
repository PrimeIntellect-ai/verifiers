"""`make_async_openai` / `_resolve_api_key`: the v1 ClientConfig -> raw AsyncOpenAI builder
shared with `resolve_client`, plus the v0 `setup_openai_client` guard that rejects a v1 config.

The api-key resolution is the bit prime-rl needs for ad-hoc engine calls (e.g. prefill scoring a
teacher via `renderers.client.score`): the configured env var, with the Prime CLI
config key as a fallback only for the default `PRIME_API_KEY` var on a Prime inference host.
"""

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import config as cfg


@pytest.fixture(autouse=True)
def _no_prime_config(monkeypatch):
    # Default: no env key and an empty Prime config, so each test opts into exactly the
    # resolution path it exercises. `apply_prime_config` runs at construction and also reads
    # this, so patch it before any config is built.
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "load_prime_config", lambda: {})


def test_env_var_key_is_used(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "sk-from-env")
    client = vf.make_async_openai(
        cfg.EvalClientConfig(base_url="http://localhost:8000/v1")
    )
    assert client.api_key == "sk-from-env"


def test_prime_config_fallback_on_prime_host(monkeypatch):
    # No env var, default PRIME_API_KEY var, Prime inference host -> fall back to the CLI config
    # key. This is the path prime-rl's hand-rolled `os.environ.get(var) or "EMPTY"` skipped,
    # 401-ing a prime-hosted teacher with no env var set.
    monkeypatch.setattr(cfg, "load_prime_config", lambda: {"api_key": "sk-from-prime"})
    client = vf.make_async_openai(
        cfg.EvalClientConfig(base_url="https://api.pinference.ai/api/v1")
    )
    assert client.api_key == "sk-from-prime"


def test_no_fallback_off_prime_host():
    # The Prime fallback must NOT fire for a non-Prime host even on the default var.
    client = vf.make_async_openai(
        cfg.EvalClientConfig(base_url="http://localhost:8000/v1")
    )
    assert client.api_key == "EMPTY"


def test_no_fallback_for_custom_api_key_var(monkeypatch):
    # A custom api_key_var never triggers the Prime-config fallback, even on a Prime host.
    monkeypatch.setattr(cfg, "load_prime_config", lambda: {"api_key": "sk-from-prime"})
    client = vf.make_async_openai(
        cfg.EvalClientConfig(
            base_url="https://api.pinference.ai/api/v1", api_key_var="OTHER_KEY"
        )
    )
    assert client.api_key == "EMPTY"


def test_base_url_and_headers_threaded():
    client = vf.make_async_openai(
        cfg.EvalClientConfig(
            base_url="http://localhost:8000/v1", headers={"X-Data-Parallel-Rank": "3"}
        )
    )
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"
    assert client.default_headers.get("X-Data-Parallel-Rank") == "3"


def test_resolve_client_shares_resolution(monkeypatch):
    # resolve_client now routes through the same _resolve_api_key, so the TrainClient's
    # underlying AsyncOpenAI inherits the identical key resolution.
    monkeypatch.setenv("PRIME_API_KEY", "sk-shared")
    client = vf.resolve_client(
        cfg.TrainClientConfig(base_url="http://localhost:8000/v1")
    )
    assert isinstance(client, vf.clients.TrainClient)
    assert client.openai.api_key == "sk-shared"


def test_v0_setup_openai_client_rejects_v1_config():
    # The v0 helper crashes cryptically on a v1 config; the guard turns it into a clear error
    # pointing at the v1 builders.
    from verifiers.utils.client_utils import setup_anthropic_client, setup_openai_client

    with pytest.raises(TypeError, match="verifiers.v1"):
        setup_openai_client(cfg.EvalClientConfig(base_url="http://localhost:8000/v1"))
    with pytest.raises(TypeError, match="verifiers.v1"):
        setup_anthropic_client(
            cfg.TrainClientConfig(base_url="http://localhost:8000/v1")
        )
