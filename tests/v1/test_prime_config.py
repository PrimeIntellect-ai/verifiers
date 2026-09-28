"""Prime CLI config resolution for the default pinference client (via the prime SDK)."""

import json
from pathlib import Path

import prime_sandboxes.core.config as sdk_config
import pytest

from verifiers.v1.configs.client import (
    PRIME_TEAM_ID_HEADER,
    EvalClientConfig,
    resolve_api_key,
)
from verifiers.v1.utils.prime import ensure_prime_auth, load_prime_config


@pytest.fixture
def home(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "home"
    config_dir = home / ".prime"
    (config_dir / "environments").mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps({"api_key": "global-key", "team_id": "global-team"})
    )
    (config_dir / "environments" / "customer.json").write_text(
        json.dumps({"api_key": "customer-key", "team_id": "customer-team"})
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    for name in (
        "PRIME_API_KEY",
        "PRIME_TEAM_ID",
        "PRIME_CONTEXT",
        "PRIME_INFERENCE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    repo = home / "code" / "repo"
    repo.mkdir(parents=True)
    monkeypatch.chdir(repo)
    return home


def test_global_config(home) -> None:
    client = EvalClientConfig()
    assert client.headers[PRIME_TEAM_ID_HEADER] == "global-team"
    assert resolve_api_key(client) == "global-key"
    assert load_prime_config()["api_key"] == "global-key"


def test_prime_context_selects_saved_context(home, monkeypatch) -> None:
    monkeypatch.setenv("PRIME_CONTEXT", "customer")
    client = EvalClientConfig()
    assert client.headers[PRIME_TEAM_ID_HEADER] == "customer-team"
    assert resolve_api_key(client) == "customer-key"


def test_empty_team_env_means_personal_account(home, monkeypatch) -> None:
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    assert PRIME_TEAM_ID_HEADER not in EvalClientConfig().headers


def test_missing_context_is_an_error_not_a_fallback(home, monkeypatch) -> None:
    monkeypatch.setenv("PRIME_CONTEXT", "missing")
    with pytest.raises(SystemExit, match="invalid prime config"):
        ensure_prime_auth()


@pytest.mark.skipif(
    not hasattr(sdk_config, "find_local_context_file"),
    reason="prime-sandboxes predates directory contexts (prime#967)",
)
def test_directory_context_pins_team(home) -> None:
    (Path.cwd() / ".prime").mkdir()
    (Path.cwd() / ".prime" / "context.json").write_text(
        json.dumps({"context": "customer", "team_id": "edison-team"})
    )
    client = EvalClientConfig()
    assert client.headers[PRIME_TEAM_ID_HEADER] == "edison-team"
    assert resolve_api_key(client) == "customer-key"
