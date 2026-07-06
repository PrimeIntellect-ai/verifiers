"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
V0 eval / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.clients.config import (
    DEFAULT_PRIME_INFERENCE_URL,
    EvalClientConfig,
    resolve_api_key,
)
from verifiers.v1.types import EnvId

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.taskset.id or config.id  # resolved to a v1 taskset or a v0 env id


def test_prime_client_context_comes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PRIME_INFERENCE_URL", "https://api.pinference.ai/api/v1")
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "prime-key")

    config = EvalClientConfig()

    assert config.base_url == "https://api.pinference.ai/api/v1"
    assert config.headers == {"X-Prime-Team-ID": "team-123"}
    assert resolve_api_key(config) == "prime-key"


def test_prime_client_ignores_prime_profile(monkeypatch, tmp_path: Path) -> None:
    prime_dir = tmp_path / ".prime"
    prime_dir.mkdir()
    (prime_dir / "config.json").write_text(
        """
{"api_key": "profile-key", "team_id": "profile-team", "inference_url": "https://profile.example/v1"}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("PRIME_INFERENCE_URL", raising=False)

    config = EvalClientConfig()

    assert config.base_url == DEFAULT_PRIME_INFERENCE_URL
    assert config.headers == {}
    assert resolve_api_key(config) == "EMPTY"


def test_environment_ids_are_local_packages() -> None:
    with pytest.raises(ValidationError, match="locally importable package"):
        TypeAdapter(EnvId).validate_python("owner/environment@1.2.3")
