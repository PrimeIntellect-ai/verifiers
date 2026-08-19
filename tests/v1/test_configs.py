"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


def test_write_config_sanitizes_headers(tmp_path: Path) -> None:
    from verifiers.v1.cli.output import write_config
    from verifiers.v1.configs.client import EvalClientConfig

    config = EvalConfig(
        client=EvalClientConfig(
            headers={"X-Prime-Team-ID": "secret-team-123", "Authorization": "Bearer secret"}
        )
    )
    written_path = write_config(config, tmp_path)
    import json
    data = json.loads(written_path.read_text())
    assert data["client"]["headers"] == {
        "X-Prime-Team-ID": "<redacted>",
        "Authorization": "<redacted>",
    }

