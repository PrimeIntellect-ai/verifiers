"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.eval import EvalConfig

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    try:
        config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    except (
        ModuleNotFoundError
    ) as e:  # workspace-only plugin (e.g. research-environments)
        pytest.skip(f"config needs a package this checkout doesn't ship: {e}")
    # Resolved to a v1 taskset, a v0 env id, or an explicit topology (seeds under
    # `topology.taskset`).
    assert config.taskset.id or config.id or config.topology is not None
