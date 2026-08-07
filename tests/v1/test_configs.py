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
from verifiers.v1.runtimes import E2BConfig, E2BRuntime

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    # resolved to a v1 taskset or a v0 env id
    assert config.env.taskset.id or config.id


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"cpu": 1.5}, "whole CPU cores"),
        ({"cpu": 3}, "1 or an even number"),
        ({"memory": 1.0009765625}, "even whole number of MB"),
        ({"disk": 0}, "greater than 0"),
    ],
)
def test_e2b_config_rejects_unsupported_resources(values: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        E2BConfig.model_validate(values)


@pytest.mark.parametrize("cpu", [1, 2, 4])
def test_e2b_config_accepts_supported_cpu_counts(cpu: int) -> None:
    assert E2BConfig(cpu=cpu).cpu == cpu


def test_e2b_runtime_revalidates_task_resource_updates() -> None:
    config = E2BConfig().model_copy(update={"cpu": 3})

    with pytest.raises(ValueError, match="1 or an even number"):
        E2BRuntime(config)
