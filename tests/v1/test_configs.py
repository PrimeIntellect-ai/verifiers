"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.eval import EvalConfig

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.taskset.id or config.id  # resolved to a v1 taskset or a v0 env id


def test_plugin_exports_resolve_and_build() -> None:
    config = EvalConfig.model_validate(
        {
            "taskset": {"id": "echo-v1", "phrases": ["hello"]},
            "harness": {"id": "default", "runtime": {"type": "subprocess"}},
        }
    )

    assert type(config.taskset).__name__ == "EchoConfig"
    assert type(config.harness).__name__ == "DefaultHarnessConfig"

    env = vf.Environment(config)
    assert type(env.taskset).__name__ == "EchoTaskset"
    assert type(env.harness).__name__ == "DefaultHarness"
    assert not hasattr(vf, "load_taskset")
    assert not hasattr(vf, "load_harness")
