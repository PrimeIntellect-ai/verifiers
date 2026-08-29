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
from verifiers.v1.runtimes.e2b import _egress_update

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


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


@pytest.mark.parametrize(
    "rule",
    ["https://api.example.com", "example.com:8443", "*.example.com"],
)
def test_e2b_config_rejects_unenforceable_egress_rules(rule: str) -> None:
    with pytest.raises(ValueError, match="hostnames, IPs, or CIDR blocks"):
        E2BConfig(allow=[rule])
    with pytest.raises(ValueError, match="hostnames, IPs, or CIDR blocks"):
        E2BConfig(block=[rule])


def test_e2b_egress_update_states_the_complete_policy() -> None:
    routes = ["https://tunnel.example.com/intercept"]

    unrestricted = _egress_update(E2BConfig(), None)
    assert unrestricted == {"allow_internet_access": True}

    blocklist = _egress_update(E2BConfig(block=["evil.example.com"]), routes)
    assert blocklist == {"deny_out": ["evil.example.com"]}

    allowlist = _egress_update(E2BConfig(allow=["api.example.com"]), routes)
    assert allowlist == {
        "allow_out": ["tunnel.example.com", "api.example.com"],
        "deny_out": ["0.0.0.0/0"],
    }

    framework_only = _egress_update(E2BConfig(allow=[]), routes)
    assert framework_only == {
        "allow_out": ["tunnel.example.com"],
        "deny_out": ["0.0.0.0/0"],
    }

    no_routes = _egress_update(E2BConfig(allow=[]), [])
    assert no_routes == {"allow_internet_access": False}
