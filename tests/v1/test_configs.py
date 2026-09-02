"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig).
"""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.runtimes import E2BConfig, E2BRuntime
from verifiers.v1.runtimes.e2b import _egress_update

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.toml"))


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
    [
        "",
        "https://api.example.com",
        "example.com:8443",
        "example.com/path",
        "user@example.com",
        "10.0.0.0/99",
        "example..com",
    ],
)
def test_e2b_config_rejects_unenforceable_egress_rules(rule: str) -> None:
    with pytest.raises(ValueError, match="allow rules"):
        E2BConfig(allow=[rule])


@pytest.mark.parametrize("rule", ["api.example.com", "*.example.com"])
def test_e2b_config_rejects_hostname_block_rules(rule: str) -> None:
    with pytest.raises(ValueError, match="block rules must be IP addresses or CIDR"):
        E2BConfig(block=[rule])


@pytest.mark.parametrize(
    "rule",
    [
        "api.example.com",
        "*.example.com",
        "10.0.0.1",
        "10.0.0.0/24",
        "2001:db8::1",
        "2001:db8::/32",
    ],
)
def test_e2b_config_accepts_supported_egress_rules(rule: str) -> None:
    assert E2BConfig(allow=[rule]).allow == [rule]


@pytest.mark.parametrize(
    "rule", ["10.0.0.1", "10.0.0.0/24", "2001:db8::1", "2001:db8::/32"]
)
def test_e2b_config_accepts_supported_block_rules(rule: str) -> None:
    assert E2BConfig(block=[rule]).block == [rule]


def test_e2b_command_combines_runtime_and_process_environments() -> None:
    runtime = E2BRuntime(E2BConfig())
    runtime.env = {
        "RUNTIME_ONLY": "runtime",
        "OVERRIDDEN": "runtime",
        "PATH": "/runtime/bin",
    }

    command, env = runtime._command(
        ["printenv"],
        {
            "PROCESS_ONLY": "process",
            "OVERRIDDEN": "process",
            "PATH": "/process/bin",
        },
    )

    assert command == 'export PATH="$VF_RUNTIME_PATH"; exec printenv'
    assert env == {
        "RUNTIME_ONLY": "runtime",
        "PROCESS_ONLY": "process",
        "OVERRIDDEN": "process",
        "VF_RUNTIME_PATH": "/process/bin",
    }


@pytest.mark.asyncio
async def test_e2b_teardown_retries_after_a_failed_kill() -> None:
    class FlakySandbox:
        def __init__(self) -> None:
            self.calls = 0

        async def kill(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient failure")

    runtime = E2BRuntime(E2BConfig())
    sandbox = FlakySandbox()
    runtime._sandbox = sandbox

    await runtime.teardown()
    assert runtime._sandbox is sandbox

    await runtime.teardown()
    assert runtime._sandbox is None


def test_e2b_egress_update_states_the_complete_policy() -> None:
    routes = ["https://tunnel.example.com/intercept"]

    unrestricted = _egress_update(E2BConfig(), None)
    assert unrestricted == {"allow_internet_access": True}

    blocklist = _egress_update(E2BConfig(block=["203.0.113.0/24"]), routes)
    assert blocklist == {
        "allow_out": ["tunnel.example.com"],
        "deny_out": ["203.0.113.0/24"],
    }

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
