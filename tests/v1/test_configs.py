"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run vf-eval @ <file>` v1 configs (EvalConfig).
"""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.runtimes import E2BConfig, VercelSandboxConfig
from verifiers.v1.runtimes.e2b import _egress_update
from verifiers.v1.runtimes.vercel import _egress_policy, _vcpus

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


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


def test_e2b_egress_update_states_the_complete_policy() -> None:
    # The security-critical policy table: whether a "restricted" sandbox actually
    # gets a deny-everything floor, with framework routes kept reachable. No e2e
    # placement runs network-restricted, so this table is pinned here.
    routes = ["https://tunnel.example.com/intercept"]

    unrestricted = _egress_update(E2BConfig(), None)
    assert unrestricted == {"allow_internet_access": True}

    blocklist = _egress_update(E2BConfig(block=["203.0.113.0/24"]), routes)
    assert blocklist == {"deny_out": ["203.0.113.0/24"]}

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


def test_vercel_config_rejects_unenforceable_egress_rules() -> None:
    for rule in ("https://api.example.com", "example.com:8443"):
        with pytest.raises(ValueError, match="allow rules"):
            VercelSandboxConfig(allow=[rule])
    with pytest.raises(ValueError, match="block rules must be IP addresses or CIDR"):
        VercelSandboxConfig(block=["example.com"])


@pytest.mark.parametrize(
    "cpu,memory,vcpus", [(1, 2, 1), (1, 3, 2), (3, 1, 4), (0.5, 0.5, 1), (8, 32, 16)]
)
def test_vercel_vcpus(cpu: float, memory: float, vcpus: int) -> None:
    assert _vcpus(cpu, memory) == vcpus


def test_vercel_egress_policy_states_the_complete_policy() -> None:
    pytest.importorskip("vercel.sandbox")
    routes = ["https://tunnel.example.com/v1", "http://127.0.0.1:9000/mcp"]

    assert _egress_policy(VercelSandboxConfig(), None).mode == "allow-all"

    blocklist = _egress_policy(VercelSandboxConfig(block=["203.0.113.0/24"]), routes)
    assert list(blocklist.allow) == ["*"]
    assert blocklist.subnets.deny == ("203.0.113.0/24",)

    allowlist = _egress_policy(
        VercelSandboxConfig(allow=["api.example.com", "10.0.0.0/8"]), routes
    )
    assert list(allowlist.allow) == ["tunnel.example.com", "api.example.com"]
    assert allowlist.subnets.allow == ("10.0.0.0/8",)

    framework_only = _egress_policy(VercelSandboxConfig(allow=[]), routes)
    assert list(framework_only.allow) == ["tunnel.example.com"]
    assert framework_only.subnets is None

    assert _egress_policy(VercelSandboxConfig(allow=[]), []).mode == "deny-all"
