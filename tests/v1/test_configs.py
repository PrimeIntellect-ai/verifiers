"""Runtime configuration validation."""

import pytest

from verifiers.v1.runtimes import E2BConfig
from verifiers.v1.runtimes.e2b import _egress_update


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
