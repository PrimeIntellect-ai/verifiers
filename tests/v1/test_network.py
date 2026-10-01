"""The system-prompt notice a rollout under a restricted network policy carries."""

from verifiers.v1.rollout import network_notice
from verifiers.v1.runtimes import NetworkPolicyConfig


def test_network_notice_names_the_restriction() -> None:
    disabled = network_notice(NetworkPolicyConfig(allow=[]))
    assert "Internet access is disabled" in disabled
    assert "do not try to work around it" in disabled
    allowed = network_notice(NetworkPolicyConfig(allow=["pypi.org", "github.com"]))
    assert "Only these destinations are reachable: pypi.org, github.com" in allowed
    blocked = network_notice(NetworkPolicyConfig(block=["example.com"]))
    assert "blocked and requests to them will fail: example.com" in blocked
