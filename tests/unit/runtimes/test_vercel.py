"""The vercel runtime's egress rules and resources: which rules it can enforce, the
policy it sends, and the vCPU count it requests."""

import pytest

from verifiers.v1.runtimes import VercelSandboxConfig
from verifiers.v1.runtimes.vercel import _egress_policy, _vcpus


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
