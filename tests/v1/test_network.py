"""The execution network policy's model-facing side: the eval entrypoints' block-by-default
and the system-prompt notice a restricted rollout carries."""

import verifiers.v1 as vf
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.rollout import network_notice
from verifiers.v1.runtimes import (
    ModalConfig,
    NetworkPolicyConfig,
    PrimeConfig,
    SubprocessConfig,
)


def env(**agent) -> vf.SingleAgentEnvConfig:
    return vf.SingleAgentEnvConfig(
        taskset=vf.TasksetConfig(id="glossary"), agent=vf.AgentConfig(**agent)
    )


def test_eval_blocks_egress_unless_the_config_speaks_to_it() -> None:
    blocked = env(runtime=PrimeConfig())
    vf.restrict_network_by_default(blocked)
    assert blocked.taskset.network_allow == []

    kept = env(runtime=PrimeConfig())
    kept.taskset.network_allow = ["*"]
    vf.restrict_network_by_default(kept)
    assert kept.taskset.network_allow == ["*"]

    for runtime in (
        SubprocessConfig(),
        PrimeConfig(allow=["pypi.org"]),
        ModalConfig(network_access=False),
    ):
        untouched = env(runtime=runtime)
        vf.restrict_network_by_default(untouched)
        assert untouched.taskset.network_allow is None, runtime


def test_eval_config_applies_the_default() -> None:
    config = EvalConfig.model_validate({"env": {"taskset": {"id": "glossary"}}})
    assert config.env.taskset.network_allow == []
    config = EvalConfig.model_validate(
        {"env": {"taskset": {"id": "glossary", "network_allow": ["*"]}}}
    )
    assert config.env.taskset.network_allow == ["*"]
    # The default survives the round trip a resume or an env server's config takes.
    again = EvalConfig.model_validate(config.model_dump(mode="json"))
    assert again.env.taskset.network_allow == ["*"]


def test_network_notice_names_the_restriction() -> None:
    disabled = network_notice(NetworkPolicyConfig(allow=[]))
    assert "Internet access is disabled" in disabled
    assert "do not try to work around it" in disabled
    allowed = network_notice(NetworkPolicyConfig(allow=["pypi.org", "github.com"]))
    assert "Only these destinations are reachable: pypi.org, github.com" in allowed
    blocked = network_notice(NetworkPolicyConfig(block=["example.com"]))
    assert "blocked and requests to them will fail: example.com" in blocked
