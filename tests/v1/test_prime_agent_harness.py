"""The prime-agent harness's release-shape selection: TS bundle vs Rust tarball."""

import pytest
from pydantic import ValidationError

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR
from verifiers.v1.harnesses.prime_agent.harness import (
    PRIME_AGENT_DIR,
    RUST_INSTALL,
    RUST_VERSION,
    TS_COMMIT,
    TS_INSTALL,
    TS_VERSION,
    PrimeAgentHarnessConfig,
    release_plan,
)
from verifiers.v1.utils.loaders import harness_config_type


def test_defaults_install_the_rust_tarball() -> None:
    config = PrimeAgentHarnessConfig()
    assert config.version == RUST_VERSION
    assert config.platform == "linux-x64"
    plan = release_plan(config.version, config.platform)
    assert plan.install is RUST_INSTALL
    # The Rust tarball puts the binary at the prefix root: no bin/ subdir.
    assert plan.bin == f"{PRIME_AGENT_DIR}/{RUST_VERSION}/prime-agent"


def test_npm_era_release_installs_the_ts_bundle() -> None:
    plan = release_plan(TS_VERSION, "linux-x64")
    assert plan.install is TS_INSTALL
    assert plan.bin == f"{PRIME_AGENT_DIR}/{TS_COMMIT}/bin/prime-agent"
    assert NODE_BIN_DIR in plan.launch_path


def test_rust_launch_path_drops_node() -> None:
    assert release_plan(RUST_VERSION, "linux-x64").launch_path == "$HOME/.local/bin"


def test_unpinned_releases_fail_fast() -> None:
    with pytest.raises(ValueError, match="no pinned install"):
        release_plan("0.9.6", "linux-x64")
    with pytest.raises(ValueError, match="no pinned linux-s390x tarball"):
        release_plan(RUST_VERSION, "linux-s390x")


def test_config_rejects_unknown_versions_and_platforms() -> None:
    with pytest.raises(ValidationError):
        PrimeAgentHarnessConfig(version="0.9.6")
    with pytest.raises(ValidationError):
        PrimeAgentHarnessConfig(platform="win32-x64")


def test_harness_id_resolves_the_specialized_config() -> None:
    assert harness_config_type("prime-agent") is PrimeAgentHarnessConfig
    assert issubclass(PrimeAgentHarnessConfig, HarnessConfig)
