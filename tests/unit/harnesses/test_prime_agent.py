from types import SimpleNamespace
from typing import Literal

import pytest

from verifiers.v1.harnesses.prime_agent import harness as prime_agent


class PackagePrimeAgentConfig(prime_agent.PrimeAgentHarnessConfig):
    commit: Literal["package-commit"] = "package-commit"
    release_version: str = "0.10.2"


@pytest.mark.asyncio
async def test_setup_uses_package_release_config(monkeypatch):
    captured = {}

    async def ensure_installed(runtime, **kwargs):
        captured.update(kwargs)

    async def prepare_uv_script(script, env=None, *, activate=True):
        return []

    monkeypatch.setattr(prime_agent, "ensure_installed", ensure_installed)
    runtime = SimpleNamespace(prepare_uv_script=prepare_uv_script)
    harness = prime_agent.PrimeAgentHarness(PackagePrimeAgentConfig())

    await harness.setup(runtime)

    assert captured["env"]["PRIME_AGENT_COMMIT"] == "package-commit"
    assert captured["env"]["PRIME_AGENT_RELEASE_VERSION"] == "0.10.2"
