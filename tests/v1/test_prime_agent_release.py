import pytest

import verifiers.v1.harnesses.prime_agent.harness as prime_agent
from verifiers.v1.harnesses.prime_agent.harness import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
    PrimeAgentRelease,
)


def release(version: str = "0.8.1") -> PrimeAgentRelease:
    return PrimeAgentRelease(
        version=version,
        sha256={
            package: f"{index:x}" * 64
            for index, package in enumerate(prime_agent.RELEASE_PACKAGES, 1)
        },
    )


def test_release_requires_every_tarball() -> None:
    resolved = release()
    entries = [
        {"file": resolved.file(package), "sha256": sha256}
        for package, sha256 in resolved.sha256.items()
    ]
    assert prime_agent._release("v0.8.1", entries) == resolved

    with pytest.raises(ValueError, match="prime-agent-tui-0.8.1.tgz"):
        prime_agent._release("0.8.1", entries[:-1])


def test_release_rejects_unsupported_versions() -> None:
    with pytest.raises(ValueError, match="older than"):
        PrimeAgentHarnessConfig(id="prime_agent", release="0.8.0")
    with pytest.raises(ValueError, match="invalid"):
        PrimeAgentHarnessConfig(id="prime_agent", release="0.8.1-beta.1")
    with pytest.raises(ValueError, match="does not match"):
        PrimeAgentHarnessConfig(
            id="prime_agent", release="0.8.2", resolved_release=release()
        )


@pytest.mark.asyncio
async def test_prepared_release_survives_config_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = release()

    async def fetch(_: str) -> PrimeAgentRelease:
        return resolved

    monkeypatch.setattr(prime_agent, "_fetch_release", fetch)
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime_agent"))
    await harness.prepare()

    saved = harness.config.model_dump(mode="json")

    async def fail(_: str) -> PrimeAgentRelease:
        raise AssertionError("frozen release was fetched again")

    monkeypatch.setattr(prime_agent, "_fetch_release", fail)
    loaded = PrimeAgentHarnessConfig.model_validate(saved)
    assert await PrimeAgentHarness(loaded)._resolve() == resolved

    saved["resolved_release"]["version"] = "0.8.0"
    with pytest.raises(ValueError, match="older than"):
        PrimeAgentHarnessConfig.model_validate(saved)
