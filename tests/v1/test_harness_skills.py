"""Hermetic skill-staging contracts shared by native harnesses."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult


class _SkillHarness(Harness[HarnessConfig]):
    async def launch(self, *args, **kwargs) -> ProgramResult:
        del args, kwargs
        return ProgramResult(exit_code=0, stdout="", stderr="")


class _Runtime:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes]] = []
        self.runs: list[list[str]] = []

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append((path, data))

    async def run(self, command: list[str], environment: dict[str, str]):
        del environment
        self.runs.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def _skill(root: Path, name: str, contents: str) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(contents)
    return skill


@pytest.mark.asyncio
async def test_skill_staging_rejects_distinct_folders_with_the_same_basename(
    tmp_path: Path,
) -> None:
    first = _skill(tmp_path / "one", "review", "one")
    second = _skill(tmp_path / "two", "review", "two")
    harness = _SkillHarness(HarnessConfig(skills=[first, second]))
    runtime = _Runtime()

    with pytest.raises(
        ValueError, match=r"duplicate skill folder name 'review'"
    ) as raised:
        await harness.install_skills(runtime, "/skills")

    message = str(raised.value)
    assert str(first.resolve()) in message
    assert str(second.resolve()) in message
    assert runtime.writes == []
    assert runtime.runs == []


@pytest.mark.asyncio
async def test_skill_staging_rejects_the_same_folder_twice(tmp_path: Path) -> None:
    skill = _skill(tmp_path, "review", "contents")
    harness = _SkillHarness(HarnessConfig(skills=[skill, skill]))
    runtime = _Runtime()

    with pytest.raises(ValueError, match=r"duplicate skill folder name 'review'"):
        await harness.install_skills(runtime, "/skills")

    assert runtime.writes == []


@pytest.mark.asyncio
async def test_skill_staging_keeps_unique_names_and_executable_modes(
    tmp_path: Path,
) -> None:
    first = _skill(tmp_path, "first", "first")
    script = first / "scripts" / "run.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o700)
    second = _skill(tmp_path, "second", "second")
    harness = _SkillHarness(HarnessConfig(skills=[first, second]))
    runtime = _Runtime()

    await harness.install_skills(runtime, "/skills")

    assert runtime.writes == [
        ("/skills/first/SKILL.md", b"first"),
        ("/skills/first/scripts/run.sh", b"#!/bin/sh\nexit 0\n"),
        ("/skills/second/SKILL.md", b"second"),
    ]
    assert runtime.runs == [["chmod", "+x", "/skills/first/scripts/run.sh"]]


def test_prime_agent_rejects_duplicate_skill_basenames_before_cli_construction(
    tmp_path: Path,
) -> None:
    from verifiers.v1.harnesses.prime_agent.harness import (
        PrimeAgentHarness,
        PrimeAgentHarnessConfig,
    )

    first = _skill(tmp_path / "one", "review", "one")
    second = _skill(tmp_path / "two", "review", "two")
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(skills=[first, second]))

    with pytest.raises(ValueError, match=r"duplicate skill folder name 'review'"):
        harness.resolved_skills()
