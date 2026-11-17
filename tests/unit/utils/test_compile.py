"""`validate_pairing` rejects a harness x task x runtime combination that can never run,
before any runtime is provisioned. The integration grid refuses the same cells."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.utils.compile import validate_pairing
from verifiers.v1.utils.loaders import harness_class, harness_config_type


def harness(harness_id: str, **config) -> vf.Harness:
    config_type = harness_config_type(harness_id)
    return harness_class(harness_id)(config_type(id=harness_id, **config))


def test_tools_need_an_mcp_harness() -> None:
    with pytest.raises(ValueError, match="does not support MCP tools"):
        validate_pairing(
            harness("mini-swe-agent"), vf.Task, DockerConfig(), tools=["echo"]
        )
    validate_pairing(harness("mini-swe-agent"), vf.Task, DockerConfig(), tools=[])


def test_skills_need_a_harness_that_discovers_them(tmp_path) -> None:
    (tmp_path / "SKILL.md").write_text("# a skill\n")
    with pytest.raises(ValueError, match="no native skill support"):
        validate_pairing(harness("bash", skills=[tmp_path]), vf.Task, DockerConfig())
    validate_pairing(harness("claude-code", skills=[tmp_path]), vf.Task, DockerConfig())
