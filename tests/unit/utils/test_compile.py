"""`validate_pairing` rejects a harness x task x runtime combination that can never run,
before any runtime is provisioned. The integration grid refuses the same cells."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import DockerConfig, SubprocessConfig
from verifiers.v1.utils.compile import validate_pairing
from verifiers.v1.utils.loaders import harness_class, harness_config_type


def harness(harness_id: str, **config) -> vf.Harness:
    config_type = harness_config_type(harness_id)
    return harness_class(harness_id)(config_type(id=harness_id, **config))


class BoxedTask(vf.Task):
    NEEDS_CONTAINER = True


def test_in_house_loops_run_on_the_host() -> None:
    for harness_id in ("null", "bash"):
        validate_pairing(harness(harness_id), vf.Task, SubprocessConfig())


def test_agent_programs_need_a_container() -> None:
    with pytest.raises(ValueError, match="needs a container runtime"):
        validate_pairing(harness("codex"), vf.Task, SubprocessConfig())
    validate_pairing(harness("codex"), vf.Task, DockerConfig())


def test_a_boxed_task_needs_a_container() -> None:
    with pytest.raises(ValueError, match="BoxedTask needs a container runtime"):
        validate_pairing(harness("bash"), BoxedTask, SubprocessConfig())
    validate_pairing(harness("bash"), BoxedTask, DockerConfig())


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
