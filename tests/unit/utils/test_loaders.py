"""Plugin resolution by id: every bundled harness, env, and judge loads, each with its
own config type, and an unknown id fails with the bundled ids in the message."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.harness import Harness
from verifiers.v1.utils.loaders import (
    builtin_harness_ids,
    env_config_type,
    environment_class,
    harness_class,
    harness_config_type,
    judge_class,
    judge_config_type,
    load_harness,
)


def test_every_builtin_harness_loads() -> None:
    ids = builtin_harness_ids()
    assert {"bash", "null", "codex", "claude-code", "rlm"} <= set(ids)
    for harness_id in ids:
        cls = harness_class(harness_id)
        assert issubclass(cls, Harness)
        config_type = harness_config_type(harness_id)
        assert isinstance(load_harness(config_type(id=harness_id)), cls)


def test_harness_config_narrows_to_its_own_type() -> None:
    config = vf.AgentConfig.model_validate(
        {"harness": {"id": "kimi-code", "dialect": "responses"}}
    )
    assert type(config.harness) is harness_config_type("kimi-code")
    assert config.harness.dialect == "responses"


def test_unknown_harness_names_the_builtins() -> None:
    with pytest.raises(Exception, match="built-in harnesses are: .*bash"):
        harness_class("no-such-harness")


@pytest.mark.parametrize(
    "env_id", ["best-of-n", "best-of-n-judge", "user-sim", "agentic-judge"]
)
def test_bundled_envs_load_with_a_role_typed_config(env_id: str) -> None:
    cls = environment_class("", env_id)
    assert issubclass(cls, vf.Env)
    config_type = env_config_type("", env_id)
    assert issubclass(config_type, vf.EnvConfig)
    seats = [
        name
        for name, field in config_type.model_fields.items()
        if isinstance(field.default, vf.AgentConfig)
    ]
    assert seats, f"{env_id} declares no agent seat"


@pytest.mark.parametrize(
    "judge_id,config", [("reference", {}), ("rubric", {"path": "rubric.toml"})]
)
def test_bundled_judges_load(judge_id: str, config: dict) -> None:
    assert issubclass(judge_class(judge_id), vf.Judge)
    assert judge_config_type(judge_id)(**config).id == judge_id
