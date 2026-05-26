"""Tests for ``vf-eval-v1`` (verifiers.scripts.eval_v1)."""

import sys
import textwrap
from pathlib import Path

import pytest

import verifiers as vf
from verifiers.scripts import eval_v1 as ev1
from verifiers.utils.v1_loader_utils import (
    HARNESS_ALIASES,
    V1_HARNESS_KEY,
    V1_TASKSET_KEY,
    build_v1_env,
    has_v1_overrides,
)
from verifiers.v1.config import HarnessConfig
from verifiers.v1.harness import Harness


# ---------------------------------------------------------------------------
# Inline v1 env module fixtures
# ---------------------------------------------------------------------------


V1_ENV_SOURCE = textwrap.dedent(
    """
    import verifiers as vf


    class DummyTasksetConfig(vf.TasksetConfig):
        difficulty: str = "easy"


    class DummyTaskset(vf.Taskset[DummyTasksetConfig]):
        def load_tasks(self) -> vf.Tasks:
            return [
                {
                    "prompt": [{"role": "user", "content": "Say ok."}],
                    "answer": "ok",
                }
            ]


    class DummyHarnessConfig(vf.HarnessConfig):
        extra_field: int = 7


    class DummyHarness(vf.Harness):
        pass


    class DummyEnvConfig(vf.EnvConfig):
        taskset: DummyTasksetConfig = DummyTasksetConfig()
        harness: DummyHarnessConfig = DummyHarnessConfig()


    def load_taskset(config: DummyTasksetConfig) -> DummyTaskset:
        return DummyTaskset(config=config)


    def load_harness(config: DummyHarnessConfig) -> DummyHarness:
        return DummyHarness(config=config)


    def load_environment(config: DummyEnvConfig) -> vf.Env:
        return vf.Env(
            taskset=load_taskset(config.taskset),
            harness=load_harness(config.harness),
        )
    """
)


V0_ENV_SOURCE = textwrap.dedent(
    """
    import verifiers as vf
    from datasets import Dataset


    def load_environment(some_arg: int = 0) -> vf.Environment:
        ds = Dataset.from_list(
            [{"prompt": [{"role": "user", "content": "ok"}], "answer": "ok"}]
        )

        async def reward(completion, answer) -> float:
            return 1.0

        return vf.SingleTurnEnv(dataset=ds, rubric=vf.Rubric(funcs=[reward]))
    """
)


def _install_module(tmp_path: Path, monkeypatch, name: str, source: str) -> None:
    module_path = tmp_path / f"{name}.py"
    module_path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)


@pytest.fixture
def dummy_v1_env(tmp_path: Path, monkeypatch):
    name = "dummy_v1_env"
    _install_module(tmp_path, monkeypatch, name, V1_ENV_SOURCE)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def dummy_v0_env(tmp_path: Path, monkeypatch):
    name = "dummy_v0_env"
    _install_module(tmp_path, monkeypatch, name, V0_ENV_SOURCE)
    yield name
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> ev1.EvalV1Config:
    """Run ``ev1.cli(EvalV1Config)`` with ``argv`` and return the config."""
    from pydantic_config import cli as pyd_cli

    saved = sys.argv
    sys.argv = ["vf-eval-v1", *argv]
    try:
        return pyd_cli(ev1.EvalV1Config)
    finally:
        sys.argv = saved


def test_cli_minimal_env_required():
    with pytest.raises(SystemExit):
        _parse([])


def test_cli_basic_flags():
    config = _parse(
        [
            "--env",
            "my-env",
            "--num-examples",
            "7",
            "--rollouts-per-example",
            "2",
            "--model",
            "gpt-test",
            "--temperature",
            "0.5",
        ]
    )
    assert config.env == "my-env"
    assert config.num_examples == 7
    assert config.rollouts_per_example == 2
    assert config.model == "gpt-test"
    assert config.temperature == 0.5


def test_cli_harness_ref_and_extras():
    config = _parse(
        [
            "--env",
            "any",
            "--harness.ref",
            "rlm",
            "--harness.rlm-max-turns",
            "12",
            "--harness.system-prompt",
            "hi",
        ]
    )
    assert config.harness.ref == "rlm"
    extras = config.harness.model_extra or {}
    assert extras["rlm_max_turns"] == "12"
    assert extras["system_prompt"] == "hi"


def test_cli_taskset_overrides():
    config = _parse(
        [
            "--env",
            "any",
            "--taskset.difficulty",
            "hard",
        ]
    )
    extras = config.taskset.model_extra or {}
    assert extras["difficulty"] == "hard"


def test_cli_positional_env_promoted():
    """``vf-eval-v1 my-env --num-examples 3`` should work like --env my-env."""
    promoted = ev1._preprocess_argv(["vf-eval-v1", "my-env", "--num-examples", "3"])
    assert promoted == ["vf-eval-v1", "--env", "my-env", "--num-examples", "3"]


def test_cli_positional_env_not_promoted_when_flag_first():
    argv = ["vf-eval-v1", "--env", "x"]
    assert ev1._preprocess_argv(argv) == argv


def test_cli_positional_env_not_promoted_for_at_file():
    argv = ["vf-eval-v1", "@", "f.toml"]
    assert ev1._preprocess_argv(argv) == argv


def test_cli_toml_load(tmp_path: Path):
    toml_path = tmp_path / "eval.toml"
    toml_path.write_text(
        textwrap.dedent(
            """
            env = "from-toml"
            num_examples = 4
            rollouts_per_example = 2
            model = "gpt-toml"

            [harness]
            ref = "verifiers.v1:Harness"
            max_turns = 9
            """
        )
    )
    config = _parse(["@", str(toml_path)])
    assert config.env == "from-toml"
    assert config.num_examples == 4
    assert config.model == "gpt-toml"
    assert config.harness.ref == "verifiers.v1:Harness"
    assert (config.harness.model_extra or {})["max_turns"] == 9


# ---------------------------------------------------------------------------
# env_args resolution
# ---------------------------------------------------------------------------


def test_resolve_env_args_v1_default_is_empty(dummy_v1_env: str):
    cfg = ev1.EvalV1Config(env=dummy_v1_env)
    assert ev1._resolve_env_args(cfg) == {}


def test_resolve_env_args_v1_with_overrides(dummy_v1_env: str):
    cfg = ev1.EvalV1Config(
        env=dummy_v1_env,
        taskset=ev1.TasksetSpec.model_validate({"difficulty": "hard"}),
        harness=ev1.HarnessSpec.model_validate({"max_turns": 3, "system_prompt": "hi"}),
    )
    env_args = ev1._resolve_env_args(cfg)
    assert env_args[V1_TASKSET_KEY] == {"difficulty": "hard"}
    assert env_args[V1_HARNESS_KEY] == {"max_turns": 3, "system_prompt": "hi"}


def test_resolve_env_args_v0_passthrough(dummy_v0_env: str):
    cfg = ev1.EvalV1Config(env=dummy_v0_env, env_args={"some_arg": 1})
    assert ev1._resolve_env_args(cfg) == {"some_arg": 1}


def test_resolve_env_args_v0_rejects_taskset_override(dummy_v0_env: str):
    cfg = ev1.EvalV1Config(
        env=dummy_v0_env,
        taskset=ev1.TasksetSpec.model_validate({"foo": "bar"}),
    )
    with pytest.raises(ValueError, match="v0 envs"):
        ev1._resolve_env_args(cfg)


def test_resolve_env_args_v0_rejects_harness_override(dummy_v0_env: str):
    cfg = ev1.EvalV1Config(
        env=dummy_v0_env,
        harness=ev1.HarnessSpec.model_validate({"ref": "rlm"}),
    )
    with pytest.raises(ValueError, match="v0 envs"):
        ev1._resolve_env_args(cfg)


def test_resolve_env_args_v1_rejects_env_args(dummy_v1_env: str):
    cfg = ev1.EvalV1Config(env=dummy_v1_env, env_args={"foo": 1})
    with pytest.raises(ValueError, match="v1 env"):
        ev1._resolve_env_args(cfg)


# ---------------------------------------------------------------------------
# v1 dispatch through load_environment
# ---------------------------------------------------------------------------


def test_load_environment_v1_default_harness(dummy_v1_env: str):
    env = vf.load_environment(dummy_v1_env)
    assert isinstance(env, vf.Env)
    assert type(env.taskset).__name__ == "DummyTaskset"
    # env's own load_harness is used
    assert type(env.harness).__name__ == "DummyHarness"
    assert env.harness.config.extra_field == 7


def test_load_environment_v1_taskset_overrides(dummy_v1_env: str):
    env = vf.load_environment(dummy_v1_env, **{V1_TASKSET_KEY: {"difficulty": "hard"}})
    assert env.taskset.config.difficulty == "hard"


def test_load_environment_v1_harness_override(dummy_v1_env: str):
    env = vf.load_environment(dummy_v1_env, **{V1_HARNESS_KEY: {"max_turns": 99}})
    # env's load_harness is used; max_turns flows into its config
    assert env.harness.config.max_turns == 99


def test_load_environment_v1_harness_ref_swap(dummy_v1_env: str):
    env = vf.load_environment(
        dummy_v1_env,
        **{V1_HARNESS_KEY: {"ref": "verifiers.v1:Harness", "max_turns": 4}},
    )
    # ref points at the base Harness class explicitly
    assert type(env.harness) is Harness
    assert isinstance(env.harness.config, HarnessConfig)
    assert env.harness.config.max_turns == 4


def test_load_environment_v1_dispatch_preserves_env_args(dummy_v1_env: str):
    """Env workers re-load the env from env_args, so dispatch markers must
    survive on the returned env."""
    env_args = {V1_HARNESS_KEY: {"max_turns": 5}}
    env = vf.load_environment(dummy_v1_env, **env_args)
    assert env.env_args == env_args


# ---------------------------------------------------------------------------
# Harness alias registry
# ---------------------------------------------------------------------------


def test_harness_aliases_registered():
    assert HARNESS_ALIASES["rlm"] == "verifiers.v1.packages.harnesses:RLM"
    assert HARNESS_ALIASES["base"] == "verifiers.v1:Harness"


def test_has_v1_overrides():
    assert not has_v1_overrides({"foo": 1})
    assert has_v1_overrides({V1_TASKSET_KEY: {}})
    assert has_v1_overrides({V1_HARNESS_KEY: {}})


# ---------------------------------------------------------------------------
# build_v1_env wiring
# ---------------------------------------------------------------------------


def test_build_v1_env_uses_env_load_harness(dummy_v1_env: str):
    env = build_v1_env(dummy_v1_env)
    assert type(env.harness).__name__ == "DummyHarness"


def test_build_v1_env_explicit_base_harness(dummy_v1_env: str):
    env = build_v1_env(dummy_v1_env, harness_spec={"ref": "base", "max_turns": 2})
    assert type(env.harness) is Harness
    assert env.harness.config.max_turns == 2
