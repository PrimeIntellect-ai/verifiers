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


    def load_taskset(config: DummyTasksetConfig) -> DummyTaskset:
        return DummyTaskset(config=config)


    def load_harness(config: DummyHarnessConfig) -> DummyHarness:
        return DummyHarness(config=config)
    """
)


# Lean v1 env: just a Taskset + load_taskset. No load_harness either.
LEAN_V1_ENV_SOURCE = textwrap.dedent(
    """
    import verifiers as vf


    class LeanTasksetConfig(vf.TasksetConfig):
        difficulty: str = "easy"


    class LeanTaskset(vf.Taskset[LeanTasksetConfig]):
        def load_tasks(self) -> vf.Tasks:
            return [
                {
                    "prompt": [{"role": "user", "content": "Say ok."}],
                    "answer": "ok",
                }
            ]


    def load_taskset(config: LeanTasksetConfig) -> LeanTaskset:
        return LeanTaskset(config=config)
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
def lean_v1_env(tmp_path: Path, monkeypatch):
    name = "lean_v1_env"
    _install_module(tmp_path, monkeypatch, name, LEAN_V1_ENV_SOURCE)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def dummy_v0_env(tmp_path: Path, monkeypatch):
    name = "dummy_v0_env"
    _install_module(tmp_path, monkeypatch, name, V0_ENV_SOURCE)
    yield name
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Argv preprocessing
# ---------------------------------------------------------------------------


def test_argv_no_positionals():
    cleaned, env, harness = ev1._extract_initial_args(["vf-eval", "--env", "x"])
    assert env == "x"
    assert harness is None
    assert cleaned == ["vf-eval", "--env", "x"]


def test_argv_single_positional_promoted():
    cleaned, env, harness = ev1._extract_initial_args(["vf-eval", "my-env"])
    assert env == "my-env"
    assert harness is None
    assert "--env" in cleaned and cleaned[cleaned.index("--env") + 1] == "my-env"


def test_argv_two_positionals_promoted():
    cleaned, env, harness = ev1._extract_initial_args(
        ["vf-eval", "my-env", "rlm", "--num-examples", "3"]
    )
    assert env == "my-env"
    assert harness == "rlm"
    assert "--env" in cleaned
    assert "--harness-name" in cleaned
    assert "--num-examples" in cleaned


def test_argv_positionals_stop_at_first_flag():
    cleaned, env, harness = ev1._extract_initial_args(
        ["vf-eval", "my-env", "--num-examples", "3", "rlm"]
    )
    # 'rlm' came after a flag — not a positional. env stays from the
    # positional; harness is None because flag-style fallback isn't here.
    assert env == "my-env"
    assert harness is None


def test_argv_explicit_flags_override_no_positionals():
    cleaned, env, harness = ev1._extract_initial_args(
        ["vf-eval", "--env", "explicit", "--harness-name", "rlm"]
    )
    assert env == "explicit"
    assert harness == "rlm"


def test_argv_positional_env_with_explicit_harness_flag():
    cleaned, env, harness = ev1._extract_initial_args(
        ["vf-eval", "my-env", "--harness-name=rlm", "--num-examples", "5"]
    )
    assert env == "my-env"
    assert harness == "rlm"


def test_argv_third_positional_rejected():
    with pytest.raises(SystemExit, match="two positionals"):
        ev1._extract_initial_args(["vf-eval", "a", "b", "c"])


def test_argv_at_file_peek_for_env(tmp_path: Path):
    toml_path = tmp_path / "eval.toml"
    toml_path.write_text('env = "from-toml"\nharness_name = "rlm"\n')
    cleaned, env, harness = ev1._extract_initial_args(["vf-eval", "@", str(toml_path)])
    assert env == "from-toml"
    assert harness == "rlm"


def test_argv_positional_wins_over_toml(tmp_path: Path):
    toml_path = tmp_path / "eval.toml"
    toml_path.write_text('env = "from-toml"\nharness_name = "from-toml-rlm"\n')
    cleaned, env, harness = ev1._extract_initial_args(
        ["vf-eval", "cli-env", "cli-rlm", "@", str(toml_path)]
    )
    assert env == "cli-env"
    assert harness == "cli-rlm"


# ---------------------------------------------------------------------------
# Config class resolution
# ---------------------------------------------------------------------------


def test_resolve_config_class_for_v1_env_is_dynamic(dummy_v1_env: str):
    cls = ev1._resolve_config_class(dummy_v1_env, harness_name=None)
    assert cls.__name__ == "ResolvedEvalConfig"
    # taskset / harness fields are typed to the env's actual subclasses
    taskset_field = cls.model_fields["taskset"].annotation
    harness_field = cls.model_fields["harness"].annotation
    assert taskset_field.__name__ == "DummyTasksetConfig"
    # No positional harness → env's load_harness gives DummyHarnessConfig
    assert harness_field.__name__ == "DummyHarnessConfig"


def test_resolve_config_class_with_positional_harness(dummy_v1_env: str):
    cls = ev1._resolve_config_class(dummy_v1_env, harness_name="base")
    harness_field = cls.model_fields["harness"].annotation
    # 'base' alias → verifiers.v1:Harness → HarnessConfig
    assert harness_field is HarnessConfig


def test_resolve_config_class_for_lean_env_uses_base_harness(lean_v1_env: str):
    cls = ev1._resolve_config_class(lean_v1_env, harness_name=None)
    harness_field = cls.model_fields["harness"].annotation
    assert harness_field is HarnessConfig


def test_resolve_config_class_for_v0_env_returns_v0_config(dummy_v0_env: str):
    cls = ev1._resolve_config_class(dummy_v0_env, harness_name=None)
    assert cls is ev1.EvalV0Config
    assert "env_args" in cls.model_fields


def test_resolve_config_class_without_env_falls_back_to_base():
    cls = ev1._resolve_config_class(None, harness_name=None)
    assert cls is ev1.EvalConfigBase


# ---------------------------------------------------------------------------
# Typed CLI parsing against the dynamic config
# ---------------------------------------------------------------------------


def _parse_cli(argv: list[str]):
    """Invoke ev1's full preprocess + dynamic-config + cli() pipeline."""
    from pydantic_config import cli as pyd_cli

    cleaned, env_id, harness_name = ev1._extract_initial_args(argv)
    cls = ev1._resolve_config_class(env_id, harness_name)
    saved = sys.argv
    sys.argv = cleaned
    try:
        return pyd_cli(cls), env_id, harness_name
    finally:
        sys.argv = saved


def test_cli_v1_positional_task(dummy_v1_env: str):
    config, env_id, harness_name = _parse_cli(["vf-eval", dummy_v1_env])
    assert env_id == dummy_v1_env
    assert harness_name is None
    assert config.env == dummy_v1_env
    assert config.taskset.difficulty == "easy"


def test_cli_v1_typed_taskset_override(dummy_v1_env: str):
    config, _, _ = _parse_cli(["vf-eval", dummy_v1_env, "--taskset.difficulty", "hard"])
    assert config.taskset.difficulty == "hard"


def test_cli_v1_typed_harness_override(dummy_v1_env: str):
    config, _, _ = _parse_cli(["vf-eval", dummy_v1_env, "--harness.extra-field", "42"])
    # DummyHarnessConfig.extra_field: int — pydantic coerces "42" -> 42
    assert config.harness.extra_field == 42


def test_cli_v1_rejects_unknown_taskset_field(dummy_v1_env: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval", dummy_v1_env, "--taskset.unknown", "x"])


def test_cli_v1_rejects_unknown_harness_field(dummy_v1_env: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval", dummy_v1_env, "--harness.unknown", "x"])


def test_cli_v1_positional_harness_swap(dummy_v1_env: str):
    config, _, harness_name = _parse_cli(["vf-eval", dummy_v1_env, "base"])
    assert harness_name == "base"
    assert isinstance(config.harness, HarnessConfig)
    assert type(config.harness).__name__ == "HarnessConfig"


def test_cli_v1_lean_env_typed_default_harness(lean_v1_env: str):
    config, _, _ = _parse_cli(["vf-eval", lean_v1_env, "--harness.max-turns", "12"])
    assert type(config.harness).__name__ == "HarnessConfig"
    assert config.harness.max_turns == 12


def test_cli_v0_env_typed(dummy_v0_env: str):
    config, _, _ = _parse_cli(
        ["vf-eval", dummy_v0_env, "--env-args", '{"some_arg": 1}']
    )
    assert isinstance(config, ev1.EvalV0Config)
    assert config.env_args == {"some_arg": 1}


def test_cli_v0_env_rejects_positional_harness(dummy_v0_env: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval", dummy_v0_env, "rlm"])


# ---------------------------------------------------------------------------
# env_args round-trip through vf.load_environment dispatch
# ---------------------------------------------------------------------------


def test_v1_env_args_taskset_dispatch(dummy_v1_env: str):
    config, _, _ = _parse_cli(["vf-eval", dummy_v1_env, "--taskset.difficulty", "hard"])
    env_args = ev1._v1_env_args(config)
    assert env_args[V1_TASKSET_KEY] == {"difficulty": "hard"}


def test_v1_env_args_harness_name_dispatch(dummy_v1_env: str):
    config, _, _ = _parse_cli(["vf-eval", dummy_v1_env, "base"])
    env_args = ev1._v1_env_args(config)
    assert env_args[V1_HARNESS_KEY] == {"name": "base"}


def test_v1_env_args_combined(dummy_v1_env: str):
    config, _, _ = _parse_cli(
        [
            "vf-eval",
            dummy_v1_env,
            "base",
            "--taskset.difficulty",
            "hard",
            "--harness.max-turns",
            "7",
        ]
    )
    env_args = ev1._v1_env_args(config)
    assert env_args[V1_TASKSET_KEY] == {"difficulty": "hard"}
    assert env_args[V1_HARNESS_KEY] == {"name": "base", "max_turns": 7}


# ---------------------------------------------------------------------------
# v1 dispatch through vf.load_environment
# ---------------------------------------------------------------------------


def test_load_environment_v1_default_harness(dummy_v1_env: str):
    env = vf.load_environment(dummy_v1_env)
    assert isinstance(env, vf.Env)
    assert type(env.taskset).__name__ == "DummyTaskset"
    assert type(env.harness).__name__ == "DummyHarness"
    assert env.harness.config.extra_field == 7


def test_load_environment_v1_taskset_overrides(dummy_v1_env: str):
    env = vf.load_environment(dummy_v1_env, **{V1_TASKSET_KEY: {"difficulty": "hard"}})
    assert env.taskset.config.difficulty == "hard"


def test_load_environment_v1_harness_name_swap(dummy_v1_env: str):
    env = vf.load_environment(
        dummy_v1_env,
        **{V1_HARNESS_KEY: {"name": "verifiers.v1:Harness", "max_turns": 4}},
    )
    assert type(env.harness) is Harness
    assert isinstance(env.harness.config, HarnessConfig)
    assert env.harness.config.max_turns == 4


def test_load_environment_v1_dispatch_preserves_env_args(dummy_v1_env: str):
    env_args = {V1_HARNESS_KEY: {"max_turns": 5}}
    env = vf.load_environment(dummy_v1_env, **env_args)
    assert env.env_args == env_args


# ---------------------------------------------------------------------------
# Lean env: only load_taskset, no load_environment, no load_harness.
# ---------------------------------------------------------------------------


def test_lean_v1_env_auto_resolves_base_harness(lean_v1_env: str):
    env = vf.load_environment(lean_v1_env)
    assert type(env.taskset).__name__ == "LeanTaskset"
    assert type(env.harness) is Harness


def test_lean_v1_env_load_environment_rejects_extra_kwargs(lean_v1_env: str):
    with pytest.raises(Exception, match="load_taskset"):
        vf.load_environment(lean_v1_env, foo=1)


# ---------------------------------------------------------------------------
# build_v1_env wiring
# ---------------------------------------------------------------------------


def test_build_v1_env_uses_env_load_harness(dummy_v1_env: str):
    env = build_v1_env(dummy_v1_env)
    assert type(env.harness).__name__ == "DummyHarness"


def test_build_v1_env_explicit_base_harness(dummy_v1_env: str):
    env = build_v1_env(dummy_v1_env, harness_spec={"name": "base", "max_turns": 2})
    assert type(env.harness) is Harness
    assert env.harness.config.max_turns == 2


# ---------------------------------------------------------------------------
# Harness alias registry + helpers
# ---------------------------------------------------------------------------


def test_harness_aliases_registered():
    assert HARNESS_ALIASES["rlm"] == "verifiers.v1.packages.harnesses:RLM"
    assert HARNESS_ALIASES["base"] == "verifiers.v1:Harness"


def test_has_v1_overrides():
    assert not has_v1_overrides({"foo": 1})
    assert has_v1_overrides({V1_TASKSET_KEY: {}})
    assert has_v1_overrides({V1_HARNESS_KEY: {}})
