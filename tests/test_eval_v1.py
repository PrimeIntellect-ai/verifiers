"""Tests for ``vf-eval-v1`` (verifiers.scripts.eval_v1)."""

import sys
import textwrap
from pathlib import Path

import pytest

import verifiers as vf
from verifiers.scripts import eval_v1 as ev1
from verifiers.utils.v1_loader_utils import (
    V1_HARNESS_KEY,
    V1_TASKSET_KEY,
    build_v1_env,
    has_v1_overrides,
)
from verifiers.v1.config import HarnessConfig
from verifiers.v1.harness import Harness


# ---------------------------------------------------------------------------
# Inline module fixtures
# ---------------------------------------------------------------------------


V1_TASKSET_SOURCE = textwrap.dedent(
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


LEAN_V1_TASKSET_SOURCE = textwrap.dedent(
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


# A standalone harness module: exposes load_harness(config) and a HarnessConfig
# subclass. Mimics what environments/v1/harnesses/<name>/ packages do.
EXTERNAL_HARNESS_SOURCE = textwrap.dedent(
    """
    import verifiers as vf


    class ExtHarnessConfig(vf.HarnessConfig):
        ext_field: str = "default"


    class ExtHarness(vf.Harness):
        pass


    def load_harness(config: ExtHarnessConfig) -> ExtHarness:
        return ExtHarness(config=config)
    """
)


def _install_module(tmp_path: Path, monkeypatch, name: str, source: str) -> None:
    module_path = tmp_path / f"{name}.py"
    module_path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)


@pytest.fixture
def dummy_taskset(tmp_path: Path, monkeypatch):
    name = "dummy_taskset"
    _install_module(tmp_path, monkeypatch, name, V1_TASKSET_SOURCE)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def lean_taskset(tmp_path: Path, monkeypatch):
    name = "lean_taskset"
    _install_module(tmp_path, monkeypatch, name, LEAN_V1_TASKSET_SOURCE)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def v0_env(tmp_path: Path, monkeypatch):
    name = "dummy_v0_env"
    _install_module(tmp_path, monkeypatch, name, V0_ENV_SOURCE)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def external_harness(tmp_path: Path, monkeypatch):
    name = "ext_harness"
    _install_module(tmp_path, monkeypatch, name, EXTERNAL_HARNESS_SOURCE)
    yield name
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Argv preprocessing
# ---------------------------------------------------------------------------


def test_argv_no_positionals():
    cleaned, ts, harness = ev1._extract_initial_args(
        ["vf-eval-v1", "--taskset-name", "x"]
    )
    assert ts == "x"
    assert harness is None
    assert cleaned == ["vf-eval-v1", "--taskset-name", "x"]


def test_argv_single_positional_taskset():
    cleaned, ts, harness = ev1._extract_initial_args(["vf-eval-v1", "my-taskset"])
    assert ts == "my-taskset"
    assert harness is None
    assert "--taskset-name" in cleaned


def test_argv_two_positionals():
    cleaned, ts, harness = ev1._extract_initial_args(
        ["vf-eval-v1", "my-taskset", "rlm", "--num-examples", "3"]
    )
    assert ts == "my-taskset"
    assert harness == "rlm"
    assert "--taskset-name" in cleaned
    assert "--harness-name" in cleaned
    assert "--num-examples" in cleaned


def test_argv_positionals_stop_at_first_flag():
    cleaned, ts, harness = ev1._extract_initial_args(
        ["vf-eval-v1", "my-taskset", "--num-examples", "3", "rlm"]
    )
    # 'rlm' came after a flag — not a positional.
    assert ts == "my-taskset"
    assert harness is None


def test_argv_explicit_flags():
    cleaned, ts, harness = ev1._extract_initial_args(
        [
            "vf-eval-v1",
            "--taskset-name",
            "explicit",
            "--harness-name",
            "rlm",
        ]
    )
    assert ts == "explicit"
    assert harness == "rlm"


def test_argv_positional_taskset_explicit_harness_flag():
    cleaned, ts, harness = ev1._extract_initial_args(
        ["vf-eval-v1", "my-taskset", "--harness-name=rlm"]
    )
    assert ts == "my-taskset"
    assert harness == "rlm"


def test_argv_third_positional_rejected():
    with pytest.raises(SystemExit, match="two positionals"):
        ev1._extract_initial_args(["vf-eval-v1", "a", "b", "c"])


def test_argv_at_file_peek(tmp_path: Path):
    toml_path = tmp_path / "eval.toml"
    toml_path.write_text('taskset_name = "from-toml"\nharness_name = "rlm-from-toml"\n')
    cleaned, ts, harness = ev1._extract_initial_args(
        ["vf-eval-v1", "@", str(toml_path)]
    )
    assert ts == "from-toml"
    assert harness == "rlm-from-toml"


def test_argv_positional_wins_over_toml(tmp_path: Path):
    toml_path = tmp_path / "eval.toml"
    toml_path.write_text('taskset_name = "from-toml"\nharness_name = "from-toml-rlm"\n')
    cleaned, ts, harness = ev1._extract_initial_args(
        ["vf-eval-v1", "cli-ts", "cli-rlm", "@", str(toml_path)]
    )
    assert ts == "cli-ts"
    assert harness == "cli-rlm"


# ---------------------------------------------------------------------------
# Config class resolution
# ---------------------------------------------------------------------------


def test_resolve_config_class_for_v1_taskset_is_dynamic(dummy_taskset: str):
    cls = ev1._resolve_config_class(dummy_taskset, harness_name=None)
    assert cls.__name__ == "ResolvedEvalConfig"
    taskset_field = cls.model_fields["taskset"].annotation
    harness_field = cls.model_fields["harness"].annotation
    assert taskset_field.__name__ == "DummyTasksetConfig"
    # No positional harness → taskset module's load_harness gives DummyHarnessConfig
    assert harness_field.__name__ == "DummyHarnessConfig"


def test_resolve_config_class_with_external_harness(
    dummy_taskset: str, external_harness: str
):
    cls = ev1._resolve_config_class(dummy_taskset, harness_name=external_harness)
    harness_field = cls.model_fields["harness"].annotation
    assert harness_field.__name__ == "ExtHarnessConfig"


def test_resolve_config_class_for_lean_taskset_uses_base_harness(lean_taskset: str):
    cls = ev1._resolve_config_class(lean_taskset, harness_name=None)
    harness_field = cls.model_fields["harness"].annotation
    assert harness_field is HarnessConfig


def test_resolve_config_class_for_v0_returns_v0_config(v0_env: str):
    cls = ev1._resolve_config_class(v0_env, harness_name=None)
    assert cls is ev1.EvalV0Config
    assert "env_args" in cls.model_fields


def test_resolve_config_class_without_taskset_falls_back_to_base():
    cls = ev1._resolve_config_class(None, harness_name=None)
    assert cls is ev1.EvalConfigBase


def test_unknown_harness_module_raises(dummy_taskset: str):
    with pytest.raises(Exception, match="not_a_module|No module named"):
        ev1._resolve_config_class(dummy_taskset, harness_name="not_a_module")


def test_harness_module_without_load_harness_raises(
    dummy_taskset: str, tmp_path: Path, monkeypatch
):
    # Module that doesn't expose load_harness should fail with a clear error.
    bad_name = "bad_harness_module"
    _install_module(tmp_path, monkeypatch, bad_name, "")  # empty module
    try:
        with pytest.raises(AttributeError, match="load_harness"):
            ev1._resolve_config_class(dummy_taskset, harness_name=bad_name)
    finally:
        sys.modules.pop(bad_name, None)


# ---------------------------------------------------------------------------
# Typed CLI parsing against the dynamic config
# ---------------------------------------------------------------------------


def _parse_cli(argv: list[str]):
    from pydantic_config import cli as pyd_cli

    cleaned, ts, harness = ev1._extract_initial_args(argv)
    cls = ev1._resolve_config_class(ts, harness)
    saved = sys.argv
    sys.argv = cleaned
    try:
        return pyd_cli(cls), ts, harness
    finally:
        sys.argv = saved


def test_cli_v1_positional_taskset(dummy_taskset: str):
    config, ts, harness = _parse_cli(["vf-eval-v1", dummy_taskset])
    assert ts == dummy_taskset
    assert harness is None
    assert config.taskset_name == dummy_taskset
    assert config.taskset.difficulty == "easy"


def test_cli_v1_typed_taskset_override(dummy_taskset: str):
    config, _, _ = _parse_cli(
        ["vf-eval-v1", dummy_taskset, "--taskset.difficulty", "hard"]
    )
    assert config.taskset.difficulty == "hard"


def test_cli_v1_typed_harness_override(dummy_taskset: str):
    config, _, _ = _parse_cli(
        ["vf-eval-v1", dummy_taskset, "--harness.extra-field", "42"]
    )
    assert config.harness.extra_field == 42


def test_cli_v1_rejects_unknown_taskset_field(dummy_taskset: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval-v1", dummy_taskset, "--taskset.unknown", "x"])


def test_cli_v1_rejects_unknown_harness_field(dummy_taskset: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval-v1", dummy_taskset, "--harness.unknown", "x"])


def test_cli_v1_external_harness_module(dummy_taskset: str, external_harness: str):
    config, _, harness_name = _parse_cli(
        [
            "vf-eval-v1",
            dummy_taskset,
            external_harness,
            "--harness.ext-field",
            "custom",
        ]
    )
    assert harness_name == external_harness
    assert type(config.harness).__name__ == "ExtHarnessConfig"
    assert config.harness.ext_field == "custom"


def test_cli_v1_client_sub_config(dummy_taskset: str):
    config, _, _ = _parse_cli(
        [
            "vf-eval-v1",
            dummy_taskset,
            "--client.model",
            "gpt-test",
            "--client.provider",
            "openai",
        ]
    )
    assert config.client.model == "gpt-test"
    assert config.client.provider == "openai"


def test_cli_v1_sampling_sub_config(dummy_taskset: str):
    config, _, _ = _parse_cli(
        [
            "vf-eval-v1",
            dummy_taskset,
            "--sampling.temperature",
            "0.5",
            "--sampling.max-tokens",
            "256",
        ]
    )
    assert config.sampling.temperature == 0.5
    assert config.sampling.max_tokens == 256


def test_cli_v1_rejects_unknown_client_field(dummy_taskset: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval-v1", dummy_taskset, "--client.unknown", "x"])


def test_cli_v1_rejects_unknown_sampling_field(dummy_taskset: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval-v1", dummy_taskset, "--sampling.unknown", "x"])


def test_cli_v0_env_typed(v0_env: str):
    config, _, _ = _parse_cli(["vf-eval-v1", v0_env, "--env-args", '{"some_arg": 1}'])
    assert isinstance(config, ev1.EvalV0Config)
    assert config.env_args == {"some_arg": 1}


def test_cli_v0_env_rejects_positional_harness(v0_env: str):
    with pytest.raises(SystemExit):
        _parse_cli(["vf-eval-v1", v0_env, "rlm"])


# ---------------------------------------------------------------------------
# Sampling/client merge into legacy EvalConfig
# ---------------------------------------------------------------------------


def test_resolve_sampling_args_merges_typed_and_extras(dummy_taskset: str):
    config, _, _ = _parse_cli(
        [
            "vf-eval-v1",
            dummy_taskset,
            "--sampling.temperature",
            "0.5",
            "--sampling.max-tokens",
            "100",
            "--sampling.extras",
            '{"top_k": 40}',
        ]
    )
    args = ev1._resolve_sampling_args(config)
    assert args == {"temperature": 0.5, "max_tokens": 100, "top_k": 40}


def test_resolve_legacy_client_config_uses_provider_defaults(dummy_taskset: str):
    config, _, _ = _parse_cli(
        ["vf-eval-v1", dummy_taskset, "--client.provider", "openai"]
    )
    legacy = ev1._resolve_legacy_client_config(config)
    assert legacy.api_base_url == "https://api.openai.com/v1"
    assert legacy.api_key_var == "OPENAI_API_KEY"


# ---------------------------------------------------------------------------
# env_args round-trip through vf.load_environment dispatch
# ---------------------------------------------------------------------------


def test_v1_env_args_taskset_dispatch(dummy_taskset: str):
    config, _, _ = _parse_cli(
        ["vf-eval-v1", dummy_taskset, "--taskset.difficulty", "hard"]
    )
    env_args = ev1._v1_env_args(config)
    assert env_args[V1_TASKSET_KEY] == {"difficulty": "hard"}


def test_v1_env_args_harness_name_dispatch(dummy_taskset: str, external_harness: str):
    config, _, _ = _parse_cli(["vf-eval-v1", dummy_taskset, external_harness])
    env_args = ev1._v1_env_args(config)
    assert env_args[V1_HARNESS_KEY] == {"name": external_harness}


def test_v1_env_args_combined(dummy_taskset: str, external_harness: str):
    config, _, _ = _parse_cli(
        [
            "vf-eval-v1",
            dummy_taskset,
            external_harness,
            "--taskset.difficulty",
            "hard",
            "--harness.ext-field",
            "x",
        ]
    )
    env_args = ev1._v1_env_args(config)
    assert env_args[V1_TASKSET_KEY] == {"difficulty": "hard"}
    assert env_args[V1_HARNESS_KEY] == {"name": external_harness, "ext_field": "x"}


# ---------------------------------------------------------------------------
# vf.load_environment dispatch
# ---------------------------------------------------------------------------


def test_load_environment_v1_default_harness(dummy_taskset: str):
    env = vf.load_environment(dummy_taskset)
    assert isinstance(env, vf.Env)
    assert type(env.taskset).__name__ == "DummyTaskset"
    assert type(env.harness).__name__ == "DummyHarness"
    assert env.harness.config.extra_field == 7


def test_load_environment_v1_taskset_overrides(dummy_taskset: str):
    env = vf.load_environment(dummy_taskset, **{V1_TASKSET_KEY: {"difficulty": "hard"}})
    assert env.taskset.config.difficulty == "hard"


def test_load_environment_v1_harness_module_swap(
    dummy_taskset: str, external_harness: str
):
    env = vf.load_environment(
        dummy_taskset,
        **{V1_HARNESS_KEY: {"name": external_harness, "ext_field": "y"}},
    )
    assert type(env.harness).__name__ == "ExtHarness"
    assert env.harness.config.ext_field == "y"


def test_load_environment_v1_dispatch_preserves_env_args(dummy_taskset: str):
    env_args = {V1_HARNESS_KEY: {"extra_field": 5}}
    env = vf.load_environment(dummy_taskset, **env_args)
    assert env.env_args == env_args


# ---------------------------------------------------------------------------
# Lean taskset: only load_taskset, no load_harness.
# ---------------------------------------------------------------------------


def test_lean_taskset_auto_resolves_base_harness(lean_taskset: str):
    env = vf.load_environment(lean_taskset)
    assert type(env.taskset).__name__ == "LeanTaskset"
    assert type(env.harness) is Harness


def test_lean_taskset_load_environment_rejects_extra_kwargs(lean_taskset: str):
    with pytest.raises(Exception, match="load_taskset"):
        vf.load_environment(lean_taskset, foo=1)


# ---------------------------------------------------------------------------
# build_v1_env wiring
# ---------------------------------------------------------------------------


def test_build_v1_env_uses_taskset_load_harness(dummy_taskset: str):
    env = build_v1_env(dummy_taskset)
    assert type(env.harness).__name__ == "DummyHarness"


def test_build_v1_env_external_harness(dummy_taskset: str, external_harness: str):
    env = build_v1_env(
        dummy_taskset, harness_spec={"name": external_harness, "ext_field": "z"}
    )
    assert type(env.harness).__name__ == "ExtHarness"
    assert env.harness.config.ext_field == "z"


def test_has_v1_overrides():
    assert not has_v1_overrides({"foo": 1})
    assert has_v1_overrides({V1_TASKSET_KEY: {}})
    assert has_v1_overrides({V1_HARNESS_KEY: {}})


# ---------------------------------------------------------------------------
# Taskset.harness_defaults() — taskset-supplied harness opinions
# ---------------------------------------------------------------------------


OPINIONATED_TASKSET_SOURCE = textwrap.dedent(
    """
    import verifiers as vf


    class OpinionatedTasksetConfig(vf.TasksetConfig):
        pass


    class OpinionatedTaskset(vf.Taskset[OpinionatedTasksetConfig]):
        def load_tasks(self) -> vf.Tasks:
            return [{"prompt": [{"role": "user", "content": "x"}], "answer": "x"}]

        def harness_defaults(self) -> dict:
            return {"max_turns": 1, "system_prompt_merge": "harness"}


    def load_taskset(config: OpinionatedTasksetConfig) -> OpinionatedTaskset:
        return OpinionatedTaskset(config=config)
    """
)


BAD_DEFAULTS_TASKSET_SOURCE = textwrap.dedent(
    """
    import verifiers as vf


    class BadDefaultsTasksetConfig(vf.TasksetConfig):
        pass


    class BadDefaultsTaskset(vf.Taskset[BadDefaultsTasksetConfig]):
        def load_tasks(self) -> vf.Tasks:
            return [{"prompt": [{"role": "user", "content": "x"}], "answer": "x"}]

        def harness_defaults(self) -> dict:
            # Field does not exist on base HarnessConfig — should fail loudly.
            return {"bogus_field": 1}


    def load_taskset(config: BadDefaultsTasksetConfig) -> BadDefaultsTaskset:
        return BadDefaultsTaskset(config=config)
    """
)


@pytest.fixture
def opinionated_taskset(tmp_path: Path, monkeypatch):
    name = "opinionated_taskset"
    _install_module(tmp_path, monkeypatch, name, OPINIONATED_TASKSET_SOURCE)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def bad_defaults_taskset(tmp_path: Path, monkeypatch):
    name = "bad_defaults_taskset"
    _install_module(tmp_path, monkeypatch, name, BAD_DEFAULTS_TASKSET_SOURCE)
    yield name
    sys.modules.pop(name, None)


def test_taskset_defaults_applied_to_base_harness(opinionated_taskset: str):
    env = build_v1_env(opinionated_taskset)
    assert env.harness.config.max_turns == 1
    assert env.harness.config.system_prompt_merge == "harness"


def test_taskset_defaults_applied_to_external_harness(
    opinionated_taskset: str, external_harness: str
):
    env = build_v1_env(opinionated_taskset, harness_spec={"name": external_harness})
    # ExtHarnessConfig inherits HarnessConfig.max_turns, which the taskset opined.
    assert env.harness.config.max_turns == 1
    assert env.harness.config.system_prompt_merge == "harness"


def test_cli_overrides_win_over_taskset_defaults(opinionated_taskset: str):
    env = build_v1_env(opinionated_taskset, harness_spec={"max_turns": 7})
    # CLI override beats taskset default
    assert env.harness.config.max_turns == 7
    # taskset default still applies for fields the user didn't set
    assert env.harness.config.system_prompt_merge == "harness"


def test_taskset_defaults_round_trip_through_load_environment(
    opinionated_taskset: str,
):
    """Workers re-run build_v1_env from env_args; taskset defaults must
    re-apply because they live on the Taskset class."""
    env = vf.load_environment(opinionated_taskset)
    assert env.harness.config.max_turns == 1


def test_bad_taskset_defaults_fail_loudly(bad_defaults_taskset: str):
    with pytest.raises(Exception, match="bogus_field"):
        build_v1_env(bad_defaults_taskset)


def test_base_taskset_harness_defaults_empty(dummy_taskset: str):
    """Tasksets that don't override harness_defaults() return an empty dict."""
    import importlib

    module = importlib.import_module(dummy_taskset)
    taskset = module.load_taskset(module.DummyTasksetConfig())
    assert taskset.harness_defaults() == {}
