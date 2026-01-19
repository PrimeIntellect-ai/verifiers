import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import verifiers.scripts.eval as vf_eval
import verifiers.utils.eval_utils
from verifiers.types import (
    GenerateMetadata,
    GenerateOutputs,
    State,
)
from verifiers.utils.eval_utils import (
    is_toml_config,
    load_toml_config,
)


def _make_metadata(eval_config) -> GenerateMetadata:
    return GenerateMetadata(
        env_id=eval_config.env.env_id,
        env_args=eval_config.env.env_args,
        model=eval_config.model.model,
        base_url=eval_config.model.client_config.api_base_url,
        num_examples=eval_config.env.num_examples,
        rollouts_per_example=eval_config.env.rollouts_per_example,
        sampling_args=eval_config.model.sampling_args,
        date="1970-01-01",
        time_ms=0.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=eval_config.env.state_columns or [],
        path_to_save=Path("test.jsonl"),
    )


def _make_generate_outputs(
    env_id: str = "test-env",
    model: str = "gpt-4.1-mini",
    num_examples: int = 1,
    rollouts_per_example: int = 1,
    rewards: list[float] | None = None,
    tasks: list[str] | None = None,
) -> GenerateOutputs:
    """Helper to create GenerateOutputs for testing."""
    n = num_examples * rollouts_per_example
    rewards = rewards or [1.0] * n
    tasks = tasks or ["default"] * n
    return GenerateOutputs(
        prompt=[[{"role": "user", "content": "p"}] for _ in range(n)],
        completion=[[{"role": "assistant", "content": "c"}] for _ in range(n)],
        answer=["" for _ in range(n)],
        state=[
            State(
                timing={
                    "generation_ms": 100.0,
                    "scoring_ms": 50.0,
                    "total_ms": 150.0,
                }
            )
            for _ in range(n)
        ],
        task=tasks,
        info=[{} for _ in range(n)],
        example_id=list(range(n)),
        reward=rewards,
        metrics={"accuracy": rewards},
        stop_conditions=[None for _ in range(n)],
        is_truncated=[False for _ in range(n)],
        metadata=GenerateMetadata(
            env_id=env_id,
            env_args={},
            model=model,
            base_url="https://api.openai.com/v1",
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            sampling_args={"max_tokens": 100},
            date="1970-01-01",
            time_ms=0.0,
            avg_reward=sum(rewards) / len(rewards),
            avg_metrics={},
            state_columns=[],
            path_to_save=Path("test.jsonl"),
        ),
    )


def _run_cli(monkeypatch, overrides, capture_all_configs: bool = False):
    """Run CLI with mocked arguments and capture config(s).

    Args:
        monkeypatch: pytest monkeypatch fixture
        overrides: dict of args to override
        capture_all_configs: if True, returns list of all configs (for multi-env)
    """
    base_args = {
        "env_id_or_path": "dummy-env",
        "env_args": {},
        "env_dir_path": "./environments",
        "endpoints_path": "./configs/endpoints.py",
        "model": "gpt-4.1-mini",
        "api_key_var": "OPENAI_API_KEY",
        "api_base_url": "https://api.openai.com/v1",
        "header": None,
        "num_examples": 1,
        "rollouts_per_example": 1,
        "max_concurrent": 1,
        "max_concurrent_generation": None,
        "max_concurrent_scoring": None,
        "independent_scoring": False,
        "max_tokens": 42,
        "temperature": 0.9,
        "sampling_args": None,
        "verbose": False,
        "print_results": False,
        "no_interleave_scoring": False,
        "state_columns": [],
        "save_results": False,
        "save_every": -1,
        "save_to_hf_hub": False,
        "hf_hub_dataset_name": "",
        "extra_env_kwargs": {},
    }
    base_args.update(overrides)
    args_namespace = SimpleNamespace(**base_args)

    captured: dict = {"sampling_args": None, "configs": []}

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: args_namespace,
    )
    monkeypatch.setattr(vf_eval, "setup_logging", lambda *_, **__: None)
    monkeypatch.setattr(vf_eval, "load_endpoints", lambda *_: {})

    async def fake_run_evaluation(eval_config, run_config):
        captured["sampling_args"] = dict(eval_config.model.sampling_args)
        captured["configs"].append(eval_config)
        metadata = _make_metadata(eval_config)
        return GenerateOutputs(
            prompt=[[{"role": "user", "content": "p"}]],
            completion=[[{"role": "assistant", "content": "c"}]],
            answer=[""],
            state=[
                State(
                    timing={
                        "generation_ms": 0.0,
                        "scoring_ms": 0.0,
                        "total_ms": 0.0,
                    }
                )
            ],
            task=["default"],
            info=[{}],
            example_id=[0],
            reward=[1.0],
            metrics={},
            stop_conditions=[None],
            is_truncated=[False],
            metadata=metadata,
        )

    monkeypatch.setattr(
        verifiers.utils.eval_utils, "run_evaluation", fake_run_evaluation
    )

    vf_eval.main()
    return captured


def test_cli_single_env_id(monkeypatch):
    """Single env ID without comma creates one config."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "env1",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 1
    assert configs[0].env.env_id == "env1"


def test_cli_sampling_args_precedence_over_flags(monkeypatch):
    """sampling_args JSON takes precedence over individual flags."""
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": {
                "enable_thinking": False,
                "max_tokens": 77,
                "temperature": 0.1,
            },
        },
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 77
    assert sa["temperature"] == 0.1
    assert sa["enable_thinking"] is False


def test_cli_sampling_args_fill_from_flags_when_missing(monkeypatch):
    """Flags fill in missing sampling_args values."""
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": {"enable_thinking": True},
            "max_tokens": 55,
            "temperature": 0.8,
        },
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 55
    assert sa["temperature"] == 0.8
    assert sa["enable_thinking"] is True


def test_cli_no_sampling_args_uses_flags(monkeypatch):
    """When no sampling_args provided, uses flag values."""
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": None,
            "max_tokens": 128,
            "temperature": 0.5,
        },
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 128
    assert sa["temperature"] == 0.5


def test_cli_temperature_not_added_when_none(monkeypatch):
    """Temperature flag with None is not added to sampling_args."""
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": None,
            "max_tokens": 100,
            "temperature": None,
        },
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 100
    assert "temperature" not in sa


def test_cli_comma_separated_env_ids(monkeypatch):
    """CLI with comma-separated env IDs creates multiple eval configs."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "env1,env2,env3",
            "num_examples": 5,
            "rollouts_per_example": 3,
        },
    )

    configs = captured["configs"]
    assert len(configs) == 3
    assert configs[0].env.env_id == "env1"
    assert configs[1].env.env_id == "env2"
    assert configs[2].env.env_id == "env3"


def test_cli_comma_separated_with_spaces(monkeypatch):
    """CLI handles spaces around commas in env IDs."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "env1 , env2 , env3",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 3
    assert configs[0].env.env_id == "env1"
    assert configs[1].env.env_id == "env2"
    assert configs[2].env.env_id == "env3"


def test_cli_comma_separated_with_org_prefix(monkeypatch):
    """CLI handles org/env format in comma-separated list."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "org/env1,org/env2",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env.env_id == "org/env1"
    assert configs[1].env.env_id == "org/env2"


def test_cli_comma_separated_shared_args(monkeypatch):
    """All comma-separated envs inherit shared CLI args."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "env1,env2,env3",
            "num_examples": 10,
            "rollouts_per_example": 4,
            "max_concurrent": 16,
            "max_tokens": 512,
            "temperature": 0.7,
        },
    )

    configs = captured["configs"]
    assert len(configs) == 3
    for config in configs:
        assert config.env.num_examples == 10
        assert config.env.rollouts_per_example == 4
        assert config.model.max_concurrent == 16
        assert config.model.sampling_args["max_tokens"] == 512
        assert config.model.sampling_args["temperature"] == 0.7


def test_cli_comma_separated_ignores_empty_entries(monkeypatch):
    """CLI ignores empty entries from trailing/double commas."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "gsm8k,,alphabet-sort,",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env.env_id == "gsm8k"
    assert configs[1].env.env_id == "alphabet-sort"


def test_is_toml_config_with_valid_toml():
    """Valid TOML file path returns True."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b"[[env]]\nenv_id = 'test'\n")
        f.flush()
        assert is_toml_config(f.name) is True


def test_is_toml_config_with_nonexistent_file():
    """Nonexistent file returns False."""
    assert is_toml_config("/nonexistent/path/config.toml") is False


def test_is_toml_config_with_non_toml_extension():
    """File with non-toml extension returns False."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"test: value\n")
        f.flush()
        assert is_toml_config(f.name) is False


def test_is_toml_config_with_directory():
    """Directory path returns False."""
    with tempfile.TemporaryDirectory() as d:
        assert is_toml_config(d) is False


def test_is_toml_config_with_env_id():
    """Simple env ID string returns False."""
    assert is_toml_config("env") is False
    assert is_toml_config("env1,env2") is False
    assert is_toml_config("org/env") is False


def test_load_toml_config_single_env():
    """Single env loads correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env1"\n')
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result["evals"]) == 1
        assert result["evals"][0]["env"]["env_id"] == "env1"


def test_load_toml_config_multi_env():
    """Multiple envs load correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env1"\n\n[[env]]\nenv_id = "env2"\n')
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result["evals"]) == 2
        assert result["evals"][0]["env"]["env_id"] == "env1"
        assert result["evals"][1]["env"]["env_id"] == "env2"


def test_load_toml_config_with_env_args():
    """Multiple sections with env_args field loads correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env1"\n[env.env_args]\nsplit = "train"\nmax_examples = 100\n'
        )
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result["evals"]) == 1
        assert result["evals"][0]["env"]["env_id"] == "env1"
        assert result["evals"][0]["env"]["env_args"]["split"] == "train"
        assert result["evals"][0]["env"]["env_args"]["max_examples"] == 100


def test_load_toml_config_missing_env_section():
    """TOML without [[env]] section raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('model = "env1"\nmax_tokens = 100\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_empty_env_list():
    """Empty env list raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write("env = []\n")
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_missing_env_id():
    """[[env]] without env_id field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nname = "env1"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_partial_missing_env_id():
    """Some [[env]] sections missing env_id raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env1"\n\n[[env]]\nname = "env2"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_invalid_field():
    """[[env]] with invalid field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env1"\ninvalid_field = "value"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_cli_multi_env_via_toml_config(monkeypatch):
    """CLI with TOML config creates multiple eval configs."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env1"\n\n[[env]]\nenv_id = "env2"\n')
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": 5,
                "rollouts_per_example": 2,
            },
            capture_all_configs=True,
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env.env_id == "env1"
    assert configs[1].env.env_id == "env2"


def test_cli_multi_env_shares_common_args(monkeypatch):
    """All env configs in multi-env inherit common CLI args."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env1"\n\n[[env]]\nenv_id = "env2"\n')
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": 10,
                "rollouts_per_example": 4,
                "max_concurrent": 16,
                "max_tokens": 512,
            },
        )

    configs = captured["configs"]
    for config in configs:
        assert config.env.num_examples == 10
        assert config.env.rollouts_per_example == 4
        assert config.model.max_concurrent == 16
        assert config.model.sampling_args["max_tokens"] == 512


def test_cli_toml_per_env_num_examples(monkeypatch):
    """TOML per-env num_examples is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env1"\nnum_examples = 10\n\n'
            '[[env]]\nenv_id = "env2"\nnum_examples = 20\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": None,  # not provided via CLI
                "rollouts_per_example": 1,
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env.env_id == "env1"
    assert configs[0].env.num_examples == 10
    assert configs[1].env.env_id == "env2"
    assert configs[1].env.num_examples == 20


def test_cli_toml_per_env_rollouts_per_example(monkeypatch):
    """TOML per-env rollouts_per_example is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env1"\nrollouts_per_example = 3\n\n'
            '[[env]]\nenv_id = "env2"\nrollouts_per_example = 5\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": 1,
                "rollouts_per_example": None,  # not provided via CLI
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env.rollouts_per_example == 3
    assert configs[1].env.rollouts_per_example == 5


def test_cli_toml_per_env_overrides_cli_args(monkeypatch):
    """TOML per-env settings take precedence over CLI args."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env-a"\nnum_examples = 100\nrollouts_per_example = 10\n\n'
            '[[env]]\nenv_id = "env-b"\nnum_examples = 200\nrollouts_per_example = 20\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": 5,  # CLI arg (lower priority)
                "rollouts_per_example": 2,  # CLI arg (lower priority)
            },
        )

    configs = captured["configs"]
    # TOML per-env settings should override CLI args
    assert configs[0].env.num_examples == 100
    assert configs[0].env.rollouts_per_example == 10
    assert configs[1].env.num_examples == 200
    assert configs[1].env.rollouts_per_example == 20


def test_cli_toml_mixed_per_env_and_cli_fallback(monkeypatch):
    """TOML with some envs having settings, others fall back to CLI args."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env-with-settings"\nnum_examples = 15\nrollouts_per_example = 4\n\n'
            '[[env]]\nenv_id = "env-without-settings"\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": 10,  # CLI fallback for envs without TOML settings
                "rollouts_per_example": 2,  # CLI fallback
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    # First env uses TOML settings (takes precedence over CLI)
    assert configs[0].env.env_id == "env-with-settings"
    assert configs[0].env.num_examples == 15
    assert configs[0].env.rollouts_per_example == 4
    # Second env uses CLI args as fallback
    assert configs[1].env.env_id == "env-without-settings"
    assert configs[1].env.num_examples == 10
    assert configs[1].env.rollouts_per_example == 2


def test_cli_toml_without_settings_uses_defaults(monkeypatch):
    """TOML envs without settings and no CLI args use global defaults."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env-a"\n\n[[env]]\nenv_id = "env-b"\n')
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": None,
                "rollouts_per_example": None,
            },
        )

    configs = captured["configs"]
    # Both envs use global defaults
    for config in configs:
        assert config.env.num_examples == 5  # DEFAULT_NUM_EXAMPLES
        assert config.env.rollouts_per_example == 3  # DEFAULT_ROLLOUTS_PER_EXAMPLE


def test_cli_toml_global_env_id(monkeypatch):
    """TOML with global env_id in [env] defaults applies to all [[eval]] entries."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[env]\nenv_id = "shared-env"\nnum_examples = 10\n\n'
            '[[eval]]\n[eval.model]\nmodel = "model-a"\n\n'
            '[[eval]]\n[eval.model]\nmodel = "model-b"\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": None,
                "rollouts_per_example": 1,
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    # Both evals should use the global env_id
    assert configs[0].env.env_id == "shared-env"
    assert configs[1].env.env_id == "shared-env"
    # And inherit env settings from defaults
    assert configs[0].env.num_examples == 10
    assert configs[1].env.num_examples == 10


def test_cli_toml_per_eval_env_id_overrides_global(monkeypatch):
    """Per-eval env_id overrides global env_id from [env] defaults."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[env]\nenv_id = "default-env"\n\n'
            '[[eval]]\n[eval.env]\nenv_id = "override-env"\n\n'
            "[[eval]]\n"
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_path": f.name,
                "num_examples": 5,
                "rollouts_per_example": 1,
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    # First eval overrides with per-eval env_id
    assert configs[0].env.env_id == "override-env"
    # Second eval uses global default
    assert configs[1].env.env_id == "default-env"
