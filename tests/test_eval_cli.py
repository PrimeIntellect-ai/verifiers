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
from verifiers.utils.eval_utils import load_toml_config


def _make_metadata(config) -> GenerateMetadata:
    return GenerateMetadata(
        env_id=config.env_id,
        env_args=config.env_args,
        model=config.model,
        base_url=config.client_config.api_base_url,
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
        sampling_args=config.sampling_args,
        date="1970-01-01",
        time_ms=0.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=config.state_columns or [],
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
        "env_id_or_config": "dummy-env",
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
        "no_interleave_scoring": False,
        "state_columns": [],
        "save_results": False,
        "save_every": -1,
        "save_to_hf_hub": False,
        "hf_hub_dataset_name": "",
        "extra_env_kwargs": {},
        "max_retries": 0,
        "tui": False,
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

    async def fake_run_evaluation(config, **kwargs):
        captured["sampling_args"] = dict(config.sampling_args)
        captured["configs"].append(config)
        metadata = _make_metadata(config)
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
            "env_id_or_config": "env1",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 1
    assert configs[0].env_id == "env1"


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


def test_load_toml_config_single_eval():
    """Single env loads correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n')
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result) == 1
        assert result[0]["env_id"] == "env1"


def test_load_toml_config_multi_env():
    """Multiple envs load correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n\n[[eval]]\nenv_id = "env2"\n')
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result) == 2
        assert result[0]["env_id"] == "env1"
        assert result[1]["env_id"] == "env2"


def test_load_toml_config_with_env_args():
    """Multiple sections with env_args field loads correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env1"\n[eval.env_args]\nsplit = "train"\nmax_examples = 100\n'
        )
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result) == 1
        assert result[0]["env_id"] == "env1"
        assert result[0]["env_args"]["split"] == "train"
        assert result[0]["env_args"]["max_examples"] == 100


def test_load_toml_config_missing_env_section():
    """TOML without [[eval]] section raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('model = "env1"\nmax_tokens = 100\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_empty_eval_list():
    """Empty eval list raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write("eval = []\n")
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_missing_env_id():
    """[[eval]] without env_id field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nname = "env1"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_partial_missing_env_id():
    """Some [[eval]] sections missing env_id raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n\n[[eval]]\nname = "env2"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_load_toml_config_invalid_field():
    """[[eval]] with invalid field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\ninvalid_field = "value"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))


def test_cli_multi_env_via_toml_config(monkeypatch):
    """CLI with TOML config creates multiple eval configs."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n\n[[eval]]\nenv_id = "env2"\n')
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": 5,
                "rollouts_per_example": 2,
            },
            capture_all_configs=True,
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env_id == "env1"
    assert configs[1].env_id == "env2"


def test_cli_toml_ignores_cli_args(monkeypatch):
    """TOML config ignores CLI args, uses defaults for unspecified values."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n\n[[eval]]\nenv_id = "env2"\n')
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": 10,  # CLI arg ignored
                "rollouts_per_example": 4,  # CLI arg ignored
                "max_concurrent": 16,  # CLI arg ignored
                "max_tokens": 512,  # CLI arg ignored
            },
        )

    configs = captured["configs"]
    for config in configs:
        # Uses global defaults, not CLI args
        assert config.num_examples == 5  # DEFAULT_NUM_EXAMPLES
        assert config.rollouts_per_example == 3  # DEFAULT_ROLLOUTS_PER_EXAMPLE
        assert config.max_concurrent == 32  # default
        assert config.sampling_args["max_tokens"] is None  # default


def test_cli_toml_per_env_num_examples(monkeypatch):
    """TOML per-env num_examples is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env1"\nnum_examples = 10\n\n'
            '[[eval]]\nenv_id = "env2"\nnum_examples = 20\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": None,  # not provided via CLI
                "rollouts_per_example": 1,
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env_id == "env1"
    assert configs[0].num_examples == 10
    assert configs[1].env_id == "env2"
    assert configs[1].num_examples == 20


def test_cli_toml_per_env_rollouts_per_example(monkeypatch):
    """TOML per-env rollouts_per_example is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env1"\nrollouts_per_example = 3\n\n'
            '[[eval]]\nenv_id = "env2"\nrollouts_per_example = 5\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": 1,
                "rollouts_per_example": None,  # not provided via CLI
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].rollouts_per_example == 3
    assert configs[1].rollouts_per_example == 5


def test_cli_toml_per_eval_settings_used(monkeypatch):
    """TOML per-eval settings are used (CLI args ignored when using config)."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env-a"\nnum_examples = 100\nrollouts_per_example = 10\n\n'
            '[[eval]]\nenv_id = "env-b"\nnum_examples = 200\nrollouts_per_example = 20\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": 5,  # CLI arg ignored
                "rollouts_per_example": 2,  # CLI arg ignored
            },
        )

    configs = captured["configs"]
    # TOML per-eval settings are used
    assert configs[0].num_examples == 100
    assert configs[0].rollouts_per_example == 10
    assert configs[1].num_examples == 200
    assert configs[1].rollouts_per_example == 20


def test_cli_toml_mixed_per_env_and_defaults_fallback(monkeypatch):
    """TOML with some evals having settings, others fall back to global defaults."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env-with-settings"\nnum_examples = 15\nrollouts_per_example = 4\n\n'
            '[[eval]]\nenv_id = "env-without-settings"\n'
        )
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": 10,  # CLI arg ignored when using config
                "rollouts_per_example": 2,  # CLI arg ignored when using config
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    # First env uses TOML per-eval settings
    assert configs[0].env_id == "env-with-settings"
    assert configs[0].num_examples == 15
    assert configs[0].rollouts_per_example == 4
    # Second env uses global defaults (CLI args ignored)
    assert configs[1].env_id == "env-without-settings"
    assert configs[1].num_examples == 5  # DEFAULT_NUM_EXAMPLES
    assert configs[1].rollouts_per_example == 3  # DEFAULT_ROLLOUTS_PER_EXAMPLE


def test_cli_toml_without_settings_uses_defaults(monkeypatch):
    """TOML evals without settings use global defaults."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env-a"\n\n[[eval]]\nenv_id = "env-b"\n')
        f.flush()
        captured = _run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": None,
                "rollouts_per_example": None,
            },
        )

    configs = captured["configs"]
    # Both evals use global defaults
    for config in configs:
        assert config.num_examples == 5  # DEFAULT_NUM_EXAMPLES
        assert config.rollouts_per_example == 3  # DEFAULT_ROLLOUTS_PER_EXAMPLE


def test_load_toml_config_global_values_with_per_eval_override():
    """Global values at top of config are inherited by evals, with per-eval overrides."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            'model = "gpt-5"\n'
            "num_examples = 100\n"
            "\n"
            "[[eval]]\n"
            'env_id = "env1"\n'
            "\n"
            "[[eval]]\n"
            'env_id = "env2"\n'
            "num_examples = 50\n"
        )
        f.flush()
        result = load_toml_config(Path(f.name))

    assert len(result) == 2
    # First eval inherits global values
    assert result[0]["env_id"] == "env1"
    assert result[0]["model"] == "gpt-5"
    assert result[0]["num_examples"] == 100
    # Second eval has per-eval override for num_examples
    assert result[1]["env_id"] == "env2"
    assert result[1]["model"] == "gpt-5"  # still inherits global
    assert result[1]["num_examples"] == 50  # per-eval override


def test_load_toml_config_invalid_global_field():
    """Invalid global field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('invalid_global = "value"\n\n[[eval]]\nenv_id = "env1"\n')
        f.flush()
        with pytest.raises(ValueError):
            load_toml_config(Path(f.name))
