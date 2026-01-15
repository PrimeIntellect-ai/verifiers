import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import verifiers.scripts.eval as vf_eval
import verifiers.utils.eval_utils
from verifiers.types import (
    EvalConfig,
    GenerateMetadata,
    GenerateOutputs,
    MultiEvalConfig,
    State,
)
from verifiers.utils.eval_utils import (
    get_results_by_task,
    is_toml_config,
    load_toml_config,
)


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

    async def fake_run_evaluation(config):
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


# =============================================================================
# Tests for sampling args CLI handling
# =============================================================================


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


# =============================================================================
# Tests for is_toml_config
# =============================================================================


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
    assert is_toml_config("gsm8k") is False
    assert is_toml_config("primeintellect/math-python") is False


# =============================================================================
# Tests for load_toml_config
# =============================================================================


def test_load_toml_config_single_env():
    """Single [[env]] section loads correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "gsm8k"\n')
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result) == 1
        assert result[0]["env_id"] == "gsm8k"


def test_load_toml_config_multi_env():
    """Multiple [[env]] sections load correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "gsm8k"\n\n[[env]]\nenv_id = "math-python"\n')
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result) == 2
        assert result[0]["env_id"] == "gsm8k"
        assert result[1]["env_id"] == "math-python"


def test_load_toml_config_with_env_args():
    """[[env]] section with env_args field loads correctly."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "test-env"\n[env.env_args]\nsplit = "train"\nmax_examples = 100\n'
        )
        f.flush()
        result = load_toml_config(Path(f.name))
        assert len(result) == 1
        assert result[0]["env_id"] == "test-env"
        assert result[0]["env_args"]["split"] == "train"
        assert result[0]["env_args"]["max_examples"] == 100


def test_load_toml_config_missing_file():
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_toml_config(Path("/nonexistent/path/config.toml"))


def test_load_toml_config_missing_env_section():
    """TOML without [[env]] section raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('model = "gpt-4"\nmax_tokens = 100\n')
        f.flush()
        with pytest.raises(ValueError, match="must contain at least one"):
            load_toml_config(Path(f.name))


def test_load_toml_config_empty_env_list():
    """Empty env list raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write("env = []\n")
        f.flush()
        with pytest.raises(ValueError, match="must contain at least one"):
            load_toml_config(Path(f.name))


def test_load_toml_config_missing_env_id():
    """[[env]] without env_id field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nname = "test"\n')
        f.flush()
        with pytest.raises(ValueError, match="must contain an env_id field"):
            load_toml_config(Path(f.name))


def test_load_toml_config_partial_missing_env_id():
    """Some [[env]] sections missing env_id raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "valid"\n\n[[env]]\nname = "no-id"\n')
        f.flush()
        with pytest.raises(ValueError, match="must contain an env_id field"):
            load_toml_config(Path(f.name))


def test_load_toml_config_invalid_field():
    """[[env]] with invalid field raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "test"\ninvalid_field = "value"\n')
        f.flush()
        with pytest.raises(ValueError, match="Invalid field"):
            load_toml_config(Path(f.name))


# =============================================================================
# Tests for get_results_by_task
# =============================================================================


def test_get_results_by_task_single_task():
    """Results with single task returns one group."""
    results = _make_generate_outputs(
        num_examples=3,
        rollouts_per_example=1,
        tasks=["default", "default", "default"],
        rewards=[0.5, 0.7, 0.9],
    )
    by_task = get_results_by_task(results)

    assert len(by_task) == 1
    assert "default" in by_task
    assert len(by_task["default"]["reward"]) == 3
    assert by_task["default"]["reward"] == [0.5, 0.7, 0.9]


def test_get_results_by_task_multiple_tasks():
    """Results with multiple tasks are grouped correctly."""
    results = _make_generate_outputs(
        num_examples=4,
        rollouts_per_example=1,
        tasks=["math", "code", "math", "code"],
        rewards=[0.5, 0.6, 0.7, 0.8],
    )
    by_task = get_results_by_task(results)

    assert len(by_task) == 2
    assert "math" in by_task
    assert "code" in by_task
    assert by_task["math"]["reward"] == [0.5, 0.7]
    assert by_task["code"]["reward"] == [0.6, 0.8]


def test_get_results_by_task_preserves_metadata():
    """Grouped results preserve metadata."""
    results = _make_generate_outputs(
        env_id="test-env",
        num_examples=2,
        rollouts_per_example=1,
        tasks=["task1", "task2"],
    )
    by_task = get_results_by_task(results)

    # Both task groups should have the same metadata
    assert by_task["task1"]["metadata"]["env_id"] == "test-env"
    assert by_task["task2"]["metadata"]["env_id"] == "test-env"


def test_get_results_by_task_preserves_metrics():
    """Grouped results preserve metrics correctly."""
    results = _make_generate_outputs(
        num_examples=4,
        rollouts_per_example=1,
        tasks=["a", "b", "a", "b"],
        rewards=[0.1, 0.2, 0.3, 0.4],
    )
    by_task = get_results_by_task(results)

    # Metrics should be split by task
    assert by_task["a"]["metrics"]["accuracy"] == [0.1, 0.3]
    assert by_task["b"]["metrics"]["accuracy"] == [0.2, 0.4]


# =============================================================================
# Tests for multi-env evaluation via TOML config
# =============================================================================


def test_cli_multi_env_via_toml_config(monkeypatch):
    """CLI with TOML config creates multiple eval configs."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env-one"\n\n[[env]]\nenv_id = "env-two"\n')
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
    assert configs[0].env_id == "env-one"
    assert configs[1].env_id == "env-two"


def test_cli_multi_env_shares_common_args(monkeypatch):
    """All env configs in multi-env inherit common CLI args."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[env]]\nenv_id = "env-a"\n\n[[env]]\nenv_id = "env-b"\n')
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
        assert config.num_examples == 10
        assert config.rollouts_per_example == 4
        assert config.max_concurrent == 16
        assert config.sampling_args["max_tokens"] == 512


def test_cli_toml_per_env_num_examples(monkeypatch):
    """TOML per-env num_examples is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env-a"\nnum_examples = 10\n\n'
            '[[env]]\nenv_id = "env-b"\nnum_examples = 20\n'
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
    assert configs[0].env_id == "env-a"
    assert configs[0].num_examples == 10
    assert configs[1].env_id == "env-b"
    assert configs[1].num_examples == 20


def test_cli_toml_per_env_rollouts_per_example(monkeypatch):
    """TOML per-env rollouts_per_example is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[env]]\nenv_id = "env-a"\nrollouts_per_example = 3\n\n'
            '[[env]]\nenv_id = "env-b"\nrollouts_per_example = 5\n'
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
    assert configs[0].rollouts_per_example == 3
    assert configs[1].rollouts_per_example == 5


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
    assert configs[0].num_examples == 100
    assert configs[0].rollouts_per_example == 10
    assert configs[1].num_examples == 200
    assert configs[1].rollouts_per_example == 20


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
    assert configs[0].env_id == "env-with-settings"
    assert configs[0].num_examples == 15
    assert configs[0].rollouts_per_example == 4
    # Second env uses CLI args as fallback
    assert configs[1].env_id == "env-without-settings"
    assert configs[1].num_examples == 10
    assert configs[1].rollouts_per_example == 2


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
        assert config.num_examples == 5  # DEFAULT_NUM_EXAMPLES
        assert config.rollouts_per_example == 3  # DEFAULT_ROLLOUTS_PER_EXAMPLE


# =============================================================================
# Tests for multi-env evaluation via comma-separated IDs
# =============================================================================


def test_cli_comma_separated_env_ids(monkeypatch):
    """CLI with comma-separated env IDs creates multiple eval configs."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "gsm8k,alphabet-sort,math-python",
            "num_examples": 5,
            "rollouts_per_example": 3,
        },
    )

    configs = captured["configs"]
    assert len(configs) == 3
    assert configs[0].env_id == "gsm8k"
    assert configs[1].env_id == "alphabet-sort"
    assert configs[2].env_id == "math-python"


def test_cli_comma_separated_env_ids_two_envs(monkeypatch):
    """CLI with two comma-separated env IDs."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "env-a,env-b",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env_id == "env-a"
    assert configs[1].env_id == "env-b"


def test_cli_comma_separated_shares_common_args(monkeypatch):
    """All comma-separated env configs inherit common CLI args."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "env-one,env-two,env-three",
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
        assert config.num_examples == 10
        assert config.rollouts_per_example == 4
        assert config.max_concurrent == 16
        assert config.sampling_args["max_tokens"] == 512
        assert config.sampling_args["temperature"] == 0.7


def test_cli_comma_separated_with_spaces(monkeypatch):
    """CLI handles spaces around commas in env IDs."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "gsm8k , alphabet-sort , math-python",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 3
    assert configs[0].env_id == "gsm8k"
    assert configs[1].env_id == "alphabet-sort"
    assert configs[2].env_id == "math-python"


def test_cli_comma_separated_with_org_prefix(monkeypatch):
    """CLI handles org/env format in comma-separated list."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "primeintellect/gsm8k,primeintellect/math-python",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env_id == "primeintellect/gsm8k"
    assert configs[1].env_id == "primeintellect/math-python"


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
    assert configs[0].env_id == "gsm8k"
    assert configs[1].env_id == "alphabet-sort"


def test_cli_single_env_id_no_comma(monkeypatch):
    """Single env ID without comma creates one config."""
    captured = _run_cli(
        monkeypatch,
        {
            "env_id_or_path": "gsm8k",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 1
    assert configs[0].env_id == "gsm8k"


# =============================================================================
# Tests for header parsing
# =============================================================================


def test_cli_header_parsing_single(monkeypatch):
    """Single header is parsed correctly."""
    captured = _run_cli(
        monkeypatch,
        {
            "header": ["X-Custom-Header: my-value"],
        },
    )

    config = captured["configs"][0]
    assert config.client_config.extra_headers == {"X-Custom-Header": "my-value"}


def test_cli_header_parsing_multiple(monkeypatch):
    """Multiple headers are parsed correctly."""
    captured = _run_cli(
        monkeypatch,
        {
            "header": [
                "Authorization: Bearer token123",
                "X-Request-ID: req-456",
            ],
        },
    )

    config = captured["configs"][0]
    assert config.client_config.extra_headers == {
        "Authorization": "Bearer token123",
        "X-Request-ID": "req-456",
    }


def test_cli_header_with_multiple_colons(monkeypatch):
    """Header with multiple colons is parsed correctly (value can contain colons)."""
    captured = _run_cli(
        monkeypatch,
        {
            "header": ["X-URL: https://example.com:8080/path"],
        },
    )

    config = captured["configs"][0]
    assert config.client_config.extra_headers == {
        "X-URL": "https://example.com:8080/path"
    }


def test_cli_header_invalid_format_no_colon(monkeypatch):
    """Header without colon raises ValueError."""
    with pytest.raises(ValueError, match="must be 'Name: Value'"):
        _run_cli(
            monkeypatch,
            {
                "header": ["InvalidHeaderNoColon"],
            },
        )


def test_cli_header_invalid_empty_name(monkeypatch):
    """Header with empty name raises ValueError."""
    with pytest.raises(ValueError, match="name cannot be empty"):
        _run_cli(
            monkeypatch,
            {
                "header": [": value-only"],
            },
        )


# =============================================================================
# Tests for state columns parsing
# =============================================================================


def test_cli_state_columns_parsed(monkeypatch):
    """State columns string is parsed into list."""
    captured = _run_cli(
        monkeypatch,
        {
            "state_columns": ["turn", "timing", "responses"],
        },
    )

    config = captured["configs"][0]
    assert config.state_columns == ["turn", "timing", "responses"]


def test_cli_state_columns_empty(monkeypatch):
    """Empty state columns list is handled."""
    captured = _run_cli(
        monkeypatch,
        {
            "state_columns": [],
        },
    )

    config = captured["configs"][0]
    assert config.state_columns == []


# =============================================================================
# Tests for endpoint resolution
# =============================================================================


def test_cli_endpoint_from_registry(monkeypatch):
    """Model found in endpoint registry uses registry values."""
    base_args = {
        "env_id_or_path": "test-env",
        "env_args": {},
        "env_dir_path": "./environments",
        "endpoints_path": "./configs/endpoints.py",
        "model": "my-alias",
        "api_key_var": None,
        "api_base_url": None,
        "header": None,
        "num_examples": 1,
        "rollouts_per_example": 1,
        "max_concurrent": 1,
        "max_concurrent_generation": None,
        "max_concurrent_scoring": None,
        "independent_scoring": False,
        "max_tokens": 100,
        "temperature": None,
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
    args_namespace = SimpleNamespace(**base_args)
    captured: dict = {"configs": []}

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: args_namespace,
    )
    monkeypatch.setattr(vf_eval, "setup_logging", lambda *_, **__: None)

    # Mock endpoint registry
    monkeypatch.setattr(
        vf_eval,
        "load_endpoints",
        lambda *_: {
            "my-alias": {
                "key": "MY_CUSTOM_KEY",
                "url": "https://custom.api.com/v1",
                "model": "actual-model-name",
            }
        },
    )

    async def fake_run_evaluation(config):
        captured["configs"].append(config)
        return _make_generate_outputs(env_id=config.env_id)

    monkeypatch.setattr(
        verifiers.utils.eval_utils, "run_evaluation", fake_run_evaluation
    )

    vf_eval.main()

    config = captured["configs"][0]
    assert config.client_config.api_key_var == "MY_CUSTOM_KEY"
    assert config.client_config.api_base_url == "https://custom.api.com/v1"
    assert config.model == "actual-model-name"


def test_cli_endpoint_cli_overrides_registry(monkeypatch):
    """CLI args override endpoint registry values."""
    base_args = {
        "env_id_or_path": "test-env",
        "env_args": {},
        "env_dir_path": "./environments",
        "endpoints_path": "./configs/endpoints.py",
        "model": "my-alias",
        "api_key_var": "OVERRIDE_KEY",
        "api_base_url": "https://override.api.com/v1",
        "header": None,
        "num_examples": 1,
        "rollouts_per_example": 1,
        "max_concurrent": 1,
        "max_concurrent_generation": None,
        "max_concurrent_scoring": None,
        "independent_scoring": False,
        "max_tokens": 100,
        "temperature": None,
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
    args_namespace = SimpleNamespace(**base_args)
    captured: dict = {"configs": []}

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: args_namespace,
    )
    monkeypatch.setattr(vf_eval, "setup_logging", lambda *_, **__: None)

    monkeypatch.setattr(
        vf_eval,
        "load_endpoints",
        lambda *_: {
            "my-alias": {
                "key": "REGISTRY_KEY",
                "url": "https://registry.api.com/v1",
                "model": "actual-model-name",
            }
        },
    )

    async def fake_run_evaluation(config):
        captured["configs"].append(config)
        return _make_generate_outputs(env_id=config.env_id)

    monkeypatch.setattr(
        verifiers.utils.eval_utils, "run_evaluation", fake_run_evaluation
    )

    vf_eval.main()

    config = captured["configs"][0]
    assert config.client_config.api_key_var == "OVERRIDE_KEY"
    assert config.client_config.api_base_url == "https://override.api.com/v1"


def test_cli_endpoint_not_in_registry_uses_defaults(monkeypatch):
    """Model not in registry uses CLI defaults or global defaults."""
    base_args = {
        "env_id_or_path": "test-env",
        "env_args": {},
        "env_dir_path": "./environments",
        "endpoints_path": "./configs/endpoints.py",
        "model": "unknown-model",
        "api_key_var": None,
        "api_base_url": None,
        "header": None,
        "num_examples": 1,
        "rollouts_per_example": 1,
        "max_concurrent": 1,
        "max_concurrent_generation": None,
        "max_concurrent_scoring": None,
        "independent_scoring": False,
        "max_tokens": 100,
        "temperature": None,
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
    args_namespace = SimpleNamespace(**base_args)
    captured: dict = {"configs": []}

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: args_namespace,
    )
    monkeypatch.setattr(vf_eval, "setup_logging", lambda *_, **__: None)
    monkeypatch.setattr(vf_eval, "load_endpoints", lambda *_: {})

    async def fake_run_evaluation(config):
        captured["configs"].append(config)
        return _make_generate_outputs(env_id=config.env_id)

    monkeypatch.setattr(
        verifiers.utils.eval_utils, "run_evaluation", fake_run_evaluation
    )

    vf_eval.main()

    config = captured["configs"][0]
    # Should use DEFAULT values from eval.py
    assert config.client_config.api_key_var == "PRIME_API_KEY"
    assert config.client_config.api_base_url == "https://api.pinference.ai/api/v1"
    assert config.model == "unknown-model"


# =============================================================================
# Tests for EvalConfig and MultiEvalConfig types
# =============================================================================


def test_eval_config_creation():
    """EvalConfig can be created with required fields."""
    from verifiers.types import ClientConfig

    config = EvalConfig(
        env_id="test-env",
        env_args={"key": "value"},
        env_dir_path="./environments",
        model="gpt-4",
        client_config=ClientConfig(),
        sampling_args={"max_tokens": 100},
        num_examples=10,
        rollouts_per_example=3,
        max_concurrent=32,
    )

    assert config.env_id == "test-env"
    assert config.num_examples == 10
    assert config.independent_scoring is False  # default


def test_multi_eval_config_creation():
    """MultiEvalConfig can be created with list of EvalConfigs."""
    from verifiers.types import ClientConfig

    configs = [
        EvalConfig(
            env_id=f"env-{i}",
            env_args={},
            env_dir_path="./environments",
            model="gpt-4",
            client_config=ClientConfig(),
            sampling_args={},
            num_examples=5,
            rollouts_per_example=1,
            max_concurrent=8,
        )
        for i in range(3)
    ]

    multi_config = MultiEvalConfig(env=configs)
    assert len(multi_config.env) == 3
    assert multi_config.env[0].env_id == "env-0"
    assert multi_config.env[2].env_id == "env-2"


# =============================================================================
# Tests for extra environment kwargs
# =============================================================================


def test_cli_extra_env_kwargs(monkeypatch):
    """Extra env kwargs are passed to config."""
    captured = _run_cli(
        monkeypatch,
        {
            "extra_env_kwargs": {"custom_key": "custom_value", "num_param": 42},
        },
    )

    config = captured["configs"][0]
    assert config.extra_env_kwargs == {"custom_key": "custom_value", "num_param": 42}


def test_cli_extra_env_kwargs_empty(monkeypatch):
    """Empty extra env kwargs is handled."""
    captured = _run_cli(
        monkeypatch,
        {
            "extra_env_kwargs": {},
        },
    )

    config = captured["configs"][0]
    assert config.extra_env_kwargs == {}


# =============================================================================
# Tests for concurrent generation/scoring options
# =============================================================================


def test_cli_concurrent_options(monkeypatch):
    """Concurrent generation/scoring options are captured."""
    captured = _run_cli(
        monkeypatch,
        {
            "max_concurrent": 64,
            "max_concurrent_generation": 32,
            "max_concurrent_scoring": 16,
        },
    )

    config = captured["configs"][0]
    assert config.max_concurrent == 64
    assert config.max_concurrent_generation == 32
    assert config.max_concurrent_scoring == 16


def test_cli_concurrent_options_none(monkeypatch):
    """None concurrent options use defaults."""
    captured = _run_cli(
        monkeypatch,
        {
            "max_concurrent": 8,
            "max_concurrent_generation": None,
            "max_concurrent_scoring": None,
        },
    )

    config = captured["configs"][0]
    assert config.max_concurrent == 8
    assert config.max_concurrent_generation is None
    assert config.max_concurrent_scoring is None


# =============================================================================
# Tests for independent scoring option
# =============================================================================


def test_cli_independent_scoring_enabled(monkeypatch):
    """Independent scoring flag is captured."""
    captured = _run_cli(
        monkeypatch,
        {
            "independent_scoring": True,
        },
    )

    config = captured["configs"][0]
    assert config.independent_scoring is True


def test_cli_independent_scoring_disabled(monkeypatch):
    """Independent scoring disabled by default."""
    captured = _run_cli(
        monkeypatch,
        {
            "independent_scoring": False,
        },
    )

    config = captured["configs"][0]
    assert config.independent_scoring is False


# =============================================================================
# Tests for save options
# =============================================================================


def test_cli_save_options(monkeypatch):
    """Save options are captured correctly."""
    captured = _run_cli(
        monkeypatch,
        {
            "save_results": True,
            "save_every": 50,
            "save_to_hf_hub": True,
            "hf_hub_dataset_name": "my-org/my-dataset",
        },
    )

    config = captured["configs"][0]
    assert config.save_results is True
    assert config.save_every == 50
    assert config.save_to_hf_hub is True
    assert config.hf_hub_dataset_name == "my-org/my-dataset"
