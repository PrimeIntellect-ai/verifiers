"""Tests for vf-gepa CLI argument parsing and configuration."""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import verifiers as vf


def require_gepa_script():
    """Import gepa script or skip tests if module is unavailable."""
    return pytest.importorskip("verifiers.scripts.gepa")


def _make_mock_env():
    """Create a mock environment for testing."""
    env = MagicMock(spec=vf.Environment)
    env.system_prompt = "Test system prompt"
    env.eval_dataset = None
    env.env_id = "test-env"
    env.oai_tools = None

    # Mock dataset methods - return enough items for all tests
    # Most tests use num_examples=10 and num_val=5, so we need at least 15 items
    mock_dataset = MagicMock()
    mock_dataset.to_list.return_value = [
        {"question": f"q{i}", "answer": f"a{i}", "task": "test", "info": {}}
        for i in range(50)  # Plenty of items for all tests
    ]
    env.get_dataset.return_value = mock_dataset
    env.get_eval_dataset.return_value = mock_dataset

    return env


def _run_cli(monkeypatch, overrides, custom_env=None):
    """
    Helper to run vf-gepa CLI with mocked dependencies.

    Args:
        monkeypatch: pytest monkeypatch fixture
        overrides: dict of CLI args to override
        custom_env: optional custom mock environment (default: _make_mock_env())

    Returns:
        dict containing captured GEPAConfig passed to run_gepa_optimization
    """
    gepa_script = require_gepa_script()

    base_args = {
        "env_id": "test-env",
        "env_args": "{}",
        "env_dir_path": "./environments",
        "num_examples": 10,
        "num_val": 5,
        "endpoints_path": "./configs/endpoints.py",
        "model": "gpt-4o-mini",
        "api_key_var": "OPENAI_API_KEY",
        "api_base_url": "https://api.openai.com/v1",
        "headers": None,
        "temperature": 1.0,
        "max_tokens": None,
        "sampling_args": None,  # Will be parsed by json.loads if not None
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "budget": "light",  # Required - mutually exclusive with max_metric_calls
        "max_metric_calls": None,
        "components": ["system_prompt"],
        "reflection_model": "gpt-4o",
        "reflection_temperature": 1.0,
        "reflection_base_url": None,
        "reflection_api_key_var": "OPENAI_API_KEY",
        "reflection_max_tokens": 8000,
        "reflection_minibatch_size": 35,
        "save_results": False,
        "save_every": -1,
        "track_stats": False,
        "verbose": False,
        "seed": 42,
        "use_wandb": False,
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_name": None,
        "wandb_api_key_var": "WANDB_API_KEY",
        "wandb_init_kwargs": None,
        "use_mlflow": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": None,
    }
    base_args.update(overrides)
    args_namespace = SimpleNamespace(**base_args)

    captured = {}

    # Mock argparse
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: args_namespace,
    )

    # Mock setup_logging
    monkeypatch.setattr(vf, "setup_logging", lambda *_, **__: None)

    # Mock load_endpoints
    from verifiers.utils import eval_utils

    monkeypatch.setattr(eval_utils, "load_endpoints", lambda *_: {})

    # Mock get_env_gepa_defaults
    from verifiers.utils import gepa_utils

    monkeypatch.setattr(gepa_utils, "get_env_gepa_defaults", lambda *_: {})

    # Mock load_environment
    mock_env = custom_env if custom_env is not None else _make_mock_env()
    monkeypatch.setattr(vf, "load_environment", lambda **kwargs: mock_env)

    # Mock os.getenv for reflection API key
    def mock_getenv(key, default=None):
        if key in ("OPENAI_API_KEY", "WANDB_API_KEY"):
            return "fake-api-key"
        return default

    monkeypatch.setattr(os, "getenv", mock_getenv)

    # Mock prepare_gepa_dataset to return non-empty datasets
    def mock_prepare_gepa_dataset(dataset):
        if dataset is None:
            raise ValueError("dataset cannot be None")
        # Return hardcoded examples instead of relying on the mock dataset
        # This ensures we always have data for the tests
        return [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "task": "test",
                "info": {},
            }
            for i in range(10)
        ]

    monkeypatch.setattr(
        gepa_utils,
        "prepare_gepa_dataset",
        mock_prepare_gepa_dataset,
    )

    # Mock run_gepa_optimization to capture config
    # Must patch in the gepa script's namespace since it's imported at module level
    async def fake_run_gepa_optimization(config):
        captured["config"] = config
        # Return immediately without running optimization
        return None

    monkeypatch.setattr(
        gepa_script,
        "run_gepa_optimization",
        fake_run_gepa_optimization,
    )

    # Run the CLI
    gepa_script.main()

    return captured


def test_cli_sampling_args_precedence_over_flags(monkeypatch):
    """Test that --sampling-args takes precedence over --temperature and --max-tokens."""
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": {"temperature": 0.5, "max_tokens": 100},
            "temperature": 0.9,
            "max_tokens": 500,
        },
    )

    config = captured["config"]
    assert config.sampling_args["temperature"] == 0.5
    assert config.sampling_args["max_tokens"] == 100


def test_cli_sampling_args_fill_from_flags_when_missing(monkeypatch):
    """Test that flags fill in when --sampling-args doesn't specify them."""
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": {"enable_thinking": True},
            "temperature": 0.7,
            "max_tokens": 200,
        },
    )

    config = captured["config"]
    assert config.sampling_args["temperature"] == 0.7
    assert config.sampling_args["max_tokens"] == 200
    assert config.sampling_args["enable_thinking"] is True


def test_cli_budget_light_conversion(monkeypatch):
    """Test that --budget light converts to expected max_metric_calls."""
    captured = _run_cli(
        monkeypatch,
        {
            "budget": "light",
            "max_metric_calls": None,
            "num_examples": 10,
            "num_val": 5,
        },
    )

    config = captured["config"]
    # Light budget should result in a positive number of metric calls
    assert config.max_metric_calls > 0
    # Light budget (~6 candidates) should be in a reasonable range
    assert config.max_metric_calls >= 300  # At least 300
    assert config.max_metric_calls <= 500  # At most 500


def test_cli_budget_medium_conversion(monkeypatch):
    """Test that --budget medium converts correctly."""
    captured = _run_cli(
        monkeypatch,
        {
            "budget": "medium",
            "max_metric_calls": None,
            "num_examples": 10,
            "num_val": 5,
        },
    )

    config = captured["config"]
    # Medium budget should result in more calls than light (~12 candidates)
    assert config.max_metric_calls >= 500  # At least 500
    assert config.max_metric_calls <= 1000  # At most 1000


def test_cli_budget_heavy_conversion(monkeypatch):
    """Test that --budget heavy converts correctly."""
    captured = _run_cli(
        monkeypatch,
        {
            "budget": "heavy",
            "max_metric_calls": None,
            "num_examples": 10,
            "num_val": 5,
        },
    )

    config = captured["config"]
    # Heavy budget should result in the most calls
    assert config.max_metric_calls > 200


def test_cli_max_metric_calls_direct(monkeypatch):
    """Test that --max-metric-calls is used directly when provided."""
    captured = _run_cli(
        monkeypatch,
        {
            "budget": None,
            "max_metric_calls": 1234,
        },
    )

    config = captured["config"]
    assert config.max_metric_calls == 1234


def test_cli_seed_candidate_extraction(monkeypatch):
    """Test that seed_candidate is extracted from env's system_prompt."""
    captured = _run_cli(
        monkeypatch,
        {
            "components": ["system_prompt"],
        },
    )

    config = captured["config"]
    assert "system_prompt" in config.seed_candidate
    assert config.seed_candidate["system_prompt"] == "Test system prompt"
    assert config.components_to_optimize == ["system_prompt"]


def test_cli_defaults_fallback(monkeypatch):
    """Test that CLI args are used when provided (not overridden by defaults)."""
    captured = _run_cli(
        monkeypatch,
        {
            "num_examples": 25,
            "num_val": 10,
            "rollouts_per_example": 3,
        },
    )

    config = captured["config"]
    assert config.num_examples == 25
    assert config.num_val == 10
    assert config.rollouts_per_example == 3


def test_cli_reflection_model_config(monkeypatch):
    """Test that reflection model configuration is captured correctly."""
    captured = _run_cli(
        monkeypatch,
        {
            "reflection_model": "gpt-4o",
            "reflection_temperature": 0.8,
            "reflection_max_tokens": 4000,
            "reflection_minibatch_size": 20,
        },
    )

    config = captured["config"]
    assert config.reflection_model == "gpt-4o"
    assert config.reflection_temperature == 0.8
    assert config.reflection_max_tokens == 4000
    assert config.reflection_minibatch_size == 20


def test_cli_experiment_tracking_config(monkeypatch):
    """Test that experiment tracking (wandb/mlflow) configuration is captured."""
    captured = _run_cli(
        monkeypatch,
        {
            "use_wandb": True,
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "wandb_name": "test-run",
            "use_mlflow": True,
            "mlflow_tracking_uri": "http://localhost:5000",
            "mlflow_experiment_name": "test-experiment",
        },
    )

    config = captured["config"]
    assert config.use_wandb is True
    assert config.wandb_project == "test-project"
    assert config.wandb_entity == "test-entity"
    assert config.wandb_name == "test-run"
    assert config.use_mlflow is True
    assert config.mlflow_tracking_uri == "http://localhost:5000"
    assert config.mlflow_experiment_name == "test-experiment"


def test_cli_env_args_parsing(monkeypatch):
    """Test that --env-args is a string that gets parsed to dict correctly."""
    # Note: env_args stays as a string in the CLI args, then gets parsed by json.loads
    # But since we're passing through SimpleNamespace, we just verify the config receives it
    captured = _run_cli(
        monkeypatch,
        {
            "env_args": '{"custom_arg": "value", "num": 42}',
        },
    )

    config = captured["config"]
    assert config.env_args["custom_arg"] == "value"
    assert config.env_args["num"] == 42


def test_cli_components_multiple(monkeypatch):
    """Test that multiple components can be specified."""
    # Create a mock env with oai_tools
    env_with_tools = _make_mock_env()
    env_with_tools.oai_tools = [
        {
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {},
            }
        }
    ]

    captured = _run_cli(
        monkeypatch,
        {
            "components": ["system_prompt", "tool_descriptions"],
        },
        custom_env=env_with_tools,
    )

    config = captured["config"]
    assert config.components_to_optimize == ["system_prompt", "tool_descriptions"]
    # Should have both system_prompt and tool descriptions in seed_candidate
    assert "system_prompt" in config.seed_candidate
    assert "tool_0_description" in config.seed_candidate
