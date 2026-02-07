import argparse
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import verifiers.scripts.eval as vf_eval
import verifiers.utils.eval_utils
from verifiers.types import GenerateOutputs
from verifiers.utils.eval_utils import load_toml_config
from verifiers.utils.save_utils import states_to_outputs


@pytest.fixture
def run_cli(make_metadata, make_state, make_input):
    def _run_cli(
        monkeypatch,
        overrides,
        capture_all_configs: bool = False,
        endpoints: dict | None = None,
    ):
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
            "endpoints_path": "./configs/endpoints.toml",
            "model": "gpt-4.1-mini",
            "api_key_var": "OPENAI_API_KEY",
            "api_base_url": "https://api.openai.com/v1",
            "header": None,
            "num_examples": 1,
            "rollouts_per_example": 1,
            "max_concurrent": 1,
            "independent_scoring": False,
            "max_tokens": 42,
            "temperature": 0.9,
            "sampling_args": None,
            "verbose": False,
            "no_interleave_scoring": False,
            "state_columns": [],
            "save_results": False,
            "resume": None,
            "save_every": -1,
            "save_to_hf_hub": False,
            "hf_hub_dataset_name": "",
            "extra_env_kwargs": {},
            "max_retries": 0,
            "tui": False,
            "debug": False,
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
        monkeypatch.setattr(vf_eval, "load_endpoints", lambda *_: endpoints or {})

        async def fake_run_evaluation(config, **kwargs):
            captured["sampling_args"] = dict(config.sampling_args)
            captured["configs"].append(config)
            _make_metadata = make_metadata
            _make_state = make_state
            _make_input = make_input
            n = config.num_examples
            r = config.rollouts_per_example
            inputs = [_make_input(example_id=i // r) for i in range(n * r)]
            states = [_make_state(**inputs[i]) for i in range(n * r)]
            rollout_outputs = states_to_outputs(states)
            metadata = _make_metadata(
                env_id=config.env_id,
                model=config.model,
                sampling_args=config.sampling_args,
                num_examples=n,
                rollouts_per_example=r,
            )
            return GenerateOutputs(outputs=rollout_outputs, metadata=metadata)

        monkeypatch.setattr(
            verifiers.utils.eval_utils, "run_evaluation", fake_run_evaluation
        )

        vf_eval.main()
        return captured

    return _run_cli


def test_cli_single_env_id(monkeypatch, run_cli):
    """Single env ID without comma creates one config."""
    captured = run_cli(
        monkeypatch,
        {
            "env_id_or_config": "env1",
        },
    )

    configs = captured["configs"]
    assert len(configs) == 1
    assert configs[0].env_id == "env1"


def test_cli_sampling_args_precedence_over_flags(monkeypatch, run_cli):
    """sampling_args JSON takes precedence over individual flags."""
    captured = run_cli(
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


def test_cli_sampling_args_fill_from_flags_when_missing(monkeypatch, run_cli):
    """Flags fill in missing sampling_args values."""
    captured = run_cli(
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


def test_cli_no_sampling_args_uses_flags(monkeypatch, run_cli):
    """When no sampling_args provided, uses flag values."""
    captured = run_cli(
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


def test_cli_temperature_not_added_when_none(monkeypatch, run_cli):
    """Temperature flag with None is not added to sampling_args."""
    captured = run_cli(
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


def test_cli_endpoint_alias_multi_variant_sets_multi_base_urls(monkeypatch, run_cli):
    captured = run_cli(
        monkeypatch,
        {
            "model": "gpt-5-mini",
            "api_key_var": None,
            "api_base_url": None,
        },
        endpoints={
            "gpt-5-mini": [
                {
                    "model": "gpt-5-mini",
                    "url": "https://a.example/v1",
                    "key": "OPENAI_API_KEY",
                },
                {
                    "model": "gpt-5-mini",
                    "url": "https://b.example/v1",
                    "key": "OPENAI_API_KEY",
                },
            ]
        },
    )

    config = captured["configs"][0]
    assert config.model == "gpt-5-mini"
    assert config.client_config.api_key_var == "OPENAI_API_KEY"
    assert config.client_config.api_base_url == "https://a.example/v1"
    assert [cfg.api_base_url for cfg in config.client_config.endpoint_configs] == [
        "https://a.example/v1",
        "https://b.example/v1",
    ]


def test_cli_model_flag_resolves_endpoint_alias_when_registry_present(
    monkeypatch, run_cli
):
    captured = run_cli(
        monkeypatch,
        {
            "model": "gpt-4.1-mini",
            "api_key_var": None,
            "api_base_url": None,
        },
        endpoints={
            "gpt-4.1-mini": [
                {
                    "model": "openai/gpt-4.1-mini",
                    "url": "https://alias.example/v1",
                    "key": "ALIAS_API_KEY",
                }
            ]
        },
    )

    config = captured["configs"][0]
    assert config.endpoint_id == "gpt-4.1-mini"
    assert config.model == "openai/gpt-4.1-mini"
    assert config.client_config.api_key_var == "ALIAS_API_KEY"
    assert config.client_config.api_base_url == "https://alias.example/v1"


def test_cli_direct_fields_work_without_endpoint_registry(monkeypatch, run_cli):
    captured = run_cli(
        monkeypatch,
        {
            "model": "my/custom-model",
            "api_key_var": "CUSTOM_API_KEY",
            "api_base_url": "https://custom.example/v1",
        },
        endpoints={},
    )

    config = captured["configs"][0]
    assert config.endpoint_id is None
    assert config.model == "my/custom-model"
    assert config.client_config.api_key_var == "CUSTOM_API_KEY"
    assert config.client_config.api_base_url == "https://custom.example/v1"


def test_cli_endpoint_alias_multi_variant_supports_mixed_keys(monkeypatch, run_cli):
    captured = run_cli(
        monkeypatch,
        {
            "model": "gpt-5-mini",
            "api_key_var": None,
            "api_base_url": None,
        },
        endpoints={
            "gpt-5-mini": [
                {
                    "model": "gpt-5-mini",
                    "url": "https://a.example/v1",
                    "key": "PRIME_API_KEY",
                },
                {
                    "model": "gpt-5-mini",
                    "url": "https://b.example/v1",
                    "key": "OPENAI_API_KEY",
                },
            ]
        },
    )

    config = captured["configs"][0]
    assert config.client_config.api_key_var == "PRIME_API_KEY"
    assert [cfg.api_key_var for cfg in config.client_config.endpoint_configs] == [
        "PRIME_API_KEY",
        "OPENAI_API_KEY",
    ]


def test_cli_endpoint_id_resolves_registry_alias(monkeypatch, run_cli):
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\nendpoint_id = "gpt-5-mini"\n')
        f.flush()
        captured = run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
            },
            endpoints={
                "gpt-5-mini": [
                    {
                        "model": "gpt-5-mini",
                        "url": "https://a.example/v1",
                        "key": "OPENAI_API_KEY",
                    },
                    {
                        "model": "gpt-5-mini",
                        "url": "https://b.example/v1",
                        "key": "OPENAI_API_KEY",
                    },
                ]
            },
        )

    config = captured["configs"][0]
    assert config.endpoint_id == "gpt-5-mini"
    assert config.model == "gpt-5-mini"
    assert config.client_config.api_key_var == "OPENAI_API_KEY"
    assert config.client_config.api_base_url == "https://a.example/v1"
    assert [cfg.api_base_url for cfg in config.client_config.endpoint_configs] == [
        "https://a.example/v1",
        "https://b.example/v1",
    ]


def test_cli_endpoint_id_accepts_directory_endpoints_path(monkeypatch, run_cli):
    with tempfile.TemporaryDirectory() as tmp_dir:
        endpoints_file = Path(tmp_dir) / "endpoints.toml"
        endpoints_file.write_text(
            (
                "[[endpoint]]\n"
                'endpoint_id = "gpt-5-mini"\n'
                'model = "gpt-5-mini"\n'
                'url = "https://a.example/v1"\n'
                'key = "OPENAI_API_KEY"\n'
            ),
            encoding="utf-8",
        )
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write(
                f'endpoints_path = "{tmp_dir}"\n\n'
                '[[eval]]\nenv_id = "env1"\nendpoint_id = "gpt-5-mini"\n'
            )
            f.flush()
            captured = run_cli(
                monkeypatch,
                {
                    "env_id_or_config": f.name,
                },
                endpoints={
                    "gpt-5-mini": [
                        {
                            "model": "gpt-5-mini",
                            "url": "https://a.example/v1",
                            "key": "OPENAI_API_KEY",
                        }
                    ]
                },
            )

    config = captured["configs"][0]
    assert config.model == "gpt-5-mini"
    assert config.client_config.api_base_url == "https://a.example/v1"


def test_cli_endpoint_id_not_found_raises(monkeypatch, run_cli):
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\nendpoint_id = "missing-id"\n')
        f.flush()
        with pytest.raises(ValueError, match="Endpoint id 'missing-id' not found"):
            run_cli(
                monkeypatch,
                {
                    "env_id_or_config": f.name,
                },
                endpoints={},
            )


def test_cli_endpoint_id_requires_toml_endpoints_path(monkeypatch, run_cli):
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            'endpoints_path = "./configs/endpoints.py"\n\n'
            '[[eval]]\nenv_id = "env1"\nendpoint_id = "gpt-5-mini"\n'
        )
        f.flush()
        with pytest.raises(
            ValueError, match="only supported with TOML endpoint registries"
        ):
            run_cli(
                monkeypatch,
                {
                    "env_id_or_config": f.name,
                },
                endpoints={
                    "gpt-5-mini": [
                        {
                            "model": "gpt-5-mini",
                            "url": "https://a.example/v1",
                            "key": "OPENAI_API_KEY",
                        }
                    ]
                },
            )


def test_toml_api_base_url_list_is_not_supported(monkeypatch, run_cli):
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            'api_base_url = ["https://a.example/v1", "https://b.example/v1"]\n\n'
            '[[eval]]\nenv_id = "env1"\n'
        )
        f.flush()
        with pytest.raises(
            ValueError, match="api_base_url lists are no longer supported"
        ):
            run_cli(
                monkeypatch,
                {
                    "env_id_or_config": f.name,
                },
                endpoints={},
            )


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


def test_cli_multi_env_via_toml_config(monkeypatch, run_cli):
    """CLI with TOML config creates multiple eval configs."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n\n[[eval]]\nenv_id = "env2"\n')
        f.flush()
        captured = run_cli(
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


def test_cli_toml_ignores_cli_args(monkeypatch, run_cli):
    """TOML config ignores CLI args, uses defaults for unspecified values."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env1"\n\n[[eval]]\nenv_id = "env2"\n')
        f.flush()
        captured = run_cli(
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


def test_cli_toml_per_env_num_examples(monkeypatch, run_cli):
    """TOML per-env num_examples is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env1"\nnum_examples = 10\n\n'
            '[[eval]]\nenv_id = "env2"\nnum_examples = 20\n'
        )
        f.flush()
        captured = run_cli(
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


def test_cli_toml_per_env_rollouts_per_example(monkeypatch, run_cli):
    """TOML per-env rollouts_per_example is used when CLI arg not provided."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env1"\nrollouts_per_example = 3\n\n'
            '[[eval]]\nenv_id = "env2"\nrollouts_per_example = 5\n'
        )
        f.flush()
        captured = run_cli(
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


def test_cli_toml_per_eval_settings_used(monkeypatch, run_cli):
    """TOML per-eval settings are used (CLI args ignored when using config)."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env-a"\nnum_examples = 100\nrollouts_per_example = 10\n\n'
            '[[eval]]\nenv_id = "env-b"\nnum_examples = 200\nrollouts_per_example = 20\n'
        )
        f.flush()
        captured = run_cli(
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


def test_cli_toml_mixed_per_env_and_defaults_fallback(monkeypatch, run_cli):
    """TOML with some evals having settings, others fall back to global defaults."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            '[[eval]]\nenv_id = "env-with-settings"\nnum_examples = 15\nrollouts_per_example = 4\n\n'
            '[[eval]]\nenv_id = "env-without-settings"\n'
        )
        f.flush()
        captured = run_cli(
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


def test_cli_toml_without_settings_uses_defaults(monkeypatch, run_cli):
    """TOML evals without settings use global defaults."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write('[[eval]]\nenv_id = "env-a"\n\n[[eval]]\nenv_id = "env-b"\n')
        f.flush()
        captured = run_cli(
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


def test_load_toml_config_resolves_endpoints_path_relative_to_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir) / "configs" / "eval"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "bench.toml"
        config_path.write_text(
            'endpoints_path = "../endpoints.toml"\n\n[[eval]]\nenv_id = "env1"\n',
            encoding="utf-8",
        )

        result = load_toml_config(config_path)
        expected = str((config_dir / "../endpoints.toml").resolve())
        assert result[0]["endpoints_path"] == expected


def test_cli_resume_explicit_path(monkeypatch, run_cli, tmp_path: Path):
    """--resume with explicit path sets resume_path."""
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir(parents=True)
    (resume_dir / "results.jsonl").write_text("", encoding="utf-8")
    (resume_dir / "metadata.json").write_text("{}", encoding="utf-8")

    captured = run_cli(
        monkeypatch,
        {
            "resume": str(resume_dir),
        },
    )

    assert captured["configs"][0].resume_path == resume_dir


def test_cli_resume_auto_detects_latest_incomplete(
    monkeypatch, run_cli, tmp_path: Path
):
    """--resume with no path auto-detects latest matching incomplete run."""
    env_id = "dummy-env"
    model = "gpt-4.1-mini"
    run_base = tmp_path / "outputs" / "evals" / f"{env_id}--{model.replace('/', '--')}"
    old_run = run_base / "oldrun"
    new_run = run_base / "newrun"
    old_run.mkdir(parents=True)
    new_run.mkdir(parents=True)

    metadata = (
        '{"env_id":"dummy-env","model":"gpt-4.1-mini",'
        '"num_examples":4,"rollouts_per_example":1}'
    )
    (old_run / "metadata.json").write_text(metadata, encoding="utf-8")
    (new_run / "metadata.json").write_text(metadata, encoding="utf-8")

    (old_run / "results.jsonl").write_text('{"example_id":0}\n', encoding="utf-8")
    (new_run / "results.jsonl").write_text(
        '{"example_id":0}\n{"example_id":1}\n', encoding="utf-8"
    )
    now = time.time()
    os.utime(old_run, (now, now))
    os.utime(new_run, (now + 1, now + 1))

    monkeypatch.chdir(tmp_path)
    captured = run_cli(
        monkeypatch,
        {
            "resume": True,
            "num_examples": 4,
            "rollouts_per_example": 1,
            "env_dir_path": str(tmp_path / "environments"),
        },
    )

    assert captured["configs"][0].resume_path is not None
    assert captured["configs"][0].resume_path.resolve() == new_run.resolve()


def test_cli_toml_resume_false_disables_global_resume(monkeypatch, run_cli):
    """Per-eval resume=false overrides global resume=true in TOML configs."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        f.write(
            "resume = true\n"
            "\n"
            "[[eval]]\n"
            'env_id = "env-a"\n'
            "\n"
            "[[eval]]\n"
            'env_id = "env-b"\n'
            "resume = false\n"
        )
        f.flush()
        captured = run_cli(
            monkeypatch,
            {
                "env_id_or_config": f.name,
                "num_examples": 1,
                "rollouts_per_example": 1,
                "env_dir_path": "./environments",
            },
        )

    configs = captured["configs"]
    assert len(configs) == 2
    assert configs[0].env_id == "env-a"
    assert configs[0].resume_path is None
    assert configs[1].env_id == "env-b"
    assert configs[1].resume_path is None
