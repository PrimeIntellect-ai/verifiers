"""The legacy (v0) eval path: one v1 parse, auto-detection at validation, one v0 mapping.

`echo-v0` / `echo-multi-v0` (tests/v1/fixtures) are classic `load_environment` envs, so
`EnvConfig._resolve_plugins` converts their taskset id to the legacy `id` and `main`
dispatches the parsed `EvalConfig` to the v0 evaluator. The evaluator itself is stubbed —
these tests assert on the v0 config the mapping produces."""

from pathlib import Path

import pytest

from verifiers.types import EvalConfig as V0EvalConfig
from verifiers.v1.cli.eval import legacy as eval_legacy
from verifiers.v1.cli.eval import main as eval_main
from verifiers.v1.configs.eval import EvalConfig


@pytest.fixture
def v0_runs(monkeypatch) -> list[V0EvalConfig]:
    """Stub the v0 evaluator and collect the configs it would have run."""
    captured: list[V0EvalConfig] = []

    async def fake_run_evaluations(run_config):
        captured.extend(run_config.evals)

    monkeypatch.setattr(eval_legacy, "run_evaluations", fake_run_evaluations)
    return captured


def test_v1_flags_map_onto_v0_eval_config(v0_runs, tmp_path: Path):
    eval_main.main(
        [
            "echo-v0",
            "-n",
            "2",
            "-r",
            "1",
            "-s",
            "-o",
            str(tmp_path),
            "--model",
            "test-model",
            "--client.api-key-var",
            "TEST_API_KEY",
            "--client.base-url",
            "http://localhost:8000/v1",
            "--retries.rollout.max-retries",
            "7",
            "--sampling.max-tokens",
            "12",
            "--sampling.temperature",
            "0.1",
        ]
    )

    (config,) = v0_runs
    assert isinstance(config, V0EvalConfig)
    assert config.env_id == "echo-v0"
    assert config.num_examples == 2
    assert config.rollouts_per_example == 1
    assert config.shuffle is True
    assert config.shuffle_seed == 0
    assert config.save_results is True
    assert config.output_dir == str(tmp_path)
    assert config.max_retries == 7
    assert config.sampling_args["max_tokens"] == 12
    assert config.sampling_args["temperature"] == 0.1
    assert config.client_config.api_key_var == "TEST_API_KEY"
    assert config.client_config.api_base_url == "http://localhost:8000/v1"


def test_flat_legacy_id_toml_runs_through_v0_evaluator(v0_runs, tmp_path: Path):
    # The shape prime's compat converter writes: a flat v1 config with a legacy `id`.
    config_path = tmp_path / "eval.toml"
    config_path.write_text('id = "echo-v0"\nmodel = "test-model"\n')

    eval_main.main(["@", str(config_path)])

    (config,) = v0_runs
    assert config.env_id == "echo-v0"
    assert config.model == "test-model"


def test_transitional_legacy_toml_converts_and_runs(v0_runs, tmp_path: Path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        f"""
model = "test-model"
api_key_var = "TEST_API_KEY"
api_base_url = "http://localhost:8000/v1"
output_dir = "{tmp_path}"
save_results = false

[[eval]]
id = "echo-v0"
num_examples = 1
rollouts_per_example = 1
""".lstrip()
    )

    eval_main.main(["@", str(config_path)])

    (config,) = v0_runs
    assert config.env_id == "echo-v0"
    assert config.num_examples == 1
    assert config.save_results is True
    assert config.output_dir == str(tmp_path)
    assert config.client_config.api_key_var == "TEST_API_KEY"


def test_v0_mapping_defaults_and_timeout():
    config = EvalConfig.model_validate(
        {"taskset": {"id": "echo-v0"}, "timeout": {"rollout": 60}}
    )
    assert config.is_legacy  # auto-detected: taskset id moved to the legacy `id`

    v0 = eval_legacy.v0_eval_config(config)
    assert v0.env_id == "echo-v0"
    assert v0.num_examples == -1  # v1 "all tasks" -> v0 -1
    assert v0.rollouts_per_example == 1
    assert v0.shuffle_seed is None
    assert v0.extra_env_kwargs == {"timeout_seconds": 60}
    assert v0.client_config.client_type == "openai_chat_completions"
    assert v0.client_config.extra_headers_from_state == {
        "X-Session-ID": "trajectory_id"
    }


def test_v0_mapping_carries_client_dialect_and_args():
    config = EvalConfig.model_validate(
        {
            "id": "echo-v0",
            "args": {"phrase": "hi"},
            "client": {"v0_client_type": "anthropic_messages", "headers": {"X-A": "b"}},
        }
    )

    v0 = eval_legacy.v0_eval_config(config)
    assert v0.env_args == {"phrase": "hi"}
    assert v0.client_config.client_type == "anthropic_messages"
    assert v0.client_config.extra_headers == {"X-A": "b"}


def test_v1_taskset_does_not_dispatch_to_legacy():
    config = EvalConfig.model_validate({"taskset": {"id": "echo-v1"}})
    assert not config.is_legacy
