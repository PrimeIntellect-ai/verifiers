from pathlib import Path

from verifiers.types import EvalConfig as LegacyEvalConfig
from verifiers.v1.cli.eval import main as eval_main
from verifiers.v1.cli.eval.legacy import (
    _eval_run_config,
    _parse_args,
    is_legacy_eval_invocation,
)


def test_legacy_eval_cli_inputs_build_v0_eval_config(tmp_path: Path):
    args = _parse_args(
        [
            "echo-v0",
            "-n",
            "2",
            "-r",
            "1",
            "-s",
            "-o",
            str(tmp_path),
            "--disable-env-server",
            "--model",
            "test-model",
            "--api-key-var",
            "TEST_API_KEY",
            "--api-base-url",
            "http://localhost:8000/v1",
            "--max-retries",
            "7",
            "--max-tokens",
            "12",
        ]
    )

    run_config = _eval_run_config(args)
    (config,) = run_config.evals

    assert isinstance(config, LegacyEvalConfig)
    assert config.env_id == "echo-v0"
    assert config.num_examples == 2
    assert config.rollouts_per_example == 1
    assert config.shuffle is True
    assert config.save_results is True
    assert config.output_dir == str(tmp_path)
    assert config.disable_env_server is True
    assert config.max_retries == 7
    assert config.sampling_args["max_tokens"] == 12
    assert config.client_config.api_key_var == "TEST_API_KEY"


def test_legacy_eval_accepts_v1_dotted_cli_aliases(tmp_path: Path):
    args = _parse_args(
        [
            "echo-v0",
            "--model",
            "test-model",
            "--client.base-url",
            "http://localhost:8000/v1",
            "--client.api-key-var",
            "TEST_API_KEY",
            "--num-tasks",
            "2",
            "--num-rollouts",
            "1",
            "--sampling.max-tokens",
            "12",
            "--sampling.temperature",
            "0.1",
            "--sampling.top-p",
            "0.9",
            "--output-dir",
            str(tmp_path),
        ]
    )

    run_config = _eval_run_config(args)
    (config,) = run_config.evals

    assert config.env_id == "echo-v0"
    assert config.num_examples == 2
    assert config.rollouts_per_example == 1
    assert config.output_dir == str(tmp_path)
    assert config.client_config.api_base_url == "http://localhost:8000/v1"
    assert config.client_config.api_key_var == "TEST_API_KEY"
    assert config.sampling_args == {
        "max_tokens": 12,
        "temperature": 0.1,
        "top_p": 0.9,
    }


def test_legacy_eval_toml_always_saves_results(tmp_path: Path):
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

    assert is_legacy_eval_invocation([str(config_path)])

    run_config = _eval_run_config(_parse_args([str(config_path)]))
    (config,) = run_config.evals

    assert isinstance(config, LegacyEvalConfig)
    assert config.env_id == "echo-v0"
    assert config.save_results is True
    assert config.output_dir == str(tmp_path)


def test_v1_toml_does_not_dispatch_to_legacy(tmp_path: Path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "test-model"

[taskset]
id = "echo-v1"
""".lstrip()
    )

    assert not is_legacy_eval_invocation([str(config_path)])


def test_eval_main_dispatches_legacy_before_v1_parse(monkeypatch):
    calls = []
    monkeypatch.setattr(eval_main, "is_legacy_eval_invocation", lambda _argv: True)
    monkeypatch.setattr(
        eval_main, "run_legacy_eval_cli", lambda argv: calls.append(argv)
    )

    eval_main.main(["echo-v0", "--save-results"])

    assert calls == [["echo-v0", "--save-results"]]


def test_direct_id_legacy_env_dispatches_to_v0_cli():
    assert is_legacy_eval_invocation(["--id", "echo-v0"])
