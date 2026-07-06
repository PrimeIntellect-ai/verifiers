import pytest

from verifiers.v1.cli.eval.compat import (
    build_v1_eval_config,
    transitional_config_to_fields,
)
from verifiers.v1.cli.eval.resume import load_resume_config
from verifiers.v1.configs.eval import EvalConfig


def test_transitional_hub_env_id_converts_to_local_legacy_id(tmp_path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "openai/gpt-oss-120b"

[[eval]]
env_id = "primeintellect/reverse-text@1.2.3"
env_args = { num_examples = 3 }
""".strip(),
        encoding="utf-8",
    )

    fields = transitional_config_to_fields(config_path)
    config, warnings = build_v1_eval_config(fields)
    parsed = EvalConfig.model_validate(config)

    assert warnings == []
    assert parsed.id == "reverse-text"
    assert parsed.is_legacy
    assert parsed.args == {"num_examples": 3}


def test_transitional_config_rejects_env_id_and_taskset(tmp_path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
env_id = "reverse-text"

[[eval]]
taskset = { id = "gsm8k-v1" }
""".strip(),
        encoding="utf-8",
    )

    fields = transitional_config_to_fields(config_path)
    with pytest.raises(ValueError, match="both a legacy env_id and a taskset"):
        build_v1_eval_config(fields)


def test_resume_normalizes_hub_era_ids(tmp_path):
    (tmp_path / "config.toml").write_text(
        """
model = "openai/gpt-oss-120b"
taskset = { id = "primeintellect/echo-v1@0.2.1" }
harness = { id = "primeintellect/default@1.0.0" }
""".strip(),
        encoding="utf-8",
    )

    config = load_resume_config(tmp_path)

    assert config.taskset.id == "echo-v1"
    assert config.harness.id == "default"
    assert config.resume == tmp_path
    assert config.output_dir == tmp_path
