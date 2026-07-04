"""The frozen v0 eval surface converts onto native v1 configs at the ingestion boundary."""

import tomllib

import pytest

from verifiers.v1.cli.eval.compat import (
    build_v1_eval_config,
    convert_transitional_config,
    is_transitional_config,
    transitional_config_to_fields,
    write_converted_eval_config,
)
from verifiers.v1.cli.eval.main import _convert_transitional_args
from verifiers.v1.configs.eval import EvalConfig

PLATFORM_TRANSITIONAL_TOML = """\
model = "openai/gpt-4.1-mini"
num_examples = 5
rollouts_per_example = 2
save_results = true

[[eval]]
env_args = { difficulty = "easy" }
max_retries = 4
headers = ["X-From-Platform: 1"]
taskset = { id = "gsm8k-v1" }
harness = { id = "default" }
group_size = 2
"""


class TestBuildV1EvalConfig:
    def test_v0_fields_map_to_v1_config(self):
        config, warnings = build_v1_eval_config(
            {
                "taskset": {"id": "reverse-text"},
                "num_examples": 10,
                "rollouts_per_example": 3,
                "model": "openai/gpt-4.1-mini",
                "env_args": {"difficulty": "hard"},
                "sampling_args": {
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "extra_body": {"provider": {"order": ["azure"]}},
                },
                "max_concurrent": 8,
                "max_retries": 2,
                "header": ["X-Custom: yes"],
            },
            tui_disabled=True,
        )

        assert warnings == []
        assert config["num_tasks"] == 10
        assert config["num_rollouts"] == 3
        assert config["rich"] is False
        assert config["retries"] == {"rollout": {"max_retries": 2}}
        assert config["client"]["headers"] == {"X-Custom": "yes"}
        # env args become taskset kwargs; explicit taskset keys win
        assert config["taskset"] == {"id": "reverse-text", "difficulty": "hard"}
        # extra_body flattens into sampling (v1 passes provider keys through)
        assert config["sampling"] == {
            "temperature": 0.7,
            "max_tokens": 512,
            "provider": {"order": ["azure"]},
        }

    def test_legacy_id_takes_env_args_as_construction_kwargs(self):
        config, _ = build_v1_eval_config(
            {"id": "reverse-text", "env_args": {"n": 1}, "num_examples": -1}
        )

        assert config["id"] == "reverse-text"
        assert config["args"] == {"n": 1}
        assert "num_tasks" not in config  # v0's -1 means all examples, v1's unset

    def test_display_only_v0_fields_warn_and_drop(self):
        config, warnings = build_v1_eval_config(
            {
                "id": "reverse-text",
                "state_columns": ["turn"],
                "independent_scoring": True,
            }
        )

        assert len(warnings) == 2
        assert config == {"id": "reverse-text"}

    def test_provider_shorthand_resolves_to_client(self):
        config, _ = build_v1_eval_config(
            {"id": "reverse-text", "provider": "openrouter"}
        )

        assert config["client"] == {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_var": "OPENROUTER_API_KEY",
        }

    def test_unmappable_fields_raise(self):
        for fields, match in [
            ({"api_client_type": "anthropic_messages"}, "api-client-type"),
            ({"save_to_hf_hub": True}, "save-to-hf-hub"),
            ({"endpoint_id": "gpt"}, "endpoint_id"),
            ({"resume": True}, "--resume"),
            ({"provider": "anthropic"}, "provider"),
        ]:
            with pytest.raises(ValueError, match=match):
                build_v1_eval_config({"id": "reverse-text", **fields})

    def test_v0_timeout_seconds_become_rollout_timeout(self):
        config, _ = build_v1_eval_config({"id": "reverse-text", "timeout": 300.0})

        assert config["timeout"] == {"rollout": 300.0}


class TestTransitionalConfigs:
    def test_platform_hosted_toml_flattens_and_maps(self, tmp_path):
        config_path = tmp_path / "hosted-eval-config.toml"
        config_path.write_text(PLATFORM_TRANSITIONAL_TOML)

        fields = transitional_config_to_fields(config_path)
        config, warnings = build_v1_eval_config(fields, tui_disabled=True)

        assert warnings == []
        assert config["taskset"] == {"id": "gsm8k-v1", "difficulty": "easy"}
        assert config["harness"] == {"id": "default"}
        assert config["num_tasks"] == 5
        assert config["num_rollouts"] == 2  # group_size wins
        assert config["retries"] == {"rollout": {"max_retries": 4}}
        assert config["client"]["headers"] == {"X-From-Platform": "1"}

    def test_multi_entry_configs_are_rejected(self, tmp_path):
        config_path = tmp_path / "multi.toml"
        config_path.write_text('[[eval]]\nenv_id = "a"\n\n[[eval]]\nenv_id = "b"\n')

        with pytest.raises(ValueError, match="single-entry"):
            transitional_config_to_fields(config_path)

    def test_written_config_validates_as_v1(self, tmp_path):
        config_path = tmp_path / "legacy.toml"
        config_path.write_text('[[eval]]\nenv_id = "reverse-text"\nnum_examples = 4\n')

        converted, warnings = convert_transitional_config(config_path)
        validated = EvalConfig.model_validate(
            tomllib.loads(converted.read_text(encoding="utf-8"))
        )

        assert warnings == []
        assert validated.is_legacy
        assert validated.env_id == "reverse-text"
        assert validated.num_tasks == 4

    def test_write_drops_nothing_and_round_trips(self):
        path = write_converted_eval_config({"id": "reverse-text", "num_tasks": 2})

        assert tomllib.loads(path.read_text(encoding="utf-8")) == {
            "id": "reverse-text",
            "num_tasks": 2,
        }


class TestMainIngestion:
    def test_transitional_at_config_is_rewritten(self, tmp_path, capsys):
        config_path = tmp_path / "old.toml"
        config_path.write_text('[[eval]]\nenv_id = "reverse-text"\n')

        for form in (["@", str(config_path)], [f"@{config_path}"]):
            args = _convert_transitional_args(form)

            assert args != form
            converted = args[-1].removeprefix("@")
            assert is_transitional_config(
                tomllib.loads(config_path.read_text(encoding="utf-8"))
            )
            assert (
                tomllib.loads(open(converted, encoding="utf-8").read())["id"]
                == "reverse-text"
            )

    def test_native_v1_config_passes_through_untouched(self, tmp_path):
        config_path = tmp_path / "v1.toml"
        config_path.write_text('[taskset]\nid = "gsm8k-v1"\n')

        args = ["@", str(config_path)]
        assert _convert_transitional_args(args) == args


def test_unknown_provider_raises_a_clear_error():
    with pytest.raises(ValueError, match="unknown provider `together`"):
        build_v1_eval_config({"id": "reverse-text", "provider": "together"})
