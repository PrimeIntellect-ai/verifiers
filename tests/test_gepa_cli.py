import argparse
from pathlib import Path

import pytest

from verifiers.scripts.gepa import load_gepa_toml_config, resolve_gepa_config_args


def test_load_gepa_toml_config_reads_env_table(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "openai/gpt-4.1-mini"',
                'endpoints_path = "../endpoints.toml"',
                "",
                "[env]",
                'env_id = "primeintellect/wiki-search"',
                'env_args = { split = "train" }',
                "extra_env_kwargs = {}",
                "",
                "[gepa]",
                "max_calls = 123",
                "num_train = 7",
                "",
                "[execution]",
                "max_concurrent = 9",
                "",
            ]
        )
    )

    loaded = load_gepa_toml_config(config_path)

    assert loaded["env_id"] == "primeintellect/wiki-search"
    assert loaded["env_args"] == {"split": "train"}
    assert loaded["extra_env_kwargs"] == {}
    assert loaded["max_calls"] == 123
    assert loaded["num_train"] == 7
    assert loaded["max_concurrent"] == 9
    assert loaded["endpoints_path"] == str((tmp_path / "../endpoints.toml").resolve())


def test_load_gepa_toml_config_requires_env_table(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text('model = "openai/gpt-4.1-mini"\n')

    with pytest.raises(ValueError, match="must contain an \\[env\\] table"):
        load_gepa_toml_config(config_path)


def test_resolve_gepa_config_args_supports_plain_env_id():
    args = argparse.Namespace(env_id_or_config="primeintellect/wordle")

    resolved = resolve_gepa_config_args(args)

    assert resolved.env_id == "primeintellect/wordle"


def test_resolve_gepa_config_args_reads_toml_and_save_results(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "openai/gpt-4.1-mini"',
                "save_results = false",
                "",
                "[env]",
                'env_id = "primeintellect/wiki-search"',
                "env_args = {}",
                "extra_env_kwargs = {}",
                "",
                "[gepa]",
                "max_calls = 321",
                "",
            ]
        )
    )

    args = argparse.Namespace(
        env_id_or_config=str(config_path),
        no_save=False,
    )

    resolved = resolve_gepa_config_args(args)

    assert resolved.env_id == "primeintellect/wiki-search"
    assert resolved.max_calls == 321
    assert resolved.no_save is True
