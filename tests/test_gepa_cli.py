import argparse
from pathlib import Path

from datasets import Dataset
import pytest

from verifiers import EnvGroup, Rubric, SingleTurnEnv
from verifiers.scripts.gepa import (
    _load_gepa_dataset,
    _gepa_extra_headers_from_group,
    load_gepa_toml_config,
    resolve_gepa_config_args,
)


def test_gepa_extra_headers_from_group_requires_consistent_variants():
    with pytest.raises(ValueError, match="different headers"):
        _gepa_extra_headers_from_group(
            [
                {"extra_headers": {"X-A": "1"}},
                {"extra_headers": {"X-A": "2"}},
            ],
            "my-alias",
        )


def test_gepa_extra_headers_from_group_returns_first_row_dict():
    h = _gepa_extra_headers_from_group(
        [
            {"extra_headers": {"X-A": "x"}},
            {"extra_headers": {"X-A": "x"}},
        ],
        "my-alias",
    )
    assert h == {"X-A": "x"}


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
                "max_concurrent = 9",
                "",
                "[sampling]",
                "max_tokens = 1024",
                "temperature = 0.7",
                "",
            ]
        )
    )

    loaded = load_gepa_toml_config(config_path)

    assert loaded["env_id"] == "primeintellect/wiki-search"
    assert loaded["envs"] == [
        {
            "env_id": "primeintellect/wiki-search",
            "env_args": {"split": "train"},
            "extra_env_kwargs": {},
        }
    ]
    assert loaded["env_args"] == {"split": "train"}
    assert loaded["extra_env_kwargs"] == {}
    assert loaded["max_calls"] == 123
    assert loaded["num_train"] == 7
    assert loaded["max_concurrent"] == 9
    assert loaded["sampling_args"] == {"max_tokens": 1024, "temperature": 0.7}
    assert loaded["endpoints_path"] == str((tmp_path / "../endpoints.toml").resolve())


def test_load_gepa_toml_config_reads_env_array(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(
        "\n".join(
            [
                "[[env]]",
                'env_id = "primeintellect/wiki-search"',
                "",
                "[[env]]",
                'env_id = "primeintellect/wordle"',
                'env_args = { split = "train" }',
                "",
            ]
        )
    )

    loaded = load_gepa_toml_config(config_path)

    assert loaded["env_id"] == "wiki-search+wordle"
    assert loaded["envs"] == [
        {
            "env_id": "primeintellect/wiki-search",
            "env_args": {},
            "extra_env_kwargs": {},
        },
        {
            "env_id": "primeintellect/wordle",
            "env_args": {"split": "train"},
            "extra_env_kwargs": {},
        },
    ]


def test_load_gepa_toml_config_accepts_execution_table_with_warning(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(
        "\n".join(
            [
                "[env]",
                'env_id = "primeintellect/wiki-search"',
                "",
                "[execution]",
                "max_concurrent = 9",
                "sampling_args = { max_tokens = 1024 }",
                "",
            ]
        )
    )

    loaded = load_gepa_toml_config(config_path)

    assert loaded["max_concurrent"] == 9
    assert loaded["sampling_args"] == {"max_tokens": 1024}
    assert "[execution]" in loaded["_warnings"][0]


def test_load_gepa_toml_config_rejects_execution_conflicts(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(
        "\n".join(
            [
                "[env]",
                'env_id = "primeintellect/wiki-search"',
                "",
                "[gepa]",
                "max_concurrent = 8",
                "",
                "[execution]",
                "max_concurrent = 9",
                "",
            ]
        )
    )

    with pytest.raises(ValueError, match="both \\[gepa\\] and \\[execution\\]"):
        load_gepa_toml_config(config_path)


def test_load_gepa_toml_config_requires_env_table(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text('model = "openai/gpt-4.1-mini"\n')

    with pytest.raises(
        ValueError, match="must contain an \\[env\\] or \\[\\[env\\]\\]"
    ):
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


def test_load_gepa_dataset_balances_multiple_envs_by_env():
    env1 = SingleTurnEnv(
        dataset=Dataset.from_dict(
            {"question": ["q1", "q2", "q3"], "answer": ["a1", "a2", "a3"]}
        ),
        rubric=Rubric(),
    )
    env2 = SingleTurnEnv(
        dataset=Dataset.from_dict({"question": ["q4"], "answer": ["a4"]}),
        rubric=Rubric(),
    )

    rows = _load_gepa_dataset(
        env=env1,
        envs=[env1, env2],
        env_names=["env1", "env2"],
        split="train",
        n=5,
        seed=0,
    )

    assert [row["task"] for row in rows].count("env1") == 3
    assert [row["task"] for row in rows].count("env2") == 2
    assert [row["example_id"] for row in rows] == list(range(5))
    assert all("source_example_id" in row for row in rows)
    assert [row["info"]["env_id"] for row in rows] == [
        "env1",
        "env1",
        "env1",
        "env2",
        "env2",
    ]

    env_group = EnvGroup(envs=[env1, env2], env_names=["env1", "env2"])
    assert [env_group._input_env_route(row) for row in rows] == [
        ("env1",),
        ("env1",),
        ("env1",),
        ("env2",),
        ("env2",),
    ]
