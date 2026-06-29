import tomllib
from pathlib import Path

from datasets import Dataset
from pydantic import ValidationError
import pytest

from verifiers import EnvGroup, Rubric, SingleTurnEnv
from verifiers.gepa.config import GEPAConfig, GEPAEnvConfig, GEPAOptimizationConfig
from verifiers.scripts.gepa import (
    _gepa_extra_headers_from_group,
    _load_gepa_dataset,
    main,
)
from verifiers.types import EndpointConfig


def test_gepa_extra_headers_from_group_requires_consistent_variants():
    with pytest.raises(ValueError, match="different headers"):
        _gepa_extra_headers_from_group(
            [
                EndpointConfig(
                    api_key_var="K",
                    base_url="https://a.example/v1",
                    model="m",
                    extra_headers={"X-A": "1"},
                ),
                EndpointConfig(
                    api_key_var="K",
                    base_url="https://a.example/v1",
                    model="m",
                    extra_headers={"X-A": "2"},
                ),
            ],
            "my-alias",
        )


def test_gepa_extra_headers_from_group_returns_first_row_dict():
    headers = _gepa_extra_headers_from_group(
        [
            EndpointConfig(
                api_key_var="K",
                base_url="https://a.example/v1",
                model="m",
                extra_headers={"X-A": "x"},
            ),
            EndpointConfig(
                api_key_var="K",
                base_url="https://a.example/v1",
                model="m",
                extra_headers={"X-A": "x"},
            ),
        ],
        "my-alias",
    )

    assert headers == {"X-A": "x"}


def test_gepa_config_accepts_direct_environment():
    config = GEPAConfig(
        id="wiki_search",
        args={"split": "train"},
    )

    assert config.environment_label == "wiki_search"
    assert config.environments == [
        GEPAEnvConfig(id="wiki_search", args={"split": "train"})
    ]


def test_gepa_config_accepts_environment_group():
    config = GEPAConfig(
        env=[
            GEPAEnvConfig(id="wiki_search"),
            GEPAEnvConfig(id="wordle", args={"split": "train"}),
        ],
        gepa=GEPAOptimizationConfig(max_calls=123, num_train=7),
        sampling={"max_tokens": 1024},
    )

    assert config.environment_label == "wiki_search+wordle"
    assert [item.id for item in config.environments] == [
        "wiki_search",
        "wordle",
    ]
    assert config.gepa.max_calls == 123
    assert config.gepa.num_train == 7
    assert config.sampling == {"max_tokens": 1024}


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {
            "id": "wiki_search",
            "env": [{"id": "wordle"}],
        },
    ],
)
def test_gepa_config_requires_exactly_one_environment_source(raw):
    with pytest.raises(ValidationError, match="exactly one"):
        GEPAConfig.model_validate(raw)


def test_gepa_config_rejects_removed_compatibility_tables():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GEPAConfig.model_validate(
            {
                "id": "wiki_search",
                "execution": {"max_concurrent": 9},
            }
        )


def test_repo_gepa_example_configs_are_valid():
    config_paths = sorted(Path("configs/gepa").glob("*.toml"))
    assert config_paths
    for config_path in config_paths:
        with config_path.open("rb") as handle:
            config = GEPAConfig.model_validate(tomllib.load(handle))
        assert config.environments


def test_gepa_main_parses_direct_environment(monkeypatch):
    captured = {}
    monkeypatch.setattr("verifiers.scripts.gepa.load_endpoints", lambda _: {})
    monkeypatch.setattr(
        "verifiers.scripts.gepa.run_gepa_optimization",
        lambda **kwargs: captured.update(kwargs),
    )

    main(["wiki_search", "--gepa.max-calls", "123"])

    assert captured["env_id"] == "wiki_search"
    assert captured["env_configs"][0].id == "wiki_search"
    assert captured["max_metric_calls"] == 123


def test_gepa_main_parses_direct_environment_after_options(monkeypatch):
    captured = {}
    monkeypatch.setattr("verifiers.scripts.gepa.load_endpoints", lambda _: {})
    monkeypatch.setattr(
        "verifiers.scripts.gepa.run_gepa_optimization",
        lambda **kwargs: captured.update(kwargs),
    )

    main(["--verbose", "--model", "test-model", "wiki_search"])

    assert captured["env_id"] == "wiki_search"
    assert captured["model"] == "test-model"


def test_gepa_main_reads_typed_toml(monkeypatch, tmp_path: Path):
    path = tmp_path / "gepa.toml"
    path.write_text(
        'model = "openai/gpt-4.1-mini"\n'
        "save_results = false\n"
        "[[env]]\n"
        'id = "wiki_search"\n'
        "[gepa]\n"
        "max_calls = 321\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr("verifiers.scripts.gepa.load_endpoints", lambda _: {})
    monkeypatch.setattr(
        "verifiers.scripts.gepa.run_gepa_optimization",
        lambda **kwargs: captured.update(kwargs),
    )

    main([str(path)])

    assert captured["max_metric_calls"] == 321
    assert captured["save_results"] is False
    assert captured["run_dir"] is None


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
