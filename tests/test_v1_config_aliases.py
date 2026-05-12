from __future__ import annotations

import pytest

from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils import eval_utils
from verifiers.utils.env_config_utils import normalize_env_config_sections


def test_normalize_env_config_sections_keeps_config_first_class() -> None:
    result = normalize_env_config_sections(
        {
            "taskset": {"tasks": "/tmp/tasks"},
            "harness": {"max_turns": 8, "sampling_args": {"max_tokens": 128}},
        },
        global_taskset={"split": "train"},
        global_harness={"max_turns": 4, "sampling_args": {"temperature": 0.2}},
    )

    assert result == {
        "taskset": {"split": "train", "tasks": "/tmp/tasks"},
        "harness": {
            "max_turns": 8,
            "sampling_args": {"temperature": 0.2, "max_tokens": 128},
        },
    }


def test_normalize_env_config_sections_rejects_non_table_aliases() -> None:
    with pytest.raises(ValueError, match="harness"):
        normalize_env_config_sections({"harness": "not-a-table"})


def test_normalize_env_config_sections_preserves_ordinary_config_kwargs() -> None:
    result = normalize_env_config_sections(
        {
            "env_args": {
                "config": {
                    "foo": "bar",
                    "taskset": {"tasks": "/tmp/tasks"},
                    "harness": {"max_turns": 8},
                }
            },
            "taskset": {"split": "eval"},
            "harness": {"sampling_args": {"max_tokens": 128}},
        },
        global_taskset={"split": "train", "dataset": "global"},
        global_harness={"max_turns": 4, "sampling_args": {"temperature": 0.2}},
    )

    assert result == {
        "taskset": {"split": "eval", "dataset": "global", "tasks": "/tmp/tasks"},
        "harness": {
            "max_turns": 8,
            "sampling_args": {"temperature": 0.2, "max_tokens": 128},
        },
        "env_args": {"config": {"foo": "bar"}},
    }


@pytest.mark.asyncio
async def test_run_evaluation_merges_aliases_with_existing_config_kwarg(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class FakeEnv:
        async def evaluate(self, **_kwargs):
            return {"outputs": [], "metadata": None}

    def fake_load_environment(env_id: str, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return FakeEnv()

    monkeypatch.setattr(eval_utils.vf, "load_environment", fake_load_environment)

    config = EvalConfig(
        env_id="env1",
        env_args={"config": {"foo": "bar", "taskset": {"source": "legacy"}}},
        taskset={"tasks": "/tmp/tasks"},
        harness={"max_turns": 4},
        env_dir_path="",
        output_dir=str(tmp_path),
        model="model",
        client_config=ClientConfig(),
        sampling_args={},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
        disable_env_server=True,
    )

    await eval_utils.run_evaluation(config)

    assert captured == {
        "env_id": "env1",
        "kwargs": {
            "config": {
                "foo": "bar",
                "taskset": {"source": "legacy", "tasks": "/tmp/tasks"},
                "harness": {"max_turns": 4},
            }
        },
    }
