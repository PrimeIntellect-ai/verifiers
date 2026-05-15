import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import verifiers as root_vf
import verifiers.v1 as vf


def load_bfcl_module() -> ModuleType:
    path = Path(__file__).parents[1] / "environments" / "bfcl_v3" / "bfcl_v3.py"
    spec = importlib.util.spec_from_file_location("bfcl_v3_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bfcl_prefers_hinted_function_schemas() -> None:
    bfcl = load_bfcl_module()
    task = {
        "function": [{"name": "plain"}],
        "function_with_hints": [{"name": "hinted"}],
    }

    assert bfcl.bfcl_functions(task) == [{"name": "hinted"}]


def test_bfcl_prefers_hinted_holdout_function_schemas() -> None:
    bfcl = load_bfcl_module()
    task = {
        "missed_function": {"1": [{"name": "plain"}]},
        "missed_function_with_hints": {"1": [{"name": "hinted"}]},
    }

    assert bfcl.bfcl_missed_function(task) == {"1": [{"name": "hinted"}]}


def test_bfcl_row_preserves_hinted_holdout_functions() -> None:
    bfcl = load_bfcl_module()
    entry = {
        "id": "case",
        "question": ["call tools"],
        "function": [{"name": "plain"}],
        "missed_function": {"1": [{"name": "plain_holdout"}]},
    }
    hinted_entry = {
        "function": [{"name": "hinted"}],
        "missed_function": {"1": [{"name": "hinted_holdout"}]},
    }

    row = bfcl.bfcl_row("multi_turn", entry, hinted_entry, None)

    assert row["function_with_hints"] == [{"name": "hinted"}]
    assert row["missed_function"] == {"1": [{"name": "plain_holdout"}]}
    assert row["missed_function_with_hints"] == {"1": [{"name": "hinted_holdout"}]}


def test_bfcl_empty_completion_has_no_tool_calls() -> None:
    bfcl = load_bfcl_module()

    assert bfcl.assistant_tool_calls({"completion": []}) == []
    assert (
        bfcl.assistant_tool_calls(
            {"completion": [{"role": "user", "content": "no assistant"}]}
        )
        == []
    )


def test_bfcl_public_loader_is_v1_only(monkeypatch: pytest.MonkeyPatch) -> None:
    bfcl = load_bfcl_module()
    seen_taskset_config: vf.TasksetConfig | None = None
    seen_harness_config: vf.HarnessConfig | None = None

    def fake_taskset(config: vf.TasksetConfig) -> vf.Taskset:
        nonlocal seen_taskset_config
        seen_taskset_config = config
        config = config.model_copy(update={"source": [], "rewards": []})
        return vf.Taskset(config=config)

    def fake_harness(config: vf.HarnessConfig) -> vf.Harness:
        nonlocal seen_harness_config
        seen_harness_config = config
        return vf.Harness(config=config)

    monkeypatch.setattr(bfcl, "load_taskset", fake_taskset)
    monkeypatch.setattr(bfcl, "load_harness", fake_harness)

    env = bfcl.load_environment(
        config=bfcl.BFCLEnvConfig(
            taskset=bfcl.BFCLTasksetConfig(
                test_category="simple_python",
                examples_per_category=0,
            ),
            harness=bfcl.BFCLHarnessConfig(),
        )
    )

    assert isinstance(env, vf.Env)
    assert isinstance(seen_taskset_config, bfcl.BFCLTasksetConfig)
    assert isinstance(seen_harness_config, bfcl.BFCLHarnessConfig)
    assert seen_taskset_config.test_category == "simple_python"
    assert seen_taskset_config.examples_per_category == 0
    assert seen_harness_config.test_category == "simple_python"
    assert not hasattr(bfcl, "load_v1_environment")


def test_bfcl_loader_supports_category_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bfcl = load_bfcl_module()
    seen_taskset_categories = []
    seen_harness_categories = []

    def fake_taskset(config: vf.TasksetConfig) -> vf.Taskset:
        assert isinstance(config, bfcl.BFCLTasksetConfig)
        seen_taskset_categories.append(config.test_category)
        config = config.model_copy(
            update={"source": [{"question": "q", "answer": "a"}], "rewards": []}
        )
        return vf.Taskset(config=config)

    def fake_harness(config: vf.HarnessConfig) -> vf.Harness:
        assert isinstance(config, bfcl.BFCLHarnessConfig)
        seen_harness_categories.append(config.test_category)
        return vf.Harness(config=config)

    monkeypatch.setattr(bfcl, "load_taskset", fake_taskset)
    monkeypatch.setattr(bfcl, "load_harness", fake_harness)

    env = bfcl.load_environment(
        config=bfcl.BFCLEnvConfig(
            taskset=bfcl.BFCLTasksetConfig(
                test_categories=["simple_python", "simple_java"],
                examples_per_category=0,
            ),
            harness=bfcl.BFCLHarnessConfig(),
        )
    )

    assert isinstance(env, root_vf.EnvGroup)
    assert env.env_names == ["simple_python", "simple_java"]
    assert seen_taskset_categories == ["simple_python", "simple_java"]
    assert seen_harness_categories == ["simple_python", "simple_java"]
