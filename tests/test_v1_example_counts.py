from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


STATIC_SOURCES = [
    (
        "environments.mcp_search_env.mcp_search_env",
        "source",
        "question",
    ),
    (
        "environments.hello_subagent_v1.hello_subagent_v1",
        "source",
        "prompt",
    ),
    (
        "environments.nested_harness_v1.nested_harness_v1",
        "source",
        "prompt",
    ),
    (
        "environments.hello_rlm_v1.hello_rlm_v1",
        "source",
        "question",
    ),
    (
        "environments.hello_parallel_sandbox_v1.hello_parallel_sandbox_v1",
        "source",
        "instruction",
    ),
    (
        "environments.hello_group_reward_v1.hello_group_reward_v1",
        "source",
        "question",
    ),
    (
        "environments.hello_self_judge_v1.hello_self_judge_v1",
        "source",
        "question",
    ),
    (
        "environments.dspy_flights.dspy_flights",
        "source",
        "user_request",
    ),
]

DEFAULT_EVAL_NUM_EXAMPLES = 5
DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE = 3


def test_static_v1_example_sources_have_at_least_ten_unique_problems() -> None:
    for module_name, source_name, key in STATIC_SOURCES:
        module = importlib.import_module(module_name)
        rows = list(getattr(module, source_name)())
        problems = {problem_text(row, key) for row in rows}

        assert len(rows) >= 10, module_name
        assert len(problems) >= 10, module_name


def test_mcp_search_env_bundles_at_least_ten_self_contained_records() -> None:
    module = importlib.import_module("environments.mcp_search_env.mcp_server")
    records = module.RECORDS

    assert len(records) >= 10
    for record in records.values():
        assert record["title"]
        assert record["summary"]


def test_environment_eval_configs_use_shared_smoke_defaults() -> None:
    pyprojects = sorted((REPO_ROOT / "environments").glob("*/pyproject.toml"))
    assert pyprojects

    for pyproject in pyprojects:
        config = tomllib.loads(pyproject.read_text())
        eval_config = config["tool"]["verifiers"]["eval"]

        env_name = pyproject.parent.name
        assert eval_config["num_examples"] == DEFAULT_EVAL_NUM_EXAMPLES, env_name
        assert (
            eval_config["rollouts_per_example"] == DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE
        ), env_name


def problem_text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    if key == "prompt":
        return prompt_text(value)
    return str(value)


def prompt_text(prompt: object) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Iterable):
        parts = []
        for item in prompt:
            if isinstance(item, Mapping):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(prompt)
