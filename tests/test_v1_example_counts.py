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
        "environments.mcp_search_env.mcp_search_v1",
        "default_dataset",
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

EVAL_CONFIGS_WITH_FULL_POOLS = [
    "hello_self_judge_v1",
    "hello_parallel_sandbox_v1",
    "hello_group_reward_v1",
    "rlm_swe_v1",
    "opencode_harbor",
    "tau2_bench_v1",
]


def test_static_v1_example_sources_have_at_least_ten_unique_problems() -> None:
    for module_name, source_name, key in STATIC_SOURCES:
        module = importlib.import_module(module_name)
        rows = list(getattr(module, source_name)())
        problems = {problem_text(row, key) for row in rows}

        assert len(rows) >= 10, module_name
        assert len(problems) >= 10, module_name


def test_mcp_search_v1_bundles_at_least_ten_self_contained_docs() -> None:
    module = importlib.import_module("environments.mcp_search_env.mcp_server")
    documents = module.DOCUMENTS

    assert len(documents) >= 10
    for document in documents.values():
        assert "Source:" in document["content"]


def test_low_v1_eval_smoke_caps_are_at_least_ten_examples() -> None:
    for env_name in EVAL_CONFIGS_WITH_FULL_POOLS:
        pyproject = REPO_ROOT / "environments" / env_name / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        num_examples = config["tool"]["verifiers"]["eval"]["num_examples"]

        assert num_examples >= 10, env_name


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
