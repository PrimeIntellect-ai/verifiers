import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_env_module(name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).parents[1] / "environments" / name / filename
    spec = importlib.util.spec_from_file_location(f"test_{name}", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dspy_rlm_empty_completion_scores_zero() -> None:
    module = load_env_module("dspy_rlm", "dspy_rlm.py")

    assert module.answer_reward({"answer": "4"}, {"completion": []}) == 0.0


def test_openai_agents_empty_completion_scores_zero() -> None:
    module = load_env_module("openai_agents_env", "openai_agents_env.py")

    assert module.answer_reward({"answer": "4"}, {"completion": []}) == 0.0


@pytest.mark.asyncio
async def test_math_python_empty_completion_scores_zero() -> None:
    module = load_env_module("math_python", "math_python_v1.py")

    assert await module.correct_answer({"answer": "4"}, {"completion": []}) == 0.0


@pytest.mark.asyncio
async def test_hello_subagent_missing_completion_scores_zero() -> None:
    module = load_env_module("hello_subagent_v1", "hello_subagent_v1.py")

    assert (
        await module.exact_answer({"answer": "hello alice"}, {"completion": None})
        == 0.0
    )


def test_hello_parallel_reward_prompt_allows_missing_completion() -> None:
    module = load_env_module(
        "hello_parallel_sandbox_v1", "hello_parallel_sandbox_v1.py"
    )

    prompt = module.reward_prompt(
        {"instruction": "write an answer", "answer": "done"},
        {"completion": None},
    )

    assert "Assistant final answer:\n\n" in prompt
