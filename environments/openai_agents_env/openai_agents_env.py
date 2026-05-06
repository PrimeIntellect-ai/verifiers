from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset

ANSWER_RE = re.compile(r"^\s*ANSWER\s*:?\s*(.+?)\s*$", re.IGNORECASE)


def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


async def run_openai_agents_program(task: Mapping[str, Any], state: Any):
    from agents import (
        Agent,
        OpenAIChatCompletionsModel,
        Runner,
        function_tool,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    endpoint_config = state.get_endpoint_config(api="chat")
    client = AsyncOpenAI(
        base_url=endpoint_config["api_base"],
        api_key=endpoint_config["api_key"],
    )
    model = OpenAIChatCompletionsModel(
        model=endpoint_config["model"],
        openai_client=client,
    )
    agent = Agent(
        name="MathSolver",
        instructions=(
            "You are a math problem solver. Use the calculate tool to evaluate "
            "expressions. Give your final numerical answer after the word ANSWER "
            "on its own line, e.g.:\nANSWER: 42"
        ),
        model=model,
        tools=[function_tool(calculate)],
    )

    result = await Runner.run(agent, input=task_question(task))
    final_output = str(result.final_output)
    state["agent_result"] = final_output
    state["completion"] = [{"role": "assistant", "content": final_output}]
    return state


def config_section(config: object | None, name: str) -> object | None:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def load_rows(split: str, num_examples: int):
    n = num_examples if num_examples > 0 else None
    return load_example_dataset("gsm8k", split=split, n=n)


def task_question(task: Mapping[str, Any]) -> str:
    question = task.get("question")
    if question is not None:
        return str(question)
    prompt = task.get("prompt")
    if isinstance(prompt, list) and prompt:
        last_message = prompt[-1]
        if isinstance(last_message, Mapping):
            return str(last_message.get("content") or "")
    return ""


def completion_text(state: Mapping[str, Any]) -> str:
    agent_result = state.get("agent_result")
    if agent_result is not None:
        return str(agent_result)
    completion = state.get("completion")
    if isinstance(completion, list) and completion:
        last_message = completion[-1]
        if isinstance(last_message, Mapping):
            return str(last_message.get("content") or "")
        return str(getattr(last_message, "content", last_message) or "")
    return ""


def extract_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        match = ANSWER_RE.match(line)
        if match:
            return match.group(1).strip()
    return ""


def answers_match(agent_answer: str, answer: str) -> float:
    try:
        parsed_agent_answer = float(agent_answer.replace(",", ""))
        parsed_answer = float(answer.replace(",", ""))
    except (ValueError, TypeError):
        return 1.0 if agent_answer.strip() == answer.strip() else 0.0
    return 1.0 if abs(parsed_agent_answer - parsed_answer) < 0.01 else 0.0


def answer_reward(task: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    """Check if the agent's final output contains the correct answer."""
    agent_answer = extract_answer(completion_text(state))
    if not agent_answer:
        return 0.0
    return answers_match(agent_answer, str(task.get("answer", "")))


def load_taskset(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    config: object | None = None,
) -> vf.Taskset:
    return vf.Taskset(
        source=lambda: load_rows("train", num_train_examples),
        eval_source=lambda: load_rows("test", num_eval_examples),
        taskset_id="gsm8k-openai-agents",
        rewards=[answer_reward],
        config=config,
    )


def load_harness(config: object | None = None) -> vf.Harness:
    return vf.Harness(program=run_openai_agents_program, config=config)


def load_environment(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    config: object | None = None,
) -> vf.Env:
    """Load the OpenAI Agents SDK V1 taskset/harness example environment."""
    return vf.Env(
        taskset=load_taskset(
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
            config=config_section(config, "taskset"),
        ),
        harness=load_harness(config_section(config, "harness")),
    )
