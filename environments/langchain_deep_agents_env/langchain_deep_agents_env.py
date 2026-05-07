from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset

ANSWER_RE = re.compile(r"^\s*ANSWER\s*:?\s*(.+?)\s*$", re.IGNORECASE)

MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 64
MAX_ABSOLUTE_VALUE = 1e100
MAX_POWER_EXPONENT = 100

BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SYSTEM_PROMPT = (
    "You are a math problem solver. Use the calculate tool to evaluate "
    "expressions and the deep-agent built-in tools (todos, files, subagents) "
    "as you see fit. Give your final numerical answer after the word ANSWER "
    "on its own line, e.g.:\nANSWER: 42"
)


def _check_numeric_limit(value: float) -> float:
    if not math.isfinite(value) or abs(value) > MAX_ABSOLUTE_VALUE:
        raise ValueError("result is too large")
    return value


def _evaluate_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_math_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("only numeric literals are allowed")
        return _check_numeric_limit(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return _check_numeric_limit(
            UNARY_OPERATORS[type(node.op)](_evaluate_math_node(node.operand))
        )
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _evaluate_math_node(node.left)
        right = _evaluate_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise ValueError("exponent is too large")
        return _check_numeric_limit(BINARY_OPERATORS[type(node.op)](left, right))
    raise ValueError("only arithmetic expressions are allowed")


def evaluate_math_expression(expression: str) -> float:
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("expression is too complex")
    return _evaluate_math_node(tree)


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = evaluate_math_expression(expression)
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


async def run_langchain_deep_agents_program(task: Mapping[str, Any], state: Any):
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    endpoint_config = state.get_endpoint_config(api="chat")
    model = ChatOpenAI(
        model=endpoint_config["model"],
        base_url=endpoint_config["api_base"],
        api_key=endpoint_config["api_key"],
    )
    agent = create_deep_agent(
        model=model,
        tools=[tool(calculate)],
        system_prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task_question(task)}]}
    )
    final_message = result["messages"][-1]
    final_output = str(getattr(final_message, "content", str(final_message)))
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
        taskset_id="gsm8k-langchain-deep-agents",
        rewards=[answer_reward],
        config=config,
    )


def load_harness(config: object | None = None) -> vf.Harness:
    return vf.Harness(program=run_langchain_deep_agents_program, config=config)


def load_environment(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    config: object | None = None,
) -> vf.Env:
    """Load the LangChain Deep Agents V1 taskset/harness example environment."""
    return vf.Env(
        taskset=load_taskset(
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
            config=config_section(config, "taskset"),
        ),
        harness=load_harness(config_section(config, "harness")),
    )
