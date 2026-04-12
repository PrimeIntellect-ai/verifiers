from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset


def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def run_agent(base_url: str, state: vf.State):
    """Run an OpenAI Agents SDK agent against the interception proxy."""
    set_tracing_disabled(True)

    client = AsyncOpenAI(base_url=base_url, api_key="intercepted")
    model = OpenAIChatCompletionsModel(
        model=state["model"],
        openai_client=client,
    )

    agent = Agent(
        name="MathSolver",
        instructions=(
            "You are a math problem solver. Use the calculate tool to "
            "evaluate expressions. Give your final numerical answer after "
            "the word ANSWER on its own line, e.g.:\nANSWER: 42"
        ),
        model=model,
        tools=[function_tool(calculate)],
    )

    prompt = state["prompt"][-1]["content"]
    result = await Runner.run(
        agent,
        input=prompt,
        run_config=RunConfig(max_turns=10),
    )
    return result.final_output


def answer_reward(completion: vf.Messages, answer: str, **kwargs) -> float:
    """Check if the agent's final output contains the correct answer."""
    if not completion:
        return 0.0
    last_content = str(completion[-1].content or "")
    # Look for "ANSWER: <number>" pattern
    for line in reversed(last_content.split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER"):
            agent_answer = line.split(":", 1)[-1].strip()
            try:
                return 1.0 if float(agent_answer) == float(answer) else 0.0
            except (ValueError, TypeError):
                return 1.0 if agent_answer.strip() == answer.strip() else 0.0
    return 0.0


def load_environment(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    timeout_seconds: float = 120.0,
) -> vf.ApiEnv:
    """Load the OpenAI Agents SDK example environment."""

    def build_dataset():
        return load_example_dataset("gsm8k", split="train", n=num_train_examples)

    def build_eval_dataset():
        return load_example_dataset("gsm8k", split="test", n=num_eval_examples)

    rubric = vf.Rubric(funcs=[answer_reward])

    return vf.ApiEnv(
        agent_fn=run_agent,
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        rubric=rubric,
        timeout_seconds=timeout_seconds,
    )
