from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as e:
        return f"Error: {e}"
    return str(result)


SYSTEM_PROMPT = (
    "You are a math problem solver. Use the calculate tool to evaluate "
    "expressions and the deep-agent built-in tools (todos, files, subagents) "
    "as you see fit. Give your final numerical answer after the word ANSWER "
    "on its own line, e.g.:\nANSWER: 42"
)


async def run_agent(base_url: str, state: vf.State):
    """Run a LangChain Deep Agent against the interception proxy."""
    model = ChatOpenAI(
        model=state["model"],
        base_url=base_url,
        api_key="intercepted",
    )

    agent = create_deep_agent(
        model=model,
        tools=[calculate],
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = state["prompt"][-1]["content"]
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
    )
    final_message = result["messages"][-1]
    return getattr(final_message, "content", str(final_message))


def answer_reward(completion: vf.Messages, answer: str, **kwargs) -> float:
    """Check if the agent's final output contains the correct answer."""
    if not completion:
        return 0.0
    last_content = str(completion[-1].content or "")
    for line in reversed(last_content.split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER"):
            agent_answer = line.split(":", 1)[-1].strip()
            try:
                return 1.0 if abs(float(agent_answer) - float(answer)) < 0.01 else 0.0
            except (ValueError, TypeError):
                return 1.0 if agent_answer.strip() == answer.strip() else 0.0
    return 0.0


def load_environment(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    timeout_seconds: float = 300.0,
) -> vf.ApiEnv:
    """Load the LangChain Deep Agents example environment."""

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
