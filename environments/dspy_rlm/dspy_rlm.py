"""
ApiEnv example using DSPy's RLM (Recursive Language Model) module.

RLM gives the model a sandboxed Python REPL to programmatically explore
problems, making multiple LLM calls that ApiEnv intercepts and routes to
the model under evaluation.

Note: RLM requires Deno for its WASM sandbox. Install with:
    curl -fsSL https://deno.land/install.sh | sh
"""

import re

import dspy

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset


async def run_agent(base_url: str, state: vf.State):
    """Run a DSPy RLM agent against the interception proxy."""
    lm = dspy.LM(
        f"openai/{state['model']}",
        api_base=base_url,
        api_key="intercepted",
        cache=False,
    )

    with dspy.context(lm=lm):
        rlm = dspy.RLM(
            "query -> answer",
            max_iterations=10,
        )

        query = state["prompt"][-1]["content"]
        result = await rlm.aforward(query=query)
        return result.answer


def answer_reward(completion: vf.Messages, answer: str, **kwargs) -> float:
    """Check if the agent's final output contains the correct answer."""
    if not completion:
        return 0.0
    last = str(completion[-1].content or "")

    # Look for DSPy structured output: [[ ## answer ## ]] value
    match = re.search(
        r"\[\[\s*##\s*answer\s*##\s*\]\]\s*(.+?)(?:\n|$)", last, re.IGNORECASE
    )
    if match:
        agent_answer = match.group(1).strip()
    else:
        # Fallback: last non-empty line
        lines = [line.strip() for line in last.strip().split("\n") if line.strip()]
        agent_answer = lines[-1] if lines else ""

    try:
        return 1.0 if abs(float(agent_answer) - float(answer)) < 0.01 else 0.0
    except (ValueError, TypeError):
        return 1.0 if agent_answer.strip() == answer.strip() else 0.0


def load_environment(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    timeout_seconds: float = 180.0,
) -> vf.ApiEnv:
    """Load the DSPy RLM example environment."""

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
