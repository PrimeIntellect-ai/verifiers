"""
Dummy RLM Environment - Example usage of the RLMEnv for testing.

This environment demonstrates how to use the Recursive Language Model (RLM)
environment for tasks involving large context analysis.

The RLM environment:
- Uses the normal "prompt" field for the query (like any other environment)
- Optionally accepts large context in info["context"]
- Sets state["final_answer"] after rollout
"""

from datasets import Dataset

import verifiers as vf
from verifiers.envs.rlm_env import RLMEnv


# Example dataset with contexts and queries
_EXAMPLE_DATA = [
    {
        "context": """
Date: Jan 1, 2024 || User: alice123 || Question: What is the capital of France?
Date: Jan 2, 2024 || User: bob456 || Question: How many planets are in our solar system?
Date: Jan 3, 2024 || User: charlie789 || Question: What is photosynthesis?
Date: Jan 4, 2024 || User: alice123 || Question: Who painted the Mona Lisa?
Date: Jan 5, 2024 || User: dave101 || Question: What is the largest ocean?
Date: Jan 6, 2024 || User: eve202 || Question: When did World War II end?
Date: Jan 7, 2024 || User: alice123 || Question: What is the speed of light?
Date: Jan 8, 2024 || User: frank303 || Question: Who wrote Romeo and Juliet?
Date: Jan 9, 2024 || User: grace404 || Question: What is the chemical symbol for gold?
Date: Jan 10, 2024 || User: henry505 || Question: What is the tallest mountain?
""".strip(),
        "query": "How many questions did user 'alice123' ask? Give just the number.",
        "answer": "3",
    },
    {
        "context": """
Product: Widget A || Price: $10.99 || Category: Electronics
Product: Gadget B || Price: $25.50 || Category: Home
Product: Tool C || Price: $15.00 || Category: Electronics
Product: Device D || Price: $99.99 || Category: Electronics
Product: Item E || Price: $5.25 || Category: Kitchen
Product: Thing F || Price: $42.00 || Category: Home
Product: Object G || Price: $18.75 || Category: Electronics
""".strip(),
        "query": "What is the total price of all Electronics products? Give just the number.",
        "answer": "144.73",
    },
    {
        "context": """
Name: John Smith | Age: 32 | Department: Engineering | Salary: $75000
Name: Jane Doe | Age: 28 | Department: Marketing | Salary: $65000
Name: Bob Wilson | Age: 45 | Department: Engineering | Salary: $95000
Name: Alice Brown | Age: 35 | Department: HR | Salary: $70000
Name: Charlie Davis | Age: 29 | Department: Engineering | Salary: $72000
Name: Diana Evans | Age: 41 | Department: Marketing | Salary: $80000
""".strip(),
        "query": "What is the average salary in the Engineering department? Round to 2 decimal places.",
        "answer": "80666.67",
    },
]


def load_environment(**kwargs) -> vf.Environment:
    """Load the dummy RLM environment."""

    # Create dataset from example data
    # Note: prompt comes from the normal "prompt" field
    # Context is optional and goes in info["context"]
    dataset_rows = []
    for i, example in enumerate(_EXAMPLE_DATA):
        dataset_rows.append(
            {
                "example_id": i,
                # The query goes in the normal prompt field
                "prompt": [{"role": "user", "content": example["query"]}],
                "task": "dummy-rlm",
                "answer": example["answer"],
                # Large context goes in info (optional)
                "info": {
                    "context": example["context"],
                },
            }
        )

    dataset = Dataset.from_list(dataset_rows)

    # ==========================================================================
    # User-defined reward functions
    #
    # The RLM environment provides:
    # - state["final_answer"]: The model's final answer (string, empty if none)
    # - state["answer"]: The expected answer from the dataset
    # ==========================================================================

    def exact_match_reward(state: vf.State) -> float:
        """Reward for exact match with expected answer."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()
        return 1.0 if final_answer == expected else 0.0

    def contains_answer_reward(state: vf.State) -> float:
        """Reward if the final answer contains the expected value."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()
        return 1.0 if expected in final_answer else 0.0

    rubric = vf.Rubric(
        funcs=[exact_match_reward, contains_answer_reward],
        weights=[1.0, 0.0],  # Only exact_match contributes to reward
    )

    env = RLMEnv(
        max_turns=30,
        max_iterations=20,
        timeout_seconds=300.0,
        request_timeout=60.0,
        max_output_length=4096,
        context_key="context",  # Key in info where context lives
        dataset=dataset,
        rubric=rubric,
        interception_host=kwargs.get("interception_host"),
    )

    return env
