"""
Needle in Haystack RLM Environment.

Tests a model's ability to find a specific piece of information (a "magic number")
hidden within a large body of random text using the RLM REPL environment.

The model must:
1. Explore the large context efficiently using Python code
2. Find the line containing "The magic number is X"
3. Extract and return the number as the final answer
"""

import random

from datasets import Dataset

import verifiers as vf
from verifiers.envs.rlm_env import RLMEnv


def generate_haystack(num_lines: int, answer: int) -> str:
    """
    Generate a large text context with random lines and a hidden magic number.

    Args:
        num_lines: Total number of lines to generate
        answer: The magic number to hide in the text

    Returns:
        A string with num_lines lines, one of which contains the magic number
    """
    random_words = [
        "blah",
        "random",
        "text",
        "data",
        "content",
        "information",
        "sample",
        "filler",
        "noise",
        "stuff",
    ]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Insert the magic number at a random position (somewhere in the middle 80%)
    magic_position = random.randint(int(num_lines * 0.1), int(num_lines * 0.9))
    lines[magic_position] = f"The magic number is {answer}"

    return "\n".join(lines)


def load_environment(
    num_samples: int = 10,
    num_lines: int = 10_000,
    **kwargs,
) -> vf.Environment:
    """
    Load the needle-in-haystack RLM environment.

    Args:
        num_samples: Number of samples to generate for the dataset
        num_lines: Number of lines in each haystack context
        **kwargs: Additional arguments passed to RLMEnv (e.g., interception_host)

    Returns:
        Configured RLMEnv instance
    """
    # Generate dataset with random answers
    dataset_rows = []
    for i in range(num_samples):
        # Generate a random 7-digit answer for each sample
        answer = random.randint(1_000_000, 9_999_999)
        context = generate_haystack(num_lines=num_lines, answer=answer)

        dataset_rows.append(
            {
                "example_id": i,
                "prompt": [
                    {
                        "role": "user",
                        "content": "I'm looking for a magic number hidden in the context. "
                        "Find it and return just the number.",
                    }
                ],
                "task": "needle-in-haystack",
                "answer": str(answer),
                "info": {
                    "context": context,
                },
            }
        )

    dataset = Dataset.from_list(dataset_rows)

    # Reward functions using state["final_answer"] set by RLMEnv.post_rollout()
    def exact_match_reward(state: vf.State) -> float:
        """Reward for exact match with expected answer."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()
        return 1.0 if final_answer == expected else 0.0

    def contains_answer_reward(state: vf.State) -> float:
        """Reward if the final answer contains the expected number."""
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
        timeout_seconds=600.0,
        request_timeout=120.0,
        max_output_length=8192,
        context_key="context",
        dataset=dataset,
        rubric=rubric,
        interception_host=kwargs.get("interception_host"),
    )

    return env
