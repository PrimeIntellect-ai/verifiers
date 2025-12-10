"""
Needle in Haystack Environment.

Tests a model's ability to find a specific piece of information (a "magic number")
hidden within a large body of random text.

Supports two modes:
- RLM mode (use_rlm=True): Uses RLMEnv with context in info["context"], model
  explores using Python code in the REPL
- Standard mode (use_rlm=False): Uses SingleTurnEnv with context directly in the
  prompt, model must search through the text directly

The model must find the line containing "The magic number is X" and return the number.
"""

import logging
import random
import re

from datasets import Dataset

import verifiers as vf
from verifiers import RLMEnv
from verifiers import SingleTurnEnv
from verifiers.utils.data_utils import extract_boxed_answer

logger = logging.getLogger(__name__)


def generate_haystack(
    num_lines: int,
    answer: int,
    needle_position: float | None = None,
    needle_variance: float = 0.0,
) -> str:
    """
    Generate a large text context with random lines and a hidden magic number.

    Args:
        num_lines: Total number of lines to generate
        answer: The magic number to hide in the text
        needle_position: Position to place the needle as fraction of total lines (0.0-1.0).
                         If None, places randomly anywhere in the context.
        needle_variance: Variance around needle_position in fraction of total lines.
                         The actual position will be uniformly sampled from
                         [needle_position - needle_variance, needle_position + needle_variance].

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

    # Calculate needle position
    if needle_position is None:
        # Fully random placement
        magic_position = random.randint(0, num_lines - 1)
    else:
        # Calculate position with variance
        pos_min = needle_position - needle_variance
        pos_max = needle_position + needle_variance

        # Clamp to valid range with warning
        if pos_min < 0.0 or pos_max > 1.0:
            original_min, original_max = pos_min, pos_max
            pos_min = max(0.0, pos_min)
            pos_max = min(1.0, pos_max)
            logger.warning(
                f"Needle position range [{original_min:.2f}, {original_max:.2f}] "
                f"exceeds valid bounds [0.0, 1.0], clamping to [{pos_min:.2f}, {pos_max:.2f}]"
            )

        # Sample position within range
        if pos_min == pos_max:
            position_frac = pos_min
        else:
            position_frac = random.uniform(pos_min, pos_max)

        magic_position = int(position_frac * (num_lines - 1))
        magic_position = max(0, min(num_lines - 1, magic_position))

    lines[magic_position] = f"The magic number is {answer}"

    return "\n".join(lines)


def _extract_number(text: str) -> str:
    """Extract a number from text, trying boxed format first, then raw numbers."""
    # Try boxed answer first
    boxed = extract_boxed_answer(text)
    if boxed != text:  # extract_boxed_answer returns original if no boxed found
        return boxed.strip()
    
    # Fall back to finding any 7-digit number in the response
    numbers = re.findall(r"\b\d{7}\b", text)
    if numbers:
        return numbers[-1]  # Return last 7-digit number found
    
    return text.strip()


def load_environment(
    num_samples: int = 10,
    num_lines: int = 10_000,
    needle_position: float | None = None,
    needle_variance: float = 0.0,
    use_rlm: bool = True,
    seed: int | None = None,
    **kwargs,
) -> vf.Environment:
    """
    Load the needle-in-haystack environment.

    Args:
        num_samples: Number of samples to generate for the dataset
        num_lines: Number of lines in each haystack context
        needle_position: Position to place the needle as fraction of context (0.0-1.0).
                         0.0 = beginning, 0.5 = middle, 1.0 = end.
                         If None (default), places randomly anywhere in the context.
        needle_variance: Variance around needle_position in fraction of context length.
                         The actual position will be uniformly sampled from
                         [needle_position - needle_variance, needle_position + needle_variance].
                         Ignored if needle_position is None.
        use_rlm: If True, use RLMEnv with context in info["context"].
                 If False, use SingleTurnEnv with context in the prompt.
        seed: Random seed for reproducible dataset generation. If None, uses random state.
        **kwargs: Additional arguments passed to the environment (e.g., interception_host for RLM)

    Returns:
        Configured environment instance (RLMEnv or SingleTurnEnv)
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Generate dataset with random answers
    dataset_rows = []
    for i in range(num_samples):
        # Generate a random 7-digit answer for each sample
        answer = random.randint(1_000_000, 9_999_999)
        context = generate_haystack(
            num_lines=num_lines,
            answer=answer,
            needle_position=needle_position,
            needle_variance=needle_variance,
        )

        if use_rlm:
            # RLM mode: context goes in info, short prompt
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
        else:
            # Standard mode: context goes directly in the prompt
            dataset_rows.append(
                {
                    "example_id": i,
                    "prompt": [
                        {
                            "role": "user",
                            "content": f"I'm looking for a magic number hidden in the following text. "
                            "Find it and return just the number inside \\boxed{}."
                            f"\n\n<text>\n{context}\n</text>",
                        }
                    ],
                    "task": "needle-in-haystack",
                    "answer": str(answer),
                }
            )

    dataset = Dataset.from_list(dataset_rows)

    if use_rlm:
        # RLM mode: reward functions use state["final_answer"] set by RLMEnv.post_rollout()
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
            max_iterations=30,
            max_output_length=8192,
            context_key="context",
            dataset=dataset,
            rubric=rubric,
            interception_host=kwargs.get("interception_host"),
        )
    else:
        # Standard mode: reward functions extract answer from completion
        def exact_match_reward(completion: str, answer: str) -> float:
            """Reward for exact match with expected answer."""
            extracted = _extract_number(completion)
            return 1.0 if extracted == answer.strip() else 0.0

        def contains_answer_reward(completion: str, answer: str) -> float:
            """Reward if the completion contains the expected number."""
            return 1.0 if answer.strip() in completion else 0.0

        rubric = vf.Rubric(
            funcs=[exact_match_reward, contains_answer_reward],
            weights=[1.0, 0.0],  # Only exact_match contributes to reward
        )

        env = SingleTurnEnv(
            dataset=dataset,
            rubric=rubric,
        )

    return env
