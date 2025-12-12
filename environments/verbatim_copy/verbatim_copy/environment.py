"""
Verbatim Copy Environment.

Tests the ability of models to accurately reproduce text verbatim.

Key design: The text to copy is included in the PROMPT for both RLM and standard
modes. This ensures both models must actually write out the text, making it a fair
comparison. The RLM's advantage is its ability to inspect and edit its answer
via the REPL, while the standard LLM generates a one-shot response.

Supports two modes:
- RLM mode (use_rlm=True): Uses RLMEnv, model can use Python to write and verify
- Standard mode (use_rlm=False): Uses SingleTurnEnv, model generates once
"""

from typing import Literal

from datasets import Dataset

import verifiers as vf
from verifiers import RLMEnv, SingleTurnEnv
from verifiers.utils.data_utils import extract_boxed_answer

from .data_generation import DifficultyLevel, generate_dataset


# =============================================================================
# System Prompts
# =============================================================================

# System prompt for standard (non-RLM) mode
_STANDARD_SYSTEM_PROMPT = """You are a precise assistant. Your task is to copy text exactly as shown, character for character.

Pay careful attention to:
- Exact spelling and capitalization
- Punctuation and special characters
- Numbers and formatting
- Whitespace and line breaks

Put your final copied text inside \\boxed{}."""

# Environment-specific tips for RLM mode (used for SFT data generation)
# These tips are wrapped in <env_tips> tags so they can be removed during training
_ENV_TIPS = """
<env_tips>
Strategy for verbatim copying:
1. Write your initial attempt to answer["content"]
2. Print answer["content"] to see exactly what you wrote
3. Compare carefully with the original text - look for typos, transpositions, missing characters
4. Fix any errors using string operations (slicing, replacement, etc.)
5. Only set answer["ready"] = True after you have verified correctness
</env_tips>"""


# =============================================================================
# Reward Functions
# =============================================================================


def _get_response(state: vf.State, completion: vf.Messages, use_rlm: bool) -> str:
    """Extract the model's response based on mode."""
    if use_rlm:
        # Apply extract_boxed_answer in case model wraps answer in \boxed{}
        return extract_boxed_answer(state.get("final_answer", ""))
    else:
        # Standard mode: extract from completion using boxed answer
        if completion and isinstance(completion, list):
            content = (
                completion[-1].get("content", "")
                if isinstance(completion[-1], dict)
                else str(completion[-1])
            )
        else:
            content = str(completion) if completion else ""
        return extract_boxed_answer(content)


def _create_exact_match_reward(use_rlm: bool):
    """Create exact match reward function."""

    def exact_match(
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Reward: 1.0 if response exactly matches expected text, 0.0 otherwise."""
        response = _get_response(state, completion, use_rlm)
        expected = state.get("answer", answer)
        return 1.0 if response == expected else 0.0

    return exact_match


def _create_char_accuracy_reward(use_rlm: bool):
    """Create character-level accuracy reward function."""

    def char_accuracy(
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Metric: proportion of characters that match (using alignment)."""
        response = _get_response(state, completion, use_rlm)
        expected = state.get("answer", answer)

        if not expected:
            return 1.0 if not response else 0.0

        # Simple character-level accuracy: count matching chars at each position
        # For different lengths, compare up to the shorter length and penalize
        matches = 0
        max_len = max(len(response), len(expected))
        min_len = min(len(response), len(expected))

        for i in range(min_len):
            if response[i] == expected[i]:
                matches += 1

        # Penalize length differences
        return matches / max_len if max_len > 0 else 1.0

    return char_accuracy


def _create_levenshtein_similarity_reward(use_rlm: bool):
    """Create Levenshtein similarity reward function."""

    def levenshtein_similarity(
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Metric: 1 - (edit_distance / max_length), giving similarity from 0 to 1."""
        response = _get_response(state, completion, use_rlm)
        expected = state.get("answer", answer)

        if not expected and not response:
            return 1.0
        if not expected or not response:
            return 0.0

        # Levenshtein distance using dynamic programming
        m, n = len(response), len(expected)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if response[i - 1] == expected[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        edit_distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (edit_distance / max_len)

    return levenshtein_similarity


# =============================================================================
# Environment Loading
# =============================================================================


def load_environment(
    num_samples: int = 100,
    difficulty: DifficultyLevel | Literal["all"] = "all",
    seed: int = 42,
    use_rlm: bool = True,
    include_env_tips: bool = False,
    max_iterations: int = 30,
    max_output_length: int = 8192,
    **kwargs,
) -> vf.Environment:
    """
    Load the verbatim copy environment.

    Args:
        num_samples: Number of samples to generate
        difficulty: Difficulty level ("easy", "medium", "hard", "mixed", or "all")
                    "all" uses the default distribution across all difficulties
        seed: Random seed for data generation
        use_rlm: If True, use RLMEnv with REPL access.
                 If False, use SingleTurnEnv for single-shot generation.
        include_env_tips: If True and use_rlm=True, include environment-specific
                          strategy tips in the prompt (wrapped in <env_tips> tags).
                          Useful for SFT data generation. Ignored if use_rlm=False.
        max_iterations: Maximum REPL iterations (RLM mode only)
        max_output_length: Maximum length of code execution output (RLM mode only)
        **kwargs: Additional arguments passed to the environment

    Returns:
        Configured environment instance (RLMEnv or SingleTurnEnv)
    """
    # Generate dataset
    if difficulty == "all":
        # Use default distribution
        samples = generate_dataset(num_samples=num_samples, seed=seed)
    else:
        # Single difficulty level
        samples = generate_dataset(
            num_samples=num_samples,
            difficulty_distribution={difficulty: 1.0},
            seed=seed,
        )

    # Build prompt for each sample
    def build_prompt(sample: dict) -> str:
        text = sample["text"]
        prompt = f"Copy the following text exactly:\n```\n{text}\n```"
        # Add environment tips for RLM mode if requested
        if use_rlm and include_env_tips:
            prompt = prompt + _ENV_TIPS
        return prompt

    # Transform samples into dataset format
    dataset_records = []
    for sample in samples:
        prompt_content = build_prompt(sample)
        record = {
            "prompt": [{"role": "user", "content": prompt_content}],
            "answer": sample["text"],  # Ground truth is the original text
            "info": {
                "difficulty": sample["difficulty"],
                "id": sample["id"],
            },
        }
        dataset_records.append(record)

    dataset = Dataset.from_list(dataset_records)

    # Create reward functions
    exact_match = _create_exact_match_reward(use_rlm)
    char_accuracy = _create_char_accuracy_reward(use_rlm)
    levenshtein_similarity = _create_levenshtein_similarity_reward(use_rlm)

    rubric = vf.Rubric(
        funcs=[exact_match, char_accuracy, levenshtein_similarity],
        weights=[1.0, 0.0, 0.0],  # Only exact_match contributes to reward
    )

    if use_rlm:
        env = RLMEnv(
            max_iterations=max_iterations,
            max_output_length=max_output_length,
            dataset=dataset,
            rubric=rubric,
            max_startup_wait_seconds=60,
            **kwargs,
        )
    else:
        env = SingleTurnEnv(
            dataset=dataset,
            rubric=rubric,
            system_prompt=_STANDARD_SYSTEM_PROMPT,
            **kwargs,
        )

    return env
