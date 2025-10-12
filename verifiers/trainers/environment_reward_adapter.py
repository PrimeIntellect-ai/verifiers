"""
EnvironmentRewardAdapter - Shared adapter for converting verifiers Environment rewards to TRL format

This adapter is used by both VerifiersGRPOTrainer and VerifiersRLOOTrainer (and any future TRL trainers)
to convert the verifiers Environment rubric scoring system to TRL's reward function format.
"""

import asyncio
import concurrent.futures
import logging
from typing import Optional

from verifiers.envs import Environment


class EnvironmentRewardAdapter:
    """
    Adapter to convert verifiers Environment scoring to TRL reward function format.

    This adapter bridges the gap between:
    - TRL's reward function signature: (prompts, completions, **kwargs) -> list[float]
    - Verifiers Environment rubric signature: (prompt_messages, completion_messages, ...) -> RolloutScores

    Used by:
    - VerifiersGRPOTrainer
    - VerifiersRLOOTrainer
    - (Future TRL trainers)
    """

    def __init__(self, env: Environment):
        self.env = env
        self.logger = logging.getLogger(__name__)
        self.__name__ = f"EnvironmentRewardAdapter_{type(env).__name__}"

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """
        Convert TRL's reward function format to verifiers format and compute rewards.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            **kwargs: Additional kwargs (ignored for now)

        Returns:
            List of reward scores
        """
        # Convert to verifiers format (list of message dictionaries)
        prompt_messages = [
            [{"role": "user", "content": str(prompt)}] for prompt in prompts
        ]
        completion_messages = [
            [{"role": "assistant", "content": str(completion)}]
            for completion in completions
        ]

        # Default parameters for verifiers (the environment doesn't use these for scoring)
        answers = [""] * len(prompts)
        states = [{"timing": {"total_ms": 0}} for _ in range(len(prompts))]
        tasks = ["default"] * len(prompts)
        infos = [{}] * len(prompts)

        # Run scoring in isolated thread to avoid event loop conflicts
        def scoring_thread():
            return asyncio.run(
                self.env.rubric.score_rollouts(
                    prompt_messages, completion_messages, answers, states, tasks, infos
                )
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(scoring_thread)
                rollout_scores = future.result(timeout=60)
        except Exception as e:
            self.logger.error(f"Error in environment reward computation: {e}")
            # Return zero rewards on error
            return [0.0] * len(prompts)

        return rollout_scores.reward
