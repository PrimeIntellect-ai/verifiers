"""
Multi-turn rollout mixin that adds true multi-turn conversation support to TRL's GRPOTrainer.

This mixin intercepts TRL's generation process to create full multi-turn conversations
instead of single completions.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Union

import torch

from verifiers.envs.environment import Environment
from verifiers.envs.multiturn_env import MultiTurnEnv


class MultiTurnMixin:
    """
    Mixin that adds multi-turn rollout support to TRL's GRPOTrainer.

    This mixin overrides the generation process to:
    1. Generate full multi-turn conversations using the environment
    2. Train on complete conversation trajectories
    3. Compute rewards based on full conversations
    """

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        super().__init__(*args, **kwargs)

    def is_multi_turn_environment(self, env: Environment) -> bool:
        """
        Detect if an environment requires multi-turn rollouts.
        """
        # Special case: SingleTurnEnv is explicitly single-turn
        if type(env).__name__ == "SingleTurnEnv":
            return False

        # Check for tools (usually indicates multi-turn)
        if hasattr(env, "oai_tools") and env.oai_tools:
            return True

        # Check if it's a MultiTurnEnv subclass (except SingleTurnEnv)
        if isinstance(env, MultiTurnEnv) and type(env).__name__ != "SingleTurnEnv":
            return True

        # Check for multi-turn interface methods
        has_multi_turn_methods = (
            hasattr(env, "is_completed")
            and callable(getattr(env, "is_completed"))
            and hasattr(env, "env_response")
            and callable(getattr(env, "env_response"))
        )

        if has_multi_turn_methods and type(env).__name__ != "SingleTurnEnv":
            return True

        return False

    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Override TRL's generation to provide multi-turn rollouts.

        For multi-turn environments, this generates full conversations.
        For single-turn environments, it falls back to standard generation.
        """
        # Check if this is a multi-turn environment
        if hasattr(self, "is_multi_turn") and self.is_multi_turn:
            return self._generate_multi_turn_rollouts(inputs)
        else:
            # Fall back to standard TRL generation
            return super()._generate_and_score_completions(inputs)

    def _generate_multi_turn_rollouts(
        self, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate full multi-turn conversations using the verifiers environment approach.

        This replicates the original verifiers GRPOTrainer logic:
        1. Get dataset from environment
        2. Use env.a_generate() to create full multi-turn conversations
        3. Use env.process_env_results_vllm() to convert to training format
        """
        device = self.accelerator.device
        batch_size = len(inputs)

        try:
            # Get fresh dataset from environment (like original implementation)
            # Handle case where environment dataset is smaller than batch_size
            try:
                dataset = self.env.get_dataset(n=batch_size)
            except Exception:
                # Fallback: get all available data and cycle/repeat if needed
                dataset = self.env.get_dataset()
                if len(dataset) < batch_size:
                    # Repeat dataset entries to match batch_size
                    dataset_list = (
                        dataset.to_list()
                        if hasattr(dataset, "to_list")
                        else list(dataset)
                    )
                    repeated_data = []
                    for i in range(batch_size):
                        repeated_data.append(dataset_list[i % len(dataset_list)])
                    from datasets import Dataset

                    dataset = Dataset.from_list(repeated_data)

            # Use environment's async generation (like AsyncBatchGenerator does)
            def run_async_generation():
                return asyncio.run(
                    self.env.a_generate(
                        dataset,
                        client=getattr(self.env, "client", None),
                        model=getattr(self.env, "model", "gpt-4"),  # Default model
                        sampling_args=getattr(self.env, "sampling_args", {}),
                        score_rollouts=True,
                        max_concurrent=getattr(self, "max_concurrent", 10),
                    )
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                env_results = executor.submit(run_async_generation).result(timeout=300)

            # Use environment's processing method (key difference from our previous attempt)
            processed_results = self.env.process_env_results_vllm(
                prompts=env_results.prompt,
                completions=env_results.completion,
                states=env_results.state,
                rewards=env_results.reward,
                processing_class=self.processing_class,
                max_seq_len=getattr(self, "max_seq_len", 2048),
                mask_env_responses=getattr(self, "mask_env_responses", False),
                mask_truncated_completions=getattr(
                    self, "mask_truncated_completions", False
                ),
                zero_truncated_completions=getattr(
                    self, "zero_truncated_completions", False
                ),
            )

            # Convert ProcessedOutputs to TRL format
            # Move tensors to the correct device
            prompt_ids = processed_results.prompt_ids.to(device)
            prompt_mask = processed_results.prompt_mask.to(device)
            completion_ids = processed_results.completion_ids.to(device)
            completion_mask = processed_results.completion_mask.to(device)

            # Combine prompt and completion
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            # Convert messages back to strings for TRL compatibility
            prompt_strings = []
            completion_strings = []

            for i in range(len(env_results.prompt)):
                # Convert prompt messages to string
                if isinstance(env_results.prompt[i], list):
                    prompt_str = self.processing_class.apply_chat_template(
                        env_results.prompt[i],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt_str = str(env_results.prompt[i])
                prompt_strings.append(prompt_str)

                # Convert completion messages to string
                if isinstance(env_results.completion[i], list):
                    completion_str = self.processing_class.apply_chat_template(
                        env_results.completion[i],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    completion_str = str(env_results.completion[i])
                completion_strings.append(completion_str)

            # Prepare output in TRL's expected format
            output = {
                "prompt": prompt_strings,
                "completion": completion_strings,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "prompt_completion_ids": prompt_completion_ids,
                "prompt_completion_mask": prompt_completion_mask,
                "rewards": torch.tensor(env_results.reward, device=device),
            }

            # Add any extra fields from original inputs
            for key in inputs[0] if inputs else {}:
                if key not in output:
                    output[key] = [
                        inputs[i % len(inputs)][key] for i in range(batch_size)
                    ]

            self.logger.info(
                f"Generated {batch_size} multi-turn conversations successfully"
            )
            return output

        except Exception as e:
            import traceback

            self.logger.warning(
                f"Multi-turn generation failed, falling back to standard: {e}"
            )
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Fall back to standard generation on error
            return super()._generate_and_score_completions(inputs)
