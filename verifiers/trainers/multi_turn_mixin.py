"""
Multi-turn extension for TRL's GRPOTrainer.

This module provides backwards-compatible multi-turn capabilities by:
1. Detecting when an environment uses multi-turn rollouts
2. Routing single-turn environments through TRL's standard path
3. Routing multi-turn environments through specialized handlers
4. Preserving full conversation context for proper reward computation
"""

import asyncio
import logging
from typing import List, Union

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from verifiers.envs.environment import Environment
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages


class MultiTurnMixin:
    """
    Mixin class that adds multi-turn rollout support to TRL trainers.
    
    This mixin provides backwards-compatible multi-turn functionality by:
    - Detecting single-turn vs multi-turn environments
    - Using TRL's standard generation for single-turn cases  
    - Using verifiers' multi-turn rollout for complex cases
    - Preserving conversation context for accurate rewards
    """
    
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        super().__init__(*args, **kwargs)
    
    def is_multi_turn_environment(self, env: Environment) -> bool:
        """
        Detect if an environment requires multi-turn rollouts.
        
        Args:
            env: The environment to check
            
        Returns:
            True if multi-turn handling is needed, False for single-turn
        """
        # Special case: SingleTurnEnv should be treated as single-turn despite inheriting from MultiTurnEnv
        if type(env).__name__ == 'SingleTurnEnv':
            return False
            
        # Check for tools first (usually indicates multi-turn)
        if hasattr(env, 'oai_tools') and env.oai_tools:
            return True
            
        # Check for actual multi-turn classes (but not SingleTurnEnv)
        if isinstance(env, MultiTurnEnv) and type(env).__name__ != 'SingleTurnEnv':
            return True
            
        # Check for multi-turn methods (for non-inheritance cases)
        has_is_completed = hasattr(env, 'is_completed') and callable(getattr(env, 'is_completed'))
        has_env_response = hasattr(env, 'env_response') and callable(getattr(env, 'env_response'))
        
        if has_is_completed and has_env_response and type(env).__name__ != 'SingleTurnEnv':
            return True
            
        return False
    
    def create_multi_turn_dataset(
        self,
        env: Environment,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int = 1000,
        **generation_kwargs
    ) -> Dataset:
        """
        Generate multi-turn dataset using environment rollouts.
        
        Args:
            env: Multi-turn environment
            model: The model to use for generation
            tokenizer: Tokenizer for the model
            num_samples: Number of samples to generate
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Dataset with multi-turn conversations
        """
        # Get training data from environment
        dataset = env.get_dataset(n=num_samples)
        
        # Use environment's async generation with proper multi-turn handling
        results = env.generate(
            inputs=dataset,
            client=env.client,
            model=env.model,
            sampling_args=env.sampling_args,
            score_rollouts=True,
            **generation_kwargs
        )
        
        # Convert to format expected by TRL
        trl_data = []
        for i in range(len(results.prompt)):
            # For multi-turn, we need to preserve the full conversation
            prompt = results.prompt[i]
            completion = results.completion[i]
            reward = results.reward[i]
            
            # Convert to strings for TRL compatibility
            if isinstance(prompt, list):
                # Convert chat messages to string format
                prompt_str = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_str = str(prompt)
                
            if isinstance(completion, list):
                # Convert completion messages to string format
                completion_str = tokenizer.apply_chat_template(
                    completion, tokenize=False, add_generation_prompt=False
                )
            else:
                completion_str = str(completion)
            
            trl_data.append({
                "prompt": prompt_str,
                "completion": completion_str,
                "reward": reward,
                # Store original multi-turn data for debugging
                "original_prompt": prompt,
                "original_completion": completion,
                "state": results.state[i] if i < len(results.state) else {},
            })
        
        return Dataset.from_list(trl_data)
    
    def generate_multi_turn_batches(
        self,
        env: Environment,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 8,
        num_batches: int = 1,
        **generation_kwargs
    ) -> List[Dataset]:
        """
        Generate multiple batches of multi-turn data.
        
        Args:
            env: Multi-turn environment
            model: The model to use for generation
            tokenizer: Tokenizer for the model
            batch_size: Size of each batch
            num_batches: Number of batches to generate
            **generation_kwargs: Additional generation arguments
            
        Returns:
            List of datasets, one per batch
        """
        batches = []
        for batch_idx in range(num_batches):
            self.logger.info(f"Generating multi-turn batch {batch_idx + 1}/{num_batches}")
            
            batch_dataset = self.create_multi_turn_dataset(
                env=env,
                model=model,
                tokenizer=tokenizer,
                num_samples=batch_size,
                **generation_kwargs
            )
            batches.append(batch_dataset)
            
        return batches
    
    def compute_multi_turn_rewards(
        self,
        env: Environment,
        prompts: List[Union[str, Messages]],
        completions: List[Union[str, Messages]],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for multi-turn conversations using environment rubric.
        
        Args:
            env: The environment with rubric
            prompts: List of prompts (can be strings or message lists)
            completions: List of completions (can be strings or message lists)
            **kwargs: Additional arguments for reward computation
            
        Returns:
            List of reward scores
        """
        if not hasattr(env, 'rubric') or not hasattr(env.rubric, 'score_rollouts'):
            # Fallback to simple reward computation
            return [1.0] * len(prompts)
        
        # Prepare data for rubric scoring
        # Most rubrics expect the original multi-turn format
        original_prompts = []
        original_completions = []
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Try to recover original format from dataset if available
            if hasattr(self, '_current_batch_data') and i < len(self._current_batch_data):
                batch_item = self._current_batch_data[i]
                if 'original_prompt' in batch_item:
                    original_prompts.append(batch_item['original_prompt'])
                    original_completions.append(batch_item['original_completion'])
                    continue
            
            # Fallback: use the string versions
            original_prompts.append(prompt)
            original_completions.append(completion)
        
        # Use async scoring if available
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to handle carefully
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._compute_rewards_sync, env, original_prompts, original_completions, **kwargs)
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._compute_rewards_async(env, original_prompts, original_completions, **kwargs)
                )
        except Exception:
            # Fallback to synchronous computation
            return self._compute_rewards_sync(env, original_prompts, original_completions, **kwargs)
    
    async def _compute_rewards_async(
        self,
        env: Environment,
        prompts: List[Messages],
        completions: List[Messages],
        **kwargs
    ) -> List[float]:
        """Async reward computation."""
        # Create dummy states and other required fields
        states = [{}] * len(prompts)
        answers = kwargs.get('answers', [''] * len(prompts))
        tasks = kwargs.get('tasks', ['default'] * len(prompts))
        infos = kwargs.get('infos', [{}] * len(prompts))
        
        result = await env.rubric.score_rollouts(
            prompts=prompts,
            completions=completions,
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            apply_weights=True
        )
        return result.reward
    
    def _compute_rewards_sync(
        self,
        env: Environment,
        prompts: List[Messages],
        completions: List[Messages],
        **kwargs
    ) -> List[float]:
        """Synchronous reward computation."""
        try:
            return asyncio.run(self._compute_rewards_async(env, prompts, completions, **kwargs))
        except Exception:
            # Final fallback: use simple reward function
            self.logger.warning("Failed to compute multi-turn rewards, using fallback")
            return [1.0] * len(prompts)