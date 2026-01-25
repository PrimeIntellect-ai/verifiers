"""
MultiAgentOrchestrator: Training integration for multi-agent environments.

This orchestrator wraps a Protocol to enable multi-agent and multi-environment
training. It delegates batch generation to Protocol.generate() which handles:
- Routing examples to correct environments (via 'task' field)
- Multi-actor turn management
- Child episode spawning
- Per-actor state flattening for credit assignment

Key differences from base Orchestrator:
- Uses Protocol's unified dataset instead of single env's dataset
- Calls protocol.generate() instead of env.generate()
- Receives flattened states (per-actor) with pre-computed advantages

Flow:
1. get_dataset_slice() pulls from Protocol's dataset
2. generate_batch() calls protocol.generate() for rollouts
3. Protocol returns flattened per-actor states with advantages
4. This class packages them into microbatches for training

"""

import time
from typing import Any

import numpy as np
from datasets import Dataset

from verifiers.envs.protocol import Protocol

from .orchestrator import Batch, Microbatch, Orchestrator


# =============================================================================
# MultiAgentOrchestrator
# =============================================================================

class MultiAgentOrchestrator(Orchestrator):
    """
    Orchestrator that delegates to Protocol for multi-agent generation.

    Extends base Orchestrator but overrides:
    - get_dataset_slice(): Use Protocol's dataset instead of env's
    - generate_batch(): Use protocol.generate() instead of env.generate()

    All other functionality (tokenizer, batch sizes, client setup) inherited
    from parent Orchestrator.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        protocol: Protocol,
        **kwargs,
    ):
        """
        Initialize orchestrator with Protocol.

        Args:
            protocol: Protocol containing actors, environments, and dataset
            **kwargs: Passed to parent Orchestrator (client, model_name,
                      sampling_args, batch sizes, etc.)
        """
        self.protocol = protocol

        # Parent Orchestrator requires an env parameter for initialization.
        # We use the first env from Protocol - parent uses it to set up
        # tokenizer and other config, but we override the actual generation.
        first_env = next(iter(protocol.envs.values()))
        super().__init__(env=first_env, **kwargs)

        # ---- Filter Protocol's dataset by prompt length ----
        # Parent's __init__ filters self.env.dataset, but we use Protocol's
        # dataset instead, so we need to apply the same filtering here.
        max_length = self.max_prompt_len

        def filter_by_prompt_length(example, processing_class):
            """Keep only examples whose prompts fit in context."""
            prompt = example["prompt"]
            if isinstance(prompt, list):
                # Chat format - apply template to get full text
                prompt_text = processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            prompt_ids = processing_class.encode(prompt_text)
            return len(prompt_ids) <= max_length

        if self.protocol._dataset is not None:
            self.protocol._dataset = self.protocol.get_dataset().filter(
                filter_by_prompt_length,
                fn_kwargs={"processing_class": self.processing_class},
            )

    # -------------------------------------------------------------------------
    # Dataset Access
    # -------------------------------------------------------------------------

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """
        Get dataset slice from Protocol's unified dataset.

        Overrides parent to use Protocol's dataset instead of env's dataset.
        This is necessary because Protocol's dataset may contain examples for
        multiple environments (routed by 'task' field).

        Args:
            batch_id: Which batch to get (determines offset into dataset)

        Returns:
            Dataset slice with prompts_per_batch examples

        """
        num_rows = self.prompts_per_batch
        dataset = self.protocol.get_dataset()
        total_rows = len(dataset)

        if total_rows == 0:
            raise ValueError("Protocol dataset is empty")

        # Calculate offset with wraparound for continuous training
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]

        return dataset.select(indices)

    # -------------------------------------------------------------------------
    # Batch Generation
    # -------------------------------------------------------------------------

    async def generate_batch(self, batch_id: int) -> Batch:
        """
        Generate training batch using protocol.generate() for multi-agent support.

        Overrides parent to use Protocol instead of single environment.
        Protocol handles:
        - Routing examples to correct environments
        - Multi-actor turn management
        - Child episode spawning
        - Per-actor state flattening with pre-computed advantages

        Args:
            batch_id: Batch identifier (determines dataset slice)

        Returns:
            Batch object containing microbatches ready for training

        Flow:
        1. Get dataset slice and repeat for GRPO rollouts
        2. Call protocol.generate() → flattened per-actor states
        3. Extract training data from trajectories
        4. Collect metrics for logging
        5. Package into microbatches for distributed training
        """
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()

        # ==== Step 1: Prepare inputs ====
        # Get batch of examples and repeat each for multiple GRPO rollouts
        # e.g., 8 examples × 4 rollouts = 32 inputs
        batch_ds = self.get_dataset_slice(batch_id)
        repeated_ds = batch_ds.repeat(self.rollouts_per_example)
        inputs = repeated_ds.to_list()

        # ==== Step 2: Run rollouts via Protocol ====
        # Protocol.generate() returns FLATTENED states:
        # - Original game states are replaced by per-actor child states
        # - Each state has pre-computed reward and advantage
        # - Advantages computed per-actor within GRPO groups
        all_states = await self.protocol.generate(
            inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            max_concurrent=self.max_concurrent,
        )

        self.is_generating = False
        wall_clock_s = time.time() - start_time

        # ==== Step 3: Extract training data from trajectories ====
        # Each trajectory step = one model call = one training example
        # Multi-agent states have trajectory filtered to single actor's turns
        prompt_ids: list[list[int]] = []
        prompt_mask: list[list[int]] = []
        completion_ids: list[list[int]] = []
        completion_mask: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        advantages: list[float] = []

        for state in all_states:
            trajectory = state.get("trajectory", [])
            for step in trajectory:
                # Skip steps without tokenized data (e.g., env-only turns)
                tokens = step.get("tokens")
                if tokens is None:
                    continue

                # Tokenized prompt and completion for this turn
                prompt_ids.append(tokens["prompt_ids"])
                prompt_mask.append(tokens["prompt_mask"])
                completion_ids.append(tokens["completion_ids"])
                completion_mask.append(tokens["completion_mask"])

                # Log probs from sampling (for importance weighting in GRPO)
                completion_logprobs.append(tokens["completion_logprobs"])

                # Advantage already computed per-actor during scoring
                advantages.append(step.get("advantage", 0.0))

        # ==== Step 4: Collect metrics for logging ====
        # Rewards per state (for logging, not training)
        rewards = [state.get("reward", 0.0) for state in all_states]
        rewards_dict: dict[str, list[float]] = {"reward": rewards}

        metrics_dict: dict[str, float] = {}

        # Reward statistics
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        # Advantage statistics (should be mean ~0 after GRPO normalization)
        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        # Token statistics
        completion_lengths = [len(ids) for ids in completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

            # Calculate fraction of tokens that are masked (padding)
            completion_mask_lengths = np.asarray(
                [sum(mask) for mask in completion_mask],
                dtype=np.float32,
            )
            valid_tokens = completion_mask_lengths.sum()
            total_tokens = completion_lengths_arr.sum()
            if total_tokens > 0:
                masked_fraction = 1.0 - (valid_tokens / total_tokens)
                metrics_dict["tokens/masked_fraction"] = float(masked_fraction)

        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)

        # Collect raw data for logging/debugging
        errors = [state.get("error") for state in all_states]
        completions = [state.get("completion") for state in all_states]
        prompts = [state.get("prompt") for state in all_states]

        # ==== Step 5: Build microbatches for distributed training ====
        # Split training examples across GPU processes, then into microbatches
        #

        N = len(advantages)  # Total training examples (trajectory steps)
        per_proc = N // self.num_processes if self.num_processes > 0 else N
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []

        for proc in range(self.num_processes):
            # Index range for this process
            ps = proc * per_proc  # process start
            pe = ps + per_proc    # process end

            proc_mbs: list[Microbatch] = []
            proc_item_total = 0

            # Split process's examples into microbatches
            for s in range(ps, pe, self.micro_batch_size):
                e = min(s + self.micro_batch_size, pe)

                # Combine prompt + completion into single sequence for training
                ids_chunk = [prompt_ids[i] + completion_ids[i] for i in range(s, e)]
                mask_chunk = [prompt_mask[i] + completion_mask[i] for i in range(s, e)]

                # Log probs: zeros for prompt (no loss), actual for completion
                logprobs_chunk = [
                    [0.0] * len(prompt_mask[i]) + completion_logprobs[i]
                    for i in range(s, e)
                ]

                # Expand scalar advantage to per-token (same value repeated)
                lengths = [len(mask) for mask in mask_chunk]
                adv_chunk = [
                    [advantages[i]] * lengths[idx]
                    for idx, i in enumerate(range(s, e))
                ]

                # Count valid (non-masked) tokens for normalization
                mb_items = sum(sum(mask) for mask in mask_chunk)

                microbatch = Microbatch(
                    input_ids=ids_chunk,
                    loss_mask=mask_chunk,
                    sampling_logprobs=logprobs_chunk,
                    advantages=adv_chunk,
                    items=mb_items,
                )
                proc_item_total += mb_items
                proc_mbs.append(microbatch)

            microbatches.append(proc_mbs)
            items_per_process.append(proc_item_total)

        global_item_count = sum(items_per_process)

        # ==== Return complete batch ====
        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=global_item_count,
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=completions,  # For logging
            prompts=prompts,          # For logging
            errors=errors,            # For debugging
            metrics_dict=metrics_dict,
        )
