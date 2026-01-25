"""
MultiAgentOrchestrator: Thin wrapper on Orchestrator that uses Protocol for generation.

Delegates batch generation to Protocol.generate() which handles multi-environment
orchestration and child state flattening.

Dataset is owned by Protocol, not environments.
"""
import time
from typing import Any

import numpy as np
from datasets import Dataset

from verifiers.envs.protocol import Protocol

from .orchestrator import Batch, Microbatch, Orchestrator


class MultiAgentOrchestrator(Orchestrator):
    """
    Thin wrapper on Orchestrator that delegates to Protocol for generation.

    Instead of a single Environment, takes a Protocol which holds multiple
    environments and actors. generate_batch() calls protocol.generate()
    which handles scheduling across environments and flattening child states.

    Dataset is owned by Protocol, not environments.
    """

    def __init__(
        self,
        protocol: Protocol,
        **kwargs,
    ):
        self.protocol = protocol

        # Pick first env as default for parent Orchestrator
        first_env = next(iter(protocol.envs.values()))
        super().__init__(env=first_env, **kwargs)

        # Override: filter Protocol's dataset instead of environment's
        # (parent __init__ filters env.dataset, we need to filter protocol's)
        max_length = self.max_prompt_len

        def filter_by_prompt_length(example, processing_class):
            prompt = example["prompt"]
            if isinstance(prompt, list):
                prompt_text = processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            prompt_ids = processing_class.encode(prompt_text)
            return len(prompt_ids) <= max_length

        # Filter Protocol's dataset
        if self.protocol._dataset is not None:
            self.protocol._dataset = self.protocol.get_dataset().filter(
                filter_by_prompt_length,
                fn_kwargs={"processing_class": self.processing_class},
            )

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """Get dataset slice from Protocol's dataset for a given batch id."""
        num_rows = self.prompts_per_batch
        dataset = self.protocol.get_dataset()
        total_rows = len(dataset)
        if total_rows == 0:
            raise ValueError("Protocol dataset is empty")
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]
        return dataset.select(indices)

    async def generate_batch(self, batch_id: int) -> Batch:
        """
        Generate batch by delegating to protocol.generate().

        Flow:
        1. MultiAgentOrchestrator.generate_batch() calls protocol.generate()
        2. protocol.generate() calls env.generate() for each environment
        3. env.generate() runs rollouts, which may call protocol.spawn() for children
        4. protocol.generate() flattens all states (including child_states)
        """
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()

        # Get dataset slice and repeat for rollouts
        batch_ds = self.get_dataset_slice(batch_id)
        repeated_ds = batch_ds.repeat(self.rollouts_per_example)
        inputs = repeated_ds.to_list()

        # Use protocol.generate() instead of env.generate()
        # This returns flattened list of all trainable states
        all_states = await self.protocol.generate(
            inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            max_concurrent=self.max_concurrent,
        )

        self.is_generating = False
        wall_clock_s = time.time() - start_time

        # Process trajectories - each step becomes a separate training example
        prompt_ids: list[list[int]] = []
        prompt_mask: list[list[int]] = []
        completion_ids: list[list[int]] = []
        completion_mask: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        advantages: list[float] = []

        for state in all_states:
            trajectory = state.get("trajectory", [])
            for step in trajectory:
                tokens = step.get("tokens")
                if tokens is None:
                    continue
                prompt_ids.append(tokens["prompt_ids"])
                prompt_mask.append(tokens["prompt_mask"])
                completion_ids.append(tokens["completion_ids"])
                completion_mask.append(tokens["completion_mask"])
                completion_logprobs.append(tokens["completion_logprobs"])
                advantages.append(step.get("advantage", 0.0))

        # Build rewards_dict from rollout-level data (for logging only)
        rewards = [state.get("reward", 0.0) for state in all_states]
        rewards_dict: dict[str, list[float]] = {"reward": rewards}

        # Collect metrics
        metrics_dict: dict[str, float] = {}
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        completion_lengths = [len(ids) for ids in completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

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

        # Collect errors and completions for logging
        errors = [state.get("error") for state in all_states]
        completions = [state.get("completion") for state in all_states]
        prompts = [state.get("prompt") for state in all_states]

        # Build per-process microbatches
        N = len(advantages)
        per_proc = N // self.num_processes if self.num_processes > 0 else N
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []

        for proc in range(self.num_processes):
            ps = proc * per_proc
            pe = ps + per_proc
            proc_mbs: list[Microbatch] = []
            proc_item_total = 0
            for s in range(ps, pe, self.micro_batch_size):
                e = min(s + self.micro_batch_size, pe)
                ids_chunk = [prompt_ids[i] + completion_ids[i] for i in range(s, e)]
                mask_chunk = [prompt_mask[i] + completion_mask[i] for i in range(s, e)]
                logprobs_chunk = [
                    [0.0] * len(prompt_mask[i]) + completion_logprobs[i]
                    for i in range(s, e)
                ]
                lengths = [len(mask) for mask in mask_chunk]
                adv_chunk = [
                    [advantages[i]] * lengths[idx]
                    for idx, i in enumerate(list(range(s, e)))
                ]
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

        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=global_item_count,
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=completions,
            prompts=prompts,
            errors=errors,
            metrics_dict=metrics_dict,
        )
