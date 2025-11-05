import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any

import httpx
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from verifiers import Environment
from verifiers.utils.processing_utils import (
    parse_chat_completion_logprobs,
    parse_chat_completion_tokens,
    parse_completion_logprobs,
    parse_completion_tokens,
)


class Microbatch(BaseModel):
    """Microbatch for batch generation"""

    input_ids: list[list[int]]
    loss_mask: list[list[int]]
    sampling_logprobs: list[list[float]]
    advantages: list[list[float]]
    items: int


class Batch(BaseModel):
    """Result from batch generation"""

    batch_id: int
    microbatches: list[list[Microbatch]]
    items_per_process: list[int]
    global_item_count: int
    # logging
    generation_time: float = 0.0
    prompts: list[Any] = Field(default_factory=list)
    completions: list[Any] = Field(default_factory=list)
    metrics_dict: dict[str, float] = Field(default_factory=dict)
    rewards_dict: dict[str, list[float]] = Field(default_factory=dict)


class Generator:
    """
    Manages asynchronous batch generation in parallel with RL training.
    """

    def __init__(
        self,
        env: Environment,
        client_base_url: str,
        client_api_key: str,
        client_limit: int,
        client_timeout: float,
        model_name: str,
        sampling_args: dict[str, Any],
        rollouts_per_example: int,
        batch_size: int,
        micro_batch_size: int,
        num_processes: int,
        generation_timeout: float,
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool,
        max_seq_len: int,
        max_prompt_len: int,
        mask_truncated_completions: bool,
        zero_truncated_completions: bool,
        max_concurrent: int,
        use_stepwise_advantage: bool,
        stepwise_gamma: float,
        stepwise_aggregation: str,
    ):
        self.env = env
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.client_limit = client_limit
        self.client_timeout = client_timeout
        self.client = None  # created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.rollouts_per_example = rollouts_per_example
        self.prompts_per_batch = batch_size // rollouts_per_example
        self.micro_batch_size = micro_batch_size
        self.num_processes = num_processes
        self.generation_timeout = generation_timeout
        self.processing_class = processing_class
        self.mask_env_responses = mask_env_responses
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.mask_truncated_completions = mask_truncated_completions
        self.zero_truncated_completions = zero_truncated_completions
        self.max_concurrent = max_concurrent
        self.use_stepwise_advantage = use_stepwise_advantage
        self.stepwise_gamma = float(stepwise_gamma)
        self.stepwise_aggregation = stepwise_aggregation

        # queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_generating = False
        self.completed_batches = {}

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.is_generating = False
        self.worker_loop = None

        max_length = self.max_prompt_len
        assert env.dataset is not None

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

        env.dataset = env.dataset.filter(
            filter_by_prompt_length,
            fn_kwargs={"processing_class": processing_class},
        )

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """Get dataset slice for a given batch id"""
        num_rows = self.prompts_per_batch
        dataset = self.env.get_dataset()
        total_rows = len(dataset)
        if total_rows == 0:
            raise ValueError("Environment dataset is empty")
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]
        return dataset.select(indices)

    def start(self):
        """Start the async generation worker thread"""
        self.worker_thread = threading.Thread(
            target=self.generation_worker, daemon=True, name="BatchGenerator"
        )
        self.worker_thread.start()

    def stop(self):
        """Stop the async generation worker thread"""
        self.stop_event.set()
        self.request_queue.put(None)  # poison pill
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def submit_batch(self, batch_id: int):
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: batch doesn't complete within timeout
            RuntimeError: generation failed
        """
        timeout = self.generation_timeout
        start_time = time.time()
        while True:
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)
            try:
                result = self.result_queue.get(timeout=0.1)
                self.completed_batches[result.batch_id] = result
                if result.batch_id == batch_id:
                    return self.completed_batches.pop(batch_id)
            except queue.Empty:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")

    def generation_worker(self):
        """Worker thread that processes generation requests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop
        self.client = AsyncOpenAI(
            base_url=self.client_base_url,
            api_key=self.client_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.client_limit),
                timeout=self.client_timeout,
            ),
        )
        try:
            while not self.stop_event.is_set():
                try:
                    batch_id = self.request_queue.get(timeout=0.1)
                    if batch_id is None:  # poison pill
                        break
                    result = loop.run_until_complete(self.generate_batch(batch_id))
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    async def generate_batch(self, batch_id: int) -> Batch:
        """
        Generate a single batch asynchronously.
        """
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()
        batch_ds = self.get_dataset_slice(batch_id)
        repeated_ds = batch_ds.repeat(self.rollouts_per_example)
        env_results = await self.env.a_generate(
            repeated_ds,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            score_rollouts=True,
            max_concurrent=self.max_concurrent,
            track_step_scores=self.use_stepwise_advantage,
        )
        self.is_generating = False
        wall_clock_s = time.time() - start_time

        if not self.use_stepwise_advantage:
            processed_results = self.env.process_env_results_vllm(
                prompts=env_results.prompt,
                completions=env_results.completion,
                states=env_results.state,
                rewards=env_results.reward,
                processing_class=self.processing_class,
                max_seq_len=self.max_seq_len,
                mask_env_responses=self.mask_env_responses,
                mask_truncated_completions=self.mask_truncated_completions,
                zero_truncated_completions=self.zero_truncated_completions,
            )

            rewards_dict = {"reward": processed_results.rewards}
            for k in env_results.metrics:
                rewards_dict[k] = env_results.metrics[k]

            rewards: list[float] = processed_results.rewards
            advantages: list[float] = [0.0] * len(rewards)
            prompts_in_batch = len(batch_ds)
            for prompt_idx in range(prompts_in_batch):
                group_indices = [
                    prompt_idx + k * prompts_in_batch
                    for k in range(self.rollouts_per_example)
                    if (prompt_idx + k * prompts_in_batch) < len(rewards)
                ]
                if not group_indices:
                    continue
                group = [rewards[i] for i in group_indices]
                gmean = sum(group) / float(len(group))
                for idx, r in zip(group_indices, group):
                    advantages[idx] = r - gmean
        else:
            # Expand each rollout into per-step training samples and compute
            # discounted MC returns per step.
            # Reference: https://arxiv.org/abs/2507.11948
            msg_type = getattr(self.env, "message_type", "chat")
            all_prompt_ids: list[list[int]] = []
            all_prompt_masks: list[list[int]] = []
            all_completion_ids: list[list[int]] = []
            all_completion_masks: list[list[int]] = []
            all_completion_logprobs: list[list[float]] = []
            all_returns: list[float] = []
            item_meta: list[tuple[int, int]] = []  # (prompt_idx, step_idx)

            # iterate in the same order as env_results arrays
            for i, (prompt, completion, state) in enumerate(
                zip(env_results.prompt, env_results.completion, env_results.state)
            ):
                # determine prompt index within this batch
                prompt_idx = i % len(batch_ds)
                # build per-step tokenization
                step_items: list[tuple[list[int], list[int], list[int], list[int], list[float]]]
                if msg_type == "chat":
                    assert isinstance(prompt, list) and isinstance(completion, list)
                    step_items = []
                    # Build context for tokenization similar to process_chat_format_vllm
                    responses = state["responses"]
                    responses_idx = 0
                    zipped_steps = []
                    for turn in completion:
                        if turn.get("role") == "assistant":
                            zipped_steps.append((turn, responses[responses_idx]))
                            responses_idx += 1
                        else:
                            zipped_steps.append((turn, None))
                    assert responses_idx == len(responses)

                    # utility to deserialize tool_calls for templates that expect JSON args
                    def _deserialize_tool_calls(message: dict) -> dict:
                        def _deserialize(tc) -> dict:
                            tc = dict(tc)
                            if (
                                "function" in tc
                                and isinstance(tc["function"], dict)
                                and "arguments" in tc["function"]
                            ):
                                args = tc["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except Exception:
                                        pass
                                tc["function"] = {**tc["function"], "arguments": args}
                            return tc

                        return {
                            **message,
                            "tool_calls": [
                                _deserialize(tc)
                                for tc in (message.get("tool_calls", []) or [])
                            ],
                        }
                    
                    def _maybe_strip_think(msg: dict) -> dict:
                        if getattr(self.env, "exclude_think", False) and hasattr(self.env, "_process_assistant_message"):
                            if msg.get("role") == "assistant":
                                return self.env._process_assistant_message(msg)
                        return msg

                    messages_consumed: list[dict] = [_maybe_strip_think(dict(m)) for m in prompt]
                    si = 0
                    j = 0
                    while j < len(zipped_steps):
                        message, response = zipped_steps[j]
                        message = _deserialize_tool_calls(message)
                        if message.get("role") == "assistant":
                            assert response is not None
                            prompt_text = self.processing_class.apply_chat_template(
                                conversation=messages_consumed,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            prompt_ids = self.processing_class.encode(prompt_text)
                            prompt_mask = [0] * len(prompt_ids)
                            completion_ids = parse_chat_completion_tokens(response)
                            completion_mask = [1] * len(completion_ids)
                            completion_logprobs = parse_chat_completion_logprobs(response)
                            step_items.append(
                                (
                                    prompt_ids,
                                    prompt_mask,
                                    completion_ids,
                                    completion_mask,
                                    completion_logprobs,
                                )
                            )
                            messages_consumed.append(_maybe_strip_think(message))
                            si += 1
                            j += 1
                        else:
                            messages_consumed.append(_maybe_strip_think(message))
                            j += 1
                else:
                    assert isinstance(prompt, str) and isinstance(completion, str)
                    responses = state.get("responses", [])
                    starts = state.get("responses_start_idx", [])
                    assert len(responses) == len(starts)
                    step_items = []
                    for ridx in range(len(responses)):
                        start_i = starts[ridx]
                        context_prefix = prompt + completion[:start_i]
                        prompt_ids = self.processing_class.encode(context_prefix)
                        prompt_mask = [0] * len(prompt_ids)
                        resp = responses[ridx]
                        completion_ids = parse_completion_tokens(resp)
                        completion_mask = [1] * len(completion_ids)
                        completion_logprobs = parse_completion_logprobs(resp)
                        step_items.append(
                            (
                                prompt_ids,
                                prompt_mask,
                                completion_ids,
                                completion_mask,
                                completion_logprobs,
                            )
                        )

                # compute immediate rewards per step and MC returns
                if msg_type == "chat":
                    assert isinstance(prompt, list) and isinstance(completion, list)
                    # Prefer precomputed per-turn scores from rollout if available
                    step_rewards: list[float] = []
                    pre_scores = state.get("step_scores", None)
                    if isinstance(pre_scores, list) and pre_scores:
                        step_rewards = [float(x) for x in pre_scores]
                    else:
                        raise RuntimeError("Per-turn scores missing in state for stepwise advantage computation")

                    returns: list[float] = self._compute_stepwise_returns(step_rewards)

                else:
                    assert isinstance(prompt, str) and isinstance(completion, str)
                    responses = state.get("responses", [])
                    starts = state.get("responses_start_idx", [])
                    step_rewards = []
                    assert len(responses) == len(starts)
                    pre_scores = state.get("step_scores", None)
                    if isinstance(pre_scores, list) and pre_scores:
                        step_rewards = [float(x) for x in pre_scores]
                        step_rewards = step_rewards[: len(starts)]
                    else:
                        for ridx, start in enumerate(starts):
                            # Include env feedback after this assistant response
                            end_i = starts[ridx + 1] if (ridx + 1) < len(starts) else len(completion)
                            partial_text = completion[:end_i]
                            rs = await self.env.rubric.score_rollout(
                                prompt=prompt,
                                completion=partial_text,
                                answer=state.get("answer", ""),
                                state=state,
                                task=state.get("task", "default"),
                                info=state.get("info", {}),
                                example_id=state.get("example_id", i),
                            )
                            step_rewards.append(float(rs.reward))

                    returns: list[float] = self._compute_stepwise_returns(step_rewards)

                for step_idx, item in enumerate(step_items):
                    p_ids, p_mask, c_ids, c_mask, c_logps = item
                    completion_truncated = False
                    if self.max_seq_len > 0:
                        max_c_possible = min(len(c_ids), self.max_seq_len)
                        keep_p = self.max_seq_len - max_c_possible
                        if keep_p < len(p_ids):
                            if keep_p <= 0:
                                p_ids, p_mask = [], []
                            else:
                                p_ids = p_ids[-keep_p:]
                                p_mask = p_mask[-keep_p:]

                        max_c_len = self.max_seq_len - len(p_ids)
                        if len(c_ids) > max_c_len:
                            completion_truncated = True
                            c_ids   = c_ids[:max_c_len] if max_c_len > 0 else []
                            c_mask  = c_mask[:max_c_len] if max_c_len > 0 else []
                            c_logps = c_logps[:max_c_len] if max_c_len > 0 else []

                    effective_c_mask = c_mask
                    if completion_truncated and self.mask_truncated_completions:
                        effective_c_mask = [0] * len(c_ids)

                    ret = float(returns[step_idx])
                    if completion_truncated and self.zero_truncated_completions:
                        ret = 0.0

                    all_prompt_ids.append(p_ids)
                    all_prompt_masks.append(p_mask)
                    all_completion_ids.append(c_ids)
                    all_completion_masks.append(effective_c_mask)
                    all_completion_logprobs.append(c_logps)
                    all_returns.append(ret)
                    item_meta.append((prompt_idx, step_idx))


            class _Proc:
                def __init__(self):
                    self.prompt_ids = all_prompt_ids
                    self.prompt_mask = all_prompt_masks
                    self.completion_ids = all_completion_ids
                    self.completion_mask = all_completion_masks
                    self.completion_logprobs = all_completion_logprobs
                    self.rewards = all_returns

            # mimic ProcessedOutputs for downstream use
            processed_results = _Proc()
            rewards_dict = {"reward": all_returns}

            rewards = all_returns
            # Compute stepwise group baseline across all m×n samples for each prompt
            advantages: list[float] = [0.0] * len(rewards)
            prompts_in_batch = len(batch_ds)
            for prompt_idx in range(prompts_in_batch):
                group = [j for j, (p, _s) in enumerate(item_meta) if p == prompt_idx]
                if not group:
                    continue
                group_vals = [rewards[j] for j in group]
                gmean = float(np.mean(group_vals))
                gstd = float(np.std(group_vals)) + 1e-8  # prevent div by zero
                for j in group:
                    advantages[j] = (rewards[j] - gmean) / gstd

        metrics_dict = {}
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if self.use_stepwise_advantage:
            metrics_dict["stepwise/turns_per_rollout"] = float(np.mean([len(s.get("step_scores", [])) for s in env_results.state]))
            metrics_dict["stepwise/rollout_length"] = float(np.mean([len(s.get("responses", [])) for s in env_results.state]))

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        for reward_name, values in env_results.metrics.items():
            if len(values) == 0:
                continue
            reward_values = np.asarray(values, dtype=np.float32)
            metrics_dict[f"reward/{reward_name}"] = float(reward_values.mean())

        completion_lengths = [len(ids) for ids in processed_results.completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

            completion_mask_lengths = np.asarray(
                [sum(mask) for mask in processed_results.completion_mask],
                dtype=np.float32,
            )
            valid_tokens = completion_mask_lengths.sum()
            total_tokens = completion_lengths_arr.sum()
            if total_tokens > 0:
                masked_fraction = 1.0 - (valid_tokens / total_tokens)
                metrics_dict["tokens/masked_fraction"] = float(masked_fraction)

        generation_ms: list[float] = []
        scoring_ms: list[float] = []
        total_ms: list[float] = []
        for state in env_results.state:
            timing = state.get("timing", {})
            if "generation_ms" in timing:
                generation_ms.append(float(timing["generation_ms"]))
            if "scoring_ms" in timing:
                scoring_ms.append(float(timing["scoring_ms"]))
            if "total_ms" in timing:
                total_ms.append(float(timing["total_ms"]))

        if generation_ms:
            metrics_dict["timing/generation_ms"] = float(np.mean(generation_ms))
        if scoring_ms:
            metrics_dict["timing/scoring_ms"] = float(np.mean(scoring_ms))
        if total_ms:
            metrics_dict["timing/total_ms"] = float(np.mean(total_ms))

        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)

        # build per-process microbatches
        N = len(processed_results.rewards)
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []
        if not self.use_stepwise_advantage:
            per_proc = N // self.num_processes
            for proc in range(self.num_processes):
                ps = proc * per_proc
                pe = ps + per_proc
                proc_mbs: list[Microbatch] = []
                proc_item_total = 0
                for s in range(ps, pe, self.micro_batch_size):
                    e = min(s + self.micro_batch_size, pe)
                    ids_chunk = [
                        processed_results.prompt_ids[i]
                        + processed_results.completion_ids[i]
                        for i in range(s, e)
                    ]
                    mask_chunk = [
                        processed_results.prompt_mask[i]
                        + processed_results.completion_mask[i]
                        for i in range(s, e)
                    ]
                    slogp_chunk = [
                        [0.0] * len(processed_results.prompt_mask[i])
                        + processed_results.completion_logprobs[i]
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
                        sampling_logprobs=slogp_chunk,
                        advantages=adv_chunk,
                        items=mb_items,
                    )
                    proc_item_total += mb_items
                    proc_mbs.append(microbatch)
                microbatches.append(proc_mbs)
                items_per_process.append(proc_item_total)
        else:
            for proc in range(self.num_processes):
                indices = list(range(proc, N, self.num_processes))
                proc_mbs: list[Microbatch] = []
                proc_item_total = 0
                for start in range(0, len(indices), self.micro_batch_size):
                    idxs = indices[start : start + self.micro_batch_size]
                    ids_chunk = [
                        processed_results.prompt_ids[i]
                        + processed_results.completion_ids[i]
                        for i in idxs
                    ]
                    mask_chunk = [
                        processed_results.prompt_mask[i]
                        + processed_results.completion_mask[i]
                        for i in idxs
                    ]
                    slogp_chunk = [
                        [0.0] * len(processed_results.prompt_mask[i])
                        + processed_results.completion_logprobs[i]
                        for i in idxs
                    ]
                    lengths = [len(mask) for mask in mask_chunk]
                    adv_chunk = [[advantages[i]] * lengths[k] for k, i in enumerate(idxs)]
                    mb_items = sum(sum(mask) for mask in mask_chunk)
                    microbatch = Microbatch(
                        input_ids=ids_chunk,
                        loss_mask=mask_chunk,
                        sampling_logprobs=slogp_chunk,
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
            completions=env_results.completion,
            prompts=env_results.prompt,
            metrics_dict=metrics_dict,
        )
    
    def _compute_stepwise_returns(self, step_rewards: list[float]) -> list[float]:
        if not step_rewards:
            return []

        g = float(self.stepwise_gamma)
        if self.stepwise_aggregation == "sum":
            # R_t=\sum_{i=t}^{T}{\gamma^{i-t} r_i}
            G = 0.0
            out = [0.0] * len(step_rewards)
            for t in range(len(step_rewards) - 1, -1, -1):
                G = float(step_rewards[t]) + g * G
                out[t] = G
            return out
    
        elif self.stepwise_aggregation == "max":
            # R_t=\max_{i=t..T}{\gamma^{i-t} r_i}
            out = [0.0] * len(step_rewards)
            R_next: float | None = None
            for t in range(len(step_rewards) - 1, -1, -1):
                r = float(step_rewards[t])
                cand_future = (g * R_next) if (R_next is not None) else None
                R_t = r if cand_future is None else max(r, cand_future)
                out[t] = R_t
                R_next = R_t
            return out
