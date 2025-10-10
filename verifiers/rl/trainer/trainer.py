import logging
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import deepspeed
import torch
from accelerate.utils import (
    broadcast_object_list,
    is_peft_model,
)
from accelerate.utils.memory import clear_device_cache
from peft import PeftConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

import verifiers as vf
import wandb
from verifiers.rl.inference.vllm_client import VLLMClient
from verifiers.rl.trainer.config import RLConfig
from verifiers.rl.trainer.generator import Generator
from verifiers.rl.trainer.utils import pad, prepare_peft_model, selective_log_softmax
from verifiers.types import Messages
from verifiers.utils.logging_utils import print_prompt_completions_sample


class RLTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | str,
        env: vf.Environment,
        args: RLConfig,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        # model + tokenizer
        if isinstance(model, str):
            model, processing_class = vf.get_model_and_tokenizer(model)
        assert isinstance(model, PreTrainedModel)
        if args.use_lora and isinstance(args.lora_config, PeftConfig):
            model = prepare_peft_model(model, args.lora_config, args)
        model.warnings_issued["estimate_tokens"] = True  # suppress warning
        self.model_name = model.config._name_or_path
        assert isinstance(processing_class, PreTrainedTokenizerBase)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token
        if processing_class.pad_token_id is None:
            processing_class.pad_token_id = processing_class.eos_token_id

        # batch args
        self.max_steps = args.max_steps
        self.rollouts_per_example = args.rollouts_per_example
        self.batch_size = args.batch_size
        self.micro_batch_size = args.micro_batch_size
        self.inner_steps = args.inner_steps
        self.max_seq_len = args.max_seq_len
        self.max_prompt_length = args.max_prompt_len or self.max_seq_len
        self.max_concurrent = args.max_concurrent
        self.sampling_args = args.sampling_args
        self.temperature = args.temperature

        # loss args
        self.epsilon = args.epsilon
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.zero_truncated_completions = args.zero_truncated_completions
        self.importance_sampling_level = args.importance_sampling_level
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap

        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # OpenAI client for Environment generation (using vLLM server)
        host = args.vllm_server_host
        port = args.vllm_server_port
        vllm_base_url = f"http://{host}:{port}/v1"
        self.client_config = {
            "base_url": vllm_base_url,
            "api_key": "EMPTY",
            "limit": args.max_concurrent,
            "timeout": args.async_generation_timeout,
        }

        # environment
        self.env = env
        self.mask_env_responses = args.mask_env_responses
        self.max_concurrent = args.max_concurrent

        # generator (main process only)
        if self.accelerator.is_main_process:
            self.client = VLLMClient(
                host=host, port=port, connection_timeout=args.vllm_server_timeout
            )
            self.client.init_communicator()
            assert isinstance(self.processing_class, PreTrainedTokenizerBase)
            self.generator = Generator(
                env=self.env,
                client_config=self.client_config,
                model_name=self.model_name,
                sampling_args=dict(self.sampling_args),
                rollouts_per_example=self.rollouts_per_example,
                batch_size=self.batch_size,
                micro_batch_size=self.micro_batch_size,
                num_processes=self.accelerator.num_processes,
                generation_timeout=args.async_generation_timeout,
                processing_class=self.processing_class,
                mask_env_responses=self.mask_env_responses,
                max_seq_len=self.max_seq_len or -1,
                max_prompt_len=args.max_prompt_len or self.max_seq_len,
                mask_truncated_completions=self.mask_truncated_completions,
                zero_truncated_completions=self.zero_truncated_completions,
                max_concurrent=self.max_concurrent,
                scale_rewards=self.scale_rewards,
            )
            self.generator.start()
            self.generator.submit_batch(0)
        else:
            self.generator = None
            self.client = None

        # metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._textual_logs = {
            "prompt": deque(),
            "completion": deque(),
            "rewards": defaultdict(lambda: deque()),
        }

    def training_step(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.update_vllm()
        if self.generator:
            self.generator.submit_batch(self.state.global_step + 1)

        broadcast_list = [None]
        if self.generator:
            broadcast_list = [self.generator.get_batch(self.state.global_step)]
        broadcast_object_list(broadcast_list)
        assert broadcast_list[0] is not None
        batch = broadcast_list[0]

        model.train()
        total_loss = torch.zeros((), device=self.accelerator.device)
        for microbatch in batch.microbatches[self.accelerator.process_index]:
            mb_input_ids = microbatch.input_ids
            mb_attention_mask = microbatch.attention_mask
            mb_sampling_logprobs = microbatch.sampling_logprobs
            mb_advantages = microbatch.advantages

            input_ids = pad(
                [torch.tensor(x, device=self.accelerator.device) for x in mb_input_ids],
                padding_value=self.processing_class.pad_token_id,  # type: ignore
                padding_side="right",
            )
            attention_mask = pad(
                [
                    torch.tensor(x, device=self.accelerator.device)
                    for x in mb_attention_mask
                ],
                padding_side="right",
            )
            sampling_logprobs = pad(
                [
                    torch.tensor(x, device=self.accelerator.device)
                    for x in mb_sampling_logprobs
                ],
                padding_value=0,
                padding_side="right",
            )

            if self.max_seq_len is not None and input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len :]
                attention_mask = attention_mask[:, -self.max_seq_len :]

            with torch.no_grad():
                completion_mask = attention_mask[:, 1:]
                logits_to_keep = completion_mask.size(1)
                logits_to_keep = min(logits_to_keep, sampling_logprobs.size(1))
                sampling_logprobs = sampling_logprobs[:, :logits_to_keep]
                model_logprobs = self.get_logprobs(
                    self.model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=self.micro_batch_size,
                )
                completion_mask = completion_mask[:, -logits_to_keep:]
                ratio = torch.exp(model_logprobs - sampling_logprobs)
                ratio = torch.clamp(ratio, max=self.vllm_importance_sampling_cap)
                ratio = torch.where(
                    completion_mask.bool(),
                    ratio,
                    torch.ones_like(ratio),
                )

            mb_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "model_logprobs": model_logprobs,
                "advantages": torch.tensor(
                    mb_advantages, device=self.accelerator.device
                ),
                "importance_sampling_ratio": ratio,
            }
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, mb_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            total_loss = total_loss + loss.detach()

        # Logging on main process using full-batch CPU metadata
        if self.accelerator.is_main_process:
            self.log_metrics(
                mode="train",
                batch_metrics=batch.metrics_dict,
            )
            self.log_rollouts(
                prompts=batch.prompts,
                completions=batch.completions,
                rewards_dict=batch.rewards_dict,
            )

        self.maybe_clear_cache()
        return total_loss

    def get_logprobs(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # chunking for memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def update_vllm(self):
        assert self.model is not None
        is_generating = False
        if self.generator:
            is_generating = self.generator.is_generating
        is_generating_list = [is_generating]
        broadcast_object_list(is_generating_list, from_process=0)
        is_generating = is_generating_list[0]

        waits = 0
        while is_generating:
            time.sleep(0.5)
            waits += 1
            if waits % 10 == 0:
                self.logger.info("Waiting for generation to finish before syncing.")
            if self.generator:
                is_generating = self.generator.is_generating
            is_generating_list = [is_generating]
            broadcast_object_list(is_generating_list, from_process=0)
            is_generating = is_generating_list[0]

        if self.state.global_step > 0:  # skip first step
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            zero_stage_3 = (
                deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
            )
            if zero_stage_3:
                gather_if_zero3 = deepspeed.zero.GatheredParameters
            else:
                gather_if_zero3 = nullcontext
            self.accelerator.wait_for_everyone()
            self.logger.info("Starting weight sync to vLLM")

            if is_peft_model(self.model):
                # PEFT: gather + merge, then update each parameter
                with gather_if_zero3(list(self.model.parameters())):
                    self.model.merge_adapter()  # type: ignore
                    for name, param in self.model.named_parameters():
                        # recover original parameter names
                        name = name.removeprefix("base_model.model.").replace(
                            ".base_layer", ""
                        )
                        if self.model.prefix in name:  # type: ignore
                            continue  # discard some parameters
                        if "original_module" in name:  # from modules_to_save
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.client:
                            self.client.update_named_param(name, param.data)
                    self.model.unmerge_adapter()  # type: ignore
            else:
                # non-PEFT models: gather + update each parameter individually
                for name, param in self.model.named_parameters():  # type: ignore
                    with gather_if_zero3([param]):
                        if self.client:
                            self.client.update_named_param(name, param.data)

            # reset cache + wait for background tasks to complete
            if self.client:
                self.client.reset_prefix_cache()
                while self.client.get_num_background_tasks() > 0:
                    time.sleep(0.5)
                    self.logger.info("Resetting prefix cache.")

        self.accelerator.wait_for_everyone()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        completion_mask = attention_mask[:, 1:]  # prompt is at least 1 token
        logits_to_keep = completion_mask.size(1)
        ratio = inputs["importance_sampling_ratio"]
        logits_to_keep = min(logits_to_keep, ratio.size(1))
        completion_mask = completion_mask[:, -logits_to_keep:]
        per_token_logps = self.get_logprobs(
            model, input_ids, attention_mask, logits_to_keep
        )
        advantages = inputs["advantages"]
        log_ratio = per_token_logps - inputs["old_per_token_logps"]
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                (log_ratio * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}"
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1,
            min=1 - self.epsilon,
            max=1 + self.epsilon,
        )
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * ratio

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_seq_len
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss

    def get_train_dataloader(self):
        class StepsDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {"labels": 0}

        return DataLoader(StepsDataset(self.max_steps))

    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # cleanup
            if self.generator:
                self.generator.stop()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        logs = {**logs, **metrics}
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process:
            print_prompt_completions_sample(
                self._textual_logs["prompt"],
                self._textual_logs["completion"],
                self._textual_logs["rewards"]["reward"],
                self.state.global_step,
            )

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)]
                    * len(self._textual_logs["prompt"]),
                    "prompt": list(self._textual_logs["prompt"]),
                    "completion": list(self._textual_logs["completion"]),
                    **{k: list(v) for k, v in self._textual_logs["rewards"].items()},
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

            # clear after logging
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

    def log_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        rewards_dict: Dict[str, Any],
    ) -> None:
        self._textual_logs["prompt"].extend(prompts)
        self._textual_logs["completion"].extend(completions)
        for reward_key in rewards_dict:
            reward_values = rewards_dict[reward_key]
            self._textual_logs["rewards"][reward_key].extend(reward_values)

    def log_metrics(
        self,
        mode: str,
        batch_metrics: Dict[str, float],
    ) -> None:
        for key, value in batch_metrics.items():
            self._metrics[mode][key].append(value)

    def maybe_clear_cache(self):
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            clear_device_cache()
