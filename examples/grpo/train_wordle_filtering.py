import argparse

import verifiers as vf
from verifiers import GRPOConfig

"""
# install
vf-install wordle (-p /path/to/environments)

# quick eval
vf-eval wordle -m (model_name in endpoints.py)

1.7b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

1.7b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle_filtering.py --size 1.7B

4b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-4B \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

4b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle_filtering.py --size 4B
"""


def main(args):
    size = args.size
    model_name = f"willcb/Qwen3-{size}-Wordle"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="wordle", use_think=True)
    run_name = f"wordle-grpo-{size}"

    training_args = GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=200,
        bf16=True,
        max_grad_norm=0.1,
        num_iterations=1,
        max_seq_len=4096,
        max_tokens=1024,
        per_device_train_batch_size=8,
        num_generations=8,
        rollout_filter_ratio=0.5,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        mask_env_responses=True,
        beta=0.0,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
    )

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        # lora_config=vf.lora_defaults()
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="1.7B")
    args = parser.parse_args()
    main(args)
