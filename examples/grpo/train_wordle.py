import argparse

import verifiers as vf

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
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B

4b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-4B-Wordle \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

4b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 4B
"""


def main(args):
    size = args.size
    model_name = f"willcb/Qwen3-{size}-Wordle"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="wordle", use_think=True)
    run_name = f"wordle-grpo-{size}"
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.per_device_train_batch_size = 8
    training_args.num_generations = 16
    training_args.gradient_accumulation_steps = 8
    training_args.max_tokens = 1024  # per turn
    training_args.max_seq_len = 4096
    training_args.max_steps = 200
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 20
    training_args.mask_env_responses = True
    training_args.max_grad_norm = 0.1
    training_args.beta = 0.0

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
