import argparse

import verifiers as vf

"""
# install
vf-install wordle (-p /path/to/environments)

# quick eval
vf-eval wordle -m (model_name in endpoints.py)

1.7b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run vf-vllm --model willcb/Qwen3-1.7B-Wordle \
    --tensor-parallel-size 6 --enforce-eager

1.7b training:
CUDA_VISIBLE_DEVICES=6,7 uv run accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/rl/train_wordle.py --size 1.7B

4b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run vf-vllm --model willcb/Qwen3-4B-Wordle \
    --data-parallel-size 6 --enforce-eager

4b training:
CUDA_VISIBLE_DEVICES=6,7 uv run accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/rl/train_wordle.py --size 4B
"""


def main(args):
    size = args.size
    model_name = f"willcb/Qwen3-{size}-Wordle"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="wordle", use_think=True)
    run_name = f"wordle-{size}"

    trainer = vf.RLTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=vf.RLConfig(run_name=run_name),
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="1.7B")
    args = parser.parse_args()
    main(args)
