import verifiers as vf

"""
# install
vf-install wordle (-p /path/to/environments)

# quick eval
vf-eval wordle -m (model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run vf-vllm --model willcb/Qwen3-1.7B-Wordle \
    --data-parallel-size 6 --enforce-eager

training:
CUDA_VISIBLE_DEVICES=6,7 uv run accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/rl/train_wordle.py
"""

trainer = vf.RLTrainer(
    model="willcb/Qwen3-1.7B-Wordle",
    env=vf.load_environment(env_id="wordle", use_think=True),
    args=vf.RLConfig(
        run_name="wordle",
        batch_size=512,
        rollouts_per_example=8,
        max_seq_len=4096,
        max_tokens=1024,
    ),
)
trainer.train()
