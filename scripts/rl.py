import verifiers as vf

"""
# install
vf-install reverse-text (-p /path/to/environments)

# quick eval
vf-eval reverse-text (-m model_name in endpoints.py)

1-GPU inference:
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT \
    --enforce-eager

1-GPU training:
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/rl/train_reverse_text.py

2-GPU inference:
CUDA_VISIBLE_DEVICES=0,1 uv run vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT \
    --data-parallel-size 2 --enforce-eager

2-GPU training:
CUDA_VISIBLE_DEVICES=2,3 uv run accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/rl/train_reverse_text.py
"""

trainer = vf.RLTrainer(
    model="willcb/Qwen2.5-0.5B-Reverse-SFT",
    env=vf.load_environment(env_id="reverse-text"),
    args=vf.RLConfig(run_name="reverse-text", batch_size=256),
)
trainer.train()
