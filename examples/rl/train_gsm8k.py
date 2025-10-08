import verifiers as vf

"""
# install
vf-install gsm8k (-p /path/to/environments)

# quick eval
vf-eval gsm8k (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py
"""

vf_env = vf.load_environment(env_id="gsm8k", num_eval_examples=100)

trainer = vf.RLTrainer(
    model="willcb/Qwen3-0.6B",
    env=vf_env,
    args=vf.RLConfig(
        run_name="gsm8k",
        batch_size=256,
        rollouts_per_example=16,
        max_seq_len=2048,
    ),
)
trainer.train()
