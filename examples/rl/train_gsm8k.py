import verifiers as vf

"""
# install
vf-install gsm8k (-p /path/to/environments)

# quick eval
vf-eval gsm8k (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model willcb/Qwen3-0.6B --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/rl/train_gsm8k.py
"""

env_id = "gsm8k"
model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
vf_env = vf.load_environment(env_id=env_id, num_eval_examples=100)
trainer = vf.RLTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.RLConfig(run_name=env_id),
)
trainer.train()
