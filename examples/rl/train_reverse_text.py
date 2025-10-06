import verifiers as vf

"""
# install
vf-install reverse-text (-p /path/to/environments)

# quick eval
vf-eval reverse-text (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT \
    --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/rl/train_reverse_text.py
"""

model_name = "willcb/Qwen2.5-0.5B-Reverse-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
vf_env = vf.load_environment(env_id="reverse-text")
trainer = vf.RLTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.RLConfig(run_name="reverse-text", batch_size=128, micro_batch_size=4),
)
trainer.train()
