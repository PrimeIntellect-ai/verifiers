import verifiers as vf

"""
# install
vf-install gsm8k (-p /path/to/environments)

# quick eval
vf-eval gsm8k (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/rl/train_gsm8k.py
"""

vf_env = vf.load_environment(env_id="gsm8k", num_eval_examples=100)

model_name = "willcb/Qwen3-0.6B"
run_name = "gsm8k_" + model_name.split("/")[-1].lower()
model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.RLTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.RLConfig(run_name=run_name),
)
trainer.train()
