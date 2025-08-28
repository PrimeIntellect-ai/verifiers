import verifiers as vf

"""
# install
vf-install self-reward (-p /path/to/environments)

# quick eval
vf-eval self-reward (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_self_reward.py
"""

model_name = "Qwen/Qwen2.5-7B-Instruct"
vf_env = vf.load_environment(env_id="self-reward", model_name=model_name)
model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=vf.grpo_defaults(run_name="self-reward"),
)
trainer.train()
