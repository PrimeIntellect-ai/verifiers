import verifiers as vf

"""
# install
vf-install math-group (-p /path/to/environments)

# quick eval
vf-eval math-group (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B \
    --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_math_group.py
"""

vf_env = vf.load_environment(env_id="math-group")

model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 16
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 8
training_args.num_iterations = 1
training_args.max_prompt_length = 512
training_args.max_completion_length = 2048
training_args.max_steps = 100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
