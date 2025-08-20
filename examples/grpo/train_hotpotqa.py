import verifiers as vf

"""
# install
vf-install vf-gsm8k (-p /path/to/environments)

# quick eval
vf-eval vf-gsm8k (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model willcb/Qwen3-0.6B --data-parallel-size 2 --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=2 uv run examples/grpo/train_hotpotqa.py
"""

# uses self-verification to judge the answer
vf_env = vf.load_environment(
    env_id="vf-hotpotqa",
    judge_model="willcb/Qwen3-0.6B",
    judge_base_url="http://0.0.0.0:8000/v1",
    judge_api_key_var="EMPTY",
    num_eval_examples=100,
    use_think=True,
)

model_name = "willcb/Qwen3-0.6B"
run_name = "hotpotqa-grpo_" + model_name.split("/")[-1].lower()

model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args = vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size = 12
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 12
training_args.max_tokens = 2048
training_args.max_seq_len = 2048
training_args.eval_strategy = "steps"
training_args.eval_steps = 10
training_args.save_strategy = "steps"
training_args.save_steps = 100
training_args.max_steps = 200
training_args.eval_strategy = "steps"
training_args.eval_steps = 10

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    lora_config=vf.lora_defaults(r=16, alpha=64),
)
trainer.train()
