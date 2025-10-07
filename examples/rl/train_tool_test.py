import verifiers as vf

"""
# install
vf-install tool-test (-p /path/to/environments)

# quick eval
vf-eval tool-test (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B \
    --enforce-eager --disable-log-requests \
    --enable-auto-tool-choice --tool-call-parser hermes

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_tool_test.py
"""

vf_env = vf.load_environment(env_id="tool-test", num_eval_examples=100)

model_name = "willcb/Qwen3-0.6B"
run_name = "tool-test_" + model_name.split("/")[-1].lower()

model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args = vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size = 12
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 8
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
    peft_config=vf.lora_defaults(),
)
trainer.train()
