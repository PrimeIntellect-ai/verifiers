import verifiers as vf

"""
# install
vf-install reverse-text (-p /path/to/environments)

# quick eval
vf-eval reverse-text (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT \
    --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_reverse_text.py
"""

model_name = "willcb/Qwen2.5-0.5B-Reverse-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.load_environment(env_id="reverse-text")

args = vf.grpo_defaults(run_name="reverse-text")
args.per_device_train_batch_size = 12
args.num_generations = 12
args.gradient_accumulation_steps = 8
args.max_steps = 100
args.eval_strategy = "steps"
args.eval_steps = 2
args.max_tokens = 1024

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(),
    args=args,
)
trainer.train()
