import verifiers as vf

"""
# install
vf-install math-python (-p /path/to/environments)

# eval
vf-eval math-python (-m model_name in endpoints.py)

# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model 'willcb/Qwen3-1.7B' \
    --data-parallel-size 6 --enforce-eager --disable-log-requests \
    --enable-auto-tool-choice --tool-call-parser hermes

# training
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_math_python.py
"""

vf_env = vf.load_environment(env_id="math-python")

model_name = "willcb/Qwen3-1.7B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-python_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 8
training_args.max_tokens = 2048
training_args.max_seq_len = 4096
training_args.max_steps = 200
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.1

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    # lora_config=vf.lora_defaults()
)
trainer.train()
