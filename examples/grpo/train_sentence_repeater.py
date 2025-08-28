import verifiers as vf

"""
# install
vf-install sentence-repeater (-p /path/to/environments)

# quick eval
vf-eval sentence-repeater (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-1.5B-Instruct \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_sentence_repeater.py
"""

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.load_environment(env_id="sentence-repeater")

run_name = "sentence-repeater-grpo-qwen1.5b"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 8
training_args.max_tokens = 1024  # per turn
training_args.max_seq_len = 4096
training_args.max_steps = 200
training_args.eval_strategy = "steps"
training_args.eval_steps = 20
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.0
training_args.async_generation_timeout = 600

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
