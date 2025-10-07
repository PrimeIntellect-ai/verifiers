import verifiers as vf

"""
# install
vf-install wiki-search (-p /path/to/environments)

# quick eval
vf-eval wiki-search (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-8B-Wiki-Search-SFT \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wiki_search.py
"""

vf_env = vf.load_environment(env_id="wiki-search")

model_name = "willcb/Qwen3-8B-Wiki-Search-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "wiki-trivia-grpo_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 16
training_args.num_iterations = 1
training_args.num_train_epochs = 5
training_args.max_seq_len = 4096
training_args.max_steps = 500
training_args.save_steps = 100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
