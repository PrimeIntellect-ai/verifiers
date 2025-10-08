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

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.RLConfig(run_name=run_name, max_seq_len=2048),
)
trainer.train()
