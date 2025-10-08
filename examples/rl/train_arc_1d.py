import verifiers as vf

"""
# install
vf-install reasoning-gym (-p /path/to/environments)

# quick eval
vf-eval reasoning-gym (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen3-14B-Arc-1D-SFT \
    --tensor-parallel-size 2 --data-parallel-size 2 \
    --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 \
    --config-file configs/zero3.yaml examples/grpo/train_arc_1d.py
"""

size = "14B"
model_name = f"willcb/Qwen3-{size}-Arc-1D-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

env_id = "reasoning-gym"
env_args = {
    "gym": "arc_1d",
    "num_samples": 4000,
    "seed": 1,
}
vf_env = vf.load_environment(env_id=env_id, **env_args)


trainer = vf.RLTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.RLConfig(run_name="RG-arc1d", max_seq_len=4096),
)
trainer.train()
