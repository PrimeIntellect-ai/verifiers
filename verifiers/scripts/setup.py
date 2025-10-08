import os

ENDPOINTS_CONTENT = """\
ENDPOINTS = {
    "gpt-4.1-nano": {
        "model": "gpt-4.1-nano",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1": {
        "model": "gpt-4.1",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5-nano": {
        "model": "gpt-4.1-nano",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5": {
        "model": "gpt-4.1",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
}
"""

ZERO3_CONTENT = """\
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

WORDLE_CONFIG_CONTENT = """\
model = "willcb/Qwen3-1.7B-Wordle"

[env]
id = "will/wordle"

[env.args]
use_think = true

[inference]
gpus = 6

[inference.args]
enforce_eager = true

[trainer]
gpus = 2

[trainer.args]
run_name = "wordle"
micro_batch_size = 8
rollouts_per_example = 16
batch_size = 512
max_steps = 500
max_tokens = 1024
max_seq_len = 4096
"""

GSM8K_CONFIG_CONTENT = """\
model = "willcb/Qwen3-0.6B"

[env]
id = "will/gsm8k"

[env.args]
use_think = true
num_eval_examples = 100

[inference]
gpus = 1

[inference.args]
enforce_eager = true
tensor_parallel_size = 1

[trainer]
gpus = 1
run_name = "gsm8k"

[trainer.args]
micro_batch_size = 12
rollouts_per_example = 12
batch_size = 144
max_steps = 100
eval_strategy = "steps"
eval_steps = 10
max_seq_len = 2048
"""


def main():
    os.makedirs("configs", exist_ok=True)
    os.makedirs("configs/rl", exist_ok=True)
    # create configs/endpoints.py if it doesn't exist
    if not os.path.exists("configs/endpoints.py"):
        with open("configs/endpoints.py", "w") as f:
            f.write(ENDPOINTS_CONTENT)
    else:
        print("configs/endpoints.py already exists")

    # create configs/zero3.yaml if it doesn't exist
    if not os.path.exists("configs/zero3.yaml"):
        # create it
        with open("configs/zero3.yaml", "w") as f:
            f.write(ZERO3_CONTENT)
    else:
        print("configs/zero3.yaml already exists")

    # create configs/rl/wordle.toml if it doesn't exist
    if not os.path.exists("configs/rl/wordle.toml"):
        with open("configs/rl/wordle.toml", "w") as f:
            f.write(WORDLE_CONFIG_CONTENT)
    else:
        print("configs/rl/wordle.toml already exists")


if __name__ == "__main__":
    main()
