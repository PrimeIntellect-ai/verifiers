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

ZERO3_CONTENT = """
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


def main():
    os.makedirs("configs", exist_ok=True)
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


if __name__ == "__main__":
    main()
