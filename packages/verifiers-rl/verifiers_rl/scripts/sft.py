"""
SFT training script following the pattern of train.py for RL.

Usage:
    vf-sft @ path/to/config.toml

Example config (configs/local/vf-sft/math-sft.toml):
    model = "Qwen/Qwen3-4B-Instruct"
    dataset = "willcb/V3-wordle"

    [sft]
    run_name = "math-sft-test"
    max_steps = 1000
    learning_rate = 2e-5
    batch_size = 512
    micro_batch_size = 8
    max_seq_len = 2048
    use_lora = true
    lora_rank = 8
"""

import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from datasets import load_dataset

import verifiers as vf
from verifiers_rl.rl.trainer import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    if args.at != "@":
        raise SystemExit("Usage: vf-sft @ path/to/file.toml")

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # Get model name
    model = config["model"]

    # Load dataset
    dataset_name = config.get("dataset")
    if dataset_name:
        dataset_split = config.get("sft", {}).get("dataset_split", "train")
        dataset = load_dataset(dataset_name, split=dataset_split)
    else:
        raise ValueError("Config must specify 'dataset'")

    # Create SFT config
    sft_config = SFTConfig(**config.get("sft", {}))

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
