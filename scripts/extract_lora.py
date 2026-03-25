"""Extract LoRA adapter from prime-rl trainer checkpoint into PEFT format for vLLM.

Usage:
    python scripts/extract_lora.py <checkpoint_dir> <output_dir> [--base-model MODEL]

Example:
    python scripts/extract_lora.py ./checkpoints_l1/v1b_step_70/trainer ./adapters/l1_v1b
    python scripts/extract_lora.py ./checkpoints_l2/v1b_step_130/trainer ./adapters/l2_v1b
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def extract_lora_from_checkpoint(ckpt_dir: str, output_dir: str, base_model: str, rank: int):
    ckpt_path = Path(ckpt_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load trainer checkpoint
    rank_file = ckpt_path / "rank_0.pt"
    if not rank_file.exists():
        raise FileNotFoundError(f"No rank_0.pt found in {ckpt_path}")

    print(f"Loading checkpoint from {rank_file}...")
    state = torch.load(rank_file, map_location="cpu", weights_only=False)

    # Extract model state dict
    if "model" in state:
        model_state = state["model"]
    elif "state_dict" in state:
        model_state = state["state_dict"]
    else:
        model_state = state

    # Find LoRA keys and convert to PEFT format
    lora_state = {}
    for key, value in model_state.items():
        if "lora_A" in key or "lora_B" in key:
            # Clean up key: remove leading prefixes, add PEFT prefix
            clean = key
            for prefix in ["_fsdp_wrapped_module.", "_orig_mod.", "model."]:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
            # Ensure PEFT format: base_model.model.<path>
            if not clean.startswith("base_model.model."):
                clean = f"base_model.model.{clean}"
            lora_state[clean] = torch.empty_like(value).copy_(value)
            print(f"  {key} -> {clean} {list(value.shape)}")

    if not lora_state:
        print("ERROR: No LoRA weights found in checkpoint!")
        print(f"Available keys (first 20): {list(model_state.keys())[:20]}")
        return

    print(f"\nFound {len(lora_state)} LoRA tensors")

    # Save as safetensors
    save_file(lora_state, out_path / "adapter_model.safetensors")
    print(f"Saved adapter_model.safetensors")

    # Detect target modules from keys
    target_modules = set()
    for key in lora_state:
        parts = key.replace("base_model.model.", "").split(".")
        for i, part in enumerate(parts):
            if part in ("lora_A", "lora_B"):
                target_modules.add(parts[i - 1])
                break

    # Write adapter_config.json
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "fan_in_fan_out": False,
    }
    with open(out_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved adapter_config.json (target_modules={sorted(target_modules)})")
    print(f"\nDone! Adapter saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LoRA from prime-rl checkpoint")
    parser.add_argument("checkpoint_dir", help="Path to trainer checkpoint dir (containing rank_0.pt)")
    parser.add_argument("output_dir", help="Where to save the PEFT adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model name")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    args = parser.parse_args()
    extract_lora_from_checkpoint(args.checkpoint_dir, args.output_dir, args.base_model, args.rank)
